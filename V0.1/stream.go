package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
)

// streamResult accumulates everything gathered from an upstream streaming
// response — equivalent to what postUpstream + extractResponseContent +
// extractCompletionTokens used to produce from one blocking call, plus a
// raw error body for the non-2xx case (errors aren't SSE-shaped even when
// stream:true was requested, since they happen before generation starts).
type streamResult struct {
	status           int
	content          string
	reasoningContent string
	finishReason     string
	completionTokens int
	promptTokens     int
	toolCalls        []map[string]interface{}
	rawErrorBody     []byte
}

// toolCallAccumulator reassembles one OpenAI-style streamed tool call from
// its fragments. The first fragment for a given index carries id/type and
// the function name; every fragment (including the first) may carry a
// piece of function.arguments, which arrives incrementally as the model
// generates the JSON argument string and must be concatenated in order —
// each individual fragment is not valid JSON on its own.
type toolCallAccumulator struct {
	id   string
	typ  string
	name string
	args strings.Builder
}

// progressEmitInterval throttles how often postUpstreamChatStreaming pushes
// a ProgressEvent, so a fast stream (many small chunks/sec) doesn't flood
// the dashboard with more redraws than a terminal can usefully show.
const progressEmitInterval = 250 * time.Millisecond

// postUpstreamChatStreaming always requests stream:true (plus
// stream_options.include_usage, so the real completion_tokens count is
// still available for the truncation-retry baseline) from upstream,
// regardless of what the client itself asked for — mirroring how the old
// blocking path always forced stream:false. This lets the proxy observe
// generation as it happens: emitting live progress (chunks received /
// elapsed time / approximate token count, throttled to
// progressEmitInterval) so a stalled remote request is visibly
// distinguishable from one that's actively working, which a single
// blocking call could never show — that's the whole point, since a local
// llama.cpp console gives you exactly this kind of visibility for free and
// a remote provider otherwise gives you none.
//
// body is read-only here; a shallow copy gets the stream fields added so
// the caller's map (reused across retries, and read for dashboard display)
// is never mutated by this call.
func (p *ProxyServer) postUpstreamChatStreaming(ctx context.Context, path, rawQuery string, body map[string]interface{}, bucket TaskBucket, attempt int) (streamResult, error) {
	streamingBody := make(map[string]interface{}, len(body)+2)
	for k, v := range body {
		streamingBody[k] = v
	}
	streamingBody["stream"] = true
	streamingBody["stream_options"] = map[string]interface{}{"include_usage": true}

	reqBytes, err := json.Marshal(streamingBody)
	if err != nil {
		return streamResult{}, fmt.Errorf("encoding streaming request: %w", err)
	}

	upstreamURL := p.upstreamBase + path
	if rawQuery != "" {
		upstreamURL += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, strings.NewReader(string(reqBytes)))
	if err != nil {
		return streamResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if p.provider != nil && p.provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.provider.APIKey)
	}

	start := time.Now()

	// p.client.Do blocks until upstream sends response headers — for a
	// provider that queues a request before generation starts (e.g.
	// OpenRouter free-tier rate-limiting), that wait can be most of the
	// total latency, and none of it happened inside the SSE read loop
	// below. Without this ticker, that whole phase emitted zero progress
	// events, so the dashboard showed "idle" the entire time even though a
	// request genuinely was in flight — indistinguishable from no request
	// running at all, which defeats the point of this feature. Emitting a
	// periodic "still waiting, N seconds elapsed, 0 tokens so far" event
	// here means the elapsed timer visibly keeps ticking during that
	// phase, which is the actual signal of life the dashboard needs to
	// show, even before upstream has sent a single byte back.
	waitDone := make(chan struct{})
	tickerStopped := make(chan struct{})
	go func() {
		defer close(tickerStopped)
		ticker := time.NewTicker(progressEmitInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				p.emitProgress(ProgressEvent{
					Bucket:    bucket,
					Attempt:   attempt,
					ElapsedMs: time.Since(start).Milliseconds(),
				})
			case <-waitDone:
				return
			}
		}
	}()

	resp, err := p.client.Do(req)
	close(waitDone)
	<-tickerStopped // don't proceed until the ticker goroutine has fully exited, so it can never still be mid-emitProgress after this function returns
	if err != nil {
		return streamResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		errBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return streamResult{}, readErr
		}
		return streamResult{status: resp.StatusCode, rawErrorBody: errBody}, nil
	}

	chunks := 0
	// multiplier corrects the chunks-based estimate below for providers
	// that batch more than one token per SSE delta (see
	// ProviderConfig.TokensSecMultiplier) — a no-op 1.0 unless the user has
	// manually tuned it for this provider.
	multiplier := p.tokensSecMultiplier()

	var contentB, reasoningB strings.Builder
	finishReason := ""
	completionTokens := 0
	promptTokens := 0
	sawUsage := false
	toolCallAccs := make(map[int]*toolCallAccumulator)
	// genStart marks when the first actual content/reasoning token
	// arrives — kept separate from start (request-sent time) specifically
	// so the tok/s rate isn't dragged down by connection/queueing/prompt-
	// processing time that has nothing to do with decode speed.
	var genStart time.Time
	generationElapsedMs := func() int64 {
		if genStart.IsZero() {
			return 0
		}
		return time.Since(genStart).Milliseconds()
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024) // some providers send large individual chunks

	// The waiting-ticker above only covers the time until p.client.Do
	// returns, i.e. until response headers arrive — it says nothing about
	// the time between headers and the first (or next) line of the body.
	// Some upstreams flush headers immediately but then delay the actual
	// first SSE data line while evaluating the prompt (prefill), which for
	// a large local prompt can be tens of seconds; scanner.Scan() blocks
	// synchronously for that whole gap with nothing else running, which
	// would silently reproduce the exact "shows idle/stale while a request
	// is genuinely in flight" bug the wait-ticker above was built to fix —
	// just one layer deeper. Reading lines on a separate goroutine lets
	// this select loop interleave a periodic ticker tick (keeping the
	// dashboard's elapsed timer visibly ticking, the same "processing
	// prompt" state as the pre-headers wait) with whatever data actually
	// arrives, however long the gap between lines turns out to be.
	//
	// The reader goroutine always runs to completion (full EOF or a real
	// error) rather than being cancelled early: even after this loop sees
	// "[DONE]" and stops caring about further lines, it keeps draining the
	// channel until the reader closes it, so scanner.Err() below reflects
	// the real terminal state exactly as it did before this loop existed,
	// and the reader can never block forever on a send nobody's receiving.
	lines := make(chan string)
	scanDone := make(chan error, 1)
	go func() {
		defer close(lines)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
		scanDone <- scanner.Err()
	}()

	ticker := time.NewTicker(progressEmitInterval)
	defer ticker.Stop()

	done := false
readLoop:
	for {
		select {
		case line, ok := <-lines:
			if !ok {
				break readLoop
			}
			if done {
				continue // draining post-"[DONE]" lines only to let the reader goroutine finish cleanly
			}
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				done = true
				continue
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue // skip a malformed line rather than aborting the whole stream
			}

			if usage, ok := chunk["usage"].(map[string]interface{}); ok {
				sawUsage = true
				completionTokens = getInt(usage, "completion_tokens", completionTokens)
				promptTokens = getInt(usage, "prompt_tokens", promptTokens)
				if _, hasPrompt := usage["prompt_tokens"]; !hasPrompt {
					log.Printf("stream (%s, attempt %d): usage object present but has no prompt_tokens field; raw usage: %v", bucket, attempt, usage)
				}
			}

			if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
				choice0, _ := choices[0].(map[string]interface{})
				if fr, ok := choice0["finish_reason"].(string); ok && fr != "" {
					finishReason = fr
				}
				if delta, ok := choice0["delta"].(map[string]interface{}); ok {
					if c, ok := delta["content"].(string); ok && c != "" {
						if genStart.IsZero() {
							genStart = time.Now()
						}
						contentB.WriteString(c)
						chunks++
					}
					if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
						if genStart.IsZero() {
							genStart = time.Now()
						}
						reasoningB.WriteString(rc)
						chunks++
					}
					if rawTCs, ok := delta["tool_calls"].([]interface{}); ok {
						for _, rawTC := range rawTCs {
							tcMap, ok := rawTC.(map[string]interface{})
							if !ok {
								continue
							}
							idx := getInt(tcMap, "index", 0)
							acc, exists := toolCallAccs[idx]
							if !exists {
								acc = &toolCallAccumulator{}
								toolCallAccs[idx] = acc
							}
							if id, ok := tcMap["id"].(string); ok && id != "" {
								acc.id = id
							}
							if typ, ok := tcMap["type"].(string); ok && typ != "" {
								acc.typ = typ
							}
							if fn, ok := tcMap["function"].(map[string]interface{}); ok {
								if name, ok := fn["name"].(string); ok && name != "" {
									acc.name = name
								}
								if args, ok := fn["arguments"].(string); ok && args != "" {
									acc.args.WriteString(args)
								}
							}
							if genStart.IsZero() {
								genStart = time.Now()
							}
							chunks++
						}
					}
				}
			}

		case <-ticker.C:
			p.emitProgress(ProgressEvent{
				Bucket:              bucket,
				Attempt:             attempt,
				ChunksReceived:      chunks,
				ApproxTokens:        int(float64(chunks) * multiplier),
				ElapsedMs:           time.Since(start).Milliseconds(),
				GenerationElapsedMs: generationElapsedMs(),
			})
		}
	}
	if err := <-scanDone; err != nil {
		if ctx.Err() != nil {
			// Downstream client disconnected; the upstream cancellation was
			// expected. Return the sentinel so the caller can exit silently
			// without logging a confusing "upstream unreachable" error or
			// trying to write a response to a gone client.
			return streamResult{}, ctx.Err()
		}
		return streamResult{}, fmt.Errorf("reading stream: %w", err)
	}
	if !sawUsage {
		log.Printf("stream (%s, attempt %d): upstream never sent a usage object despite stream_options.include_usage being requested — prompt/completion token counts will show as 0", bucket, attempt)
	}

	// The final progress event can use the real completion_tokens count
	// (from the usage chunk, already parsed above) instead of the
	// chunks-received approximation used for every event up to this
	// point — it's strictly more accurate, and by now the stream is done
	// so there's no "estimate before we have real data" reason not to.
	// The multiplier only applies to the chunks fallback: real usage data
	// needs no correction, only the chunk-count estimate does.
	finalTokens := completionTokens
	if finalTokens == 0 {
		finalTokens = int(float64(chunks) * multiplier)
	}
	p.emitProgress(ProgressEvent{
		Bucket:              bucket,
		Attempt:             attempt,
		ChunksReceived:      chunks,
		ApproxTokens:        finalTokens,
		ElapsedMs:           time.Since(start).Milliseconds(),
		GenerationElapsedMs: generationElapsedMs(),
		Done:                true,
	})

	var toolCalls []map[string]interface{}
	if len(toolCallAccs) > 0 {
		indices := make([]int, 0, len(toolCallAccs))
		for idx := range toolCallAccs {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		toolCalls = make([]map[string]interface{}, 0, len(indices))
		for _, idx := range indices {
			acc := toolCallAccs[idx]
			typ := acc.typ
			if typ == "" {
				typ = "function"
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   acc.id,
				"type": typ,
				"function": map[string]interface{}{
					"name":      acc.name,
					"arguments": acc.args.String(),
				},
			})
		}
	}

	return streamResult{
		status:           resp.StatusCode,
		content:          contentB.String(),
		reasoningContent: reasoningB.String(),
		finishReason:     finishReason,
		completionTokens: completionTokens,
		promptTokens:     promptTokens,
		toolCalls:        toolCalls,
	}, nil
}

// buildChatCompletionJSON synthesizes a standard, complete chat.completion
// JSON object from accumulated streaming deltas, for clients that didn't
// request streaming themselves. Needed because upstream is now always
// asked to stream (postUpstreamChatStreaming) regardless of what the
// client wants, so there's no longer a single raw upstream response body
// to just pass through for a non-streaming client — this reconstructs the
// equivalent of what a non-streaming upstream call would have returned.
func buildChatCompletionJSON(content, reasoningContent, finishReason string, promptTokens, completionTokens int, toolCalls []map[string]interface{}) []byte {
	message := map[string]interface{}{"role": "assistant", "content": content}
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	if finishReason == "" {
		finishReason = "stop"
	}
	resp := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-proxy-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"choices": []map[string]interface{}{
			{"index": 0, "message": message, "finish_reason": finishReason},
		},
		// prompt_tokens/total_tokens matter beyond bookkeeping: Cline (and
		// most OpenAI-compatible clients) reads usage.prompt_tokens off
		// every response to track how full the context window is and
		// decide when to auto-compact. Omitting it (as this used to)
		// silently reads as 0% utilization forever — auto-compact never
		// fires, and the request eventually fails outright once the real
		// upstream context fills up.
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}
