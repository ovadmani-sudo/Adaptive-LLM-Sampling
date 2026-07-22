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
	"sync/atomic"
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
	// generationElapsedMs is time from the first content/reasoning token
	// arriving to the stream's end — i.e. real decode time, deliberately
	// excluding connection/queueing/prompt-processing (prefill) time before
	// any token appeared. This is what throughput tok/s must be divided by
	// (see logThroughput/updateThroughputStats) — using the total request
	// latency instead (as this proxy previously did) silently dilutes the
	// reported rate by however long prefill took, which for a large prompt
	// can be comparable to or longer than generation itself, and is exactly
	// why the persisted throughput report read roughly half of what the
	// live in-flight indicator showed for the same requests: the live
	// indicator already correctly used generation-only time (see
	// ProgressEvent.GenerationElapsedMs), but this figure did not.
	generationElapsedMs int64
}

// reasoningBudgetExceededFinishReason is an internal sentinel value stored
// transiently in streamResult.finishReason when postUpstreamChatStreaming's
// reasoning guard aborts a stuck generation early (see
// ReasoningGuardConfig) — never a real upstream finish_reason. It must
// never reach client-facing JSON: handleClassified (proxy.go) checks for it
// immediately upon reading the result and rewrites it to a real value
// before it can leak into buildChatCompletionJSON or lastGoodFinishReason.
const reasoningBudgetExceededFinishReason = "proxy_reasoning_budget_exceeded"

// reasoningChunkCap returns how many reasoning-content chunks
// postUpstreamChatStreaming should tolerate before treating the generation
// as a runaway loop, or 0 if the guard should not apply to this request at
// all (thinking_budget_tokens absent and no fallback configured). This is
// only ever an approximation of real tokens (one delta chunk is usually,
// but not guaranteed to be, one token) — good enough for a "this has gone
// on far, far longer than requested" guard, not meant to be exact.
func reasoningChunkCap(body map[string]interface{}, cfg *ReasoningGuardConfig) int {
	if budget := getInt(body, "thinking_budget_tokens", 0); budget > 0 {
		return int(float64(budget) * cfg.BudgetMultiplier)
	}
	return cfg.FallbackMaxReasoningTokens
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
// p.idleTimeout governs this call as a genuine INACTIVITY timeout, not an
// absolute wall-clock ceiling: it only fires if no progress at all (no
// response headers, no SSE line — not even a blank one) arrives for that
// long, and resets every time real progress is observed. A generation
// that's slow but steadily producing tokens can run indefinitely; only
// genuine silence this long gets cut off. idleCtx is a CHILD of ctx
// (created via context.WithCancel) used only for the outbound request —
// canceling it on idle-timeout must never mark ctx itself (ultimately
// r.Context()) as done, since handleClassified relies on r.Context().Err()
// staying nil to distinguish "our own idle-timeout fired" from "the
// downstream client actually disconnected".
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

	// model is the request's actual "model" field (post any provider
	// config/live-override/vision_describe override) — surfaced on every
	// ProgressEvent below so the live in-flight indicator shows which
	// model is doing the work, not just the completed-request log.
	model := getString(body, "model", "")

	if h := systemPromptHash(streamingBody); h != "" {
		log.Printf("system_prompt_hash egress (%s, attempt %d): %s", bucket, attempt, h)
	}

	reqBytes, err := json.Marshal(streamingBody)
	if err != nil {
		return streamResult{}, fmt.Errorf("encoding streaming request: %w", err)
	}

	// In forward-proxy mode (mitm.go), the destination and credential
	// aren't p.upstreamBase/p.provider.APIKey — there's no single
	// registered vendor for an arbitrary intercepted host. The override
	// carried on ctx (attached per-request by ForwardProxyServer) points
	// at whatever real host the client's CONNECT actually targeted, and
	// carries the client's own Authorization header (already correct,
	// since the client believes it's talking directly to that vendor).
	_, currentProvider, upstreamBase, _ := p.currentProvider()
	authHeader := ""
	if fc, ok := forwardProxyOverrideFromCtx(ctx); ok {
		upstreamBase = fc.upstreamBase
		authHeader = fc.authHeader
	}

	upstreamURL := upstreamBase + path
	if rawQuery != "" {
		upstreamURL += "?" + rawQuery
	}

	idleCtx, cancelIdle := context.WithCancel(ctx)
	defer cancelIdle()

	var lastProgressAt atomic.Int64 // UnixNano; touched from this goroutine and read/written from both ticker goroutines below
	lastProgressAt.Store(time.Now().UnixNano())
	touchProgress := func() {
		lastProgressAt.Store(time.Now().UnixNano())
	}
	// idleRemaining is how long until the idle timeout fires if no further
	// progress arrives — recomputed on every tick and surfaced on every
	// ProgressEvent so the dashboard can render a live countdown. -1 means
	// no idle timeout is configured (p.idleTimeout <= 0), distinct from 0
	// ("about to fire").
	idleRemaining := func() time.Duration {
		if p.idleTimeout <= 0 {
			return -1
		}
		remaining := p.idleTimeout - time.Since(time.Unix(0, lastProgressAt.Load()))
		if remaining < 0 {
			remaining = 0
		}
		return remaining
	}
	var timedOut atomic.Bool
	// checkIdle piggybacks on the existing 250ms progress tickers (both the
	// pre-headers wait ticker and the main read-loop ticker below) rather
	// than running a separate timer goroutine — the same cadence that
	// already drives dashboard updates is exactly what a countdown display
	// needs anyway.
	checkIdle := func() {
		if p.idleTimeout > 0 && time.Since(time.Unix(0, lastProgressAt.Load())) >= p.idleTimeout {
			timedOut.Store(true)
			cancelIdle()
		}
	}

	req, err := http.NewRequestWithContext(idleCtx, http.MethodPost, upstreamURL, strings.NewReader(string(reqBytes)))
	if err != nil {
		return streamResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if authHeader != "" {
		req.Header.Set("Authorization", authHeader)
	} else if currentProvider != nil && currentProvider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+currentProvider.APIKey)
	}

	start := time.Now()

	// doneEmitted guards against a double Done:true emission: the success
	// path at the end of this function emits a rich final event (real
	// token/elapsed counts) and sets this itself. The deferred fallback
	// below only fires if this function returns via any of its several
	// error paths (idle timeout, client disconnect, stream-read error, a
	// non-2xx status, ...) without ever reaching that point — every one of
	// those used to return with NO final progress event at all, leaving
	// the dashboard's in-flight indicator (cleared only on Done:true)
	// stuck forever showing stale progress for a request that has already
	// failed. Confirmed regression: this is exactly why the dashboard
	// could show a live "generating..."/countdown line indefinitely after
	// llama-server had already cancelled the task and nothing was
	// actually happening anymore.
	var doneEmitted bool
	defer func() {
		if !doneEmitted {
			p.emitProgress(ProgressEvent{
				Bucket:                 bucket,
				Attempt:                attempt,
				ElapsedMs:              time.Since(start).Milliseconds(),
				IdleTimeoutRemainingMs: -1,
				Model:                  model,
				Done:                   true,
			})
		}
	}()

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
	// show, even before upstream has sent a single byte back. This same
	// tick also drives the idle-timeout check for this phase — no
	// response headers at all within p.idleTimeout means genuine silence.
	waitDone := make(chan struct{})
	tickerStopped := make(chan struct{})
	go func() {
		defer close(tickerStopped)
		ticker := time.NewTicker(progressEmitInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				checkIdle()
				p.emitProgress(ProgressEvent{
					Bucket:                 bucket,
					Attempt:                attempt,
					ElapsedMs:              time.Since(start).Milliseconds(),
					IdleTimeoutRemainingMs: idleRemaining().Milliseconds(),
					Model:                  model,
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
		if timedOut.Load() {
			return streamResult{}, fmt.Errorf("no response from upstream within %s (idle timeout)", p.idleTimeout)
		}
		return streamResult{}, err
	}
	touchProgress() // headers arrived — real progress, resets the idle clock before the read loop starts
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

	// reasoningCap/reasoningChunks/reasoningGuardTripped implement
	// ReasoningGuard (see reasoningChunkCap's doc comment) — computed once
	// up front since thinking_budget_tokens doesn't change mid-stream.
	reasoningCap := 0
	if p.cfg.ReasoningGuard.Enabled {
		reasoningCap = reasoningChunkCap(body, &p.cfg.ReasoningGuard)
	}
	reasoningChunks := 0
	reasoningGuardTripped := false

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
			touchProgress() // any line at all (including blank keepalive lines) is real progress
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
						reasoningChunks++
						if reasoningCap > 0 && !reasoningGuardTripped && reasoningChunks > reasoningCap {
							reasoningGuardTripped = true
							log.Printf("reasoning-guard (%s, attempt %d): aborting after %d reasoning chunks (cap %d) with no real content yet — likely a runaway reasoning loop", bucket, attempt, reasoningChunks, reasoningCap)
							cancelIdle()
						}
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
			checkIdle()
			p.emitProgress(ProgressEvent{
				Bucket:                 bucket,
				Attempt:                attempt,
				ChunksReceived:         chunks,
				ApproxTokens:           int(float64(chunks) * multiplier),
				ElapsedMs:              time.Since(start).Milliseconds(),
				GenerationElapsedMs:    generationElapsedMs(),
				IdleTimeoutRemainingMs: idleRemaining().Milliseconds(),
				Model:                  model,
			})
		}
	}
	if err := <-scanDone; err != nil && !reasoningGuardTripped {
		if timedOut.Load() {
			return streamResult{}, fmt.Errorf("no data from upstream for %s (idle timeout)", p.idleTimeout)
		}
		if ctx.Err() != nil {
			// Downstream client disconnected; the upstream cancellation was
			// expected. Return the sentinel so the caller can exit silently
			// without logging a confusing "upstream unreachable" error or
			// trying to write a response to a gone client.
			return streamResult{}, ctx.Err()
		}
		return streamResult{}, fmt.Errorf("reading stream: %w", err)
	}
	if reasoningGuardTripped {
		// cancelIdle() above deliberately aborted the read mid-stream; this
		// is a successful (if truncated) result, not an error — handled as
		// IssueReasoningLoop by the retry loop in proxy.go, same as any
		// other detected issue.
		finishReason = reasoningBudgetExceededFinishReason
	}
	if !sawUsage && !reasoningGuardTripped {
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
	doneEmitted = true
	p.emitProgress(ProgressEvent{
		Bucket:                 bucket,
		Attempt:                attempt,
		ChunksReceived:         chunks,
		ApproxTokens:           finalTokens,
		ElapsedMs:              time.Since(start).Milliseconds(),
		GenerationElapsedMs:    generationElapsedMs(),
		IdleTimeoutRemainingMs: -1, // request is finished; no countdown applies anymore
		Model:                  model,
		Done:                   true,
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
		status:              resp.StatusCode,
		content:             contentB.String(),
		reasoningContent:    reasoningB.String(),
		finishReason:        finishReason,
		completionTokens:    completionTokens,
		promptTokens:        promptTokens,
		toolCalls:           toolCalls,
		generationElapsedMs: generationElapsedMs(),
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
