package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

type endpointKind int

const (
	kindChat endpointKind = iota
	kindCompletion
)

// ProxyServer holds the shared state for the reverse proxy: config, the
// upstream target, an HTTP client, and the channel used to push dashboard
// updates without blocking request handling. When provider is non-nil, the
// upstream target is a remote OpenAI-compatible backend (claude/gemini/
// openai/openrouter) selected via CLI argument instead of the local
// llama-server.
type ProxyServer struct {
	cfg           *Config
	provider      *ProviderConfig
	providerLabel string
	upstreamBase  string
	client        *http.Client
	events        chan<- UIEvent
	progress      chan<- ProgressEvent
	passthrough   *httputil.ReverseProxy

	retryLogMu   sync.Mutex
	retryLogFile *os.File
}

// NewProxyServer wires up the proxy. retryLogPath, if non-empty, is where
// every request that retried at least once gets its full adjustment
// trajectory appended as a JSON line (see logRetryTrajectory) — opened
// once here in append mode and written to under a mutex, since requests
// are handled concurrently. An empty path disables trajectory logging
// entirely (used by tests that don't want a file on disk).
func NewProxyServer(cfg *Config, provider *ProviderConfig, providerLabel string, events chan<- UIEvent, progressCh chan<- ProgressEvent, retryLogPath string) (*ProxyServer, error) {
	p := &ProxyServer{
		cfg:           cfg,
		provider:      provider,
		providerLabel: providerLabel,
		client: &http.Client{
			Timeout: time.Duration(cfg.Server.RequestTimeoutSeconds) * time.Second,
		},
		events:   events,
		progress: progressCh,
	}

	if retryLogPath != "" {
		f, err := os.OpenFile(retryLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("opening retry log: %w", err)
		}
		p.retryLogFile = f
	}

	if provider != nil {
		p.upstreamBase = provider.BaseURL
		// Passthrough is deliberately unsupported in remote-provider mode:
		// each provider mounts its OpenAI-compatible surface at a different
		// prefix (see [provider.*] comments in config.ini), so naively
		// joining an incoming path onto base_url isn't reliable beyond the
		// one endpoint (/v1/chat/completions) we've verified per provider.
		return p, nil
	}

	p.upstreamBase = fmt.Sprintf("http://%s:%d", cfg.Server.UpstreamHost, cfg.Server.UpstreamPort)
	target, err := url.Parse(p.upstreamBase)
	if err != nil {
		return nil, fmt.Errorf("invalid upstream address: %w", err)
	}

	rp := httputil.NewSingleHostReverseProxy(target)
	origErrHandler := rp.ErrorHandler
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("passthrough error for %s: %v", r.URL.Path, err)
		if origErrHandler != nil {
			origErrHandler(w, r, err)
			return
		}
		w.WriteHeader(http.StatusBadGateway)
	}
	p.passthrough = rp

	return p, nil
}

// tokensSecMultiplier returns the correction factor applied to the live
// chunk-count-based tok/s estimate (see ProviderConfig.TokensSecMultiplier).
// Local llama-server has no ProviderConfig at all (p.provider is nil) and
// its one-delta-per-token behavior is already verified against source, so
// it always gets the no-op 1.0 rather than needing an entry in config.ini.
// A zero value (e.g. a ProviderConfig built as a struct literal that
// doesn't set this field, as several tests do) is treated the same as
// "unset" rather than a real 0x multiplier — 0 would always report 0
// tokens, which is never a sane setting, only a missing one.
func (p *ProxyServer) tokensSecMultiplier() float64 {
	if p.provider == nil || p.provider.TokensSecMultiplier == 0 {
		return 1.0
	}
	return p.provider.TokensSecMultiplier
}

// Close releases the retry log file handle, if one was opened. Safe to
// call even when trajectory logging is disabled.
func (p *ProxyServer) Close() error {
	if p.retryLogFile == nil {
		return nil
	}
	return p.retryLogFile.Close()
}

func (p *ProxyServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", p.handleClassified(kindChat))
	mux.HandleFunc("/v1/models", p.handleModels)

	if p.provider != nil {
		// Some clients (Page Assist, unlike Cline) GET the bare base URL
		// as a reachability check before calling /v1/models. There's no
		// real upstream endpoint to proxy this to — it's not a documented
		// part of any provider's API — so answer it directly rather than
		// rejecting a harmless probe. Deliberately NOT a subtree pattern
		// ("/v1/"): any other /v1/* path we don't explicitly support
		// should still hit the clear 501 below rather than a misleading
		// fake-success empty response.
		mux.HandleFunc("/v1", p.handleRootProbe)
		mux.HandleFunc("/completion", p.rejectRemoteOnly)
		mux.HandleFunc("/", p.rejectRemoteOnly)
		return withCORS(mux)
	}

	mux.HandleFunc("/completion", p.handleClassified(kindCompletion))
	mux.HandleFunc("/", p.passthrough.ServeHTTP)
	return withCORS(mux)
}

// withCORS adds permissive CORS headers to every response and answers
// preflight OPTIONS requests directly. Cline runs in the VS Code extension
// host and is never subject to CORS, but browser-based clients (e.g. Page
// Assist) are: without Access-Control-Allow-Origin, the browser silently
// discards a perfectly successful response before the client's JS ever
// sees it — the request succeeds at the network level but looks like an
// empty/broken response to the client. This is a local, single-user dev
// proxy, so a permissive "*" origin carries no real multi-tenant risk.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// handleRootProbe answers a bare GET /v1 (or /v1/) with 200 OK, for
// clients that ping the base URL itself before calling a real endpoint.
// Not a real provider endpoint — just enough to satisfy a reachability
// check without misrepresenting anything as a genuine API response.
func (p *ProxyServer) handleRootProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"object":"list","data":[]}`))
}

// handleModels proxies the client's model-list request (Cline calls this
// on startup to populate its model picker). In local mode this is just the
// existing passthrough. In remote-provider mode, GET <base_url>/models is
// a well-documented endpoint on OpenAI, OpenRouter, and Anthropic's
// OpenAI-compat layer; Gemini's compat layer is expected to support it too
// but wasn't independently verified the way the chat-completions path was
// — worth checking if model listing fails specifically on gemini.
func (p *ProxyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	if p.provider == nil {
		p.passthrough.ServeHTTP(w, r)
		return
	}

	req, err := http.NewRequest(r.Method, p.upstreamBase+"/models", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if p.provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.provider.APIKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		log.Printf("models request to %q failed: %v", p.providerLabel, err)
		http.Error(w, fmt.Sprintf("upstream unreachable: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if resp.StatusCode >= 300 {
		log.Printf("models request to %q returned %d: %s", p.providerLabel, resp.StatusCode, truncateForLog(respBytes))
	}
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	w.Write(respBytes)
}

// rejectRemoteOnly responds clearly to any path other than
// /v1/chat/completions and /v1/models when running against a remote
// provider backend, rather than silently mis-routing (e.g. llama.cpp's
// native /completion has no equivalent on these APIs; generic passthrough
// can't be mapped reliably across providers with different URL prefixes).
// Logged to stdout so a rejected request is diagnosable even though it
// never reaches the dashboard's request log (that's reserved for the
// classify/inject/detect/retry cycle on the chat-completions route).
func (p *ProxyServer) rejectRemoteOnly(w http.ResponseWriter, r *http.Request) {
	log.Printf("rejected %s %s: not available against remote provider %q", r.Method, r.URL.Path, p.providerLabel)
	http.Error(w, fmt.Sprintf(
		"path %q is not available when running against remote provider %q; only /v1/chat/completions and /v1/models are supported in this mode",
		r.URL.Path, p.providerLabel,
	), http.StatusNotImplemented)
}

func (p *ProxyServer) emit(ev UIEvent) {
	select {
	case p.events <- ev:
	default:
		// dashboard is behind; drop rather than block request handling
	}
}

func (p *ProxyServer) emitProgress(ev ProgressEvent) {
	if p.progress == nil {
		return
	}
	select {
	case p.progress <- ev:
	default:
		// dashboard is behind; drop rather than block the stream read loop
	}
}

func (p *ProxyServer) handleClassified(kind endpointKind) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		r.Body.Close()

		path := p.chatPath(kind)

		var body map[string]interface{}
		if err := json.Unmarshal(rawBody, &body); err != nil {
			// Not a JSON body we understand; pass through unmodified.
			p.forwardRaw(w, r, rawBody, path)
			return
		}

		clientWantsStream, _ := body["stream"].(bool)

		classificationText := extractClassificationText(body, kind)
		bucket := Classify(&p.cfg.Classification, classificationText)
		applyPreset(body, p.cfg.Presets[bucket])

		// A model name valid for the local llama-server almost certainly
		// isn't valid on a remote provider, so a configured provider model
		// always overrides whatever the client requested. If the provider
		// has no model configured, the client's request passes through
		// unchanged.
		if p.provider != nil && p.provider.Model != "" {
			body["model"] = p.provider.Model
		}

		// For kindChat, upstream is always asked to stream internally now
		// (see postUpstreamChatStreaming), regardless of what the client
		// itself wants — this is what makes live progress tracking
		// possible. kindCompletion (llama.cpp's native endpoint, local-only
		// — remote providers never register this route) keeps the original
		// single blocking call; it's a low-traffic legacy path and local
		// llama.cpp already gives console-level progress visibility for it.
		if kind != kindChat {
			body["stream"] = false
		}

		maxRetries := p.cfg.Server.MaxRetries
		var (
			respBytes        []byte
			respStatus       int
			finalIssue       Issue
			content          string
			reasoningContent string
			finishReason     string
			toolCalls        []map[string]interface{}
			retryCount       int
			// finalPromptTokens/finalCompletionTokens are the real counts
			// from upstream's usage object (kindChat only), kept from
			// whichever attempt was last — reported on the completed-request
			// UIEvent, not the transient live indicator.
			finalPromptTokens     int
			finalCompletionTokens int
			// streamable is only true once a 2xx response was successfully
			// parsed into usable content/finish_reason. If the upstream
			// response was an error or an unexpected shape, we must forward
			// it as-is rather than synthesizing a fake SSE stream around it.
			streamable  bool
			upstreamErr string
			adjustments []RetryAdjustment
		)

		for attempt := 0; ; attempt++ {
			var completionTokens int

			if kind == kindChat {
				result, sErr := p.postUpstreamChatStreaming(r.Context(), path, r.URL.RawQuery, body, bucket, attempt)
				if sErr != nil {
					if errors.Is(sErr, context.Canceled) || errors.Is(sErr, context.DeadlineExceeded) {
						return // downstream client disconnected; nothing to send
					}
					p.emit(UIEvent{
						Timestamp:  time.Now(),
						Bucket:     bucket,
						RetryCount: attempt,
						Error:      sErr.Error(),
					})
					http.Error(w, fmt.Sprintf("upstream unreachable: %v", sErr), http.StatusBadGateway)
					return
				}

				respStatus = result.status
				if respStatus >= 300 {
					respBytes = result.rawErrorBody
					finalIssue = IssueNone
					upstreamErr = fmt.Sprintf("upstream returned %d: %s", respStatus, truncateForLog(respBytes))
					break
				}

				content = result.content
				reasoningContent = result.reasoningContent
				finishReason = result.finishReason
				completionTokens = result.completionTokens
				toolCalls = result.toolCalls
				streamable = true
				respBytes = buildChatCompletionJSON(content, reasoningContent, finishReason, result.promptTokens, completionTokens, toolCalls)

				finalPromptTokens = result.promptTokens
				finalCompletionTokens = result.completionTokens
			} else {
				reqBytes, err := json.Marshal(body)
				if err != nil {
					http.Error(w, "failed to encode upstream request", http.StatusInternalServerError)
					return
				}

				var lastErr error
				respStatus, respBytes, lastErr = p.postUpstream(r.Context(), path, r.URL.RawQuery, reqBytes)
				if lastErr != nil {
					p.emit(UIEvent{
						Timestamp:  time.Now(),
						Bucket:     bucket,
						RetryCount: attempt,
						Error:      lastErr.Error(),
					})
					http.Error(w, fmt.Sprintf("upstream unreachable: %v", lastErr), http.StatusBadGateway)
					return
				}

				if respStatus >= 300 {
					// Upstream itself rejected the request (bad params, etc.);
					// nothing to inspect or retry. Forward its error verbatim.
					finalIssue = IssueNone
					upstreamErr = fmt.Sprintf("upstream returned %d: %s", respStatus, truncateForLog(respBytes))
					break
				}

				var respBody map[string]interface{}
				if err := json.Unmarshal(respBytes, &respBody); err != nil {
					// Can't inspect it; pass through as-is.
					finalIssue = IssueNone
					upstreamErr = fmt.Sprintf("upstream response not valid JSON: %v", err)
					break
				}

				var ok bool
				content, reasoningContent, finishReason, ok = extractResponseContent(respBody, kind)
				if !ok {
					finalIssue = IssueNone
					upstreamErr = "upstream response missing expected choices/content shape"
					break
				}
				streamable = true
				completionTokens = extractCompletionTokens(respBody, kind)
			}

			var issueDetail string
			finalIssue, issueDetail = DetectIssue(&p.cfg.Detection, content, finishReason)
			if finalIssue == IssueNone || attempt >= maxRetries {
				break
			}
			if issueDetail != "" {
				// Goes to proxy_<backend>.log — the first thing to check
				// when retries seem too frequent: if the logged n-gram is
				// legitimate structure (XML tool tags, file paths) rather
				// than degenerate looping, the detector is false-positiving
				// and [detection] needs tuning, not the sampling params.
				log.Printf("detect (%s, attempt %d): %s triggered by repeated n-gram: %q", bucket, attempt, finalIssue, issueDetail)
			}

			newAdjustments := adjustForIssue(body, finalIssue, p.cfg, kind, attempt, completionTokens)
			for i := range newAdjustments {
				newAdjustments[i].Detail = issueDetail
			}
			adjustments = append(adjustments, newAdjustments...)
			retryCount = attempt + 1
		}

		latency := time.Since(start).Milliseconds()

		if len(adjustments) > 0 {
			p.logRetryTrajectory(bucket, adjustments, streamable && finalIssue == IssueNone, retryCount, latency)
		}

		temp, topP, topK, penalty, budget := effectiveDynamicValues(body)
		p.emit(UIEvent{
			Timestamp:            time.Now(),
			Bucket:               bucket,
			RetryCount:           retryCount,
			Issue:                finalIssue,
			LatencyMs:            latency,
			Temperature:          temp,
			TopP:                 topP,
			TopK:                 topK,
			RepeatPenalty:        penalty,
			ThinkingBudgetTokens: budget,
			PromptTokens:         finalPromptTokens,
			CompletionTokens:     finalCompletionTokens,
			Error:                upstreamErr,
		})

		p.writeResponse(w, respStatus, respBytes, content, reasoningContent, finishReason, toolCalls, finalPromptTokens, finalCompletionTokens, clientWantsStream && streamable, kind)
	}
}

// chatPath returns the outbound request path for the given endpoint kind.
// In local mode this mirrors the incoming route exactly (llama-server
// mounts its OAI-compatible surface at /v1/chat/completions and its native
// completion endpoint at /completion, off a bare host:port base with no
// path of its own). In remote-provider mode, kind is always kindChat (the
// only route registered against a provider — see Handler), and the path is
// always exactly "/chat/completions": base_url already encodes each
// provider's own mount prefix up to that point, verified per provider in
// config.ini, so the incoming request's own path is irrelevant here.
func (p *ProxyServer) chatPath(kind endpointKind) string {
	if p.provider != nil {
		return "/chat/completions"
	}
	if kind == kindChat {
		return "/v1/chat/completions"
	}
	return "/completion"
}

// forwardRaw ships a request body upstream unmodified when it can't be
// parsed as JSON, still going through the same client/timeout plumbing.
func (p *ProxyServer) forwardRaw(w http.ResponseWriter, r *http.Request, rawBody []byte, path string) {
	status, respBytes, err := p.postUpstream(r.Context(), path, r.URL.RawQuery, rawBody)
	if err != nil {
		http.Error(w, fmt.Sprintf("upstream unreachable: %v", err), http.StatusBadGateway)
		return
	}
	w.WriteHeader(status)
	w.Write(respBytes)
}

func (p *ProxyServer) postUpstream(ctx context.Context, path string, rawQuery string, body []byte) (int, []byte, error) {
	upstreamURL := p.upstreamBase + path
	if rawQuery != "" {
		upstreamURL += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.provider != nil && p.provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.provider.APIKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, err
	}
	return resp.StatusCode, respBytes, nil
}

// writeResponse sends the final validated response to the client, either
// as a plain JSON body or, if the original client request asked for
// streaming, re-chunked as SSE events.
func (p *ProxyServer) writeResponse(w http.ResponseWriter, status int, respBytes []byte, content, reasoningContent, finishReason string, toolCalls []map[string]interface{}, promptTokens, completionTokens int, clientWantsStream bool, kind endpointKind) {
	if !clientWantsStream {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(respBytes)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(status)
	flusher, _ := w.(http.Flusher)

	if kind == kindChat {
		streamChatSSE(w, flusher, content, reasoningContent, finishReason, toolCalls, promptTokens, completionTokens)
	} else {
		streamCompletionSSE(w, flusher, content, finishReason)
	}
}

// sseChunkRunes controls how many runes each synthesized SSE delta carries
// when re-chunking an already-complete response for a streaming client.
// Chosen to land in roughly the same delta size as the previous
// word-based chunking, without any of its data loss (see chunkText).
const sseChunkRunes = 20

// streamChatSSE re-chunks an already-complete chat response into an
// OpenAI-style SSE stream so streaming clients (e.g. Cline) see incremental
// deltas instead of a long silent wait followed by one huge event.
// reasoning_content (the model's thinking/reasoning trace, kept distinct
// from content by llama-server per the DeepSeek-style convention Cline
// already understands) is sent as its own delta chunk before the content
// deltas, mirroring generation order — dropping it here previously meant
// any UI that renders reasoning/plan output separately from the final
// answer (e.g. Cline's Plan Mode) silently lost that formatting whenever
// the client asked for streaming.
//
// toolCalls (already fully reassembled from upstream's fragmented deltas
// by postUpstreamChatStreaming) is sent as one complete delta chunk right
// before the finish_reason chunk. This isn't token-by-token incremental
// the way upstream's own stream was, but it's the same trade-off already
// made for content via chunkText: a client that accumulates delta pieces
// by index (which is what the tool_calls streaming format requires
// clients to do anyway) ends up with the identical final tool_calls array
// either way.
//
// A final usage-only chunk (empty choices, populated usage — the same
// shape upstream itself sends because every request forces
// stream_options.include_usage) is emitted right before [DONE]. Without
// it, a streaming client has no way to learn prompt_tokens/total_tokens at
// all: Cline reads usage.prompt_tokens off every response to track how
// full the model's context window is and decide when to auto-compact.
// Silently omitting it here (as this used to) reads as permanent 0%
// context utilization, so auto-compact never fires — the request just
// keeps growing until it fails outright once the real upstream context
// fills up.
func streamChatSSE(w io.Writer, flusher http.Flusher, content, reasoningContent, finishReason string, toolCalls []map[string]interface{}, promptTokens, completionTokens int) {
	id := fmt.Sprintf("chatcmpl-proxy-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(delta map[string]interface{}, finish interface{}) {
		chunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"choices": []map[string]interface{}{
				{"index": 0, "delta": delta, "finish_reason": finish},
			},
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}

	writeChunk(map[string]interface{}{"role": "assistant", "content": ""}, nil)

	if reasoningContent != "" {
		writeChunk(map[string]interface{}{"reasoning_content": reasoningContent}, nil)
	}

	for _, piece := range chunkText(content, sseChunkRunes) {
		writeChunk(map[string]interface{}{"content": piece}, nil)
	}

	if len(toolCalls) > 0 {
		writeChunk(map[string]interface{}{"tool_calls": toolCalls}, nil)
	}

	if finishReason == "" {
		finishReason = "stop"
	}
	writeChunk(map[string]interface{}{}, finishReason)

	usageChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"choices": []interface{}{},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(usageChunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// streamCompletionSSE re-chunks a completed native /completion response
// into llama-server's own streaming event shape.
func streamCompletionSSE(w io.Writer, flusher http.Flusher, content, finishReason string) {
	for _, piece := range chunkText(content, sseChunkRunes) {
		event := map[string]interface{}{"content": piece, "stop": false}
		b, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	final := map[string]interface{}{"content": "", "stop": true, "truncated": finishReason == "length"}
	b, _ := json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}
}

// chunkText splits text into contiguous rune-count-sized pieces with zero
// loss: concatenating every returned piece reproduces the original string
// exactly, including all whitespace, indentation, and line breaks.
//
// The previous implementation (chunkWords) split on strings.Fields — any
// run of whitespace — and rejoined pieces with a single ASCII space. That
// silently destroyed every newline, every indentation level, and every
// multi-space run in the original text: a multi-line code block came back
// as one flattened line with no indentation, for any client that requests
// streaming (Cline does, by default). A non-streaming request to the same
// backend looked completely fine, which is what made this easy to miss —
// the corruption only showed up in the SSE re-chunking path, not in the
// content itself. Slicing by rune (not byte) index avoids splitting a
// multi-byte UTF-8 character across two chunks.
func chunkText(text string, n int) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var out []string
	for i := 0; i < len(runes); i += n {
		end := i + n
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}

func extractClassificationText(body map[string]interface{}, kind endpointKind) string {
	if kind == kindChat {
		return extractLastUserMessage(body)
	}
	if prompt, ok := body["prompt"].(string); ok {
		return prompt
	}
	return ""
}

// extractResponseContent pulls the generated text, reasoning/thinking
// trace, and finish reason out of an upstream response body, whether it
// came from the OAI-compatible chat endpoint or llama-server's native
// /completion endpoint. reasoningContent is always "" for /completion,
// which has no equivalent concept.
func extractResponseContent(respBody map[string]interface{}, kind endpointKind) (content, reasoningContent, finishReason string, ok bool) {
	if kind == kindChat {
		choices, has := respBody["choices"].([]interface{})
		if !has || len(choices) == 0 {
			return "", "", "", false
		}
		choice0, has := choices[0].(map[string]interface{})
		if !has {
			return "", "", "", false
		}
		finishReason, _ = choice0["finish_reason"].(string)
		message, has := choice0["message"].(map[string]interface{})
		if !has {
			return "", "", "", false
		}
		content, _ = message["content"].(string)
		reasoningContent, _ = message["reasoning_content"].(string)
		return content, reasoningContent, finishReason, true
	}

	content, has := respBody["content"].(string)
	if !has {
		return "", "", "", false
	}
	truncated, _ := respBody["truncated"].(bool)
	finishReason = "stop"
	if truncated {
		finishReason = "length"
	}
	return content, "", finishReason, true
}

// extractCompletionTokens reads how many tokens the model actually
// generated in this response — usage.completion_tokens for the OAI chat
// endpoint, tokens_predicted for llama.cpp's native /completion — so a
// truncation retry can escalate from the real length that got cut off
// instead of a guess. Returns 0 if the field is missing, so callers know
// to fall back to whatever's already in the request body.
func extractCompletionTokens(respBody map[string]interface{}, kind endpointKind) int {
	if kind == kindChat {
		usage, ok := respBody["usage"].(map[string]interface{})
		if !ok {
			return 0
		}
		return getInt(usage, "completion_tokens", 0)
	}
	return getInt(respBody, "tokens_predicted", 0)
}

// applyPreset fills in only the fields the client did not already set
// explicitly; an explicit client value always wins. repeat_penalty, DRY,
// and min_p are deliberately never injected here — they're either left to
// llama-server's own default (min_p) or reserved for the retry path only.
func applyPreset(body map[string]interface{}, preset Preset) {
	setIfAbsent(body, "temperature", preset.Temperature)
	setIfAbsent(body, "top_p", preset.TopP)
	setIfAbsent(body, "top_k", preset.TopK)
	setIfAbsent(body, "thinking_budget_tokens", preset.ThinkingBudgetTokens)
}

func setIfAbsent[T any](body map[string]interface{}, key string, val *T) {
	if val == nil {
		return
	}
	if _, exists := body[key]; exists {
		return
	}
	body[key] = *val
}

// maxTokensKey returns the field name used for the generation length cap
// on each endpoint flavor.
func maxTokensKey(kind endpointKind) string {
	if kind == kindChat {
		return "max_tokens"
	}
	return "n_predict"
}

// RetryAdjustment records one parameter change made by adjustForIssue, for
// the retry trajectory log (see logRetryTrajectory). OldValue/NewValue are
// always numeric even for fields that are conceptually integers (top_k,
// max_tokens, dry_allowed_length), so a single log shape covers every
// adjustable param.
type RetryAdjustment struct {
	Attempt  int     `json:"attempt"`
	Issue    Issue   `json:"issue"`
	Param    string  `json:"param"`
	OldValue float64 `json:"old_value"`
	NewValue float64 `json:"new_value"`
	// Detail says what specifically triggered the issue — for repetition,
	// the exact n-gram that repeated. This is the field to read when
	// deciding whether a retry was a genuine catch or a false positive
	// (e.g. legitimately repeated XML tool tags or file paths in agentic
	// output tripping the detector on healthy content).
	Detail string `json:"detail,omitempty"`
}

// RetryLogEntry captures one request's full retry trajectory: every
// adjustment made, in order, and whether the request ultimately came back
// clean. This is the data retry_step_exponent should be tuned against —
// compare Resolved rates and TotalAttempts across requests logged at
// different exponent values to see which converges faster without
// overshooting.
type RetryLogEntry struct {
	Timestamp     time.Time         `json:"timestamp"`
	Bucket        TaskBucket        `json:"bucket"`
	StepExponent  float64           `json:"step_exponent"`
	Adjustments   []RetryAdjustment `json:"adjustments"`
	Resolved      bool              `json:"resolved"`
	TotalAttempts int               `json:"total_attempts"`
	LatencyMs     int64             `json:"latency_ms"`
}

// logRetryTrajectory appends one JSON line to the retry log for a request
// that retried at least once. Requests that succeeded on the first attempt
// never call this — there's nothing to learn from a request that needed no
// adjustment. Safe for concurrent use; writes are serialized under a mutex
// so lines from different in-flight requests never interleave.
func (p *ProxyServer) logRetryTrajectory(bucket TaskBucket, adjustments []RetryAdjustment, resolved bool, totalAttempts int, latencyMs int64) {
	if p.retryLogFile == nil {
		return
	}

	entry := RetryLogEntry{
		Timestamp:     time.Now(),
		Bucket:        bucket,
		StepExponent:  p.cfg.Retry.StepExponent,
		Adjustments:   adjustments,
		Resolved:      resolved,
		TotalAttempts: totalAttempts,
		LatencyMs:     latencyMs,
	}

	line, err := json.Marshal(entry)
	if err != nil {
		log.Printf("failed to marshal retry log entry: %v", err)
		return
	}

	p.retryLogMu.Lock()
	defer p.retryLogMu.Unlock()
	p.retryLogFile.Write(line)
	p.retryLogFile.Write([]byte("\n"))
}

// adjustForIssue applies the reactive-only [retry] adjustments on top of
// whatever preset is already in the body. repeat_penalty and DRY only ever
// appear here, never on a clean first-pass request.
//
// attempt is the 0-indexed retry number (0 for the first retry, 1 for the
// second, ...), used to scale the adjustment magnitude by
// retry_step_exponent^attempt: with the default exponent of 1.0 every
// retry takes the same flat step (original behavior); a higher exponent
// makes later retries within the same request take bigger steps than
// earlier ones. max_tokens is deliberately excluded from this scaling —
// its multiplier already compounds naturally attempt over attempt, since
// each retry multiplies the previous attempt's value rather than a fixed
// base, so applying the exponent on top would compound two escalating
// factors together.
//
// actualCompletionTokens is how many tokens the model actually generated
// before hitting the cap that just got flagged as truncated (0 if
// unavailable). Since no preset ever injects max_tokens on the initial
// request, the request body itself usually has no prior value to read for
// the IssueTruncated case — using the real generated length here instead
// of a made-up fallback (previously a hardcoded 512, regardless of the
// backend's actual default cap, which is often much higher) means the
// escalated cap is grounded in what the response actually needed rather
// than a guess that could already be smaller than what was truncated.
func adjustForIssue(body map[string]interface{}, issue Issue, cfg *Config, kind endpointKind, attempt int, actualCompletionTokens int) []RetryAdjustment {
	det := &cfg.Detection
	rty := &cfg.Retry
	step := math.Pow(rty.StepExponent, float64(attempt))

	switch issue {
	case IssueRepetition:
		if rty.PreferDryOverRepeatPenalty {
			// Must escalate off the real accumulated old value, the same
			// way repeat_penalty does below — not recompute from
			// DryMultiplierOnRetry*step alone. At the shipped default
			// retry_step_exponent=1.0, step is 1^attempt=1 for every
			// attempt, so a formula that ignores old would produce the
			// exact same dry_multiplier on every retry (regression: found
			// in real retry_log_*.jsonl data showing two consecutive
			// repetition retries both landing on dry_multiplier=0.8 —
			// identical, no escalation at all).
			old := getFloat(body, "dry_multiplier", 0)
			next := old + rty.DryMultiplierOnRetry*step
			body["dry_multiplier"] = next
			body["dry_base"] = rty.DryBase
			body["dry_allowed_length"] = rty.DryAllowedLength
			return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "dry_multiplier", OldValue: old, NewValue: next}}
		}
		old := getFloat(body, "repeat_penalty", 1.0)
		next := old + rty.RepeatPenaltyIncrement*step
		body["repeat_penalty"] = next
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "repeat_penalty", OldValue: old, NewValue: next}}
	case IssueTruncated:
		key := maxTokensKey(kind)
		old := float64(actualCompletionTokens)
		if old <= 0 {
			old = getFloat(body, key, 512)
		}
		next := old * det.MaxTokensRetryMultiplier
		if next > float64(det.MaxTokensCeiling) {
			next = float64(det.MaxTokensCeiling)
		}
		nextInt := int(next)
		body[key] = nextInt
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: key, OldValue: old, NewValue: float64(nextInt)}}
	case IssueBadSyntax:
		// temperature_floor exists specifically so repeated bad-syntax
		// retries can never land on exactly 0 (greedy/degenerate decoding).
		old := getFloat(body, "temperature", 0.8)
		next := old - rty.TemperatureDecrementOnBadSyntax*step
		if next < rty.TemperatureFloor {
			next = rty.TemperatureFloor
		}
		body["temperature"] = next
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "temperature", OldValue: old, NewValue: next}}
	}
	return nil
}

// truncateForLog keeps a raw upstream error body short enough to be
// readable in the dashboard's one-line log.
func truncateForLog(b []byte) string {
	const max = 200
	s := strings.TrimSpace(string(b))
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func getFloat(body map[string]interface{}, key string, def float64) float64 {
	v, ok := body[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	default:
		return def
	}
}

func getInt(body map[string]interface{}, key string, def int) int {
	return int(getFloat(body, key, float64(def)))
}

// effectiveDynamicValues reads back the sampling params currently in the
// (possibly preset-filled, possibly retry-adjusted) request body, for
// display on the dashboard. repeat_penalty naturally reads back as the
// neutral baseline (1.0) whenever no retry has injected it.
func effectiveDynamicValues(body map[string]interface{}) (temp, topP float64, topK int, penalty float64, budget int) {
	temp = getFloat(body, "temperature", 0)
	topP = getFloat(body, "top_p", 0)
	topK = getInt(body, "top_k", 0)
	penalty = getFloat(body, "repeat_penalty", 1.0)
	budget = getInt(body, "thinking_budget_tokens", 0)
	return
}
