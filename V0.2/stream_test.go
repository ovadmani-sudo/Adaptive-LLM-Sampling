package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestApproxTokensTracksChunkCountNotWords is the regression test for the
// dashboard reporting ~20 tok/s against a server whose own console showed
// ~70 tok/s: the original ApproxTokens implementation counted
// whitespace-separated words, which badly undercounts code — brackets,
// operators, and identifiers each tokenize separately but don't split on
// spaces (e.g. "func foo(x int){return x*2}" is ~9 words but many more
// real tokens). Simulating 10 real per-token SSE deltas (llama.cpp's
// actual streaming granularity — one token per delta) that only produce 2
// whitespace-separated words once concatenated: ApproxTokens must track
// the 10 real delta events, not the 2 words.
func TestApproxTokensTracksChunkCountNotWords(t *testing.T) {
	var b strings.Builder
	tokens := []string{"func", " foo", "(", "x", " int", ")", "{", "return", " x*2", "}"}
	for _, tok := range tokens {
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"index": 0, "delta": map[string]interface{}{"content": tok}, "finish_reason": nil},
			},
		}
		data, _ := json.Marshal(chunk)
		b.WriteString("data: ")
		b.Write(data)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(b.String()))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	wordCount := len(strings.Fields(result.content))
	if wordCount >= len(tokens) {
		t.Fatalf("test setup broken: expected word count (%d) to be well below real token count (%d)", wordCount, len(tokens))
	}

	close(progressCh)
	var doneEvent ProgressEvent
	for ev := range progressCh {
		if ev.Done {
			doneEvent = ev
		}
	}

	if doneEvent.ApproxTokens != len(tokens) {
		t.Errorf("ApproxTokens = %d, want %d (real delta count) — got word count (%d) instead, the exact undercounting bug this guards against",
			doneEvent.ApproxTokens, len(tokens), wordCount)
	}
}

// TestTokensSecMultiplierScalesChunkBasedEstimate covers the manual
// correction knob added after observing OpenRouter paid models never
// showing above ~25 tok/s on the live dashboard: the live estimate counts
// SSE delta events assuming one token per delta (true for llama.cpp, not
// guaranteed for a provider that batches). With TokensSecMultiplier set to
// 2.0 and no usage object in the response (forcing the chunks-based
// fallback rather than real completion_tokens), the final ApproxTokens
// must be exactly chunks*2.0, not the raw chunk count.
func TestTokensSecMultiplierScalesChunkBasedEstimate(t *testing.T) {
	var b strings.Builder
	tokens := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
	for _, tok := range tokens {
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"index": 0, "delta": map[string]interface{}{"content": tok}, "finish_reason": nil},
			},
		}
		data, _ := json.Marshal(chunk)
		b.WriteString("data: ")
		b.Write(data)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n") // no usage object: forces chunks-based fallback, not real completion_tokens

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(b.String()))
	}))
	defer upstream.Close()

	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: upstream.URL, TokensSecMultiplier: 2.0}

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, provider, "openrouter", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	var doneEvent ProgressEvent
	for ev := range progressCh {
		if ev.Done {
			doneEvent = ev
		}
	}

	want := int(float64(len(tokens)) * 2.0)
	if doneEvent.ApproxTokens != want {
		t.Errorf("ApproxTokens = %d, want %d (chunk count %d * multiplier 2.0)", doneEvent.ApproxTokens, want, len(tokens))
	}
}

// TestTokensSecMultiplierDefaultsToOneForZeroValueProviderConfig guards
// against the footgun of a ProviderConfig built as a struct literal (as
// several other tests do) that doesn't set TokensSecMultiplier: the Go
// zero value is 0.0, which would make the live estimate always report 0
// tokens if treated as a real multiplier rather than "unset".
func TestTokensSecMultiplierDefaultsToOneForZeroValueProviderConfig(t *testing.T) {
	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: "https://example.invalid"} // TokensSecMultiplier left at zero value
	proxy, err := NewProxyServer(cfg, provider, "openrouter", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	if got := proxy.tokensSecMultiplier(); got != 1.0 {
		t.Errorf("tokensSecMultiplier() = %v, want 1.0 for a zero-value ProviderConfig", got)
	}
}

// TestPostUpstreamChatStreamingAccumulatesContent verifies content,
// reasoning_content, finish_reason, and completion_tokens are correctly
// accumulated from a multi-chunk SSE response — the same data a blocking
// call used to hand back in one piece, now assembled from a real stream.
func TestPostUpstreamChatStreamingAccumulatesContent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("the full answer", "thinking about it", "stop", 42))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	if result.content != "the full answer" {
		t.Errorf("content = %q, want %q", result.content, "the full answer")
	}
	if result.reasoningContent != "thinking about it" {
		t.Errorf("reasoningContent = %q, want %q", result.reasoningContent, "thinking about it")
	}
	if result.finishReason != "stop" {
		t.Errorf("finishReason = %q, want %q", result.finishReason, "stop")
	}
	if result.completionTokens != 42 {
		t.Errorf("completionTokens = %d, want 42", result.completionTokens)
	}
	if result.promptTokens != 10 {
		t.Errorf("promptTokens = %d, want 10 (sseChatStream's fixed usage.prompt_tokens)", result.promptTokens)
	}
	if result.status != http.StatusOK {
		t.Errorf("status = %d, want 200", result.status)
	}
}

// TestPostUpstreamChatStreamingEmitsProgressEvents verifies at least one
// progress event fires during a streaming request, and the final one is
// marked Done — this is the actual mechanism behind the dashboard's live
// activity indicator, so a request that never emits anything would mean
// the "is it hung" signal silently doesn't exist.
func TestPostUpstreamChatStreamingEmitsProgressEvents(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("some words here to count", "", "stop", 5))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	var lastEvent ProgressEvent
	count := 0
	for ev := range progressCh {
		lastEvent = ev
		count++
	}

	if count == 0 {
		t.Fatal("expected at least one progress event, got none")
	}
	if !lastEvent.Done {
		t.Error("expected the final progress event to have Done=true")
	}
	if lastEvent.Bucket != BucketStrictCode {
		t.Errorf("progress event Bucket = %q, want %q", lastEvent.Bucket, BucketStrictCode)
	}
}

// TestPostUpstreamChatStreamingProgressEventsCarryModel verifies every
// emitted ProgressEvent carries the request's actual "model" field — the
// live in-flight indicator's only source for which model is doing the
// work right now (see formatInFlightLine), most useful when it differs
// from whatever's "current" for the listener (vision_describe or a live
// model override redirecting an individual request).
func TestPostUpstreamChatStreamingProgressEventsCarryModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("clean answer", "", "stop", 5))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"model":    "qwen3-vl",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	count := 0
	for ev := range progressCh {
		count++
		if ev.Model != "qwen3-vl" {
			t.Errorf("progress event Model = %q, want %q", ev.Model, "qwen3-vl")
		}
	}
	if count == 0 {
		t.Fatal("expected at least one progress event, got none")
	}
}

// TestPostUpstreamChatStreamingEmitsProgressWhileWaitingForFirstByte is the
// regression test for the dashboard showing "idle" the entire time against
// OpenRouter even while a message was actively running in Cline: p.client.Do
// blocks until upstream sends response headers, and for a provider that
// queues a request before generation starts (free-tier rate-limiting is
// exactly this), that wait can be most of the total latency. Progress
// tracking used to start only after Do() returned, so that whole
// queueing phase emitted nothing — indistinguishable from no request being
// in flight at all. This verifies at least one progress event fires with
// elapsed time ticking up and zero tokens, before upstream ever sends a
// byte.
func TestPostUpstreamChatStreamingEmitsProgressWhileWaitingForFirstByte(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(600 * time.Millisecond) // simulate provider-side queueing before any bytes are sent
		w.Write(sseChatStream("quick answer", "", "stop", 3))
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Server.RequestTimeoutSeconds = 5
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	var sawWaitingEvent bool
	for ev := range progressCh {
		if !ev.Done && ev.ApproxTokens == 0 && ev.ElapsedMs > 0 {
			sawWaitingEvent = true
			if ev.GenerationElapsedMs != 0 {
				t.Errorf("expected GenerationElapsedMs=0 while still waiting for the first byte (nothing generated yet), got %d", ev.GenerationElapsedMs)
			}
		}
	}

	if !sawWaitingEvent {
		t.Error("expected a progress event while waiting for upstream's first byte (elapsed>0, tokens=0, not Done) — without this, the dashboard shows 'idle' during provider-side queueing delay even though a request is genuinely in flight")
	}
}

// TestPostUpstreamChatStreamingEmitsProgressWhileWaitingForFirstLine is the
// regression test for a gap one layer deeper than
// TestPostUpstreamChatStreamingEmitsProgressWhileWaitingForFirstByte above:
// that test covers upstream delaying its response *headers* (p.client.Do
// blocking); this one covers upstream sending headers immediately but then
// delaying the first SSE *body* line — e.g. actually evaluating a large
// prompt (prefill) after already flushing a 200 response. Before this fix,
// scanner.Scan() blocked synchronously for that whole gap with no ticker
// running (the pre-Do wait-ticker had already stopped once headers came
// back), so the dashboard would show stale/idle for the entire
// prompt-processing duration despite a request genuinely being in flight —
// confirmed against real llama-server logs showing 40+s of prompt
// processing on a large prompt before the first generated token. This
// verifies at least one progress event fires (elapsed ticking up, zero
// tokens, not Done) during that specific post-headers, pre-first-line gap.
func TestPostUpstreamChatStreamingEmitsProgressWhileWaitingForFirstLine(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush() // headers sent now, before any body content exists
		}
		time.Sleep(600 * time.Millisecond) // simulate prompt processing after headers already went out
		w.Write(sseChatStream("answer after prefill", "", "stop", 4))
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Server.RequestTimeoutSeconds = 5
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	var sawWaitingEvent bool
	for ev := range progressCh {
		if !ev.Done && ev.ApproxTokens == 0 && ev.GenerationElapsedMs == 0 && ev.ElapsedMs > 0 {
			sawWaitingEvent = true
		}
	}

	if !sawWaitingEvent {
		t.Error("expected a progress event while waiting for the first SSE body line after headers already arrived (elapsed>0, tokens=0, not Done) — without this, the dashboard shows stale/idle during prompt processing even though a request is genuinely in flight")
	}
}

// TestPostUpstreamChatStreamingDrainsExtraLinesAfterDoneWithoutHanging
// guards against a goroutine-lifecycle bug the waiting-for-first-line fix
// above could have introduced: the reader goroutine sends each scanned
// line to an unbuffered channel, and once this function's read loop sees
// "[DONE]" it stops caring about further lines — if it stopped *reading*
// from that channel too, a misbehaving upstream that keeps sending lines
// after "[DONE]" would leave the reader goroutine blocked forever on a
// send nobody receives (exactly the kind of goroutine leak this
// codebase's `go test -race` discipline exists to catch). The fix keeps
// draining the channel until the reader goroutine finishes on its own, so
// this must return promptly rather than hang, and correctly ignore the
// bogus trailing content.
func TestPostUpstreamChatStreamingDrainsExtraLinesAfterDoneWithoutHanging(t *testing.T) {
	var b strings.Builder
	b.Write(sseChatStream("real answer", "", "stop", 3))
	// A compliant upstream wouldn't do this, but nothing stops one from
	// trying — this is exactly what the drain loop must survive.
	for i := 0; i < 5; i++ {
		b.WriteString("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"bogus-post-done\"},\"finish_reason\":null}]}\n\n")
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(b.String()))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	resultCh := make(chan streamResult, 1)
	errCh := make(chan error, 1)
	go func() {
		result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- result
	}()

	select {
	case err := <-errCh:
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	case result := <-resultCh:
		if result.content != "real answer" {
			t.Errorf("content = %q, want %q (trailing post-[DONE] lines must not be appended)", result.content, "real answer")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("postUpstreamChatStreaming hung — reader goroutine likely blocked sending a post-[DONE] line nobody drains")
	}
}

// TestGenerationElapsedExcludesPreFirstByteWait is the regression test for
// the rate-calculation bug: GenerationElapsedMs must only cover time since
// the first real token arrived, not the full request duration — otherwise
// a slow-to-start upstream silently drags the reported tok/s down even
// though decode itself was fast (a request that waits 600ms then decodes
// in ~50ms should report GenerationElapsedMs well under half of the total
// ElapsedMs, not roughly equal to it).
func TestGenerationElapsedExcludesPreFirstByteWait(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(600 * time.Millisecond) // pre-generation wait, e.g. provider queueing
		w.Write(sseChatStream("fast decode", "", "stop", 2))
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Server.RequestTimeoutSeconds = 5
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}
	if result.content != "fast decode" {
		t.Fatalf("expected content to still be correctly accumulated, got %q", result.content)
	}

	close(progressCh)
	var doneEvent ProgressEvent
	for ev := range progressCh {
		if ev.Done {
			doneEvent = ev
		}
	}

	if doneEvent.ElapsedMs < 600 {
		t.Fatalf("sanity check failed: total ElapsedMs (%d) should be at least the 600ms pre-generation sleep", doneEvent.ElapsedMs)
	}
	if doneEvent.GenerationElapsedMs >= doneEvent.ElapsedMs/2 {
		t.Errorf("GenerationElapsedMs (%d) should be much smaller than total ElapsedMs (%d) — it must exclude the 600ms pre-generation wait, or the rate calculation silently understates real decode speed", doneEvent.GenerationElapsedMs, doneEvent.ElapsedMs)
	}
}

// TestPostUpstreamChatStreamingErrorStatusSkipsSSEParsing verifies a
// non-2xx response is read as a raw error body, not fed through the SSE
// parser — error responses aren't SSE-shaped even when stream:true was
// requested, since the error happens before generation starts.
func TestPostUpstreamChatStreamingErrorStatusSkipsSSEParsing(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	if result.status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", result.status)
	}
	if !strings.Contains(string(result.rawErrorBody), "rate limited") {
		t.Errorf("expected raw error body forwarded, got: %s", result.rawErrorBody)
	}
}

// TestKindCompletionDoesNotStream verifies the legacy llama.cpp-native
// /completion path is untouched by this change: it still forces
// stream:false and uses the original blocking call, never emitting
// progress events. That's an intentional scope boundary — local llama.cpp
// already gives console-level progress visibility for this rarely-used
// path, and remote providers never register it at all.
func TestKindCompletionDoesNotStream(t *testing.T) {
	var sawStream interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawStream = body["stream"]
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":"fine","stop":true,"truncated":false}`))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	reqBody, _ := json.Marshal(map[string]interface{}{"prompt": "hi"})
	req := httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if sawStream != false {
		t.Errorf("expected stream=false forwarded for /completion, got %v", sawStream)
	}

	select {
	case ev := <-progressCh:
		t.Errorf("expected no progress events for /completion, got: %+v", ev)
	default:
	}
}

func TestBuildChatCompletionJSON(t *testing.T) {
	b := buildChatCompletionJSON("hello", "thinking", "stop", 15, 7, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(b, &parsed); err != nil {
		t.Fatalf("buildChatCompletionJSON produced invalid JSON: %v", err)
	}

	choices, ok := parsed["choices"].([]interface{})
	if !ok || len(choices) != 1 {
		t.Fatalf("expected exactly 1 choice, got: %v", parsed["choices"])
	}
	choice0 := choices[0].(map[string]interface{})
	if choice0["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice0["finish_reason"])
	}
	message := choice0["message"].(map[string]interface{})
	if message["content"] != "hello" {
		t.Errorf("content = %v, want hello", message["content"])
	}
	if message["reasoning_content"] != "thinking" {
		t.Errorf("reasoning_content = %v, want thinking", message["reasoning_content"])
	}

	usage := parsed["usage"].(map[string]interface{})
	if getInt(usage, "completion_tokens", -1) != 7 {
		t.Errorf("usage.completion_tokens = %v, want 7", usage["completion_tokens"])
	}
	if getInt(usage, "prompt_tokens", -1) != 15 {
		t.Errorf("usage.prompt_tokens = %v, want 15 — Cline reads this to track context window utilization", usage["prompt_tokens"])
	}
	if getInt(usage, "total_tokens", -1) != 22 {
		t.Errorf("usage.total_tokens = %v, want 22 (prompt + completion)", usage["total_tokens"])
	}
}

func TestBuildChatCompletionJSONOmitsEmptyReasoningContent(t *testing.T) {
	b := buildChatCompletionJSON("hello", "", "stop", 0, 1, nil)
	var parsed map[string]interface{}
	json.Unmarshal(b, &parsed)
	choice0 := parsed["choices"].([]interface{})[0].(map[string]interface{})
	message := choice0["message"].(map[string]interface{})
	if _, has := message["reasoning_content"]; has {
		t.Error("expected reasoning_content omitted when empty")
	}
}

// TestUIEventCarriesRealPromptAndCompletionTokens is the full
// request-cycle version: verifies the completed-request UIEvent reports
// the exact prompt/completion token counts from upstream's usage object,
// not the in-flight indicator's word-count estimate.
func TestUIEventCarriesRealPromptAndCompletionTokens(t *testing.T) {
	cfg := testConfig()
	proxy, events, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("a short answer", "", "stop", 55))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	ev := <-events
	if ev.CompletionTokens != 55 {
		t.Errorf("CompletionTokens = %d, want 55", ev.CompletionTokens)
	}
	if ev.PromptTokens != 10 {
		t.Errorf("PromptTokens = %d, want 10 (sseChatStream's fixed usage.prompt_tokens)", ev.PromptTokens)
	}
}

// TestClientContextCancellationCancelsUpstreamRequest is the regression test
// for the confirmed bug: when the downstream client (Cline) times out and
// closes its connection, the proxy kept the upstream HTTP request to
// llama-server alive — it kept generating indefinitely, occupying the only
// generation slot and making all subsequent requests time out too. Fix:
// upstream requests are now made with the incoming request's context
// (http.NewRequestWithContext), so when the client disappears, Go's HTTP
// transport tears down the upstream TCP connection within one round-trip.
//
// This test uses a fake upstream that blocks forever and a context that's
// cancelled immediately. Without the fix, postUpstreamChatStreaming would
// hang for the full http.Client.Timeout (600s). The test enforces that it
// returns within 1s of cancellation.
func TestClientContextCancellationCancelsUpstreamRequest(t *testing.T) {
	started := make(chan struct{})
	// release is closed only after the test's assertions complete, so the
	// handler goroutine never lingers past the test — httptest.Server.Close
	// would otherwise block waiting for it, since whether/when the Go HTTP
	// server itself notices a torn-down client connection while a handler
	// is mid-flight (via r.Context().Done()) isn't the thing under test
	// here and isn't reliably fast. What's under test is purely the client
	// side: that the proxy's RoundTrip returns promptly once its context is
	// cancelled, which is what stops it blocking upstream for the full
	// 600s http.Client.Timeout.
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Server.RequestTimeoutSeconds = 60 // much longer than the test deadline
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	events := make(chan UIEvent, 8)
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", events, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		_, err := proxy.postUpstreamChatStreaming(ctx, "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		errCh <- err
	}()

	// Wait until the upstream handler is actually running before cancelling,
	// so we're testing mid-stream cancellation not pre-connect cancellation.
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream never received the request")
	}

	cancel()

	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("expected an error on context cancellation, got nil")
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v; want context.Canceled (or wrapping it)", err)
		}
	case <-time.After(time.Second):
		t.Fatal("postUpstreamChatStreaming did not return within 1s of context cancellation — upstream request was not cancelled, llama-server would keep generating")
	}

	close(release)
}

// sseToolCallStream builds a fake upstream SSE response with a tool call
// split across multiple delta fragments the way real OpenAI-compatible
// servers stream them: the first fragment carries id/type/function.name,
// every fragment (including the first) may carry a piece of
// function.arguments that must be concatenated in arrival order to
// reassemble the full JSON argument string.
func sseToolCallStream(id, name string, argPieces []string, completionTokens int) []byte {
	var b strings.Builder
	writeChunk := func(delta map[string]interface{}, finish interface{}) {
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"index": 0, "delta": delta, "finish_reason": finish},
			},
		}
		data, _ := json.Marshal(chunk)
		b.WriteString("data: ")
		b.Write(data)
		b.WriteString("\n\n")
	}

	writeChunk(map[string]interface{}{"role": "assistant"}, nil)
	writeChunk(map[string]interface{}{
		"tool_calls": []map[string]interface{}{
			{"index": 0, "id": id, "type": "function", "function": map[string]interface{}{"name": name, "arguments": ""}},
		},
	}, nil)
	for _, piece := range argPieces {
		writeChunk(map[string]interface{}{
			"tool_calls": []map[string]interface{}{
				{"index": 0, "function": map[string]interface{}{"arguments": piece}},
			},
		}, nil)
	}
	writeChunk(map[string]interface{}{}, "tool_calls")

	usageChunk := map[string]interface{}{
		"choices": []interface{}{},
		"usage":   map[string]interface{}{"completion_tokens": completionTokens, "prompt_tokens": 10},
	}
	data, _ := json.Marshal(usageChunk)
	b.WriteString("data: ")
	b.Write(data)
	b.WriteString("\n\n")

	b.WriteString("data: [DONE]\n\n")
	return []byte(b.String())
}

// TestNonStreamingClientReceivesReassembledToolCalls is the regression test
// for the reported bug: a non-streaming client got finish_reason:"tool_calls"
// with an empty message and no tool_calls array at all, because
// postUpstreamChatStreaming never read delta.tool_calls from upstream's SSE
// fragments and buildChatCompletionJSON never had anywhere to put them even
// if it had. A downstream MCP agent loop silently got zero tool calls back
// on every turn as a result.
func TestNonStreamingClientReceivesReassembledToolCalls(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(sseToolCallStream("call_abc123", "get_weather", []string{`{"lat`, `itude":1}`}, 12))
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug in weather tool"}},
	})

	var parsed map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("response is not valid JSON: %v, body: %s", err, rec.Body.String())
	}
	choices, _ := parsed["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected exactly 1 choice, got: %v", parsed["choices"])
	}
	choice0 := choices[0].(map[string]interface{})
	if choice0["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", choice0["finish_reason"])
	}
	message, _ := choice0["message"].(map[string]interface{})
	toolCalls, ok := message["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected exactly 1 reassembled tool call in message.tool_calls, got: %v", message["tool_calls"])
	}
	tc := toolCalls[0].(map[string]interface{})
	if tc["id"] != "call_abc123" {
		t.Errorf("tool_calls[0].id = %v, want call_abc123", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_calls[0].type = %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("tool_calls[0].function.name = %v, want get_weather", fn["name"])
	}
	if fn["arguments"] != `{"latitude":1}` {
		t.Errorf(`tool_calls[0].function.arguments = %v, want {"latitude":1} (fragments must be concatenated in order)`, fn["arguments"])
	}
}

// TestStreamingClientReceivesToolCallsDelta verifies the SSE re-chunking
// path (streamChatSSE) also forwards the reassembled tool_calls, as one
// complete delta chunk before the finish_reason chunk — a streaming client
// must see the exact same final tool_calls array as a non-streaming client
// would, just delivered as an SSE event instead of a JSON body.
func TestStreamingClientReceivesToolCallsDelta(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(sseToolCallStream("call_xyz", "run_query", []string{`{"sql":"select 1"}`}, 5))
	})
	defer closeFn()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	var foundToolCalls []interface{}
	var sawFinishReason interface{}
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		choice0, _ := choices[0].(map[string]interface{})
		if delta, ok := choice0["delta"].(map[string]interface{}); ok {
			if tc, ok := delta["tool_calls"].([]interface{}); ok {
				foundToolCalls = tc
			}
		}
		if fr, ok := choice0["finish_reason"]; ok && fr != nil {
			sawFinishReason = fr
		}
	}

	if len(foundToolCalls) != 1 {
		t.Fatalf("expected exactly 1 tool call in an SSE delta chunk, got: %v", foundToolCalls)
	}
	tc := foundToolCalls[0].(map[string]interface{})
	if tc["id"] != "call_xyz" {
		t.Errorf("tool_calls[0].id = %v, want call_xyz", tc["id"])
	}
	fn, _ := tc["function"].(map[string]interface{})
	if fn["arguments"] != `{"sql":"select 1"}` {
		t.Errorf("tool_calls[0].function.arguments = %v, want {\"sql\":\"select 1\"}", fn["arguments"])
	}
	if sawFinishReason != "tool_calls" {
		t.Errorf("finish_reason chunk = %v, want tool_calls", sawFinishReason)
	}
}

// TestNonStreamingClientReceivesPromptTokensForContextTracking is the
// regression test for Cline showing 0% context utilization and never
// auto-compacting: usage.prompt_tokens (and total_tokens) used to be
// omitted from the synthesized non-streaming response entirely, only
// completion_tokens was ever included. A client that reads prompt_tokens
// off every response to track how full the context window is would read
// that permanent absence as 0% usage forever.
func TestNonStreamingClientReceivesPromptTokensForContextTracking(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(sseChatStream("hello there", "", "stop", 4))
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})

	var parsed map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("response is not valid JSON: %v, body: %s", err, rec.Body.String())
	}
	usage, ok := parsed["usage"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected a usage object, got: %v", parsed["usage"])
	}
	// sseChatStream's fixed usage.prompt_tokens is 10 (see proxy_test.go).
	if got := getInt(usage, "prompt_tokens", -1); got != 10 {
		t.Errorf("usage.prompt_tokens = %d, want 10 — without this, Cline can't track context window utilization and auto-compact never fires", got)
	}
	if got := getInt(usage, "completion_tokens", -1); got != 4 {
		t.Errorf("usage.completion_tokens = %d, want 4", got)
	}
	if got := getInt(usage, "total_tokens", -1); got != 14 {
		t.Errorf("usage.total_tokens = %d, want 14 (prompt + completion)", got)
	}
}

// TestStreamingClientReceivesFinalUsageChunkForContextTracking is the
// streaming-path counterpart: streamChatSSE used to never emit a usage
// object at all, so a streaming client (Cline's default mode) had no way
// to learn prompt_tokens/total_tokens from any SSE event, not even the
// final one — the same silent "0% context used" failure mode as the
// non-streaming path, for the code path Cline actually uses by default.
func TestStreamingClientReceivesFinalUsageChunkForContextTracking(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(sseChatStream("hello there", "", "stop", 4))
	})
	defer closeFn()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	var sawUsage map[string]interface{}
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			sawUsage = usage
		}
	}

	if sawUsage == nil {
		t.Fatal("expected a final SSE chunk carrying a usage object, found none — a streaming client has no way to learn prompt_tokens for context tracking")
	}
	if got := getInt(sawUsage, "prompt_tokens", -1); got != 10 {
		t.Errorf("usage.prompt_tokens = %d, want 10", got)
	}
	if got := getInt(sawUsage, "total_tokens", -1); got != 14 {
		t.Errorf("usage.total_tokens = %d, want 14 (prompt + completion)", got)
	}
}

// TestErrorExitPathsAlwaysEmitFinalDoneEvent is the regression test for the
// confirmed bug behind the dashboard getting stuck showing stale
// "generating..."/countdown text forever after a request had already
// failed: several of postUpstreamChatStreaming's error-return paths (idle
// timeout, a non-2xx upstream status, a genuine mid-stream stall) used to
// return without ever emitting a final Done:true progress event — the only
// signal that clears the dashboard's in-flight indicator. Confirmed via a
// real incident: llama-server logged cancelling the task, but the proxy's
// dashboard never went back to idle, and only a full proxy restart cleared
// it. Every attempt below must produce at least one Done:true event
// regardless of how the attempt ends.
func TestErrorExitPathsAlwaysEmitFinalDoneEvent(t *testing.T) {
	t.Run("idle timeout waiting for headers", func(t *testing.T) {
		release := make(chan struct{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			<-release
		}))
		defer upstream.Close()

		cfg := testConfig()
		u, _ := url.Parse(upstream.URL)
		host, portStr, _ := strings.Cut(u.Host, ":")
		port, _ := strconv.Atoi(portStr)
		cfg.Server.UpstreamHost = host
		cfg.Server.UpstreamPort = port

		progressCh := make(chan ProgressEvent, 32)
		proxy, err := NewProxyServer(cfg, nil, "local", nil, progressCh, "")
		if err != nil {
			t.Fatalf("NewProxyServer: %v", err)
		}
		proxy.idleTimeout = 100 * time.Millisecond

		_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		if err == nil {
			t.Fatal("expected an idle-timeout error")
		}
		close(release)

		close(progressCh)
		var sawDone bool
		for ev := range progressCh {
			if ev.Done {
				sawDone = true
			}
		}
		if !sawDone {
			t.Error("expected a Done:true progress event after an idle-timeout-while-waiting-for-headers error, got none — dashboard would show this request as in-flight forever")
		}
	})

	t.Run("non-2xx upstream status", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":"boom"}`))
		}))
		defer upstream.Close()

		cfg := testConfig()
		u, _ := url.Parse(upstream.URL)
		host, portStr, _ := strings.Cut(u.Host, ":")
		port, _ := strconv.Atoi(portStr)
		cfg.Server.UpstreamHost = host
		cfg.Server.UpstreamPort = port

		progressCh := make(chan ProgressEvent, 32)
		proxy, err := NewProxyServer(cfg, nil, "local", nil, progressCh, "")
		if err != nil {
			t.Fatalf("NewProxyServer: %v", err)
		}

		result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		if err != nil {
			t.Fatalf("postUpstreamChatStreaming: %v", err)
		}
		if result.status != http.StatusInternalServerError {
			t.Fatalf("status = %d, want 500", result.status)
		}

		close(progressCh)
		var sawDone bool
		for ev := range progressCh {
			if ev.Done {
				sawDone = true
			}
		}
		if !sawDone {
			t.Error("expected a Done:true progress event after a non-2xx upstream status, got none — dashboard would show this request as in-flight forever")
		}
	})

	t.Run("client context cancellation", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			close(started)
			<-release
		}))
		defer upstream.Close()

		cfg := testConfig()
		cfg.Server.RequestTimeoutSeconds = 60
		u, _ := url.Parse(upstream.URL)
		host, portStr, _ := strings.Cut(u.Host, ":")
		port, _ := strconv.Atoi(portStr)
		cfg.Server.UpstreamHost = host
		cfg.Server.UpstreamPort = port

		progressCh := make(chan ProgressEvent, 32)
		proxy, err := NewProxyServer(cfg, nil, "local", nil, progressCh, "")
		if err != nil {
			t.Fatalf("NewProxyServer: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		errCh := make(chan error, 1)
		go func() {
			_, err := proxy.postUpstreamChatStreaming(ctx, "/v1/chat/completions", "", map[string]interface{}{
				"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
			}, BucketStrictCode, 0)
			errCh <- err
		}()

		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatal("upstream never received the request")
		}
		cancel()

		select {
		case err := <-errCh:
			if err == nil {
				t.Fatal("expected an error on context cancellation")
			}
		case <-time.After(time.Second):
			t.Fatal("postUpstreamChatStreaming did not return promptly")
		}
		close(release)

		close(progressCh)
		var sawDone bool
		for ev := range progressCh {
			if ev.Done {
				sawDone = true
			}
		}
		if !sawDone {
			t.Error("expected a Done:true progress event after a client disconnect, got none — dashboard would show this request as in-flight forever")
		}
	})
}

// TestIdleTimeoutFiresWhenNoResponseHeadersArrive is the regression test for
// the idle-timeout mechanism's pre-headers phase: p.idleTimeout must cut off
// an upstream that never responds at all once that duration elapses, with an
// error that identifies it as an idle timeout rather than a generic
// context-canceled message a caller could misdiagnose as something else.
func TestIdleTimeoutFiresWhenNoResponseHeadersArrive(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release // never responds within the test's idle window
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.idleTimeout = 100 * time.Millisecond

	errCh := make(chan error, 1)
	go func() {
		_, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		errCh <- err
	}()

	select {
	case err := <-errCh:
		if err == nil || !strings.Contains(err.Error(), "idle timeout") {
			t.Errorf("err = %v, want an error mentioning idle timeout", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("postUpstreamChatStreaming did not return promptly after idleTimeout elapsed with no response headers ever arriving")
	}

	close(release)
}

// TestIdleTimeoutFiresAfterGenuineStallMidStream verifies a stream that
// sends some real progress and then goes truly silent for longer than
// idleTimeout gets cut off with an idle-timeout error. Crucially, the
// context passed in (context.Background()) is never cancelled by this test
// — this is what distinguishes an idle-timeout-induced cancellation from a
// genuine downstream client disconnect, the exact distinction
// handleClassified relies on via r.Context().Err() to avoid silently
// dropping a real error as if the client had gone away.
func TestIdleTimeoutFiresAfterGenuineStallMidStream(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"index": 0, "delta": map[string]interface{}{"content": "partial"}, "finish_reason": nil},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
		<-release // then goes genuinely silent, well past idleTimeout
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.idleTimeout = 100 * time.Millisecond

	errCh := make(chan error, 1)
	go func() {
		_, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		}, BucketStrictCode, 0)
		errCh <- err
	}()

	select {
	case err := <-errCh:
		if err == nil || !strings.Contains(err.Error(), "idle timeout") {
			t.Errorf("err = %v, want an error mentioning idle timeout", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("postUpstreamChatStreaming did not return promptly after a genuine mid-stream stall past idleTimeout")
	}

	close(release)
}

// TestIdleTimeoutDoesNotFireDuringSteadyProgress is the core idle-vs-
// absolute-timeout regression test: a stream whose gaps between chunks
// never individually exceed idleTimeout must complete normally even though
// its TOTAL duration is several times longer than idleTimeout — this is
// exactly the case (a slow but steadily-progressing generation) that an
// absolute wall-clock ceiling would have wrongly cut off, and the entire
// reason this mechanism was changed from p.client.Timeout to a resettable
// idle timer.
func TestIdleTimeoutDoesNotFireDuringSteadyProgress(t *testing.T) {
	const chunkCount = 8
	const chunkGap = 40 * time.Millisecond // well under idleTimeout below

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for i := 0; i < chunkCount; i++ {
			chunk := map[string]interface{}{
				"choices": []map[string]interface{}{
					{"index": 0, "delta": map[string]interface{}{"content": "x"}, "finish_reason": nil},
				},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(chunkGap)
		}
		usage := map[string]interface{}{"choices": []interface{}{}, "usage": map[string]interface{}{"completion_tokens": chunkCount, "prompt_tokens": 10}}
		data, _ := json.Marshal(usage)
		fmt.Fprintf(w, "data: %s\n\n", data)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	// idleTimeout is well over any single chunkGap, but well under the
	// total stream duration (chunkCount*chunkGap) — the total duration
	// exceeding idleTimeout must not matter, only individual gaps do.
	proxy.idleTimeout = 150 * time.Millisecond

	result, err := proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v (idle timeout must not fire when no single gap between chunks exceeds idleTimeout, regardless of total stream duration)", err)
	}
	if result.content != strings.Repeat("x", chunkCount) {
		t.Errorf("content = %q, want %d x's", result.content, chunkCount)
	}
}

// TestProgressEventsCarryIdleTimeoutCountdown verifies IdleTimeoutRemainingMs
// is populated with a sane value on in-flight progress events — this is the
// mechanism behind the dashboard's live countdown display, so a request that
// never populates it would mean the countdown silently doesn't exist.
func TestProgressEventsCarryIdleTimeoutCountdown(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond) // longer than progressEmitInterval (250ms) so the pre-headers wait ticker fires at least once
		w.Write(sseChatStream("done", "", "stop", 1))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", nil, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.idleTimeout = 5 * time.Second

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	var sawCountdown bool
	for ev := range progressCh {
		if !ev.Done && ev.IdleTimeoutRemainingMs > 0 && ev.IdleTimeoutRemainingMs <= proxy.idleTimeout.Milliseconds() {
			sawCountdown = true
		}
	}
	if !sawCountdown {
		t.Error("expected at least one in-flight progress event with a positive IdleTimeoutRemainingMs no greater than idleTimeout — this is what the dashboard's countdown display renders")
	}
}

// TestIdleTimeoutDisabledWhenZeroOmitsCountdown verifies
// IdleTimeoutRemainingMs is -1 (not 0, which would misleadingly read as
// "about to fire") on every progress event when idleTimeout is unconfigured.
func TestIdleTimeoutDisabledWhenZeroOmitsCountdown(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("done", "", "stop", 1))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, nil, "local", nil, progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.idleTimeout = 0

	_, err = proxy.postUpstreamChatStreaming(context.Background(), "/v1/chat/completions", "", map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}, BucketStrictCode, 0)
	if err != nil {
		t.Fatalf("postUpstreamChatStreaming: %v", err)
	}

	close(progressCh)
	for ev := range progressCh {
		if ev.IdleTimeoutRemainingMs != -1 {
			t.Errorf("IdleTimeoutRemainingMs = %d, want -1 (no idle timeout configured) on every event, got one with a real countdown value", ev.IdleTimeoutRemainingMs)
		}
	}
}
