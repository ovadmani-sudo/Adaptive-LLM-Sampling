package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testConfig() *Config {
	temp := 0.2
	budget := 512
	return &Config{
		Server: ServerConfig{MaxRetries: 2, RequestTimeoutSeconds: 5, InjectThinkingBudget: true},
		Classification: ClassificationConfig{
			Keywords: map[TaskBucket][]string{
				BucketStrictCode: {"fix bug"},
			},
			PriorityOrder: []TaskBucket{BucketStrictCode},
			DefaultBucket: BucketStrictCode,
		},
		Presets: map[TaskBucket]Preset{
			BucketStrictCode: {
				Temperature:          &temp,
				ThinkingBudgetTokens: &budget,
			},
		},
		Detection: DetectionConfig{
			RepetitionNgramSize:   3,
			RepetitionMinRepeats:  3,
			RepetitionWindowWords: 96,
			// Deliberately false here (production default is true): the
			// retry-pipeline tests simulate repetition with finish_reason
			// "stop", and their job is exercising the retry machinery, not
			// the finish-reason gate — which has its own dedicated tests.
			RepetitionRequiresLengthFinish: false,
			MaxTokensRetryMultiplier:       1.5,
			MaxTokensCeiling:               8192,
		},
		Retry: RetryConfig{
			PreferDryOverRepeatPenalty:      true,
			DryMultiplierOnRetry:            0.8,
			DryBase:                         1.75,
			DryAllowedLength:                2,
			RepeatPenaltyIncrement:          0.15,
			TemperatureFloor:                0.1,
			TemperatureDecrementOnBadSyntax: 0.15,
			TemperatureIncrementOnEmpty:     0.2,
			TemperatureCeiling:              1.5,
			StepExponent:                    1.0,
		},
	}
}

// sseChatStream builds a minimal valid OpenAI-style SSE stream for a fake
// upstream test server: role delta, content delta (if any), reasoning
// delta (if any), a finish_reason chunk, a usage-only chunk (matching
// stream_options.include_usage — see postUpstreamChatStreaming), and the
// closing [DONE] sentinel. Every fake upstream in this test suite must
// return this shape, not a plain non-streaming JSON body — kindChat
// requests always ask upstream to stream now (see stream.go), so a plain
// JSON response has no "data: " lines for the parser to find and reads
// back as empty content.
func sseChatStream(content, reasoningContent, finishReason string, completionTokens int) []byte {
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
	if content != "" {
		writeChunk(map[string]interface{}{"content": content}, nil)
	}
	if reasoningContent != "" {
		writeChunk(map[string]interface{}{"reasoning_content": reasoningContent}, nil)
	}
	if finishReason == "" {
		finishReason = "stop"
	}
	writeChunk(map[string]interface{}{}, finishReason)

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

func chatResponse(content, finishReason string) []byte {
	return sseChatStream(content, "", finishReason, len(strings.Fields(content)))
}

func chatResponseWithReasoning(content, reasoningContent, finishReason string) []byte {
	return sseChatStream(content, reasoningContent, finishReason, len(strings.Fields(content))+len(strings.Fields(reasoningContent)))
}

func newFixtureProxy(t *testing.T, cfg *Config, handler http.HandlerFunc) (*ProxyServer, chan UIEvent, func()) {
	t.Helper()
	upstream := httptest.NewServer(handler)

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
	return proxy, events, upstream.Close
}

func postChat(proxy *ProxyServer, body map[string]interface{}) *httptest.ResponseRecorder {
	reqBody, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)
	return rec
}

// TestRetryOnRepetitionPrefersDRY verifies that with the default
// prefer_dry_over_repeat_penalty=true, a repetitive first response causes
// the retry to set dry_multiplier/dry_base/dry_allowed_length rather than
// bumping repeat_penalty, and the clean second response reaches the client.
func TestRetryOnRepetitionPrefersDRY(t *testing.T) {
	var callCount int32
	var sawBodies []map[string]interface{}

	cfg := testConfig()
	proxy, events, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)

		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawBodies = append(sawBodies, body)

		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			w.Write(chatResponse(strings.Repeat("loop loop loop ", 5), "stop"))
		} else {
			w.Write(chatResponse("this is a clean, non-repeating answer", "stop"))
		}
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug in this function"},
		},
	})

	if callCount != 2 {
		t.Fatalf("expected 2 upstream calls (initial + 1 retry), got %d", callCount)
	}

	if _, has := sawBodies[0]["dry_multiplier"]; has {
		t.Error("dry_multiplier must not appear on the initial request")
	}
	if _, has := sawBodies[0]["repeat_penalty"]; has {
		t.Error("repeat_penalty must not appear on the initial request")
	}

	if got := getFloat(sawBodies[1], "dry_multiplier", -1); got != 0.8 {
		t.Errorf("retry dry_multiplier = %v, want 0.8", got)
	}
	if got := getFloat(sawBodies[1], "dry_base", -1); got != 1.75 {
		t.Errorf("retry dry_base = %v, want 1.75", got)
	}
	if got := getInt(sawBodies[1], "dry_allowed_length", -1); got != 2 {
		t.Errorf("retry dry_allowed_length = %v, want 2", got)
	}
	if _, has := sawBodies[1]["repeat_penalty"]; has {
		t.Error("repeat_penalty should not be set when DRY is preferred")
	}

	var respBody map[string]interface{}
	json.NewDecoder(rec.Body).Decode(&respBody)
	choices := respBody["choices"].([]interface{})
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	if msg["content"] != "this is a clean, non-repeating answer" {
		t.Errorf("expected clean response body to be returned to client, got %v", msg["content"])
	}

	ev := <-events
	if ev.RetryCount != 1 {
		t.Errorf("expected UIEvent.RetryCount=1, got %d", ev.RetryCount)
	}
	if ev.Issue != IssueNone {
		t.Errorf("expected final issue to be clean, got %q", ev.Issue)
	}
}

// TestRetryOnRepetitionFallsBackToRepeatPenalty verifies that when
// prefer_dry_over_repeat_penalty=false, the retry bumps repeat_penalty by
// repeat_penalty_increment instead of touching DRY fields.
func TestRetryOnRepetitionFallsBackToRepeatPenalty(t *testing.T) {
	var callCount int32
	var sawPenalties []float64

	cfg := testConfig()
	cfg.Retry.PreferDryOverRepeatPenalty = false
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)

		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawPenalties = append(sawPenalties, getFloat(body, "repeat_penalty", -1))

		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			w.Write(chatResponse(strings.Repeat("loop loop loop ", 5), "stop"))
		} else {
			w.Write(chatResponse("this is a clean, non-repeating answer", "stop"))
		}
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug in this function"},
		},
	})

	if callCount != 2 {
		t.Fatalf("expected 2 upstream calls, got %d", callCount)
	}
	if sawPenalties[0] != -1 {
		t.Errorf("repeat_penalty must be absent on initial request, got %v", sawPenalties[0])
	}
	if sawPenalties[1] < 1.14 || sawPenalties[1] > 1.16 {
		t.Errorf("retry repeat_penalty = %v, want ~1.15 (1.0 + 0.15)", sawPenalties[1])
	}
}

// TestCleanResponseCarriesNoReactiveParams verifies that a clean first-pass
// response never sends repeat_penalty, dry_multiplier, or min_p at all.
func TestCleanResponseCarriesNoReactiveParams(t *testing.T) {
	var sawBody map[string]interface{}

	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&sawBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("a perfectly normal answer", "stop"))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	for _, key := range []string{"repeat_penalty", "dry_multiplier", "dry_base", "dry_allowed_length", "min_p"} {
		if _, has := sawBody[key]; has {
			t.Errorf("clean request must not carry %q", key)
		}
	}
}

// TestClientExplicitValueWins verifies a client-supplied temperature is
// never overwritten by the preset.
func TestClientExplicitValueWins(t *testing.T) {
	var sawTemp float64
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawTemp = getFloat(body, "temperature", -1)
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("fine", "stop"))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"temperature": 0.99,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if sawTemp != 0.99 {
		t.Errorf("expected client temperature 0.99 to survive merge, got %v", sawTemp)
	}
}

// TestStreamingPreservesReasoningContent verifies that when a client
// requests streaming, the model's reasoning/thinking trace
// (message.reasoning_content from llama-server, kept distinct from the
// final answer per the DeepSeek-style convention) survives the SSE
// re-chunking as its own delta.reasoning_content chunk, rather than being
// silently dropped (regression: streamChatSSE previously only forwarded
// content, discarding reasoning_content entirely — breaking any client,
// e.g. Cline's Plan Mode, that renders reasoning separately from the
// final answer).
func TestStreamingPreservesReasoningContent(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponseWithReasoning("final answer", "here is my plan: step one, step two", "stop"))
	})
	defer closeFn()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"stream": true,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, `"reasoning_content":"here is my plan: step one, step two"`) {
		t.Errorf("expected reasoning_content to survive SSE re-chunking, got body: %s", body)
	}
	if !strings.Contains(body, `"content":"final answer"`) && !strings.Contains(body, `final answer`) {
		t.Errorf("expected final answer content to still be present, got body: %s", body)
	}
}

// TestChunkTextPreservesWhitespaceExactly is the unit-level regression test
// for the formatting-loss bug: the old chunkWords split on strings.Fields
// (any run of whitespace) and rejoined with single ASCII spaces, silently
// destroying every newline, indentation level, and multi-space run —
// reported live as "any formatting is lost, including code formatting" for
// clients that request streaming (Cline does, by default). Concatenating
// every chunk must reproduce the original string exactly, byte for byte.
func TestChunkTextPreservesWhitespaceExactly(t *testing.T) {
	original := "func foo() {\n\tif x {\n\t\treturn  1\n\t}\n}\n\nfunc bar() {}"
	chunks := chunkText(original, 5)
	got := strings.Join(chunks, "")
	if got != original {
		t.Errorf("chunkText did not preserve the original text exactly.\ngot:  %q\nwant: %q", got, original)
	}
}

// TestStreamingPreservesCodeFormatting is the end-to-end version of the
// same regression, through the real SSE re-chunking path a streaming
// client actually sees: reconstructing the message from the delta.content
// pieces (the way a real client concatenates them) must reproduce the
// original multi-line, indented code exactly — newlines, tabs, and all.
func TestStreamingPreservesCodeFormatting(t *testing.T) {
	cfg := testConfig()
	code := "func foo() {\n\tif x {\n\t\treturn 1\n\t}\n}\n"
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse(code, "stop"))
	})
	defer closeFn()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"stream": true,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	var reconstructed strings.Builder
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
		delta, _ := choice0["delta"].(map[string]interface{})
		if c, ok := delta["content"].(string); ok {
			reconstructed.WriteString(c)
		}
	}

	if reconstructed.String() != code {
		t.Errorf("streamed content did not reconstruct the original formatting.\ngot:  %q\nwant: %q", reconstructed.String(), code)
	}
}

// TestUpstreamErrorNeverWrappedInFakeStream verifies that when upstream
// returns a non-2xx response to a streaming client, the proxy forwards the
// real error body/status instead of synthesizing an empty "clean" SSE
// stream around it (regression: this previously hid llama-server's actual
// 400 error behind a fake stream with empty content and finish_reason=stop).
func TestUpstreamErrorNeverWrappedInFakeStream(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"invalid param: thinking_budget_tokens"}}`))
	})
	defer closeFn()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"stream": true,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected the real upstream status 400 to be forwarded, got %d", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "data:") {
		t.Errorf("error response must not be wrapped in an SSE stream, got body: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid param") {
		t.Errorf("expected the real error message to be forwarded, got: %s", rec.Body.String())
	}
}

// TestBadSyntaxRetryRespectsTemperatureFloor verifies repeated bad_syntax
// retries never push temperature below the configured floor (and never to
// exactly 0, which would mean greedy/degenerate decoding).
func TestBadSyntaxRetryRespectsTemperatureFloor(t *testing.T) {
	var sawTemps []float64

	badJSON := "```json\n{\"a\": 1,}\n```"

	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawTemps = append(sawTemps, getFloat(body, "temperature", -1))
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse(badJSON, "stop"))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if len(sawTemps) != cfg.Server.MaxRetries+1 {
		t.Fatalf("expected %d upstream calls, got %d", cfg.Server.MaxRetries+1, len(sawTemps))
	}
	for i, temp := range sawTemps {
		if temp < cfg.Retry.TemperatureFloor {
			t.Errorf("call %d temperature = %v, must never drop below floor %v", i, temp, cfg.Retry.TemperatureFloor)
		}
		if temp == 0 {
			t.Errorf("call %d temperature is exactly 0 (degenerate decoding)", i)
		}
	}
}

// TestAdjustForIssueRepeatPenaltyScalesByExponent verifies each successive
// attempt's repeat_penalty step grows by step_exponent^attempt, not the
// flat base increment every time.
func TestAdjustForIssueRepeatPenaltyScalesByExponent(t *testing.T) {
	cfg := testConfig()
	cfg.Retry.PreferDryOverRepeatPenalty = false
	cfg.Retry.RepeatPenaltyIncrement = 0.1
	cfg.Retry.StepExponent = 2.0

	cases := []struct {
		attempt  int
		wantStep float64 // increment actually applied on top of the 1.0 baseline
	}{
		{0, 0.1}, // 0.1 * 2^0
		{1, 0.2}, // 0.1 * 2^1
		{2, 0.4}, // 0.1 * 2^2
	}
	for _, c := range cases {
		body := map[string]interface{}{}
		adjustments := adjustForIssue(body, IssueRepetition, cfg, kindChat, c.attempt, 0)
		if len(adjustments) != 1 {
			t.Fatalf("attempt %d: expected 1 adjustment, got %d", c.attempt, len(adjustments))
		}
		got := getFloat(body, "repeat_penalty", -1)
		want := 1.0 + c.wantStep
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("attempt %d: repeat_penalty = %v, want %v", c.attempt, got, want)
		}
		if adjustments[0].NewValue != got {
			t.Errorf("attempt %d: RetryAdjustment.NewValue = %v, want %v", c.attempt, adjustments[0].NewValue, got)
		}
	}
}

// TestAdjustForIssueDRYScalesByExponent verifies dry_multiplier escalates
// across attempts the same way repeat_penalty does.
func TestAdjustForIssueDRYScalesByExponent(t *testing.T) {
	cfg := testConfig()
	cfg.Retry.PreferDryOverRepeatPenalty = true
	cfg.Retry.DryMultiplierOnRetry = 0.5
	cfg.Retry.StepExponent = 2.0

	body := map[string]interface{}{}
	adjustForIssue(body, IssueRepetition, cfg, kindChat, 0, 0)
	if got := getFloat(body, "dry_multiplier", -1); got != 0.5 {
		t.Errorf("attempt 0: dry_multiplier = %v, want 0.5", got)
	}

	body = map[string]interface{}{}
	adjustForIssue(body, IssueRepetition, cfg, kindChat, 1, 0)
	if got := getFloat(body, "dry_multiplier", -1); got != 1.0 {
		t.Errorf("attempt 1: dry_multiplier = %v, want 1.0 (0.5 * 2^1)", got)
	}
}

// TestAdjustForIssueDRYEscalatesAcrossRealAttemptsAtDefaultStepExponent is
// the regression test for a bug found in real retry_log_*.jsonl data: two
// consecutive DRY-preferred repetition retries within the same request
// (attempt 0 then attempt 1) both landed on dry_multiplier=0.8 — identical,
// no escalation — at the shipped default retry_step_exponent=1.0. The old
// formula computed dry_multiplier_on_retry*step from scratch every
// attempt, discarding whatever the body already had; since step is
// 1^attempt=1 for every attempt at the default exponent, that reproduces
// the exact same value forever, meaning a "retry" resent identical
// sampling params with nothing but token-level randomness to save it.
// Unlike TestAdjustForIssueDRYScalesByExponent above (which uses a fresh
// body per case and so can't see this), this test reuses one body across
// both calls, matching how the real retry loop in handleClassified mutates
// body in place across attempts.
func TestAdjustForIssueDRYEscalatesAcrossRealAttemptsAtDefaultStepExponent(t *testing.T) {
	cfg := testConfig() // StepExponent defaults to 1.0 here, matching config.ini's shipped default
	cfg.Retry.PreferDryOverRepeatPenalty = true
	cfg.Retry.DryMultiplierOnRetry = 0.8

	body := map[string]interface{}{}
	adjustForIssue(body, IssueRepetition, cfg, kindChat, 0, 0)
	afterAttempt0 := getFloat(body, "dry_multiplier", -1)

	adjustForIssue(body, IssueRepetition, cfg, kindChat, 1, 0)
	afterAttempt1 := getFloat(body, "dry_multiplier", -1)

	if afterAttempt1 <= afterAttempt0 {
		t.Errorf("dry_multiplier after attempt 1 (%v) must be greater than after attempt 0 (%v) — a repetition retry that resends identical sampling params has no reason to succeed where the previous identical attempt just failed", afterAttempt1, afterAttempt0)
	}
}

// TestAdjustForIssueBadSyntaxScalesByExponentButRespectsFloor verifies the
// temperature decrement also escalates, while still never crossing the
// configured floor.
func TestAdjustForIssueBadSyntaxScalesByExponentButRespectsFloor(t *testing.T) {
	cfg := testConfig()
	cfg.Retry.TemperatureDecrementOnBadSyntax = 0.1
	cfg.Retry.TemperatureFloor = 0.1
	cfg.Retry.StepExponent = 3.0

	body := map[string]interface{}{"temperature": 0.8}
	adjustForIssue(body, IssueBadSyntax, cfg, kindChat, 0, 0)
	if got := getFloat(body, "temperature", -1); math.Abs(got-0.7) > 1e-9 {
		t.Errorf("attempt 0: temperature = %v, want 0.7 (0.8 - 0.1*3^0)", got)
	}

	// attempt 1 would subtract 0.1*3 = 0.3 from 0.7, landing at 0.4 — still
	// above the floor, should not be clamped.
	adjustForIssue(body, IssueBadSyntax, cfg, kindChat, 1, 0)
	if got := getFloat(body, "temperature", -1); math.Abs(got-0.4) > 1e-9 {
		t.Errorf("attempt 1: temperature = %v, want 0.4 (0.7 - 0.1*3^1)", got)
	}

	// attempt 2 would subtract 0.1*9 = 0.9, landing well below the floor —
	// must clamp to exactly the floor, never below.
	adjustForIssue(body, IssueBadSyntax, cfg, kindChat, 2, 0)
	if got := getFloat(body, "temperature", -1); got != cfg.Retry.TemperatureFloor {
		t.Errorf("attempt 2: temperature = %v, want floor %v", got, cfg.Retry.TemperatureFloor)
	}
}

// TestAdjustForIssueEmptyScalesByExponentButRespectsCeiling mirrors
// TestAdjustForIssueBadSyntaxScalesByExponentButRespectsFloor for the
// opposite direction: IssueEmpty retries push temperature UP (an empty
// completion is more often an unlucky early-EOS draw than something
// determinism would fix), capped at temperature_ceiling instead of floored.
func TestAdjustForIssueEmptyScalesByExponentButRespectsCeiling(t *testing.T) {
	cfg := testConfig()
	cfg.Retry.TemperatureIncrementOnEmpty = 0.1
	cfg.Retry.TemperatureCeiling = 1.0
	cfg.Retry.StepExponent = 3.0

	body := map[string]interface{}{"temperature": 0.5}
	adjustForIssue(body, IssueEmpty, cfg, kindChat, 0, 0)
	if got := getFloat(body, "temperature", -1); math.Abs(got-0.6) > 1e-9 {
		t.Errorf("attempt 0: temperature = %v, want 0.6 (0.5 + 0.1*3^0)", got)
	}

	// attempt 1 would add 0.1*3 = 0.3 to 0.6, landing at 0.9 — still below
	// the ceiling, should not be clamped.
	adjustForIssue(body, IssueEmpty, cfg, kindChat, 1, 0)
	if got := getFloat(body, "temperature", -1); math.Abs(got-0.9) > 1e-9 {
		t.Errorf("attempt 1: temperature = %v, want 0.9 (0.6 + 0.1*3^1)", got)
	}

	// attempt 2 would add 0.1*9 = 0.9, landing well above the ceiling — must
	// clamp to exactly the ceiling, never above.
	adjustForIssue(body, IssueEmpty, cfg, kindChat, 2, 0)
	if got := getFloat(body, "temperature", -1); got != cfg.Retry.TemperatureCeiling {
		t.Errorf("attempt 2: temperature = %v, want ceiling %v", got, cfg.Retry.TemperatureCeiling)
	}
}

// TestAdjustForIssueMaxTokensNotScaledByExponent verifies truncation's
// max_tokens multiplier is unaffected by step_exponent — it already
// compounds naturally attempt over attempt via its own multiplier.
func TestAdjustForIssueMaxTokensNotScaledByExponent(t *testing.T) {
	cfg := testConfig()
	cfg.Detection.MaxTokensRetryMultiplier = 1.5
	cfg.Retry.StepExponent = 10.0 // if this leaked in, the result would be wildly different

	body := map[string]interface{}{"max_tokens": float64(512)}
	adjustForIssue(body, IssueTruncated, cfg, kindChat, 3, 0) // high attempt number on purpose; 0 = no real completion_tokens known, exercises the body[key] fallback
	if got := getFloat(body, "max_tokens", -1); got != 768 {
		t.Errorf("max_tokens = %v, want 768 (512 * 1.5, unaffected by exponent)", got)
	}
}

// TestAdjustForIssueTruncatedPrefersActualCompletionTokens verifies the
// truncation retry escalates from the real number of tokens the model
// generated (regression: previously always guessed a hardcoded 512 when
// max_tokens wasn't already in the request body — which it never is, since
// no preset injects it — so the "escalated" cap could already be smaller
// than what the backend's own default limit had already produced,
// guaranteeing the retry got truncated again too).
func TestAdjustForIssueTruncatedPrefersActualCompletionTokens(t *testing.T) {
	cfg := testConfig()
	cfg.Detection.MaxTokensRetryMultiplier = 1.5

	body := map[string]interface{}{} // no max_tokens set, matching a real first-pass request
	adjustments := adjustForIssue(body, IssueTruncated, cfg, kindChat, 0, 2200)

	if len(adjustments) != 1 {
		t.Fatalf("expected 1 adjustment, got %d", len(adjustments))
	}
	if adjustments[0].OldValue != 2200 {
		t.Errorf("OldValue = %v, want 2200 (the real completion length), not the old hardcoded 512 guess", adjustments[0].OldValue)
	}
	if got := getFloat(body, "max_tokens", -1); got != 3300 {
		t.Errorf("max_tokens = %v, want 3300 (2200 * 1.5)", got)
	}
}

// TestAdjustForIssueTruncatedLoggedValueMatchesActualIntSent is the
// regression test for a real bug found via live retry_log.jsonl data: the
// RetryAdjustment.NewValue recorded a pre-truncation float (e.g. 457.5 for
// 305*1.5) while body[key] correctly stored the truncated int actually
// sent upstream (457) — the log was silently lying about what request was
// really made. Any old_value*multiplier that isn't a whole number
// reproduces this.
func TestAdjustForIssueTruncatedLoggedValueMatchesActualIntSent(t *testing.T) {
	cfg := testConfig()
	cfg.Detection.MaxTokensRetryMultiplier = 1.5

	body := map[string]interface{}{}
	adjustments := adjustForIssue(body, IssueTruncated, cfg, kindChat, 0, 305) // 305*1.5 = 457.5

	sentValue := getFloat(body, "max_tokens", -1)
	if sentValue != 457 {
		t.Fatalf("sanity check: max_tokens actually sent = %v, want 457 (int(457.5))", sentValue)
	}
	if adjustments[0].NewValue != sentValue {
		t.Errorf("RetryAdjustment.NewValue = %v, want %v (must match the truncated int actually sent, not the pre-truncation float 457.5)",
			adjustments[0].NewValue, sentValue)
	}
}

// TestAdjustForIssueTruncatedFallsBackWhenCompletionTokensUnknown verifies
// the old body[key]-or-512 behavior still applies when the real completion
// length genuinely isn't available (e.g. upstream response missing usage
// info) — the fix should only change behavior when better data exists.
func TestAdjustForIssueTruncatedFallsBackWhenCompletionTokensUnknown(t *testing.T) {
	cfg := testConfig()
	cfg.Detection.MaxTokensRetryMultiplier = 1.5

	body := map[string]interface{}{}
	adjustForIssue(body, IssueTruncated, cfg, kindChat, 0, 0) // 0 = unknown
	if got := getFloat(body, "max_tokens", -1); got != 768 {
		t.Errorf("max_tokens = %v, want 768 (falls back to the 512 default * 1.5)", got)
	}
}

// TestExtractCompletionTokensReadsUsageField verifies the OAI chat
// endpoint's usage.completion_tokens is read correctly, and that it's 0
// (not a crash) when usage is absent — llama.cpp's native /completion has
// no usage object at all, and some responses may omit it.
func TestExtractCompletionTokensReadsUsageField(t *testing.T) {
	respBody := map[string]interface{}{
		"choices": []interface{}{},
		"usage":   map[string]interface{}{"completion_tokens": float64(2048), "prompt_tokens": float64(100)},
	}
	if got := extractCompletionTokens(respBody, kindChat); got != 2048 {
		t.Errorf("extractCompletionTokens = %d, want 2048", got)
	}

	missing := map[string]interface{}{"choices": []interface{}{}}
	if got := extractCompletionTokens(missing, kindChat); got != 0 {
		t.Errorf("extractCompletionTokens with no usage field = %d, want 0", got)
	}
}

// TestExtractCompletionTokensNativeCompletion verifies the
// llama.cpp-native /completion response uses tokens_predicted instead of
// an OAI-style usage object.
func TestExtractCompletionTokensNativeCompletion(t *testing.T) {
	respBody := map[string]interface{}{
		"content":          "some text",
		"tokens_predicted": float64(900),
	}
	if got := extractCompletionTokens(respBody, kindCompletion); got != 900 {
		t.Errorf("extractCompletionTokens = %d, want 900", got)
	}
}

// TestTruncationRetryEndToEndUsesRealCompletionLength is the full
// integration version of the regression: an upstream response truncated
// at ~2200 real tokens (with no max_tokens ever set by the client) must
// escalate the retry from that real length, not a hardcoded guess smaller
// than what was already generated.
func TestTruncationRetryEndToEndUsesRealCompletionLength(t *testing.T) {
	var sawMaxTokens []float64

	cfg := testConfig()
	cfg.Detection.MaxTokensRetryMultiplier = 1.5
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawMaxTokens = append(sawMaxTokens, getFloat(body, "max_tokens", -1))

		w.Write(sseChatStream("unterminated code fence:\n```go\nfunc foo() {", "", "length", 2200))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if len(sawMaxTokens) < 2 {
		t.Fatalf("expected at least 2 upstream calls, got %d", len(sawMaxTokens))
	}
	if sawMaxTokens[0] != -1 {
		t.Errorf("initial request should not carry max_tokens, got %v", sawMaxTokens[0])
	}
	if sawMaxTokens[1] != 3300 {
		t.Errorf("retry max_tokens = %v, want 3300 (2200 real completion tokens * 1.5), not a guess based on a hardcoded 512 baseline", sawMaxTokens[1])
	}
}

// TestEmptyResponseRetryEndToEnd is the regression test for a confirmed
// real-world failure: a completion with blank content and finish_reason
// "stop" previously reached the client unchanged (retry=0, issue=clean),
// with neither the retry loop nor alert-continuation ever noticing — the
// client's own generic "please resend" error was all the user saw. Verifies
// the full pipeline now retries an empty first attempt, bumping temperature
// upward, and delivers the second attempt's real content to the client.
func TestEmptyResponseRetryEndToEnd(t *testing.T) {
	var sawTemps []float64
	attempt := 0

	cfg := testConfig()
	cfg.Retry.TemperatureIncrementOnEmpty = 0.2
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawTemps = append(sawTemps, getFloat(body, "temperature", -1))

		if attempt == 0 {
			attempt++
			w.Write(sseChatStream("", "", "stop", 0)) // genuinely empty completion
			return
		}
		w.Write(sseChatStream("here is the actual answer", "", "stop", 4))
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if len(sawTemps) < 2 {
		t.Fatalf("expected the empty first attempt to trigger a retry (at least 2 upstream calls), got %d", len(sawTemps))
	}
	if sawTemps[1] <= sawTemps[0] {
		t.Errorf("retry temperature = %v, want higher than the first attempt's %v (IssueEmpty pushes temperature up)", sawTemps[1], sawTemps[0])
	}
	if !strings.Contains(rec.Body.String(), "here is the actual answer") {
		t.Errorf("expected the client to receive the retry's real content, got: %s", rec.Body.String())
	}
}

// TestUpstreamUnreachableDoesNotCrash verifies a connection-refused
// upstream produces a 502 and a UIEvent with an Error, not a panic.
func TestUpstreamUnreachableDoesNotCrash(t *testing.T) {
	cfg := testConfig()
	cfg.Server.UpstreamHost = "127.0.0.1"
	cfg.Server.UpstreamPort = 1 // nothing listens here

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if rec.Code != http.StatusBadGateway {
		t.Errorf("expected 502, got %d", rec.Code)
	}
	ev := <-events
	if ev.Error == "" {
		t.Error("expected UIEvent to carry a connection error")
	}
}

// TestUpstreamClientTimeoutReportsErrorNotSilentDrop is the regression
// test for a confirmed bug: p.client.Timeout firing produces a Go error
// that satisfies errors.Is(err, context.DeadlineExceeded) — the same
// check previously used (imprecisely) to detect a genuine downstream
// client disconnect — so a plain client-side timeout talking to upstream
// used to be silently treated as "the client went away" and dropped with
// an empty response, even though the downstream client (this test) is
// still there and waiting. The fix checks r.Context().Err() directly
// instead of inspecting the upstream error's type.
func TestUpstreamClientTimeoutReportsErrorNotSilentDrop(t *testing.T) {
	cfg := testConfig()
	cfg.Server.RequestTimeoutSeconds = 1 // fires long before keepaliveGracePeriod's 60s default
	proxy, events, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second) // longer than RequestTimeoutSeconds
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 (a client-side upstream timeout must be reported, not silently dropped)", rec.Code)
	}
	if rec.Body.Len() == 0 {
		t.Error("expected a non-empty error body, got an empty response — this is exactly the silent-drop bug")
	}
	ev := <-events
	if ev.Error == "" {
		t.Error("expected UIEvent to carry the timeout error")
	}
}

// TestRemoteProviderRoutingAndAuth verifies that in remote-provider mode
// the proxy hits exactly "<base_url>/chat/completions" — never the
// incoming client path (which would double up the "/v1" prefix most
// providers' base_url already includes) — and sends the configured API
// key as a Bearer token.
func TestRemoteProviderRoutingAndAuth(t *testing.T) {
	var sawPath string
	var sawAuth string

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		sawAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("fine", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	provider := &ProviderConfig{
		BaseURL: upstream.URL, // e.g. http://127.0.0.1:PORT, mimicking base_url = https://.../v1
		APIKey:  "sk-test-123",
	}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openrouter", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	// The client hits our proxy at /v1/chat/completions (as Cline would);
	// that path must NOT leak into the outbound request.
	reqBody, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if sawPath != "/chat/completions" {
		t.Errorf("expected outbound path /chat/completions, got %q", sawPath)
	}
	if sawAuth != "Bearer sk-test-123" {
		t.Errorf("expected Authorization: Bearer sk-test-123, got %q", sawAuth)
	}
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
}

// TestRemoteProviderModelOverride verifies that a configured provider
// model always overrides whatever model the client requested, since a
// model name valid for local llama-server won't be valid on a remote API.
func TestRemoteProviderModelOverride(t *testing.T) {
	var sawModel string

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawModel, _ = body["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("fine", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: upstream.URL, Model: "gpt-5"}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openai", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"model": "local-llama-model",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if sawModel != "gpt-5" {
		t.Errorf("expected provider model override to win, got %q", sawModel)
	}
}

// TestRemoteProviderRejectsCompletionAndPassthrough verifies that in
// remote-provider mode, /completion and any other path fail clearly
// instead of being silently mis-routed (these providers don't support
// llama.cpp's native completion endpoint, and generic passthrough can't
// be mapped reliably across providers with different URL prefixes).
func TestRemoteProviderRejectsCompletionAndPassthrough(t *testing.T) {
	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: "https://api.openai.com/v1"}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openai", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	for _, path := range []string{"/completion", "/health", "/props", "/v1/embeddings", "/v1/"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		proxy.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Errorf("path %q: expected 501, got %d", path, rec.Code)
		}
	}
}

// TestRemoteProviderWithAllowPassthroughForwardsUnrecognizedPaths is the
// regression test for the confirmed clinepass bug: "Error: Token refresh
// failed: 501" happened because clinepass is Cline's own account gateway,
// not a plain model vendor — Cline's extension calls auxiliary endpoints
// (token refresh, recommended-models, remote-config) against the same
// base_url beyond /chat/completions, and the chat/models-only allowlist
// rejected all of them with 501. With AllowPassthrough set, any path other
// than /chat/completions and /models must reach upstream verbatim instead
// of being rejected.
func TestRemoteProviderWithAllowPassthroughForwardsUnrecognizedPaths(t *testing.T) {
	var sawPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: upstream.URL + "/api/v1", AllowPassthrough: true}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "clinepass", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/users/me/token/refresh", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (path should be forwarded to upstream, not rejected with 501)", rec.Code)
	}
	if sawPath != "/api/v1/users/me/token/refresh" {
		t.Errorf("upstream saw path %q, want %q (base_url's own path joined with the incoming path)", sawPath, "/api/v1/users/me/token/refresh")
	}
}

// TestRemoteProviderWithoutAllowPassthroughStillRejects verifies the
// default (AllowPassthrough unset/false) is completely unaffected by this
// feature — every other provider (claude/gemini/openai/openrouter) must
// keep rejecting unrecognized paths with 501 exactly as before.
func TestRemoteProviderWithoutAllowPassthroughStillRejects(t *testing.T) {
	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: "https://api.openai.com/v1"} // AllowPassthrough left at zero value (false)

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openai", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/some/other/path", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501 (AllowPassthrough is false; unrecognized paths must still be rejected)", rec.Code)
	}
}

// TestRemoteProviderAnswersBareV1Probe verifies GET /v1 (the exact bare
// path, no trailing slash) gets a 200 instead of the generic 501 — some
// clients (e.g. Page Assist) ping the base URL itself as a reachability
// check before calling /v1/models, and there's no real provider endpoint
// to proxy that probe to.
func TestRemoteProviderAnswersBareV1Probe(t *testing.T) {
	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: "https://openrouter.ai/api/v1"}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openrouter", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for bare /v1 probe, got %d", rec.Code)
	}
}

// TestCORSHeadersPresent verifies every response carries permissive CORS
// headers. Cline (VS Code extension host) is never subject to CORS, but
// browser-based clients (e.g. Page Assist) are: without
// Access-Control-Allow-Origin, the browser discards an otherwise-successful
// response before the client's own JS ever sees it, which looks like an
// empty/broken model list even though the server did everything right.
func TestCORSHeadersPresent(t *testing.T) {
	cfg := testConfig()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("fine", "stop"))
	}))
	defer upstream.Close()

	provider := &ProviderConfig{BaseURL: upstream.URL}
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openrouter", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Access-Control-Allow-Origin = %q, want \"*\"", got)
	}
}

// TestCORSReflectsRequestedHeadersOnEveryListener verifies withCORS's
// header-reflection (see the forward-proxy regression test in mitm_test.go
// for the original failure) applies through Handler() too — i.e. to every
// reverse-proxy listener (local, claude, gemini, openai, openrouter,
// clinepass), not only the forward-proxy's own pipeline handler. Both
// return paths in Handler() (provider != nil and the local/nil-provider
// branch) share the same withCORS call, so one fix covers all of them; this
// exercises both branches directly to prove it rather than infer it from
// mitm_test.go's coverage alone.
func TestCORSReflectsRequestedHeadersOnEveryListener(t *testing.T) {
	cfg := testConfig()
	requested := "content-type, authorization, x-title, http-referer"

	assertReflected := func(t *testing.T, proxy *ProxyServer, label string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
		req.Header.Set("Access-Control-Request-Headers", requested)
		rec := httptest.NewRecorder()
		proxy.Handler().ServeHTTP(rec, req)

		got := strings.ToLower(rec.Header().Get("Access-Control-Allow-Headers"))
		for _, want := range []string{"x-title", "http-referer"} {
			if !strings.Contains(got, want) {
				t.Errorf("[%s] Access-Control-Allow-Headers = %q, missing requested header %q", label, got, want)
			}
		}
	}

	localProxy, err := NewProxyServer(cfg, nil, "local", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer (local): %v", err)
	}
	assertReflected(t, localProxy, "local (nil provider)")

	providerProxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://example.invalid"}, "openrouter", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer (provider): %v", err)
	}
	assertReflected(t, providerProxy, "provider mode (openrouter)")
}

// TestCORSPreflightHandled verifies an OPTIONS preflight request (which a
// browser sends before a cross-origin POST carrying Content-Type/
// Authorization headers) gets a clean 204 without hitting the classify/
// inject/detect logic at all.
func TestCORSPreflightHandled(t *testing.T) {
	cfg := testConfig()
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Errorf("expected 204 for OPTIONS preflight, got %d", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Methods"); got == "" {
		t.Error("expected Access-Control-Allow-Methods on preflight response")
	}
}

// TestRemoteProviderProxiesModelsEndpoint verifies GET /v1/models (which
// Cline calls on startup to populate its model picker) is forwarded to
// <base_url>/models with auth, rather than rejected like other passthrough
// paths — this was the actual cause of a "connection failed" report where
// the dashboard showed zero requests because the rejection wasn't a
// classified chat-completions call.
func TestRemoteProviderProxiesModelsEndpoint(t *testing.T) {
	var sawPath, sawAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		sawAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[{"id":"gpt-5","object":"model"}]}`))
	}))
	defer upstream.Close()

	cfg := testConfig()
	provider := &ProviderConfig{BaseURL: upstream.URL, APIKey: "sk-test-456"}

	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, provider, "openai", events, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if sawPath != "/models" {
		t.Errorf("expected outbound path /models, got %q", sawPath)
	}
	if sawAuth != "Bearer sk-test-456" {
		t.Errorf("expected Authorization: Bearer sk-test-456, got %q", sawAuth)
	}
	if !strings.Contains(rec.Body.String(), "gpt-5") {
		t.Errorf("expected model list body to be forwarded, got: %s", rec.Body.String())
	}
}

// TestKeepaliveFastPathUnaffected is a direct regression guard for the
// constraint that shaped the keepalive design: a request resolving well
// under keepaliveGracePeriod must produce byte-for-byte the same output
// as before the goroutine/channel restructuring — no eager headers, no
// keepalive lines, real status code forwarded as-is. Shrinking
// keepaliveGracePeriod itself isn't needed here (the fake upstream
// responds instantly, well under even a real 60s), so this uses the
// production default.
func TestKeepaliveFastPathUnaffected(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("a quick answer", "stop"))
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "keepalive") {
		t.Errorf("fast-resolving request must never see a keepalive line, got: %s", rec.Body.String())
	}
}

// TestKeepaliveSlowPathSendsHeartbeatChunksBeforeRealContent is the core new
// behavior: a request still running past keepaliveGracePeriod must get
// eager SSE headers and at least one heartbeat chunk before the real
// content arrives, so a client with its own hard idle-connection timeout
// (what this whole feature exists to fix) sees a steady trickle of bytes
// instead of dead silence for the entire classify/inject/detect/retry
// duration.
//
// The heartbeat is a real, well-formed empty-delta "data: {...}"
// chat-completion-chunk — NOT a bare SSE comment line (the original
// design here): a comment line is spec-legal SSE, but a confirmed
// real-world client ("Stream decode error") only handled the "data: {...}"
// shape every other chunk in the stream already uses, having apparently
// never been exercised against an unfamiliar bare-colon line at all. An
// empty delta needs no new parsing support from any client that already
// handles the rest of this stream.
func TestKeepaliveSlowPathSendsHeartbeatChunksBeforeRealContent(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(120 * time.Millisecond) // comfortably past the shrunk grace period below
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("the real answer", "stop"))
	})
	defer closeFn()

	proxy.keepaliveGracePeriod = 20 * time.Millisecond
	proxy.keepaliveTickInterval = 10 * time.Millisecond

	rec := postChat(proxy, map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream (eager headers must have committed to SSE)", ct)
	}

	body := rec.Body.String()
	if strings.Contains(body, ": keepalive") {
		t.Error("must not use a bare SSE comment line anymore — a real client failed to parse it")
	}

	// Walk every "data: " line in order (each is a complete, compact JSON
	// object — json.Marshal never emits raw newlines inside one — so
	// splitting on "\n" always lands on line boundaries) and confirm at
	// least one well-formed empty-delta heartbeat chunk appears before the
	// chunk carrying the real content.
	heartbeatsBeforeContent := 0
	sawRealContent := false
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("line is not valid JSON: %v, line: %s", err, line)
		}
		choices, _ := chunk["choices"].([]interface{})
		if len(choices) != 1 {
			continue // the final usage-only chunk (empty choices), not a content/heartbeat chunk
		}
		choice0, _ := choices[0].(map[string]interface{})
		delta, _ := choice0["delta"].(map[string]interface{})
		if c, _ := delta["content"].(string); strings.Contains(c, "the real answer") {
			sawRealContent = true
			break
		}
		if len(delta) == 0 {
			heartbeatsBeforeContent++
		}
	}
	if !sawRealContent {
		t.Fatalf("expected the real content to still arrive after the heartbeat ticks, got: %s", body)
	}
	if heartbeatsBeforeContent == 0 {
		t.Fatalf("expected at least one empty-delta heartbeat chunk before the real content, got: %s", body)
	}
}

// TestKeepaliveSlowPathSurfacesErrorAsSSEContent covers the documented
// tradeoff: once headers are committed to 200 SSE (past
// keepaliveGracePeriod), a subsequent transport-level failure can no
// longer be reported via a real status code — it must surface as
// synthesized SSE content instead of hanging or writing a raw error onto
// an SSE-typed connection.
func TestKeepaliveSlowPathSurfacesErrorAsSSEContent(t *testing.T) {
	cfg := testConfig()
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		// Hijack and close without ever writing a response — produces a
		// clean, unambiguous I/O error (not a context.Canceled/
		// DeadlineExceeded-wrapping one) once past the shrunk grace
		// period, exercising the fatalErr path deliberately rather than
		// the clientGone path.
		time.Sleep(100 * time.Millisecond)
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("ResponseWriter does not support hijacking")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatalf("hijack: %v", err)
		}
		conn.Close()
	})
	defer closeFn()

	proxy.keepaliveGracePeriod = 20 * time.Millisecond
	proxy.keepaliveTickInterval = 10 * time.Millisecond

	rec := postChat(proxy, map[string]interface{}{
		"stream":   true,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers were already committed before the failure occurred)", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data:") {
		t.Errorf("expected a well-formed SSE stream even for the error case, got: %s", body)
	}
	if !strings.Contains(body, "Error:") {
		t.Errorf("expected the failure to surface as an assistant-visible error message, got: %s", body)
	}
}

// TestApplyPresetOmitsThinkingBudgetWhenDisabled is the regression test for
// a backend launched with reasoning disabled server-side (e.g. llama-server
// --reasoning off): sending thinking_budget_tokens anyway risks silently
// re-enabling a reasoning pass that eats into the shared max_tokens budget
// before the model's visible answer, which reads as truncated/incomplete
// output rather than an outright error. injectThinkingBudget=false must
// mean the field is never set, even though the preset defines a value.
func TestApplyPresetOmitsThinkingBudgetWhenDisabled(t *testing.T) {
	budget := 512
	preset := Preset{ThinkingBudgetTokens: &budget}
	body := map[string]interface{}{}

	applyPreset(body, preset, false)

	if _, has := body["thinking_budget_tokens"]; has {
		t.Errorf("expected thinking_budget_tokens omitted when injectThinkingBudget=false, got %v", body["thinking_budget_tokens"])
	}
}

// TestApplyPresetInjectsThinkingBudgetWhenEnabled verifies the opposite
// case and existing behavior otherwise unchanged: injectThinkingBudget=true
// with no client-supplied value still fills it in from the preset.
func TestApplyPresetInjectsThinkingBudgetWhenEnabled(t *testing.T) {
	budget := 512
	preset := Preset{ThinkingBudgetTokens: &budget}
	body := map[string]interface{}{}

	applyPreset(body, preset, true)

	if got := getInt(body, "thinking_budget_tokens", -1); got != 512 {
		t.Errorf("thinking_budget_tokens = %d, want 512", got)
	}
}

// TestApplyPresetClientValueAlwaysWinsOverThinkingBudget verifies an
// explicit client-supplied thinking_budget_tokens is never overwritten,
// regardless of injectThinkingBudget — this toggle only controls whether
// the preset FILLS IN a missing value, not whether it clobbers one the
// client already set.
func TestApplyPresetClientValueAlwaysWinsOverThinkingBudget(t *testing.T) {
	budget := 512
	preset := Preset{ThinkingBudgetTokens: &budget}
	body := map[string]interface{}{"thinking_budget_tokens": 999}

	applyPreset(body, preset, true)

	if got := getInt(body, "thinking_budget_tokens", -1); got != 999 {
		t.Errorf("thinking_budget_tokens = %d, want 999 (client's explicit value must win)", got)
	}
}

// TestThinkingBudgetNotSentToUpstreamWhenDisabledInConfig is the full
// request-cycle version: with [server].inject_thinking_budget_tokens =
// false, upstream must never receive a thinking_budget_tokens field at
// all, confirming the config wiring (not just the isolated applyPreset
// call) actually reaches the outbound request.
func TestThinkingBudgetNotSentToUpstreamWhenDisabledInConfig(t *testing.T) {
	var sawBudget bool
	cfg := testConfig()
	cfg.Server.InjectThinkingBudget = false
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		_, sawBudget = reqBody["thinking_budget_tokens"]
		w.Write(sseChatStream("ok", "", "stop", 1))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if sawBudget {
		t.Error("expected upstream to never receive thinking_budget_tokens when InjectThinkingBudget is false")
	}
}

// TestSystemPromptHashStableAndSensitive verifies systemPromptHash's core
// diagnostic property: identical system-message content always hashes the
// same, any change in content changes the hash, and a request with no
// system message (or no messages at all) returns "" rather than panicking.
func TestSystemPromptHashStableAndSensitive(t *testing.T) {
	bodyA := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "TERMINAL RULES: commit after deliver"},
			map[string]interface{}{"role": "user", "content": "hi"},
		},
	}
	bodyB := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "TERMINAL RULES: commit after deliver"},
			map[string]interface{}{"role": "user", "content": "something totally different"},
		},
	}
	bodyC := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "TERMINAL RULES: commit BEFORE deliver"},
			map[string]interface{}{"role": "user", "content": "hi"},
		},
	}
	bodyNoSystem := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "hi"},
		},
	}
	bodyNoMessages := map[string]interface{}{}

	hashA := systemPromptHash(bodyA)
	hashB := systemPromptHash(bodyB)
	hashC := systemPromptHash(bodyC)

	if hashA == "" {
		t.Fatal("expected a non-empty hash for a body with a system message")
	}
	if hashA != hashB {
		t.Errorf("expected identical system-message content to hash the same regardless of other messages, got %q vs %q", hashA, hashB)
	}
	if hashA == hashC {
		t.Errorf("expected different system-message content to hash differently, both got %q", hashA)
	}
	if got := systemPromptHash(bodyNoSystem); got != "" {
		t.Errorf("expected \"\" for a body with no system message, got %q", got)
	}
	if got := systemPromptHash(bodyNoMessages); got != "" {
		t.Errorf("expected \"\" for a body with no messages field at all, got %q", got)
	}
}

// TestSwitchProviderRedirectsToNewBackend verifies a request made after
// SwitchProvider goes to the NEW backend with the new API key and model
// override, not the one this instance was originally constructed with —
// this is the mechanism behind letting a single-provider agent (e.g.
// Codex, hardcoded to one API) be redirected to a different real backend
// without restarting the proxy.
func TestSwitchProviderRedirectsToNewBackend(t *testing.T) {
	var sawAuthA, sawAuthB string
	var sawModelB string

	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuthA = r.Header.Get("Authorization")
		w.Write(chatResponse("from A", "stop"))
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuthB = r.Header.Get("Authorization")
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		sawModelB, _ = reqBody["model"].(string)
		w.Write(chatResponse("from B", "stop"))
	}))
	defer upstreamB.Close()

	cfg := testConfig()
	providerA := &ProviderConfig{BaseURL: upstreamA.URL, APIKey: "key-a"}
	proxy, err := NewProxyServer(cfg, providerA, "provider-a", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if !strings.Contains(rec.Body.String(), "from A") {
		t.Fatalf("expected first request to hit upstream A, got: %s", rec.Body.String())
	}
	if sawAuthA != "Bearer key-a" {
		t.Errorf("upstream A saw Authorization = %q, want %q", sawAuthA, "Bearer key-a")
	}

	if err := proxy.SwitchProvider("provider-b", ProviderConfig{BaseURL: upstreamB.URL, APIKey: "key-b", Model: "model-b"}); err != nil {
		t.Fatalf("SwitchProvider: %v", err)
	}

	rec = postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if !strings.Contains(rec.Body.String(), "from B") {
		t.Fatalf("expected second request (after SwitchProvider) to hit upstream B, got: %s", rec.Body.String())
	}
	if sawAuthB != "Bearer key-b" {
		t.Errorf("upstream B saw Authorization = %q, want %q", sawAuthB, "Bearer key-b")
	}
	if sawModelB != "model-b" {
		t.Errorf("upstream B saw model = %q, want %q (the new provider's configured model override)", sawModelB, "model-b")
	}
}

// TestSwitchProviderUpdatesAllowPassthroughRouting verifies switching to a
// provider with a different AllowPassthrough setting takes effect on the
// catch-all route immediately — Handler()'s mux is only built once, so
// this must be re-checked per request rather than baked in at
// construction time (see the dynamic closure in Handler()).
func TestSwitchProviderUpdatesAllowPassthroughRouting(t *testing.T) {
	var sawPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	cfg := testConfig()
	// Constructed WITHOUT passthrough — must 501 before switching.
	proxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://example.invalid"}, "no-passthrough", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/some/aux/endpoint", nil)
	rec := httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("before switching: status = %d, want 501", rec.Code)
	}

	if err := proxy.SwitchProvider("with-passthrough", ProviderConfig{BaseURL: upstream.URL, AllowPassthrough: true}); err != nil {
		t.Fatalf("SwitchProvider: %v", err)
	}

	req = httptest.NewRequest(http.MethodGet, "/some/aux/endpoint", nil)
	rec = httptest.NewRecorder()
	proxy.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("after switching to an AllowPassthrough provider: status = %d, want 200 (forwarded, not rejected)", rec.Code)
	}
	if sawPath != "/some/aux/endpoint" {
		t.Errorf("upstream saw path %q, want %q", sawPath, "/some/aux/endpoint")
	}
}

// TestForcedBucketOverridesClassification verifies SetForcedBucket pins
// classification regardless of what the message content would otherwise
// match, and ClearForcedBucket returns to automatic classification.
func TestForcedBucketOverridesClassification(t *testing.T) {
	var sawTemp float64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		sawTemp = getFloat(reqBody, "temperature", -1)
		w.Write(chatResponse("ok", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	temp := 0.8
	cfg.Presets[BucketArchitecture] = Preset{Temperature: &temp}
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	proxy, err := NewProxyServer(cfg, nil, "local", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	// A message that would normally classify as strict_code (matches
	// "fix bug" in testConfig's keyword list) — forcing architecture must
	// override that and apply architecture's preset instead.
	proxy.SetForcedBucket(BucketArchitecture)
	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if sawTemp != 0.8 {
		t.Errorf("temperature = %v, want 0.8 (architecture preset) — forced bucket was not applied", sawTemp)
	}

	proxy.ClearForcedBucket()
	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if sawTemp != 0.2 {
		t.Errorf("temperature = %v, want 0.2 (strict_code preset, from testConfig) — ClearForcedBucket did not return to auto-detect", sawTemp)
	}
}

func TestCurrentForcedBucketReflectsState(t *testing.T) {
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	if _, ok := proxy.CurrentForcedBucket(); ok {
		t.Error("expected no forced bucket initially")
	}

	proxy.SetForcedBucket(BucketExplanation)
	if b, ok := proxy.CurrentForcedBucket(); !ok || b != BucketExplanation {
		t.Errorf("CurrentForcedBucket() = (%q, %v), want (%q, true)", b, ok, BucketExplanation)
	}

	proxy.ClearForcedBucket()
	if _, ok := proxy.CurrentForcedBucket(); ok {
		t.Error("expected no forced bucket after Clear")
	}
}

// TestOutboundClientIgnoresProxyEnvVars is the regression test for a
// confirmed real-world failure: this process's own outbound client MUST
// NOT honor HTTP_PROXY/HTTPS_PROXY, since forward-proxy mode's whole job is
// BEING that proxy for other tools — if this process itself inherits those
// same env vars (e.g. started from a shell that also exports them for a
// client tool like Cline), Go's http.DefaultTransport default of
// Proxy: ProxyFromEnvironment would route this process's own outbound
// upstream calls back through itself, hit its own MITM leaf cert, and fail
// TLS verification (confirmed in production: "certificate signed by
// unknown authority"), since the outbound client never trusts this
// process's own CA. Proves the fix: an env var pointing at a bogus/unused
// proxy address must have zero effect on where the request actually goes.
func TestOutboundClientIgnoresProxyEnvVars(t *testing.T) {
	var sawUpstream bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawUpstream = true
		w.Write(chatResponse("direct, not proxied", "stop"))
	}))
	defer upstream.Close()

	bogusProxyHit := make(chan struct{}, 1)
	bogusProxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case bogusProxyHit <- struct{}{}:
		default:
		}
		w.WriteHeader(http.StatusTeapot)
	}))
	defer bogusProxy.Close()

	t.Setenv("HTTP_PROXY", bogusProxy.URL)
	t.Setenv("http_proxy", bogusProxy.URL)

	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: upstream.URL}, "test-provider", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if !sawUpstream {
		t.Error("expected the real upstream to receive the request directly")
	}
	if !strings.Contains(rec.Body.String(), "direct, not proxied") {
		t.Errorf("expected response from the real upstream, got: %s", rec.Body.String())
	}
	select {
	case <-bogusProxyHit:
		t.Error("outbound client honored HTTP_PROXY env var — it must always dial upstream directly, never through an ambient env-configured proxy (including itself, in forward-proxy mode)")
	default:
	}
}

// TestBypassSamplingSkipsPresetInjection verifies the control-panel "pass
// through dynamic sampling" toggle: normally the bucket preset fills in
// sampling params the client omitted (temperature 0.2 for strict_code); with
// bypass on, nothing is injected and the request goes upstream as-is.
func TestBypassSamplingSkipsPresetInjection(t *testing.T) {
	var tempPresent bool
	var sawTemp float64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		v, ok := body["temperature"].(float64)
		tempPresent = ok
		sawTemp = v
		w.Write(chatResponse("ok", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig() // strict_code preset injects temperature 0.2
	proxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: upstream.URL}, "local", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	// Normal mode: preset injects the temperature the client omitted.
	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if !tempPresent || sawTemp != 0.2 {
		t.Fatalf("normal mode: temp present=%v val=%v, want present 0.2 (preset injected)", tempPresent, sawTemp)
	}

	// Bypass mode: no preset injection — the omitted param stays omitted.
	proxy.SetBypassSampling(true)
	tempPresent, sawTemp = false, 0
	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	if tempPresent {
		t.Fatalf("bypass mode: upstream saw temperature=%v, want none (no preset injection)", sawTemp)
	}
}

// TestModelOverrideWinsOverConfiguredModel verifies the control-panel model
// picker: a live SetModelOverride replaces the outgoing model, beating the
// static config.ini provider Model; clearing it reverts to the configured one.
func TestModelOverrideWinsOverConfiguredModel(t *testing.T) {
	var sawModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		sawModel, _ = body["model"].(string)
		w.Write(chatResponse("ok", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: upstream.URL, Model: "static-model"}, "clinepass", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	msg := map[string]interface{}{"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}}}

	postChat(proxy, msg)
	if sawModel != "static-model" {
		t.Fatalf("no override: model = %q, want static-model", sawModel)
	}
	if proxy.EffectiveModel() != "static-model" {
		t.Errorf("EffectiveModel() = %q, want static-model", proxy.EffectiveModel())
	}

	proxy.SetModelOverride("cline-pass/qwen3.7-max")
	postChat(proxy, msg)
	if sawModel != "cline-pass/qwen3.7-max" {
		t.Fatalf("with override: model = %q, want cline-pass/qwen3.7-max", sawModel)
	}
	if proxy.EffectiveModel() != "cline-pass/qwen3.7-max" {
		t.Errorf("EffectiveModel() = %q, want the override", proxy.EffectiveModel())
	}

	proxy.SetModelOverride("")
	postChat(proxy, msg)
	if sawModel != "static-model" {
		t.Fatalf("after clearing override: model = %q, want static-model", sawModel)
	}
}
