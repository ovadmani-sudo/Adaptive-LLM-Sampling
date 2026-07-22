package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func TestLooksLikeConfirmation(t *testing.T) {
	cases := []struct {
		content string
		want    bool
	}{
		{"", true},
		{"Yes, the request is fully completed.", true},
		{"Done.", true},
		{"Nothing more to add here.", true},
		{strings.Repeat("more real continuation content ", 20), false}, // long, no confirmation keywords
		{"Here's the rest of the function:\n\nfunc foo() {}\n", false},
	}
	for _, c := range cases {
		if got := looksLikeConfirmation(c.content); got != c.want {
			t.Errorf("looksLikeConfirmation(%q) = %v, want %v", c.content, got, c.want)
		}
	}
}

func TestModelInAlertList(t *testing.T) {
	list := []string{"cline-pass/deepseek-v4-pro", " Some-Model "}
	if !modelInAlertList(list, "cline-pass/deepseek-v4-pro") {
		t.Error("expected exact match to be found")
	}
	if !modelInAlertList(list, "SOME-MODEL") {
		t.Error("expected case-insensitive, trimmed match to be found")
	}
	if modelInAlertList(list, "cline-pass/deepseek-v4-flash") {
		t.Error("expected a different (even similar) model name not to match")
	}
	if modelInAlertList(list, "") {
		t.Error("expected empty model name never to match")
	}
	if modelInAlertList(nil, "any-model") {
		t.Error("expected an empty list to match nothing")
	}
}

func TestModelInAlertListWildcard(t *testing.T) {
	for _, wildcard := range []string{"*", "any", " ANY ", "Any"} {
		list := []string{wildcard}
		if !modelInAlertList(list, "literally-anything") {
			t.Errorf("wildcard %q: expected any model name to match", wildcard)
		}
		if !modelInAlertList(list, "another-model") {
			t.Errorf("wildcard %q: expected a second, different model name to also match", wildcard)
		}
	}
	if modelInAlertList([]string{"*"}, "") {
		t.Error("expected empty model name never to match, even with a wildcard configured")
	}
}

// alertTestConfig returns a testConfig() with alert-continuation enabled
// for "test-model", a small max-rounds cap, and the default probe message.
func alertTestConfig() *Config {
	cfg := testConfig()
	cfg.Server.AlertEnabled = true
	cfg.Server.AlertModels = []string{"test-model"}
	cfg.Server.AlertMaxRounds = 3
	cfg.Server.AlertProbeMessage = defaultAlertProbeMessage
	return cfg
}

// TestAlertContinuationAppendsGenuineContinuation is the core scenario:
// round 0 stops prematurely (finish_reason stop) on a genuinely
// unfinished task, the alert probe's reply is substantive continuation
// content (not a bare confirmation), and that continuation must be
// appended to the delivered response — followed by a real "stop" that
// ends the exchange because the probe reply is itself accepted (only 2
// rounds needed here; the loop must not ask a third time once nothing
// prompted it to).
func TestAlertContinuationAppendsGenuineContinuation(t *testing.T) {
	var callCount int32
	// Varied (not repeated) text — a repeated phrase would trip the
	// existing repetition detector (testConfig's RepetitionNgramSize=3,
	// RepetitionRequiresLengthFinish=false applies it regardless of
	// finish_reason) and retry WITHIN this round, breaking this test's
	// call-count assumptions about which call corresponds to which round.
	continuation := "func handleRequest(w http.ResponseWriter, r *http.Request) {\n" +
		"\tvalidateInput(r)\n\tprocessBusinessLogic(r)\n\twriteJSONResponse(w)\n}\n" +
		"This finishes the handler implementation with proper error checks."
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		msgs, _ := reqBody["messages"].([]interface{})

		switch n {
		case 1:
			if len(msgs) != 1 {
				t.Errorf("round 1: expected 1 original message, got %d", len(msgs))
			}
			w.Write(sseChatStream("partial work, stopped early", "", "stop", 10))
		case 2:
			if len(msgs) != 3 {
				t.Errorf("round 2: expected 3 messages (original + assistant + probe), got %d", len(msgs))
			}
			last, _ := msgs[len(msgs)-1].(map[string]interface{})
			if last["content"] != defaultAlertProbeMessage {
				t.Errorf("round 2: expected the probe message appended as the last turn, got %v", last["content"])
			}
			// A genuine (non-confirmation) continuation with finish_reason
			// "stop" — the loop must ask again (round 3) rather than
			// treating this as done, since nothing here signals completion.
			w.Write(sseChatStream(continuation, "", "stop", 20))
		case 3:
			if len(msgs) != 5 {
				t.Errorf("round 3: expected 5 messages (original + 2 assistant/probe pairs), got %d", len(msgs))
			}
			// Now it genuinely confirms — the loop must stop here and
			// must NOT append this reply's text to the delivered content.
			w.Write(sseChatStream("Yes, that completes the request.", "", "stop", 8))
		default:
			t.Fatalf("unexpected extra call %d — the confirmation reply in round 3 should have stopped the loop", n)
		}
	}))
	defer upstream.Close()

	cfg := alertTestConfig()
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	var parsed map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("response not valid JSON: %v, body: %s", err, rec.Body.String())
	}
	choice0 := parsed["choices"].([]interface{})[0].(map[string]interface{})
	message := choice0["message"].(map[string]interface{})
	got := message["content"].(string)
	want := "partial work, stopped early" + continuation
	if got != want {
		t.Errorf("content = %q, want %q (initial + continuation concatenated, confirmation reply excluded)", got, want)
	}
	if int(callCount) != 3 {
		t.Errorf("expected exactly 3 upstream calls (initial + continuation + confirmation), got %d", callCount)
	}
}

// TestAlertContinuationDropsConfirmationReply verifies a short,
// confirmation-flavored reply to the probe is NOT appended to the
// delivered content, and stops the alert loop without asking again.
func TestAlertContinuationDropsConfirmationReply(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		switch n {
		case 1:
			w.Write(sseChatStream("the actual answer", "", "stop", 10))
		case 2:
			w.Write(sseChatStream("Yes, the request is fully completed.", "", "stop", 5))
		default:
			t.Fatalf("unexpected extra call %d — a confirmation reply must stop the loop", n)
		}
	}))
	defer upstream.Close()

	cfg := alertTestConfig()
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	var parsed map[string]interface{}
	json.Unmarshal(rec.Body.Bytes(), &parsed)
	choice0 := parsed["choices"].([]interface{})[0].(map[string]interface{})
	message := choice0["message"].(map[string]interface{})
	got := message["content"].(string)
	if got != "the actual answer" {
		t.Errorf("content = %q, want %q (confirmation reply must not be appended)", got, "the actual answer")
	}
	if int(callCount) != 2 {
		t.Errorf("expected exactly 2 upstream calls (initial + one probe that confirmed), got %d", callCount)
	}
}

// TestAlertContinuationDisabledByDefault verifies a request to a model
// that WOULD match alert_models never gets a follow-up probe when
// AlertEnabled is false (the default, absent --alert) — the master
// kill-switch must actually gate everything.
func TestAlertContinuationDisabledByDefault(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		w.Write(sseChatStream("stopped early", "", "stop", 10))
	}))
	defer upstream.Close()

	cfg := alertTestConfig()
	cfg.Server.AlertEnabled = false // the master switch, off
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if int(callCount) != 1 {
		t.Errorf("expected exactly 1 upstream call with AlertEnabled=false, got %d", callCount)
	}
}

// TestAlertContinuationSkippedForUnlistedModel verifies --alert being on
// globally still does nothing for a model not present in alert_models —
// this is opt-in per model, not a blanket behavior change.
func TestAlertContinuationSkippedForUnlistedModel(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		w.Write(sseChatStream("stopped early", "", "stop", 10))
	}))
	defer upstream.Close()

	cfg := alertTestConfig() // alert_models = ["test-model"]
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model":    "some-other-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if int(callCount) != 1 {
		t.Errorf("expected exactly 1 upstream call for a model not in alert_models, got %d", callCount)
	}
}

// TestAlertContinuationCapsAtMaxRounds verifies a model that keeps
// producing genuine (non-confirmation) continuation content on every
// round still stops once alert_max_rounds is reached, rather than looping
// forever.
func TestAlertContinuationCapsAtMaxRounds(t *testing.T) {
	var callCount int32
	// Varied (not repeated) text — see the comment in
	// TestAlertContinuationAppendsGenuineContinuation for why a repeated
	// phrase would trip the unrelated repetition detector instead.
	continuation := "Next I'll refactor the validation layer, then wire up the new " +
		"error handling path, and finally add integration tests covering " +
		"the edge cases around empty payloads and malformed headers."
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		w.Write(sseChatStream(continuation, "", "stop", 20))
	}))
	defer upstream.Close()

	cfg := alertTestConfig()
	cfg.Server.AlertMaxRounds = 2
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	// 1 initial attempt + 2 alert rounds (the cap) = 3 total calls.
	if int(callCount) != 3 {
		t.Errorf("expected exactly 3 upstream calls (1 initial + alert_max_rounds=2), got %d", callCount)
	}
}

// TestAlertContinuationFallsBackWhenProbeRoundFails verifies a failed
// alert-probe follow-up (upstream returns a non-2xx status on that round)
// doesn't turn an already-successful response into an error — the client
// must still get the real, complete content from the round(s) before it.
func TestAlertContinuationFallsBackWhenProbeRoundFails(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		if n == 1 {
			w.Write(sseChatStream("the real answer", "", "stop", 10))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"boom"}`))
	}))
	defer upstream.Close()

	cfg := alertTestConfig()
	proxy, _, closeFn := newFixtureProxyWith(t, cfg, upstream)
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 — a failed follow-up probe must not turn a successful response into an error", rec.Code)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("response not valid JSON: %v, body: %s", err, rec.Body.String())
	}
	choice0 := parsed["choices"].([]interface{})[0].(map[string]interface{})
	message := choice0["message"].(map[string]interface{})
	if got := message["content"].(string); got != "the real answer" {
		t.Errorf("content = %q, want %q", got, "the real answer")
	}
}

// newFixtureProxyWith mirrors newFixtureProxy (proxy_test.go) but takes an
// already-started httptest.Server instead of a bare handler, since several
// alert tests need call-count state shared across the test function.
func newFixtureProxyWith(t *testing.T, cfg *Config, upstream *httptest.Server) (*ProxyServer, chan UIEvent, func()) {
	t.Helper()
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
