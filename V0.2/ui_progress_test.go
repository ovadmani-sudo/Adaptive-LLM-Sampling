package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestFormatInFlightLineIdle(t *testing.T) {
	got := formatInFlightLine(nil)
	if !strings.Contains(got, "idle") {
		t.Errorf("expected idle indicator for nil progress, got: %q", got)
	}
}

func TestFormatInFlightLineActive(t *testing.T) {
	ev := &ProgressEvent{
		Bucket:              BucketStrictCode,
		Attempt:             0,
		ApproxTokens:        120,
		ElapsedMs:           5000,
		GenerationElapsedMs: 2000,
	}
	got := formatInFlightLine(ev)
	if !strings.Contains(got, "120") {
		t.Errorf("expected token count in output, got: %q", got)
	}
	if !strings.Contains(got, "5.0s") {
		t.Errorf("expected total elapsed seconds in output, got: %q", got)
	}
	if !strings.Contains(got, "60.0 tok/s") {
		t.Errorf("expected rate computed from GenerationElapsedMs (120 tokens / 2s = 60 tok/s), not total ElapsedMs (which would wrongly give 24 tok/s), got: %q", got)
	}
	if !strings.Contains(got, string(BucketStrictCode)) {
		t.Errorf("expected bucket name in output, got: %q", got)
	}
	if !strings.Contains(got, "attempt 1") {
		t.Errorf("expected attempt+1 (0-indexed Attempt=0 shown as attempt 1), got: %q", got)
	}
}

// TestFormatInFlightLineWaitingForFirstByte verifies the pre-generation
// state (GenerationElapsedMs still 0, upstream hasn't sent any content
// yet) renders a distinct "processing prompt" message rather than
// computing a rate from zero — this is the state during OpenRouter's
// free-tier queueing delay and during local prompt-eval/prefill, where
// ElapsedMs ticks up but nothing has been generated yet. Named
// "processing prompt" rather than a generic "waiting for upstream" since
// that's what's actually happening upstream during this window (verified
// locally: a ~54k-token prompt took 40+s of prompt processing before the
// first generated token).
func TestFormatInFlightLineWaitingForFirstByte(t *testing.T) {
	ev := &ProgressEvent{Bucket: BucketStrictCode, Attempt: 0, ApproxTokens: 0, ElapsedMs: 3000, GenerationElapsedMs: 0}
	got := formatInFlightLine(ev)
	if !strings.Contains(got, "processing prompt") {
		t.Errorf("expected a distinct 'processing prompt' message before generation starts, got: %q", got)
	}
	if !strings.Contains(got, "3.0s") {
		t.Errorf("expected elapsed seconds still shown while waiting, got: %q", got)
	}
	if strings.Contains(got, "tok/s") {
		t.Errorf("expected no rate shown before generation starts, got: %q", got)
	}
}

// TestFormatInFlightLineShowsModel verifies which model is actually doing
// the work is visible in the live in-flight indicator, not just after the
// request completes — most useful when it differs from whatever's
// "current" for the listener (vision_describe or a live model override
// redirecting an individual request). Covers all three branches
// (Label-override, pre-generation, and generating), since each builds its
// string separately.
func TestFormatInFlightLineShowsModel(t *testing.T) {
	labelEv := &ProgressEvent{Label: "passthrough", ElapsedMs: 1000, Model: "qwen3-vl"}
	if got := formatInFlightLine(labelEv); !strings.Contains(got, "[qwen3-vl]") {
		t.Errorf("expected model shown in Label branch, got: %q", got)
	}

	prefillEv := &ProgressEvent{Bucket: BucketStrictCode, ElapsedMs: 1000, GenerationElapsedMs: 0, Model: "qwen3-vl"}
	if got := formatInFlightLine(prefillEv); !strings.Contains(got, "[qwen3-vl]") {
		t.Errorf("expected model shown in pre-generation branch, got: %q", got)
	}

	generatingEv := &ProgressEvent{Bucket: BucketStrictCode, ApproxTokens: 10, ElapsedMs: 1000, GenerationElapsedMs: 500, Model: "qwen3-vl"}
	if got := formatInFlightLine(generatingEv); !strings.Contains(got, "[qwen3-vl]") {
		t.Errorf("expected model shown in generating branch, got: %q", got)
	}

	noModelEv := &ProgressEvent{Bucket: BucketStrictCode, ApproxTokens: 10, ElapsedMs: 1000, GenerationElapsedMs: 500}
	if got := formatInFlightLine(noModelEv); strings.Contains(got, "[]") {
		t.Errorf("expected no empty bracket when Model is unset, got: %q", got)
	}
}

func TestFormatInFlightLineZeroElapsedNoDivideByZero(t *testing.T) {
	ev := &ProgressEvent{Bucket: BucketStrictCode, ApproxTokens: 0, ElapsedMs: 0}
	got := formatInFlightLine(ev)
	if strings.Contains(got, "Inf") || strings.Contains(got, "NaN") {
		t.Errorf("expected no divide-by-zero artifacts, got: %q", got)
	}
}

// TestDashboardModelClearsInFlightOnDone verifies the model's Update logic
// treats a Done progress event as "clear the indicator," and a non-Done
// event as "show it" — this is what makes the dashboard line disappear
// once a request finishes instead of freezing on its last value forever.
func TestDashboardModelClearsInFlightOnDone(t *testing.T) {
	events := make(chan UIEvent)
	progressCh := make(chan ProgressEvent)
	m := newDashboardModel(events, progressCh, "127.0.0.1:9090", "local", nil, DashboardControls{})

	updated, _ := m.Update(progressMsg(ProgressEvent{Bucket: BucketStrictCode, ApproxTokens: 10}))
	m = updated.(dashboardModel)
	if m.inFlight == nil {
		t.Fatal("expected inFlight to be set after a non-Done progress event")
	}

	updated, _ = m.Update(progressMsg(ProgressEvent{Bucket: BucketStrictCode, Done: true}))
	m = updated.(dashboardModel)
	if m.inFlight != nil {
		t.Error("expected inFlight to be cleared after a Done progress event")
	}
}

func TestFormatLastRequestTokensLineEmptyWhenNoData(t *testing.T) {
	got := formatLastRequestTokensLine(UIEvent{})
	if got != "" {
		t.Errorf("expected empty string for a zero-value UIEvent, got: %q", got)
	}
}

func TestFormatLastRequestTokensLineShowsRealCounts(t *testing.T) {
	got := formatLastRequestTokensLine(UIEvent{PromptTokens: 727, CompletionTokens: 1437})
	if !strings.Contains(got, "727") || !strings.Contains(got, "1437") {
		t.Errorf("expected both prompt and completion counts present, got: %q", got)
	}
	if !strings.Contains(got, "2164") {
		t.Errorf("expected total (727+1437=2164), got: %q", got)
	}
}

func TestFormatAlertCountsSingularPlural(t *testing.T) {
	got := formatAlertCounts(map[string]int{"model-a": 1, "model-b": 3})
	if !strings.Contains(got, "model-a was alerted 1 time\n") {
		t.Errorf("expected singular \"1 time\" for model-a, got: %q", got)
	}
	if !strings.Contains(got, "model-b was alerted 3 times\n") {
		t.Errorf("expected plural \"3 times\" for model-b, got: %q", got)
	}
}

func TestFormatAlertCountsSortedByModel(t *testing.T) {
	got := formatAlertCounts(map[string]int{"zeta": 1, "alpha": 2})
	if strings.Index(got, "alpha") > strings.Index(got, "zeta") {
		t.Errorf("expected models sorted alphabetically, got: %q", got)
	}
}

func TestFormatAlertCountsEmpty(t *testing.T) {
	if got := formatAlertCounts(nil); got != "" {
		t.Errorf("expected empty string for no alert counts, got: %q", got)
	}
}

// TestDashboardModelAccumulatesAlertCounts verifies alertCounts sums
// AlertRounds across multiple UIEvents for the same model, rather than
// overwriting — this is a running total for the whole session, not a
// per-request snapshot.
func TestDashboardModelAccumulatesAlertCounts(t *testing.T) {
	events := make(chan UIEvent)
	progressCh := make(chan ProgressEvent)
	m := newDashboardModel(events, progressCh, "127.0.0.1:9090", "local", nil, DashboardControls{})

	updated, _ := m.Update(uiEventMsg(UIEvent{Model: "test-model", AlertRounds: 2}))
	m = updated.(dashboardModel)
	updated, _ = m.Update(uiEventMsg(UIEvent{Model: "test-model", AlertRounds: 1}))
	m = updated.(dashboardModel)
	updated, _ = m.Update(uiEventMsg(UIEvent{Model: "other-model", AlertRounds: 0}))
	m = updated.(dashboardModel)

	if m.alertCounts["test-model"] != 3 {
		t.Errorf("alertCounts[test-model] = %d, want 3 (2+1 accumulated)", m.alertCounts["test-model"])
	}
	if _, ok := m.alertCounts["other-model"]; ok {
		t.Error("expected an event with AlertRounds=0 not to create an entry at all")
	}
}

func keyRune(r rune) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}}
}

// TestDashboardForceBucketKeybindings verifies keys 1-4 call
// SetForcedBucket with the right bucket and update local display state,
// and "0" clears back to auto-detect.
func TestDashboardForceBucketKeybindings(t *testing.T) {
	var lastForced TaskBucket
	var forceCalls, clearCalls int
	controls := DashboardControls{
		SetForcedBucket: func(b TaskBucket) {
			lastForced = b
			forceCalls++
		},
		ClearForcedBucket: func() {
			clearCalls++
		},
	}
	m := newDashboardModel(make(chan UIEvent), make(chan ProgressEvent), "127.0.0.1:9090", "local", nil, controls)

	cases := []struct {
		key    rune
		bucket TaskBucket
	}{
		{'1', BucketStrictCode},
		{'2', BucketExploratoryCode},
		{'3', BucketExplanation},
		{'4', BucketArchitecture},
	}
	for _, c := range cases {
		updated, _ := m.Update(keyRune(c.key))
		m = updated.(dashboardModel)
		if lastForced != c.bucket {
			t.Errorf("key %q: SetForcedBucket called with %q, want %q", c.key, lastForced, c.bucket)
		}
		if m.forcedBucket == nil || *m.forcedBucket != c.bucket {
			t.Errorf("key %q: dashboard's own forcedBucket display state = %v, want %q", c.key, m.forcedBucket, c.bucket)
		}
	}
	if forceCalls != len(cases) {
		t.Errorf("SetForcedBucket called %d times, want %d", forceCalls, len(cases))
	}

	updated, _ := m.Update(keyRune('0'))
	m = updated.(dashboardModel)
	if clearCalls != 1 {
		t.Errorf("ClearForcedBucket called %d times, want 1", clearCalls)
	}
	if m.forcedBucket != nil {
		t.Errorf("expected forcedBucket display state cleared, got %v", m.forcedBucket)
	}
}

// TestDashboardToggleThroughputKeybinding verifies "r" toggles the
// throughput panel's visibility flag.
func TestDashboardToggleThroughputKeybinding(t *testing.T) {
	m := newDashboardModel(make(chan UIEvent), make(chan ProgressEvent), "127.0.0.1:9090", "local", nil, DashboardControls{})
	if m.showThroughput {
		t.Fatal("expected showThroughput to start false")
	}
	updated, _ := m.Update(keyRune('r'))
	m = updated.(dashboardModel)
	if !m.showThroughput {
		t.Error("expected showThroughput = true after first 'r' press")
	}
	updated, _ = m.Update(keyRune('r'))
	m = updated.(dashboardModel)
	if m.showThroughput {
		t.Error("expected showThroughput = false after second 'r' press")
	}
}
