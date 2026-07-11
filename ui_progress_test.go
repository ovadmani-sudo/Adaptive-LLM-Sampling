package main

import (
	"strings"
	"testing"
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
	m := newDashboardModel(events, progressCh, "127.0.0.1:9090", "local")

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
