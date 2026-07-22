package main

import (
	"strings"
	"testing"
	"time"
)

// TestFormatLogLinePreservesFullErrorText verifies a log entry always
// shows the complete error message (debug-friendly setting: an entry may
// wrap across multiple visual rows if it's long, rather than being cut
// short with "..." — maxLogLines' small count is what keeps total
// rendered height bounded instead of a per-entry length budget).
func TestFormatLogLinePreservesFullErrorText(t *testing.T) {
	longError := `upstream returned 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your own rate limits: https://openrouter.ai/settings/integrations"}}}`

	ev := UIEvent{
		Timestamp:  time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		RetryCount: 0,
		Error:      longError,
	}

	line := formatLogLine(ev)
	if strings.Contains(line, "\n") {
		t.Errorf("formatLogLine must never embed a literal newline (wrapping is lipgloss's job at render time), got: %q", line)
	}
	if !strings.Contains(line, longError) {
		t.Errorf("expected the full error text preserved untruncated, got: %q", line)
	}
}

// TestCondenseForLogLineCollapsesWhitespace verifies embedded
// newlines/tabs are collapsed to spaces without truncating the content.
func TestCondenseForLogLineCollapsesWhitespace(t *testing.T) {
	messy := "line one\nline two\ttabbed   spaced"
	got := condenseForLogLine(messy)
	if strings.ContainsAny(got, "\n\t") {
		t.Errorf("expected newlines/tabs collapsed, got: %q", got)
	}

	long := strings.Repeat("a", 200)
	got = condenseForLogLine(long)
	if len(got) != 200 {
		t.Errorf("expected full text preserved without truncation, got %d chars: %q", len(got), got)
	}
}

// TestFormatLogLineCleanRequestNeverWraps verifies the non-error path
// never embeds a literal newline.
func TestFormatLogLineCleanRequestNeverWraps(t *testing.T) {
	ev := UIEvent{
		Timestamp:  time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		Bucket:     BucketArchitecture,
		RetryCount: 2,
		Issue:      IssueNone,
		LatencyMs:  1234,
	}
	line := formatLogLine(ev)
	if strings.Contains(line, "\n") {
		t.Errorf("formatLogLine must never embed a newline, got: %q", line)
	}
}

// TestFormatLogLineWithHostShowsFullHost verifies forward-proxy mode's
// host= suffix (UIEvent.Host) is shown in full, not truncated.
// generativelanguage.googleapis.com:443 (38 chars) is one of the actual
// configured allowed_hosts, not a synthetic worst case.
func TestFormatLogLineWithHostShowsFullHost(t *testing.T) {
	const geminiHost = "generativelanguage.googleapis.com:443"

	clean := UIEvent{
		Timestamp:  time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		Bucket:     BucketStrictCode,
		RetryCount: 0,
		Issue:      IssueNone,
		LatencyMs:  22683,
		Host:       geminiHost,
	}
	line := formatLogLine(clean)
	if !strings.Contains(line, geminiHost) {
		t.Errorf("expected the full host %q to appear untruncated, got: %q", geminiHost, line)
	}

	withError := UIEvent{
		Timestamp:  time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		RetryCount: 1,
		Error:      `upstream returned 429: {"error":{"message":"rate limited"}}`,
		Host:       geminiHost,
	}
	errLine := formatLogLine(withError)
	if !strings.Contains(errLine, geminiHost) {
		t.Errorf("expected the full host %q to appear untruncated, got: %q", geminiHost, errLine)
	}
}

// TestFormatLogLineShowsModel verifies which model actually handled a
// request is visible in the completed-request log row — most useful when
// it differs from whatever's "current" for the listener (vision_describe or
// a live model override redirecting an individual request). Covers both
// the clean and error branches, since each builds its line separately.
func TestFormatLogLineShowsModel(t *testing.T) {
	clean := UIEvent{
		Timestamp: time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		Bucket:    BucketStrictCode,
		Issue:     IssueNone,
		LatencyMs: 1000,
		Model:     "qwen3-vl",
	}
	if line := formatLogLine(clean); !strings.Contains(line, "qwen3-vl") {
		t.Errorf("expected model %q to appear, got: %q", "qwen3-vl", line)
	}

	withError := UIEvent{
		Timestamp: time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		Error:     "upstream unreachable",
		Model:     "qwen3-vl",
	}
	if line := formatLogLine(withError); !strings.Contains(line, "qwen3-vl") {
		t.Errorf("expected model %q to appear in the error branch, got: %q", "qwen3-vl", line)
	}
}
