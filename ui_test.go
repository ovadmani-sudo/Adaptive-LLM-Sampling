package main

import (
	"strings"
	"testing"
	"time"
)

// TestFormatLogLineNeverWraps verifies a single log entry always renders
// as one line, even for a long, verbose upstream error body (regression:
// long 429/error messages previously word-wrapped into 4-5 rows inside
// the fixed-width dashboard box; with maxLogLines capping entry count but
// not rendered height, that pushed the bars panel above it off the top of
// the terminal).
func TestFormatLogLineNeverWraps(t *testing.T) {
	longError := `upstream returned 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your own rate limits: https://openrouter.ai/settings/integrations"}}}`

	ev := UIEvent{
		Timestamp:  time.Date(2026, 1, 1, 17, 32, 16, 0, time.UTC),
		RetryCount: 0,
		Error:      longError,
	}

	line := formatLogLine(ev)
	if strings.Contains(line, "\n") {
		t.Errorf("formatLogLine must never embed a newline, got: %q", line)
	}
	// logBoxWidth (80) minus border (2) minus padding (2) = 76 usable
	// columns; the whole formatted line must fit inside that or lipgloss
	// word-wraps it into multiple rows, which is exactly the bug this
	// guards against.
	const boxContentWidth = logBoxWidth - 4
	if len(line) > boxContentWidth {
		t.Errorf("formatLogLine result %d chars, exceeds box content width %d, will wrap: %q", len(line), boxContentWidth, line)
	}
}

// TestCondenseForLogLineCollapsesWhitespaceAndTruncates verifies embedded
// newlines/tabs are collapsed to spaces and the result is capped, so it
// can never itself introduce wrapping regardless of box width.
func TestCondenseForLogLineCollapsesWhitespaceAndTruncates(t *testing.T) {
	messy := "line one\nline two\ttabbed   spaced"
	got := condenseForLogLine(messy)
	if strings.ContainsAny(got, "\n\t") {
		t.Errorf("expected newlines/tabs collapsed, got: %q", got)
	}

	long := strings.Repeat("a", 200)
	got = condenseForLogLine(long)
	if len(got) > 31 { // 28 chars + "..."
		t.Errorf("expected truncation to ~28 chars, got %d chars: %q", len(got), got)
	}
	if !strings.HasSuffix(got, "...") {
		t.Errorf("expected truncated result to end with \"...\", got: %q", got)
	}
}

// TestFormatLogLineCleanRequestNeverWraps verifies the non-error path also
// stays within a safe single-line bound (bucket names and issue strings
// are bounded by config, but worth pinning down regardless).
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
