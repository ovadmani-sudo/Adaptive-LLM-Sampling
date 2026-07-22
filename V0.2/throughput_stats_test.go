package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestThroughputStatsPathForDerivesFromRetryLogPath(t *testing.T) {
	got := throughputStatsPathFor("/tmp/somedir/retry_log_local.jsonl", "local")
	want := "/tmp/somedir/throughput_stats_local.json"
	if got != want {
		t.Errorf("throughputStatsPathFor = %q, want %q", got, want)
	}
}

func TestThroughputStatsPathForEmptyWhenRetryLogPathEmpty(t *testing.T) {
	if got := throughputStatsPathFor("", "local"); got != "" {
		t.Errorf("throughputStatsPathFor(\"\", ...) = %q, want \"\" (trajectory logging disabled means stats persistence is too)", got)
	}
}

// TestSaveAndLoadThroughputStatsRoundTrip verifies entries written via
// saveThroughputStatsAtomic come back identical via loadThroughputStats —
// the basic contract the whole persistence feature depends on.
func TestSaveAndLoadThroughputStatsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "throughput_stats_local.json")
	entries := []ThroughputStatsEntry{
		{Provider: "local", Model: "gpt-oss-120b", Bucket: BucketStrictCode, Samples: 3, SumPromptTokens: 300, SumCompletionTokens: 150, SumLatencyMs: 3000, MinTokensPerSecond: 40, MaxTokensPerSecond: 60},
	}

	if err := saveThroughputStatsAtomic(path, entries); err != nil {
		t.Fatalf("saveThroughputStatsAtomic: %v", err)
	}

	loaded, err := loadThroughputStats(path)
	if err != nil {
		t.Fatalf("loadThroughputStats: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 loaded entry, got %d", len(loaded))
	}
	if loaded[0] != entries[0] {
		t.Errorf("loaded entry = %+v, want %+v", loaded[0], entries[0])
	}
}

// TestSaveThroughputStatsAtomicLeavesNoTempFile verifies the
// temp-file-then-rename write doesn't leave the intermediate .tmp file
// behind — a leftover .tmp would be confusing clutter and, if ever loaded
// by mistake, a source of stale data.
func TestSaveThroughputStatsAtomicLeavesNoTempFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "throughput_stats_local.json")
	if err := saveThroughputStatsAtomic(path, []ThroughputStatsEntry{{Provider: "local"}}); err != nil {
		t.Fatalf("saveThroughputStatsAtomic: %v", err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("expected no leftover .tmp file, stat error = %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected the final file to exist: %v", err)
	}
}

// TestLoadThroughputStatsMissingFileIsNotAnError verifies a first-run
// (file doesn't exist yet) returns a nil slice with no error, rather than
// failing startup.
func TestLoadThroughputStatsMissingFileIsNotAnError(t *testing.T) {
	entries, err := loadThroughputStats(filepath.Join(t.TempDir(), "does_not_exist.json"))
	if err != nil {
		t.Errorf("expected no error for a missing file, got %v", err)
	}
	if entries != nil {
		t.Errorf("expected nil entries for a missing file, got %+v", entries)
	}
}

func TestThroughputStatsEntryAverageTokensPerSecond(t *testing.T) {
	e := ThroughputStatsEntry{SumCompletionTokens: 300, SumLatencyMs: 6000}
	if got := e.AverageTokensPerSecond(); got != 50 {
		t.Errorf("AverageTokensPerSecond = %v, want 50 (300 tokens / 6s)", got)
	}

	zero := ThroughputStatsEntry{SumCompletionTokens: 100, SumLatencyMs: 0}
	if got := zero.AverageTokensPerSecond(); got != 0 {
		t.Errorf("AverageTokensPerSecond with 0 latency = %v, want 0 (guarded against division by zero)", got)
	}
}

// TestFormatThroughputReportSortsEntries verifies the report is sorted by
// provider, then model, then bucket, so repeated views are stable and
// easy to scan/diff rather than reflecting arbitrary map/slice order.
func TestFormatThroughputReportSortsEntries(t *testing.T) {
	entries := []ThroughputStatsEntry{
		{Provider: "openrouter", Model: "z-model", Bucket: BucketStrictCode, Samples: 1, SumCompletionTokens: 10, SumLatencyMs: 1000},
		{Provider: "local", Model: "a-model", Bucket: BucketArchitecture, Samples: 1, SumCompletionTokens: 10, SumLatencyMs: 1000},
		{Provider: "local", Model: "a-model", Bucket: BucketStrictCode, Samples: 1, SumCompletionTokens: 10, SumLatencyMs: 1000},
	}
	report := formatThroughputReport(entries)

	localArchIdx := strings.Index(report, "local")
	if localArchIdx == -1 {
		t.Fatal("expected 'local' provider to appear in the report")
	}
	strictIdx := strings.Index(report, string(BucketStrictCode))
	archIdx := strings.Index(report, string(BucketArchitecture))
	openrouterIdx := strings.Index(report, "openrouter")

	if !(localArchIdx < strictIdx && strictIdx < openrouterIdx) {
		t.Errorf("expected report rows ordered local/architecture, local/strict_code, openrouter — got:\n%s", report)
	}
	if archIdx > strictIdx {
		t.Errorf("expected architecture bucket (same provider+model) sorted before strict_code — got:\n%s", report)
	}
}

func TestFormatThroughputReportEmptyEntries(t *testing.T) {
	if got := formatThroughputReport(nil); got != "(no throughput data yet)" {
		t.Errorf("formatThroughputReport(nil) = %q, want the empty-state message", got)
	}
}
