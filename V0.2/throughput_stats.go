package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ThroughputStatsEntry accumulates running throughput totals for one
// (provider, model, bucket) combination — persisted to disk (see
// loadThroughputStats/saveThroughputStatsAtomic) so this data survives the
// raw per-request retry_log_*.jsonl being cleared or rotated away, and
// across proxy restarts (reloaded on startup). SumCompletionTokens/
// SumLatencyMs (not a running average) let AverageTokensPerSecond compute
// a weighted average across all samples, which is more statistically
// correct than averaging each request's own rate — a handful of tiny/fast
// requests would otherwise skew a plain average disproportionately.
type ThroughputStatsEntry struct {
	Provider            string     `json:"provider"`
	Model               string     `json:"model"`
	Bucket              TaskBucket `json:"bucket"`
	Samples             int        `json:"samples"`
	SumPromptTokens     int64      `json:"sum_prompt_tokens"`
	SumCompletionTokens int64      `json:"sum_completion_tokens"`
	SumLatencyMs        int64      `json:"sum_latency_ms"`
	MinTokensPerSecond  float64    `json:"min_tokens_per_second"`
	MaxTokensPerSecond  float64    `json:"max_tokens_per_second"`
	LastUpdated         time.Time  `json:"last_updated"`
}

// AverageTokensPerSecond is the weighted average across every sample this
// entry has accumulated: total completion tokens generated divided by
// total time spent generating them.
func (e ThroughputStatsEntry) AverageTokensPerSecond() float64 {
	if e.SumLatencyMs <= 0 {
		return 0
	}
	return float64(e.SumCompletionTokens) / (float64(e.SumLatencyMs) / 1000.0)
}

// throughputStatsPathFor derives the persistent stats file path from the
// retry log's own directory and the provider label — e.g.
// ".../retry_log_local.jsonl" with providerLabel "local" yields
// ".../throughput_stats_local.json". A distinct filename/pattern from the
// *.jsonl retry log is deliberate: if whatever process or habit
// occasionally clears retry_log_*.jsonl doesn't also match this name, the
// aggregated history survives that. Empty retryLogPath (trajectory
// logging disabled, e.g. most tests) disables this too — same "no file on
// disk" intent, and avoids a constructor signature change since this
// derives from a parameter NewProxyServer already takes.
func throughputStatsPathFor(retryLogPath, providerLabel string) string {
	if retryLogPath == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(retryLogPath), fmt.Sprintf("throughput_stats_%s.json", providerLabel))
}

// loadThroughputStats reads a previously-persisted stats file. A missing
// file is not an error (first run) — returns a nil slice.
func loadThroughputStats(path string) ([]ThroughputStatsEntry, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	var entries []ThroughputStatsEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

// saveThroughputStatsAtomic writes the full stats slice to path via a
// temp-file-then-rename, so a process killed mid-write can never leave a
// half-written, corrupt stats file behind for the next load to choke on.
func saveThroughputStatsAtomic(path string, entries []ThroughputStatsEntry) error {
	if path == "" {
		return nil
	}
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// formatThroughputReport renders a plain-text table of every stats entry,
// sorted by provider then model then bucket — shared by the --report CLI
// mode (main.go) and the dashboard's summary panel (ui.go) so both always
// show identical figures.
func formatThroughputReport(entries []ThroughputStatsEntry) string {
	if len(entries) == 0 {
		return "(no throughput data yet)"
	}

	sorted := make([]ThroughputStatsEntry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Provider != sorted[j].Provider {
			return sorted[i].Provider < sorted[j].Provider
		}
		if sorted[i].Model != sorted[j].Model {
			return sorted[i].Model < sorted[j].Model
		}
		return sorted[i].Bucket < sorted[j].Bucket
	})

	const rowFmt = "%-12s  %-28s  %-15s  %7s  %8s  %8s  %8s\n"
	var b strings.Builder
	fmt.Fprintf(&b, rowFmt, "PROVIDER", "MODEL", "BUCKET", "SAMPLES", "AVG T/S", "MIN T/S", "MAX T/S")
	for _, e := range sorted {
		fmt.Fprintf(&b, rowFmt,
			e.Provider, e.Model, e.Bucket,
			fmt.Sprintf("%d", e.Samples),
			fmt.Sprintf("%.1f", e.AverageTokensPerSecond()),
			fmt.Sprintf("%.1f", e.MinTokensPerSecond),
			fmt.Sprintf("%.1f", e.MaxTokensPerSecond),
		)
	}
	return b.String()
}
