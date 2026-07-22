package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// readJSONLines returns only the retry_trajectory-shaped lines from the
// log — it shares a file with ThroughputLogEntry (see RetryLogEntry.Type),
// so every reader must filter by Type rather than assume every line
// unmarshals into the shape it wants. An empty Type is also accepted as a
// retry trajectory for backward compatibility with log files written
// before this field existed.
func readJSONLines(t *testing.T, path string) []RetryLogEntry {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatalf("opening retry log: %v", err)
	}
	defer f.Close()

	var entries []RetryLogEntry
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var e RetryLogEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("retry log line is not valid JSON: %v\nline: %s", err, line)
		}
		if e.Type != "" && e.Type != "retry_trajectory" {
			continue
		}
		entries = append(entries, e)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanning retry log: %v", err)
	}
	return entries
}

// readThroughputJSONLines is readJSONLines' counterpart for the other
// entry shape sharing the same file.
func readThroughputJSONLines(t *testing.T, path string) []ThroughputLogEntry {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatalf("opening retry log: %v", err)
	}
	defer f.Close()

	var entries []ThroughputLogEntry
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var e ThroughputLogEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("retry log line is not valid JSON: %v\nline: %s", err, line)
		}
		if e.Type != "throughput" {
			continue
		}
		entries = append(entries, e)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanning retry log: %v", err)
	}
	return entries
}

// TestRetryTrajectoryLoggedOnSuccessfulRetry verifies a request that
// retried at least once gets its full adjustment trajectory written to the
// retry log as a single JSON line, with the shape needed to compare
// outcomes across different retry_step_exponent values.
func TestRetryTrajectoryLoggedOnSuccessfulRetry(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			w.Write(chatResponse(strings.Repeat("loop loop loop ", 5), "stop"))
		} else {
			w.Write(chatResponse("clean answer", "stop"))
		}
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Retry.StepExponent = 2.0
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	entries := readJSONLines(t, logPath)
	if len(entries) != 1 {
		t.Fatalf("expected 1 retry log entry, got %d", len(entries))
	}

	e := entries[0]
	if e.Bucket != BucketStrictCode {
		t.Errorf("Bucket = %q, want %q", e.Bucket, BucketStrictCode)
	}
	if e.StepExponent != 2.0 {
		t.Errorf("StepExponent = %v, want 2.0", e.StepExponent)
	}
	if !e.Resolved {
		t.Error("expected Resolved=true (the retry did fix it)")
	}
	if e.TotalAttempts != 1 {
		t.Errorf("TotalAttempts = %d, want 1", e.TotalAttempts)
	}
	if len(e.Adjustments) != 1 {
		t.Fatalf("expected 1 adjustment, got %d", len(e.Adjustments))
	}
	adj := e.Adjustments[0]
	if adj.Issue != IssueRepetition {
		t.Errorf("adjustment Issue = %q, want %q", adj.Issue, IssueRepetition)
	}
	if adj.Attempt != 0 {
		t.Errorf("adjustment Attempt = %d, want 0", adj.Attempt)
	}
	if !strings.Contains(adj.Detail, "loop") {
		t.Errorf("adjustment Detail = %q, want the repeated n-gram that triggered detection — without it, the log can't distinguish genuine degeneration from a false positive on legitimately repeated structure", adj.Detail)
	}
}

// TestRetryTrajectoryNotLoggedOnCleanFirstPass verifies a request that
// never retried writes no *retry_trajectory* entry — there's no trajectory
// to record for a request that needed no adjustment. It does still get a
// throughput entry (see TestThroughputLoggedOnCleanFirstPass below) since
// the two are logged independently to the same file.
func TestRetryTrajectoryNotLoggedOnCleanFirstPass(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse("clean answer", "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	entries := readJSONLines(t, logPath)
	if len(entries) != 0 {
		t.Errorf("expected no retry log entries for a clean first-pass request, got %d", len(entries))
	}
}

// TestRetryTrajectoryLoggedOnExhaustedRetries verifies a request that used
// up all its retries without resolving still gets logged, with
// Resolved=false — this negative signal matters as much as successes for
// judging whether an exponent is too timid or too aggressive.
func TestRetryTrajectoryLoggedOnExhaustedRetries(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse(strings.Repeat("loop loop loop ", 5), "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	entries := readJSONLines(t, logPath)
	if len(entries) != 1 {
		t.Fatalf("expected 1 retry log entry, got %d", len(entries))
	}
	if entries[0].Resolved {
		t.Error("expected Resolved=false when retries were exhausted without success")
	}
	if entries[0].TotalAttempts != cfg.Server.MaxRetries {
		t.Errorf("TotalAttempts = %d, want %d (max_retries)", entries[0].TotalAttempts, cfg.Server.MaxRetries)
	}
}

// TestReasoningGuardAbortsRunawayLoopAndRetries is the regression test for
// the confirmed-via-packet-capture Gemma failure mode: a model stuck
// emitting reasoning_content indefinitely, never reaching a finish_reason
// or [DONE] at all. Every other detection mechanism only ever inspects a
// finished response, so this fake upstream deliberately never finishes its
// first response — if ReasoningGuard didn't abort it, this test would hang
// until the suite's own timeout. The second call simulates the retry
// succeeding normally.
func TestReasoningGuardAbortsRunawayLoopAndRetries(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		flusher := w.(http.Flusher)
		writeChunk := func(delta map[string]interface{}, finish interface{}) bool {
			chunk := map[string]interface{}{
				"choices": []map[string]interface{}{
					{"index": 0, "delta": delta, "finish_reason": finish},
				},
			}
			data, _ := json.Marshal(chunk)
			if _, err := w.Write([]byte("data: ")); err != nil {
				return false
			}
			if _, err := w.Write(data); err != nil {
				return false
			}
			if _, err := w.Write([]byte("\n\n")); err != nil {
				return false
			}
			flusher.Flush()
			return true
		}

		if n == 1 {
			// Never emits real content, never a finish_reason, never
			// [DONE] — only ReasoningGuard's live abort can end this.
			for i := 0; i < 50; i++ {
				if !writeChunk(map[string]interface{}{"reasoning_content": "loop "}, nil) {
					return
				}
				time.Sleep(2 * time.Millisecond)
			}
		} else {
			w.Write(sseChatStream("clean answer after retry", "", "stop", 5))
		}
	}))
	defer upstream.Close()

	cfg := testConfig()
	// The strict_code preset injects thinking_budget_tokens=512; a small
	// multiplier keeps the cap tiny so the test doesn't need 512+ chunks.
	cfg.ReasoningGuard = ReasoningGuardConfig{
		Enabled:                    true,
		BudgetMultiplier:           0.01, // 512 * 0.01 = ~5
		FallbackMaxReasoningTokens: 5,
	}
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	rec := postChat(proxy, map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "please fix bug"},
		},
	})

	if got := atomic.LoadInt32(&callCount); got < 2 {
		t.Fatalf("expected at least 2 upstream calls (runaway loop aborted, then a retry), got %d", got)
	}
	if body := rec.Body.String(); !strings.Contains(body, "clean answer after retry") {
		t.Errorf("expected the retried clean answer in the final response, got: %s", body)
	}

	entries := readJSONLines(t, logPath)
	if len(entries) != 1 {
		t.Fatalf("expected 1 retry log entry, got %d", len(entries))
	}
	if len(entries[0].Adjustments) == 0 {
		t.Fatalf("expected at least 1 adjustment, got 0")
	}
	if adj := entries[0].Adjustments[0]; adj.Issue != IssueReasoningLoop {
		t.Errorf("adjustment Issue = %q, want %q", adj.Issue, IssueReasoningLoop)
	}
	if !entries[0].Resolved {
		t.Error("expected Resolved=true (the retry did produce a clean answer)")
	}
}

// TestRetryTrajectoryConcurrentWritesDontInterleave fires many
// retry-triggering requests concurrently and verifies every resulting log
// line is still independently valid JSON — the mutex around the log file
// write must fully serialize writes, or concurrent requests could produce
// a torn/interleaved line that breaks offline analysis.
func TestRetryTrajectoryConcurrentWritesDontInterleave(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(chatResponse(strings.Repeat("loop loop loop ", 5), "stop"))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 64)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	const n = 20
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			postChat(proxy, map[string]interface{}{
				"messages": []interface{}{
					map[string]interface{}{"role": "user", "content": "please fix bug"},
				},
			})
		}()
	}
	wg.Wait()

	entries := readJSONLines(t, logPath) // fails the test via t.Fatalf on any invalid line
	if len(entries) != n {
		t.Errorf("expected %d retry log entries, got %d", n, len(entries))
	}
}

// TestThroughputLoggedOnCleanFirstPass verifies a clean (no retry) chat
// request logs exactly one throughput entry with the real provider/model/
// bucket/token counts and a tokens/sec figure derived from them — and that
// it's independent of the retry-trajectory log (zero retry_trajectory
// entries for the same request, per TestRetryTrajectoryNotLoggedOnCleanFirstPass).
func TestThroughputLoggedOnCleanFirstPass(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("a clean answer here", "", "stop", 55))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	throughput := readThroughputJSONLines(t, logPath)
	if len(throughput) != 1 {
		t.Fatalf("expected 1 throughput entry, got %d", len(throughput))
	}
	e := throughput[0]
	if e.Provider != "local" {
		t.Errorf("Provider = %q, want %q", e.Provider, "local")
	}
	if e.Model != "test-model" {
		t.Errorf("Model = %q, want %q", e.Model, "test-model")
	}
	if e.Bucket != BucketStrictCode {
		t.Errorf("Bucket = %q, want %q", e.Bucket, BucketStrictCode)
	}
	if e.PromptTokens != 10 {
		t.Errorf("PromptTokens = %d, want 10 (sseChatStream's fixed usage.prompt_tokens)", e.PromptTokens)
	}
	if e.CompletionTokens != 55 {
		t.Errorf("CompletionTokens = %d, want 55", e.CompletionTokens)
	}
	// A fast in-process test can measure 0ms latency; production code
	// guards against dividing by that (see logThroughput), so the expected
	// value must mirror the same guard rather than assume latency > 0.
	var wantRate float64
	if e.LatencyMs > 0 {
		wantRate = float64(55) / (float64(e.LatencyMs) / 1000.0)
	}
	if e.TokensPerSecond != wantRate {
		t.Errorf("TokensPerSecond = %v, want %v (completion_tokens / latency)", e.TokensPerSecond, wantRate)
	}

	if retryEntries := readJSONLines(t, logPath); len(retryEntries) != 0 {
		t.Errorf("expected 0 retry_trajectory entries for a clean request, got %d", len(retryEntries))
	}
}

// TestThroughputNotLoggedWhenRequestRetried verifies a request that needed
// a retry does NOT get a throughput entry — its total latency includes a
// discarded attempt's wasted regeneration time, which would understate the
// real tokens/sec of the generation that actually reached the client.
func TestThroughputNotLoggedWhenRequestRetried(t *testing.T) {
	var callCount int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		if n == 1 {
			w.Write(sseChatStream(strings.Repeat("loop loop loop ", 5), "", "stop", 15))
		} else {
			w.Write(sseChatStream("clean answer", "", "stop", 5))
		}
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})

	if throughput := readThroughputJSONLines(t, logPath); len(throughput) != 0 {
		t.Errorf("expected 0 throughput entries for a retried request, got %d: %+v", len(throughput), throughput)
	}
	if retryEntries := readJSONLines(t, logPath); len(retryEntries) != 1 {
		t.Errorf("expected 1 retry_trajectory entry (sanity check the retry log itself still works), got %d", len(retryEntries))
	}
	if stats := proxy.ThroughputStatsSnapshot(); len(stats) != 0 {
		t.Errorf("expected no persistent throughput stats entry for a retried request either, got %+v", stats)
	}
}

// TestThroughputStatsAccumulateAcrossRequests verifies two clean requests
// against the same model/bucket accumulate into one entry (not two), with
// sums, min/max, and sample count all correctly updated — this is the
// data behind the --report CLI flag and the dashboard's summary panel.
func TestThroughputStatsAccumulateAcrossRequests(t *testing.T) {
	completionTokens := []int{50, 100}
	var call int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := completionTokens[call]
		call++
		w.Write(sseChatStream("clean answer", "", "stop", n))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	for i := 0; i < 2; i++ {
		postChat(proxy, map[string]interface{}{
			"model":    "test-model",
			"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
		})
	}

	stats := proxy.ThroughputStatsSnapshot()
	if len(stats) != 1 {
		t.Fatalf("expected 1 accumulated entry (same provider/model/bucket), got %d: %+v", len(stats), stats)
	}
	e := stats[0]
	if e.Samples != 2 {
		t.Errorf("Samples = %d, want 2", e.Samples)
	}
	if e.SumCompletionTokens != 150 {
		t.Errorf("SumCompletionTokens = %d, want 150 (50+100)", e.SumCompletionTokens)
	}
	if e.SumPromptTokens != 20 {
		t.Errorf("SumPromptTokens = %d, want 20 (sseChatStream's fixed 10 per request x2)", e.SumPromptTokens)
	}
}

// TestThroughputUsesGenerationTimeNotTotalRequestLatency is the regression
// test for the bug where the persisted throughput report read roughly half
// of what the live in-flight indicator showed for the same requests: the
// fake upstream here simulates a slow prompt-processing (prefill) phase
// BEFORE the first token, followed by a much shorter real generation phase
// — total request latency is dominated by the prefill sleep, but tok/s
// must be computed from the generation phase alone (see
// streamResult.generationElapsedMs and finishClassifiedRequest's
// generationMs). If this ever regresses back to dividing by total latency,
// the reported rate collapses toward completionTokens/totalLatency, well
// below the assertions below.
func TestThroughputUsesGenerationTimeNotTotalRequestLatency(t *testing.T) {
	const completionTokens = 100
	const prefillDelay = 300 * time.Millisecond
	const generationDelay = 60 * time.Millisecond

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher := w.(http.Flusher)
		writeChunk := func(delta map[string]interface{}, finish interface{}) {
			chunk := map[string]interface{}{
				"choices": []map[string]interface{}{
					{"index": 0, "delta": delta, "finish_reason": finish},
				},
			}
			data, _ := json.Marshal(chunk)
			w.Write([]byte("data: "))
			w.Write(data)
			w.Write([]byte("\n\n"))
			flusher.Flush()
		}

		// Simulated prompt processing: nothing hits the wire yet, so
		// genStart (stream.go) hasn't started — this time must NOT count
		// toward generation elapsed time.
		time.Sleep(prefillDelay)
		writeChunk(map[string]interface{}{"role": "assistant"}, nil)
		writeChunk(map[string]interface{}{"content": "a clean answer here"}, nil)

		// Real decode time: genStart is now running (triggered by the
		// content delta above). This IS what tok/s must be divided by.
		time.Sleep(generationDelay)
		writeChunk(map[string]interface{}{}, "stop")

		usageChunk := map[string]interface{}{
			"choices": []interface{}{},
			"usage":   map[string]interface{}{"completion_tokens": completionTokens, "prompt_tokens": 10},
		}
		data, _ := json.Marshal(usageChunk)
		w.Write([]byte("data: "))
		w.Write(data)
		w.Write([]byte("\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)
	proxy, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	start := time.Now()
	postChat(proxy, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	totalLatencyMs := float64(time.Since(start).Milliseconds())

	throughput := readThroughputJSONLines(t, logPath)
	if len(throughput) != 1 {
		t.Fatalf("expected 1 throughput entry, got %d", len(throughput))
	}
	e := throughput[0]

	// The persisted LatencyMs field (semantics changed by this fix — see
	// logThroughput's doc comment) must reflect only the generation phase,
	// not the full request including the simulated prefill sleep.
	if float64(e.LatencyMs) >= totalLatencyMs {
		t.Errorf("LatencyMs = %v, want well under total request latency %v (prefill time must be excluded)", e.LatencyMs, totalLatencyMs)
	}
	if float64(e.LatencyMs) > totalLatencyMs*0.6 {
		t.Errorf("LatencyMs = %v is too close to total latency %v; prefillDelay (%v) doesn't look excluded", e.LatencyMs, totalLatencyMs, prefillDelay)
	}

	totalBasedRate := float64(completionTokens) / (totalLatencyMs / 1000.0)
	if e.TokensPerSecond <= totalBasedRate*1.3 {
		t.Errorf("TokensPerSecond = %v, want meaningfully higher than the total-latency-based rate %v (the historical bug's value)", e.TokensPerSecond, totalBasedRate)
	}

	stats := proxy.ThroughputStatsSnapshot()
	if len(stats) != 1 {
		t.Fatalf("expected 1 accumulated throughput-stats entry, got %d", len(stats))
	}
	if got := stats[0].AverageTokensPerSecond(); got <= totalBasedRate*1.3 {
		t.Errorf("persisted stats AverageTokensPerSecond() = %v, want meaningfully higher than the total-latency-based rate %v", got, totalBasedRate)
	}
}

// TestThroughputStatsPersistAcrossRestart verifies stats accumulated by
// one ProxyServer instance are correctly loaded back by a brand new
// instance pointed at the same retryLogPath — the core "survives a
// restart" guarantee this feature exists for.
func TestThroughputStatsPersistAcrossRestart(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("clean answer", "", "stop", 42))
	}))
	defer upstream.Close()

	cfg := testConfig()
	u, _ := url.Parse(upstream.URL)
	host, portStr, _ := strings.Cut(u.Host, ":")
	port, _ := strconv.Atoi(portStr)
	cfg.Server.UpstreamHost = host
	cfg.Server.UpstreamPort = port

	logPath := filepath.Join(t.TempDir(), "retry_log.jsonl")
	events := make(chan UIEvent, 8)

	proxy1, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer (first instance): %v", err)
	}
	postChat(proxy1, map[string]interface{}{
		"model":    "test-model",
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "please fix bug"}},
	})
	proxy1.Close()

	// A brand new instance, same retryLogPath — simulating a proxy restart.
	proxy2, err := NewProxyServer(cfg, nil, "local", events, nil, logPath)
	if err != nil {
		t.Fatalf("NewProxyServer (second instance): %v", err)
	}
	defer proxy2.Close()

	stats := proxy2.ThroughputStatsSnapshot()
	if len(stats) != 1 {
		t.Fatalf("expected the new instance to load 1 persisted entry, got %d", len(stats))
	}
	if stats[0].Samples != 1 || stats[0].SumCompletionTokens != 42 {
		t.Errorf("loaded entry = %+v, want Samples=1 SumCompletionTokens=42", stats[0])
	}
}
