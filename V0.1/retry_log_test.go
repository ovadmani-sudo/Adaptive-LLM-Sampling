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
)

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
// never retried writes nothing to the log — there's no trajectory to
// record for a request that needed no adjustment.
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
