package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// newTestPanel wires a WebPanel over a supervisor holding one real backend
// ProxyServer, served via httptest (no TUI, no real ports).
func newTestPanel(t *testing.T) (*httptest.Server, *Supervisor, *ProxyServer) {
	t.Helper()
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), proxy.Handler(), proxy)
	broker := NewBroker()
	panel := NewWebPanel(sup, broker, func() []ThroughputStatsEntry { return nil }, "https://api.cline.bot/api/v1")
	return httptest.NewServer(panel.Handler()), sup, proxy
}

func TestWebPanelStatus(t *testing.T) {
	srv, _, _ := newTestPanel(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/status")
	if err != nil {
		t.Fatalf("GET status: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Listeners    []ListenerStatus `json:"listeners"`
		ForcedBucket string           `json:"forced_bucket"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Listeners) != 1 || out.Listeners[0].Name != "local" {
		t.Fatalf("listeners = %+v, want one named local", out.Listeners)
	}
	if !out.Listeners[0].SupportsBypass {
		t.Error("local backend should support bypass")
	}
}

func TestWebPanelBucketToggle(t *testing.T) {
	srv, sup, _ := newTestPanel(t)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/bucket?bucket=strict_code", "", nil)
	if err != nil {
		t.Fatalf("POST bucket: %v", err)
	}
	resp.Body.Close()
	if b, ok := sup.CurrentForcedBucket(); !ok || b != BucketStrictCode {
		t.Errorf("forced bucket = (%q,%v), want strict_code", b, ok)
	}

	// clear
	resp, _ = http.Post(srv.URL+"/api/bucket?bucket=", "", nil)
	resp.Body.Close()
	if _, ok := sup.CurrentForcedBucket(); ok {
		t.Error("bucket should be cleared")
	}
}

func TestWebPanelBypassToggle(t *testing.T) {
	srv, _, proxy := newTestPanel(t)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/bypass?name=local&on=true", "", nil)
	if err != nil {
		t.Fatalf("POST bypass: %v", err)
	}
	resp.Body.Close()
	if !proxy.BypassSampling() {
		t.Error("proxy should have sampling bypassed after POST bypass on=true")
	}
}

func TestWebPanelAlertToggle(t *testing.T) {
	srv, _, proxy := newTestPanel(t)
	defer srv.Close()

	if proxy.AlertEnabled() {
		t.Fatal("alert should start disabled")
	}
	resp, err := http.Post(srv.URL+"/api/alert?name=local&on=true", "", nil)
	if err != nil {
		t.Fatalf("POST alert: %v", err)
	}
	resp.Body.Close()
	if !proxy.AlertEnabled() {
		t.Error("alert should be enabled after POST alert on=true")
	}
	resp, _ = http.Post(srv.URL+"/api/alert?name=local&on=false", "", nil)
	resp.Body.Close()
	if proxy.AlertEnabled() {
		t.Error("alert should be disabled after POST alert on=false")
	}
}

// TestWebPanelVisionToggle verifies the global vision-describe switch:
// POST /api/vision flips every backend in lock-step, and /api/status
// reports the current state for the panel's checkbox.
func TestWebPanelVisionToggle(t *testing.T) {
	srv, sup, proxy := newTestPanel(t)
	defer srv.Close()

	// testConfig has VisionDescribe zero-valued, so it starts disabled;
	// with enabled=true in a real config.ini the seed works the same way
	// (see NewProxyServer) — the toggle just flips it live from there.
	if proxy.VisionDescribeEnabled() {
		t.Fatal("vision describe should start disabled with a zero-value config")
	}

	resp, err := http.Post(srv.URL+"/api/vision?on=true", "", nil)
	if err != nil {
		t.Fatalf("POST vision: %v", err)
	}
	resp.Body.Close()
	if !proxy.VisionDescribeEnabled() {
		t.Error("vision describe should be enabled after POST vision on=true")
	}
	if !sup.VisionDescribeEnabled() {
		t.Error("supervisor should report vision describe enabled")
	}

	// Status must expose it for the panel checkbox.
	sresp, err := http.Get(srv.URL + "/api/status")
	if err != nil {
		t.Fatalf("GET status: %v", err)
	}
	var out struct {
		VisionDescribe bool `json:"vision_describe"`
	}
	if err := json.NewDecoder(sresp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	sresp.Body.Close()
	if !out.VisionDescribe {
		t.Error("status vision_describe = false, want true")
	}

	resp, _ = http.Post(srv.URL+"/api/vision?on=false", "", nil)
	resp.Body.Close()
	if proxy.VisionDescribeEnabled() {
		t.Error("vision describe should be disabled after POST vision on=false")
	}
}

// TestVisionDescribeLiveToggleGatesRequestPath verifies the toggle takes
// effect on the very next request without a restart: flipping it off
// mid-session means an image-bearing request passes through untouched.
func TestVisionDescribeLiveToggleGatesRequestPath(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "a thing", &vlmCalls)
	defer vlm.Close()

	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("ok", "", "stop", 5))
	})
	defer closeFn()

	req := map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{imageMsg("what is this?", "data:image/png;base64,live-toggle-test")},
	}
	postChat(proxy, req)
	if got := atomic.LoadInt32(&vlmCalls); got != 1 {
		t.Fatalf("VLM calls = %d, want 1 while enabled", got)
	}

	proxy.SetVisionDescribe(false)
	// A DIFFERENT image, so a cache hit can't mask a wrongly-active
	// pipeline.
	postChat(proxy, map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{imageMsg("and this?", "data:image/png;base64,other-image")},
	})
	if got := atomic.LoadInt32(&vlmCalls); got != 1 {
		t.Errorf("VLM calls = %d, want still 1 after live-disabling", got)
	}
}

func TestWebPanelSetModel(t *testing.T) {
	srv, _, proxy := newTestPanel(t)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/model?name=local&model=cline-pass/qwen3.7-max", "", nil)
	if err != nil {
		t.Fatalf("POST model: %v", err)
	}
	resp.Body.Close()
	if proxy.ModelOverride() != "cline-pass/qwen3.7-max" {
		t.Errorf("model override = %q, want the posted model", proxy.ModelOverride())
	}
	// clear
	resp, _ = http.Post(srv.URL+"/api/model?name=local&model=", "", nil)
	resp.Body.Close()
	if proxy.ModelOverride() != "" {
		t.Error("model override should be cleared by empty model")
	}
}

func TestWebPanelIndexServesHTML(t *testing.T) {
	srv, _, _ := newTestPanel(t)
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "control panel") {
		t.Error("index should serve the control panel HTML")
	}
}
