package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
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
	srv, _, proxy := newTestPanel(t)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/bucket?name=local&bucket=strict_code", "", nil)
	if err != nil {
		t.Fatalf("POST bucket: %v", err)
	}
	resp.Body.Close()
	if b, ok := proxy.CurrentForcedBucket(); !ok || b != BucketStrictCode {
		t.Errorf("forced bucket = (%q,%v), want strict_code", b, ok)
	}

	// clear
	resp, _ = http.Post(srv.URL+"/api/bucket?name=local&bucket=", "", nil)
	resp.Body.Close()
	if _, ok := proxy.CurrentForcedBucket(); ok {
		t.Error("bucket should be cleared")
	}
}

// TestWebPanelBucketIsPerListener verifies the API-level guarantee behind
// TestSupervisorForcedBucketIsPerListener: forcing a mode on one
// listener's port must not affect another listener sharing the same
// panel — the scenario that motivated making this per-listener at all
// (multiple agents, each pointed at a different port).
func TestWebPanelBucketIsPerListener(t *testing.T) {
	cfg := testConfig()
	p1, _ := NewProxyServer(cfg, nil, "local", nil, nil, "")
	p2, _ := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://x.invalid"}, "claude", nil, nil, "")
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), p1.Handler(), p1)
	sup.Register("claude", "provider", freePort(t), p2.Handler(), p2)
	broker := NewBroker()
	panel := NewWebPanel(sup, broker, func() []ThroughputStatsEntry { return nil }, "https://api.cline.bot/api/v1")
	srv := httptest.NewServer(panel.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/bucket?name=local&bucket=architecture", "", nil)
	if err != nil {
		t.Fatalf("POST bucket: %v", err)
	}
	resp.Body.Close()
	if b, ok := p1.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Error("local should have the forced bucket")
	}
	if _, ok := p2.CurrentForcedBucket(); ok {
		t.Error("claude must be unaffected by local's forced bucket")
	}

	// An unknown listener name is rejected, not silently applied anywhere.
	resp, err = http.Post(srv.URL+"/api/bucket?name=does-not-exist&bucket=architecture", "", nil)
	if err != nil {
		t.Fatalf("POST bucket: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an unknown listener name", resp.StatusCode)
	}
	resp.Body.Close()
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

// TestWebPanelVisionToggle verifies the per-listener vision-describe
// switch: POST /api/vision?name=<listener> flips only that listener, and
// /api/status's per-row field reports the current state for the panel's
// checkbox.
func TestWebPanelVisionToggle(t *testing.T) {
	srv, _, proxy := newTestPanel(t)
	defer srv.Close()

	// testConfig has VisionDescribe zero-valued, so it starts disabled;
	// with enabled=true in a real config.ini the seed works the same way
	// (see NewProxyServer) — the toggle just flips it live from there.
	if proxy.VisionDescribeEnabled() {
		t.Fatal("vision describe should start disabled with a zero-value config")
	}

	resp, err := http.Post(srv.URL+"/api/vision?name=local&on=true", "", nil)
	if err != nil {
		t.Fatalf("POST vision: %v", err)
	}
	resp.Body.Close()
	if !proxy.VisionDescribeEnabled() {
		t.Error("vision describe should be enabled after POST vision on=true")
	}

	// Status must expose it (per-listener) for the panel checkbox.
	sresp, err := http.Get(srv.URL + "/api/status")
	if err != nil {
		t.Fatalf("GET status: %v", err)
	}
	var out struct {
		Listeners []ListenerStatus `json:"listeners"`
	}
	if err := json.NewDecoder(sresp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	sresp.Body.Close()
	if len(out.Listeners) != 1 || !out.Listeners[0].VisionDescribe {
		t.Errorf("status listeners = %+v, want local's vision_describe = true", out.Listeners)
	}

	resp, _ = http.Post(srv.URL+"/api/vision?name=local&on=false", "", nil)
	resp.Body.Close()
	if proxy.VisionDescribeEnabled() {
		t.Error("vision describe should be disabled after POST vision on=false")
	}
}

// TestWebPanelVisionIsPerListener verifies toggling one listener's
// vision-describe never affects another — the per-listener guarantee
// this feature exists for.
func TestWebPanelVisionIsPerListener(t *testing.T) {
	cfg := testConfig()
	p1, _ := NewProxyServer(cfg, nil, "local", nil, nil, "")
	p2, _ := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://x.invalid"}, "claude", nil, nil, "")
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), p1.Handler(), p1)
	sup.Register("claude", "provider", freePort(t), p2.Handler(), p2)
	broker := NewBroker()
	panel := NewWebPanel(sup, broker, func() []ThroughputStatsEntry { return nil }, "https://api.cline.bot/api/v1")
	srv := httptest.NewServer(panel.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/vision?name=local&on=true", "", nil)
	if err != nil {
		t.Fatalf("POST vision: %v", err)
	}
	resp.Body.Close()
	if !p1.VisionDescribeEnabled() {
		t.Error("local should have vision describe enabled")
	}
	if p2.VisionDescribeEnabled() {
		t.Error("claude must be unaffected by local's vision-describe toggle")
	}
}

// TestWebPanelSystemPromptToggle verifies the per-listener system-prompt
// selection: POST /api/system_prompt?name=<listener>&prompt=<name>
// selects it only on that listener, /api/status exposes both the
// per-listener selection and the global list of configured names for
// the dropdown, and an empty prompt clears it.
func TestWebPanelSystemPromptToggle(t *testing.T) {
	cfg := testConfig()
	cfg.SystemPrompts = map[string]string{"research": "You are a research assistant.", "code": "Be precise."}
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), proxy.Handler(), proxy)
	broker := NewBroker()
	panel := NewWebPanel(sup, broker, func() []ThroughputStatsEntry { return nil }, "https://api.cline.bot/api/v1")
	srv := httptest.NewServer(panel.Handler())
	defer srv.Close()

	if proxy.SystemPromptOverride() != "" {
		t.Fatal("system prompt override should start empty")
	}

	resp, err := http.Post(srv.URL+"/api/system_prompt?name=local&prompt=research", "", nil)
	if err != nil {
		t.Fatalf("POST system_prompt: %v", err)
	}
	resp.Body.Close()
	if proxy.SystemPromptOverride() != "research" {
		t.Errorf("proxy SystemPromptOverride() = %q, want %q", proxy.SystemPromptOverride(), "research")
	}

	sresp, err := http.Get(srv.URL + "/api/status")
	if err != nil {
		t.Fatalf("GET status: %v", err)
	}
	var out struct {
		Listeners     []ListenerStatus `json:"listeners"`
		SystemPrompts []string         `json:"system_prompts"`
	}
	if err := json.NewDecoder(sresp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	sresp.Body.Close()
	if len(out.Listeners) != 1 || out.Listeners[0].SystemPrompt != "research" {
		t.Errorf("status listeners = %+v, want local's system_prompt = research", out.Listeners)
	}
	wantNames := []string{"code", "research"} // sorted, per SystemPromptNames
	if !reflect.DeepEqual(out.SystemPrompts, wantNames) {
		t.Errorf("status system_prompts = %v, want %v", out.SystemPrompts, wantNames)
	}

	resp, _ = http.Post(srv.URL+"/api/system_prompt?name=local&prompt=", "", nil)
	resp.Body.Close()
	if proxy.SystemPromptOverride() != "" {
		t.Error("system prompt override should be cleared by an empty prompt")
	}
}

// TestWebPanelSystemPromptIsPerListener verifies selecting a prompt on
// one listener never affects another — the exact scenario this whole
// per-listener conversion exists for: two different agents, two
// different ports, two different dedicated prompts.
func TestWebPanelSystemPromptIsPerListener(t *testing.T) {
	cfg := testConfig()
	cfg.SystemPrompts = map[string]string{"research": "You are a research assistant.", "code": "Be precise."}
	p1, _ := NewProxyServer(cfg, nil, "local", nil, nil, "")
	p2, _ := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://x.invalid"}, "claude", nil, nil, "")
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), p1.Handler(), p1)
	sup.Register("claude", "provider", freePort(t), p2.Handler(), p2)
	broker := NewBroker()
	panel := NewWebPanel(sup, broker, func() []ThroughputStatsEntry { return nil }, "https://api.cline.bot/api/v1")
	srv := httptest.NewServer(panel.Handler())
	defer srv.Close()

	post := func(name, prompt string) {
		resp, err := http.Post(srv.URL+"/api/system_prompt?name="+name+"&prompt="+prompt, "", nil)
		if err != nil {
			t.Fatalf("POST system_prompt: %v", err)
		}
		resp.Body.Close()
	}
	post("local", "research")
	post("claude", "code")

	if p1.SystemPromptOverride() != "research" {
		t.Errorf("local SystemPromptOverride() = %q, want %q", p1.SystemPromptOverride(), "research")
	}
	if p2.SystemPromptOverride() != "code" {
		t.Errorf("claude SystemPromptOverride() = %q, want %q", p2.SystemPromptOverride(), "code")
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
