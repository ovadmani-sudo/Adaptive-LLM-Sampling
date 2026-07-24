package main

import (
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"
)

// freePort grabs an OS-assigned free TCP port and returns it (the listener is
// closed immediately; a brief reuse race is acceptable for tests).
func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port
}

func TestSupervisorStartStopBindsAndUnbinds(t *testing.T) {
	sup := NewSupervisor()
	port := freePort(t)
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	sup.Register("test", "provider", port, h, nil)

	url := fmt.Sprintf("http://127.0.0.1:%d/", port)

	// Not running yet → connection refused.
	if _, err := http.Get(url); err == nil {
		t.Fatal("expected connection to fail before Start")
	}

	if err := sup.Start("test"); err != nil {
		t.Fatalf("Start: %v", err)
	}
	// give the goroutine a moment to begin serving
	deadline := time.Now().Add(2 * time.Second)
	var got string
	for time.Now().Before(deadline) {
		if resp, err := http.Get(url); err == nil {
			resp.Body.Close()
			got = "up"
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got != "up" {
		t.Fatal("server not reachable after Start")
	}
	if st := sup.Status(); !st[0].Running {
		t.Error("Status should report running=true after Start")
	}

	if err := sup.Stop("test"); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	// after stop, the port should be free again (bindable)
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatalf("port not released after Stop: %v", err)
	}
	ln.Close()
	if st := sup.Status(); st[0].Running {
		t.Error("Status should report running=false after Stop")
	}
}

func TestSupervisorStartIsIdempotentAndRestartable(t *testing.T) {
	sup := NewSupervisor()
	port := freePort(t)
	sup.Register("x", "provider", port, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}), nil)

	if err := sup.Start("x"); err != nil {
		t.Fatalf("first Start: %v", err)
	}
	if err := sup.Start("x"); err != nil {
		t.Fatalf("second Start (idempotent) should be a no-op, got: %v", err)
	}
	if err := sup.Stop("x"); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if err := sup.Start("x"); err != nil {
		t.Fatalf("restart after Stop should work: %v", err)
	}
	sup.Stop("x")
}

func TestSupervisorSetBypassReflectedInStatus(t *testing.T) {
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), proxy.Handler(), proxy)

	st := sup.Status()[0]
	if !st.SupportsBypass || st.BypassSampling {
		t.Fatalf("initial: SupportsBypass=%v BypassSampling=%v, want true/false", st.SupportsBypass, st.BypassSampling)
	}
	if err := sup.SetBypass("local", true); err != nil {
		t.Fatalf("SetBypass: %v", err)
	}
	if !sup.Status()[0].BypassSampling {
		t.Error("BypassSampling should be true after SetBypass(true)")
	}
	if !proxy.BypassSampling() {
		t.Error("proxy.BypassSampling() should agree")
	}
}

// TestSupervisorForcedBucketIsPerListener verifies forcing a bucket on
// one listener never affects another — the whole point of making this
// per-listener instead of the old "apply to every backend at once"
// behavior: several agents can share this process on different ports
// without one agent's forced mode leaking into another's traffic.
func TestSupervisorForcedBucketIsPerListener(t *testing.T) {
	cfg := testConfig()
	p1, _ := NewProxyServer(cfg, nil, "local", nil, nil, "")
	p2, _ := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://x.invalid"}, "claude", nil, nil, "")
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), p1.Handler(), p1)
	sup.Register("claude", "provider", freePort(t), p2.Handler(), p2)

	if err := sup.SetForcedBucket("local", BucketArchitecture); err != nil {
		t.Fatalf("SetForcedBucket: %v", err)
	}
	if b, ok := p1.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Error("p1 (local) forced bucket not set")
	}
	if _, ok := p2.CurrentForcedBucket(); ok {
		t.Error("p2 (claude) must be unaffected by local's forced bucket")
	}

	status := sup.Status()
	var localStatus, claudeStatus ListenerStatus
	for _, st := range status {
		switch st.Name {
		case "local":
			localStatus = st
		case "claude":
			claudeStatus = st
		}
	}
	if localStatus.ForcedBucket != string(BucketArchitecture) {
		t.Errorf("local status ForcedBucket = %q, want %q", localStatus.ForcedBucket, BucketArchitecture)
	}
	if claudeStatus.ForcedBucket != "" {
		t.Errorf("claude status ForcedBucket = %q, want empty (unaffected)", claudeStatus.ForcedBucket)
	}

	if err := sup.SetForcedBucket("local", ""); err != nil {
		t.Fatalf("SetForcedBucket clear: %v", err)
	}
	if _, ok := p1.CurrentForcedBucket(); ok {
		t.Error("expected no forced bucket on local after clearing")
	}

	if err := sup.SetForcedBucket("does-not-exist", BucketArchitecture); err == nil {
		t.Error("expected an error for an unknown listener name")
	}
}

// TestSupervisorPersistsOnEveryToggle verifies every per-listener Set*
// call writes the new state to disk immediately, so a later restart (see
// TestSupervisorEnablePersistenceRestoresState) has something current to
// read — persistence would be silently useless if these calls didn't
// actually save.
func TestSupervisorPersistsOnEveryToggle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "listener_state.json")
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), proxy.Handler(), proxy)
	sup.EnablePersistence(path) // no file yet — arms persistence, nothing to restore

	if err := sup.SetBypass("local", true); err != nil {
		t.Fatalf("SetBypass: %v", err)
	}
	if err := sup.SetForcedBucket("local", BucketArchitecture); err != nil {
		t.Fatalf("SetForcedBucket: %v", err)
	}
	if err := sup.SetVisionDescribe("local", true); err != nil {
		t.Fatalf("SetVisionDescribe: %v", err)
	}
	if err := sup.SetSystemPrompt("local", "code"); err != nil {
		t.Fatalf("SetSystemPrompt: %v", err)
	}
	if err := sup.SetAlert("local", true); err != nil {
		t.Fatalf("SetAlert: %v", err)
	}
	if err := sup.SetModelOverride("local", "gpt-oss-120b"); err != nil {
		t.Fatalf("SetModelOverride: %v", err)
	}

	states, err := loadListenerStates(path)
	if err != nil {
		t.Fatalf("loadListenerStates: %v", err)
	}
	got, ok := states["local"]
	if !ok {
		t.Fatal(`expected a saved entry for "local"`)
	}
	want := ListenerState{
		BypassSampling: true, Alert: true, Model: "gpt-oss-120b",
		ForcedBucket: string(BucketArchitecture), VisionDescribe: true,
		SystemPrompt: "code",
	}
	if got != want {
		t.Errorf("saved state = %+v, want %+v", got, want)
	}
}

// TestSupervisorEnablePersistenceRestoresState verifies a fresh Supervisor
// reads a previously-saved state file (as if the process had just
// restarted) and reapplies every per-listener switch — including
// auto-starting a listener that was running when the state was last
// saved — so the user doesn't have to manually re-toggle every agent's
// mode/vision/prompt/running state after a restart.
func TestSupervisorEnablePersistenceRestoresState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "listener_state.json")
	saved := map[string]ListenerState{
		"local": {
			Running: true, BypassSampling: true, Alert: true,
			Model: "gpt-oss-120b", ForcedBucket: string(BucketArchitecture),
			VisionDescribe: true, SystemPrompt: "research",
		},
	}
	if err := saveListenerStatesAtomic(path, saved); err != nil {
		t.Fatalf("seed state file: %v", err)
	}

	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, nil, "local", nil, nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	sup := NewSupervisor()
	port := freePort(t)
	sup.Register("local", "local", port, proxy.Handler(), proxy)

	sup.EnablePersistence(path)

	if !proxy.BypassSampling() {
		t.Error("BypassSampling not restored")
	}
	if !proxy.AlertEnabled() {
		t.Error("AlertEnabled not restored")
	}
	if got := proxy.ModelOverride(); got != "gpt-oss-120b" {
		t.Errorf("ModelOverride = %q, want gpt-oss-120b", got)
	}
	if b, ok := proxy.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Errorf("CurrentForcedBucket = %q/%v, want architecture/true", b, ok)
	}
	if !proxy.VisionDescribeEnabled() {
		t.Error("VisionDescribeEnabled not restored")
	}
	if got := proxy.SystemPromptOverride(); got != "research" {
		t.Errorf("SystemPromptOverride = %q, want research", got)
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/", port)
	deadline := time.Now().Add(2 * time.Second)
	var up bool
	for time.Now().Before(deadline) {
		if resp, err := http.Get(url); err == nil {
			resp.Body.Close()
			up = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !up {
		t.Fatal("listener should have auto-started because its saved state had running=true")
	}
	sup.Stop("local")
}
