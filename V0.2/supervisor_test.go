package main

import (
	"fmt"
	"net"
	"net/http"
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

func TestSupervisorForcedBucketAllAcrossBackends(t *testing.T) {
	cfg := testConfig()
	p1, _ := NewProxyServer(cfg, nil, "local", nil, nil, "")
	p2, _ := NewProxyServer(cfg, &ProviderConfig{BaseURL: "https://x.invalid"}, "claude", nil, nil, "")
	sup := NewSupervisor()
	sup.Register("local", "local", freePort(t), p1.Handler(), p1)
	sup.Register("claude", "provider", freePort(t), p2.Handler(), p2)

	sup.SetForcedBucketAll(BucketArchitecture)
	if b, ok := p1.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Error("p1 forced bucket not set")
	}
	if b, ok := p2.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Error("p2 forced bucket not set")
	}
	if b, ok := sup.CurrentForcedBucket(); !ok || b != BucketArchitecture {
		t.Errorf("CurrentForcedBucket() = (%q,%v)", b, ok)
	}
	sup.ClearForcedBucketAll()
	if _, ok := sup.CurrentForcedBucket(); ok {
		t.Error("expected no forced bucket after ClearForcedBucketAll")
	}
}
