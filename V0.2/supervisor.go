package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// Supervisor manages a set of named listeners that can be started and stopped
// at runtime (the control panel's live enable/disable). Each listener wraps a
// long-lived handler (a ProxyServer's Handler, or the forward-proxy handler);
// Start binds its TCP port(s) and serves, Stop gracefully unbinds them. The
// handler/ProxyServer itself outlives start/stop cycles, so per-listener state
// (forced bucket, sampling-bypass) persists across toggles.
type Supervisor struct {
	mu    sync.Mutex
	items map[string]*managedListener
	order []string
}

// managedListener is one supervised port. proxy is the backing ProxyServer
// when the listener is an adaptive backend (nil for a pure passthrough), used
// to expose per-listener toggles in status.
type managedListener struct {
	name    string
	kind    string // "local", "provider", "forward-proxy"
	port    int
	handler http.Handler
	proxy   *ProxyServer

	srv     *http.Server
	lns     []net.Listener
	running bool
	lastErr string
}

// ListenerStatus is a snapshot of one listener for the control panel / TUI.
type ListenerStatus struct {
	Name           string `json:"name"`
	Kind           string `json:"kind"`
	Port           int    `json:"port"`
	Running        bool   `json:"running"`
	SupportsBypass bool   `json:"supports_bypass"`
	BypassSampling bool   `json:"bypass_sampling"`
	Alert          bool   `json:"alert"` // alert-continuation enabled for this backend
	Model          string `json:"model"` // effective outgoing model (override or configured)
	LastErr        string `json:"last_err,omitempty"`
}

func NewSupervisor() *Supervisor {
	return &Supervisor{items: make(map[string]*managedListener)}
}

// Register adds a listener definition (stopped). proxy may be nil. Registering
// a name that already exists is a no-op.
func (s *Supervisor) Register(name, kind string, port int, handler http.Handler, proxy *ProxyServer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[name]; ok {
		return
	}
	s.items[name] = &managedListener{name: name, kind: kind, port: port, handler: handler, proxy: proxy}
	s.order = append(s.order, name)
}

// Start binds the listener's port and begins serving. It binds both IPv4
// (127.0.0.1) and, best-effort, IPv6 (::1) loopback, since agent HTTP clients
// vary in how they resolve "localhost" (see the same fix in the connector).
// Already-running is a no-op; a bind failure is returned and recorded.
func (s *Supervisor) Start(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.running {
		return nil
	}

	var lns []net.Listener
	for _, host := range []string{"127.0.0.1", "[::1]"} {
		ln, err := net.Listen("tcp", fmt.Sprintf("%s:%d", host, ml.port))
		if err != nil {
			if host == "127.0.0.1" { // IPv4 is mandatory; IPv6 is best-effort
				for _, l := range lns {
					l.Close()
				}
				ml.lastErr = err.Error()
				return fmt.Errorf("bind %s:%d: %w", host, ml.port, err)
			}
			continue
		}
		lns = append(lns, ln)
	}

	srv := &http.Server{Handler: ml.handler}
	ml.srv = srv
	ml.lns = lns
	ml.running = true
	ml.lastErr = ""
	for _, ln := range lns {
		go func(ln net.Listener) {
			if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
				s.mu.Lock()
				ml.lastErr = err.Error()
				ml.running = false
				s.mu.Unlock()
			}
		}(ln)
	}
	return nil
}

// Stop gracefully shuts the listener's server down (closing its bound ports),
// leaving the underlying handler/ProxyServer intact for a later Start.
func (s *Supervisor) Stop(name string) error {
	s.mu.Lock()
	ml, ok := s.items[name]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("no such listener %q", name)
	}
	if !ml.running {
		s.mu.Unlock()
		return nil
	}
	srv := ml.srv
	lns := ml.lns
	ml.running = false
	ml.srv = nil
	ml.lns = nil
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := srv.Shutdown(ctx)
	// Also close the listeners explicitly: if Stop raced ahead of the
	// Serve goroutines, Shutdown may not have registered (and thus not
	// closed) them, which would leak the bound port. Close is safe to call
	// even if Serve already closed them (the extra error is ignored).
	for _, ln := range lns {
		_ = ln.Close()
	}
	return err
}

// SetBypass toggles adaptive-sampling pass-through on a backend listener.
func (s *Supervisor) SetBypass(name string, on bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support sampling bypass", name)
	}
	ml.proxy.SetBypassSampling(on)
	return nil
}

// SetAlert toggles alert-continuation on a backend live (control panel).
func (s *Supervisor) SetAlert(name string, on bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support alert", name)
	}
	ml.proxy.SetAlertEnabled(on)
	return nil
}

// SetModelOverride pins a backend's outgoing model live (control-panel model
// picker); "" clears it.
func (s *Supervisor) SetModelOverride(name, model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support a model override", name)
	}
	ml.proxy.SetModelOverride(model)
	return nil
}

// SetForcedBucketAll pins the classification bucket on every backend (the
// same "force mode" the TUI applies), and ClearForcedBucketAll reverts them
// all to auto-detect. Forward-proxy backends are included too.
func (s *Supervisor) SetForcedBucketAll(b TaskBucket) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ml := range s.items {
		if ml.proxy != nil {
			ml.proxy.SetForcedBucket(b)
		}
	}
}

func (s *Supervisor) ClearForcedBucketAll() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ml := range s.items {
		if ml.proxy != nil {
			ml.proxy.ClearForcedBucket()
		}
	}
}

// SetVisionDescribeAll flips [vision_describe] on every backend in
// lock-step — the feature is global by design (see VisionDescribeConfig),
// so unlike bypass/alert there is deliberately no per-listener variant.
func (s *Supervisor) SetVisionDescribeAll(on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ml := range s.items {
		if ml.proxy != nil {
			ml.proxy.SetVisionDescribe(on)
		}
	}
}

// VisionDescribeEnabled reports the global vision-describe state (all
// backends are kept in lock-step by SetVisionDescribeAll, so the first
// one's value is everyone's value), for panel display.
func (s *Supervisor) VisionDescribeEnabled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, name := range s.order {
		ml := s.items[name]
		if ml.proxy != nil {
			return ml.proxy.VisionDescribeEnabled()
		}
	}
	return false
}

// CurrentForcedBucket returns the forced bucket if any backend has one set
// (they're kept in lock-step by SetForcedBucketAll), for display.
func (s *Supervisor) CurrentForcedBucket() (TaskBucket, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, name := range s.order {
		ml := s.items[name]
		if ml.proxy != nil {
			if b, ok := ml.proxy.CurrentForcedBucket(); ok {
				return b, true
			}
		}
	}
	return "", false
}

// Status returns a snapshot of all listeners in registration order.
func (s *Supervisor) Status() []ListenerStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ListenerStatus, 0, len(s.order))
	for _, name := range s.order {
		ml := s.items[name]
		st := ListenerStatus{
			Name: ml.name, Kind: ml.kind, Port: ml.port,
			Running: ml.running, LastErr: ml.lastErr,
		}
		if ml.proxy != nil {
			st.SupportsBypass = true
			st.BypassSampling = ml.proxy.BypassSampling()
			st.Alert = ml.proxy.AlertEnabled()
			st.Model = ml.proxy.EffectiveModel()
		}
		out = append(out, st)
	}
	return out
}

// StopAll shuts every running listener down (process exit).
func (s *Supervisor) StopAll() {
	for _, st := range s.Status() {
		if st.Running {
			_ = s.Stop(st.Name)
		}
	}
}
