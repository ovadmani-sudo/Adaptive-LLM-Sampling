package main

import (
	"context"
	"fmt"
	"log"
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
	mu        sync.Mutex
	items     map[string]*managedListener
	order     []string
	statePath string // set by EnablePersistence; "" = persistence off
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
	// ForcedBucket, VisionDescribe, SystemPrompt are all per-listener (see
	// SetForcedBucket/SetVisionDescribe/SetSystemPrompt) — deliberately
	// independent settings per backend, so running several agents through
	// this process on different ports never means one agent's forced
	// mode/prompt/vision setting leaks into another's traffic.
	ForcedBucket   string `json:"forced_bucket"` // "" = auto-detect
	VisionDescribe bool   `json:"vision_describe"`
	SystemPrompt   string `json:"system_prompt"` // "" = no injection
	LastErr        string `json:"last_err,omitempty"`
}

func NewSupervisor() *Supervisor {
	return &Supervisor{items: make(map[string]*managedListener)}
}

// EnablePersistence loads any previously-saved per-listener switch state
// from path and applies it to every currently-registered listener that has
// a saved entry, then arms every subsequent Start/Stop/Set* call to save
// state back to the same file. Call once, after every Register — so a
// listener registered later than this call would never receive its saved
// state. A listener with no saved entry (new name, or first run with no
// file yet) is left exactly as its caller set it up (config.ini / CLI-flag
// defaults), so a fresh install behaves the same as before this feature.
func (s *Supervisor) EnablePersistence(path string) {
	s.mu.Lock()
	s.statePath = path
	states, err := loadListenerStates(path)
	if err != nil {
		s.mu.Unlock()
		log.Printf("web: could not load listener state from %s: %v", path, err)
		return
	}
	var toStart []string
	for name, st := range states {
		ml, ok := s.items[name]
		if !ok || ml.proxy == nil {
			continue
		}
		ml.proxy.SetBypassSampling(st.BypassSampling)
		ml.proxy.SetAlertEnabled(st.Alert)
		ml.proxy.SetModelOverride(st.Model)
		if st.ForcedBucket == "" {
			ml.proxy.ClearForcedBucket()
		} else {
			ml.proxy.SetForcedBucket(TaskBucket(st.ForcedBucket))
		}
		ml.proxy.SetVisionDescribe(st.VisionDescribe)
		ml.proxy.SetSystemPromptOverride(st.SystemPrompt)
		if st.Running {
			toStart = append(toStart, name)
		}
	}
	s.mu.Unlock()

	// Start() binds ports and must not be called while s.mu is held.
	for _, name := range toStart {
		if err := s.Start(name); err != nil {
			log.Printf("web: could not restore running state for %q: %v", name, err)
		}
	}
}

// persist saves current per-listener switch state to disk, if persistence
// is enabled (see EnablePersistence). Caller must already hold s.mu.
func (s *Supervisor) persist() {
	if s.statePath == "" {
		return
	}
	states := make(map[string]ListenerState, len(s.items))
	for name, ml := range s.items {
		if ml.proxy == nil {
			continue
		}
		st := ListenerState{
			Running:        ml.running,
			BypassSampling: ml.proxy.BypassSampling(),
			Alert:          ml.proxy.AlertEnabled(),
			Model:          ml.proxy.ModelOverride(),
			VisionDescribe: ml.proxy.VisionDescribeEnabled(),
			SystemPrompt:   ml.proxy.SystemPromptOverride(),
		}
		if b, ok := ml.proxy.CurrentForcedBucket(); ok {
			st.ForcedBucket = string(b)
		}
		states[name] = st
	}
	if err := saveListenerStatesAtomic(s.statePath, states); err != nil {
		log.Printf("web: could not save listener state to %s: %v", s.statePath, err)
	}
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
				s.persist()
				s.mu.Unlock()
			}
		}(ln)
	}
	s.persist()
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
	s.persist()
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
	s.persist()
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
	s.persist()
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
	s.persist()
	return nil
}

// SetForcedBucket pins the classification bucket on ONE listener (the same
// "force mode" the TUI applies, scoped to a single backend); an empty
// bucket clears it back to auto-detect. Per-listener — deliberately NOT
// lock-stepped across backends: forcing a mode for one agent's task must
// not silently affect a different agent that happens to be routed
// through this same process on another port.
func (s *Supervisor) SetForcedBucket(name string, b TaskBucket) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support classification mode", name)
	}
	if b == "" {
		ml.proxy.ClearForcedBucket()
	} else {
		ml.proxy.SetForcedBucket(b)
	}
	s.persist()
	return nil
}

// SetVisionDescribe toggles [vision_describe] on ONE listener. Per-listener.
func (s *Supervisor) SetVisionDescribe(name string, on bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support vision describe", name)
	}
	ml.proxy.SetVisionDescribe(on)
	s.persist()
	return nil
}

// SetSystemPrompt selects a [system_prompt.*] entry by name on ONE
// listener; "" clears it back to no injection. Per-listener.
func (s *Supervisor) SetSystemPrompt(name, promptName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ml, ok := s.items[name]
	if !ok {
		return fmt.Errorf("no such listener %q", name)
	}
	if ml.proxy == nil {
		return fmt.Errorf("listener %q does not support a system prompt override", name)
	}
	ml.proxy.SetSystemPromptOverride(promptName)
	s.persist()
	return nil
}

// SystemPromptNames returns every configured [system_prompt.*] name
// (every backend shares the same cfg, so any one of them has the full
// list), for the web panel's per-listener dropdown options. This one
// stays a plain lookup, not per-listener state — it's the menu of
// available choices, identical everywhere, not a selection.
func (s *Supervisor) SystemPromptNames() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, name := range s.order {
		ml := s.items[name]
		if ml.proxy != nil {
			return ml.proxy.SystemPromptNames()
		}
	}
	return nil
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
			if b, ok := ml.proxy.CurrentForcedBucket(); ok {
				st.ForcedBucket = string(b)
			}
			st.VisionDescribe = ml.proxy.VisionDescribeEnabled()
			st.SystemPrompt = ml.proxy.SystemPromptOverride()
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
