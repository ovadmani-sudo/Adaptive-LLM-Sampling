package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type endpointKind int

const (
	kindChat endpointKind = iota
	kindCompletion
)

// ProxyServer holds the shared state for the reverse proxy: config, the
// upstream target, an HTTP client, and the channel used to push dashboard
// updates without blocking request handling. When provider is non-nil, the
// upstream target is a remote OpenAI-compatible backend (claude/gemini/
// openai/openrouter) selected via CLI argument instead of the local
// llama-server.
type ProxyServer struct {
	cfg      *Config
	client   *http.Client
	events   chan<- UIEvent
	progress chan<- ProgressEvent

	// providerMu guards provider/providerLabel/upstreamBase/passthrough —
	// together they describe the currently active remote-provider target,
	// which SwitchProvider can hot-swap at runtime (a single-provider
	// agent like Codex, hardcoded to one API, can be redirected to a
	// different real backend without restarting the proxy). Always
	// uncontended in local mode and in forward-proxy mode's own instance,
	// neither of which ever switches — provider's nilness itself (local
	// vs. provider mode) is fixed for the process lifetime and safe to
	// read without this lock; only the *contents* once non-nil change.
	providerMu    sync.RWMutex
	provider      *ProviderConfig
	providerLabel string
	upstreamBase  string
	passthrough   *httputil.ReverseProxy

	// listenerName is this instance's fixed identity — the same name it
	// was registered under in the Supervisor (see web_mode.go's
	// mkBackend) — stamped onto every UIEvent/ProgressEvent so the web
	// panel can attribute activity to a specific listener/session (the
	// "In-flight" card's session selector). Deliberately a separate,
	// immutable field from providerLabel: providerLabel can change live
	// via SwitchProvider (a single-provider agent redirected to a
	// different real backend), but which *listener* a request came in on
	// never does.
	listenerName string

	// forcedBucketMu guards forcedBucket — set via the dashboard to pin a
	// classification bucket for every request, bypassing Classify()
	// entirely, when auto-detection doesn't pick the right one. nil means
	// "auto-detect" (the default, original behavior).
	forcedBucketMu sync.RWMutex
	forcedBucket   *TaskBucket

	// bypassSampling, when true, makes this instance forward chat requests
	// with the client's own sampling params untouched and no retry loop —
	// the adaptive pipeline (preset injection, detect, exponential retry) is
	// skipped, but streaming/progress/throughput tracking still work. It's a
	// live, per-listener toggle (see SetBypassSampling), used by the control
	// panel's "pass through dynamic sampling" switch. Atomic so it can be
	// flipped concurrently with in-flight requests.
	bypassSampling atomic.Bool

	// modelOverride, when non-empty, replaces the outgoing "model" on chat
	// requests — a live, per-listener choice from the control panel's model
	// picker that wins over the static config.ini Model. Empty means "use the
	// configured provider Model (or the client's own model)". Guarded by
	// modelMu so it can change concurrently with in-flight requests.
	modelMu       sync.RWMutex
	modelOverride string

	// alertEnabled gates alert-continuation for THIS instance (see alert.go).
	// It starts from the global --alert flag (cfg.Server.AlertEnabled) but is
	// then a live per-listener toggle from the control panel, so alert can be
	// enabled on the local backend alone without affecting the others (which
	// all share the same cfg). Atomic — flipped concurrently with requests.
	alertEnabled atomic.Bool

	// visionDescribeEnabled gates [vision_describe] (vision_describe.go).
	// Seeded from cfg.VisionDescribe.Enabled, then a live, per-listener
	// toggle from the control panel (see Supervisor.SetVisionDescribe) —
	// independent per instance, so one agent's vision handling never
	// affects another sharing this process. Atomic for the same
	// lock-free read the other toggles use on the request path.
	visionDescribeEnabled atomic.Bool

	// systemPromptOverride, when non-empty, names a [system_prompt.*]
	// entry (see Config.SystemPrompts) whose text is prepended to every
	// outgoing chat request — a live, per-listener choice from the web
	// panel's dropdown (see Supervisor.SetSystemPrompt). Guarded by
	// systemPromptMu, same pattern as modelOverride, since a plain string
	// has no atomic type.
	systemPromptMu       sync.RWMutex
	systemPromptOverride string

	retryLogMu   sync.Mutex
	retryLogFile *os.File

	// idleTimeout is an INACTIVITY timeout for the streaming upstream call
	// (postUpstreamChatStreaming, stream.go): it only fires if no progress
	// at all — no response headers, no SSE line — arrives for this long,
	// and resets every time real progress is observed. A generation that's
	// slow but steadily producing tokens can run indefinitely; only
	// genuine silence this long gets cut off. This replaces the old
	// p.client.Timeout absolute wall-clock ceiling, which cut off a still-
	// progressing generation just because total elapsed time exceeded it.
	// The two non-streaming call sites (postUpstream, handleModels) have no
	// per-byte progress signal to reset against, so they reuse this same
	// duration as a plain absolute ceiling via context.WithTimeout instead.
	idleTimeout time.Duration

	// keepaliveGracePeriod is how long handleClassified lets the
	// classify/inject/detect/retry cycle run silently before committing to
	// SSE headers and starting keepalive ticks (see classifiedResult /
	// finishClassifiedRequest). keepaliveTickInterval is how often a tick
	// is sent once in that state. Both are fields rather than bare
	// constants so tests can shrink them to exercise the slow path without
	// actually waiting out the real durations.
	keepaliveGracePeriod  time.Duration
	keepaliveTickInterval time.Duration

	// throughputStatsPath/throughputStats persist per-(provider,model,bucket)
	// throughput aggregates (see throughput_stats.go) independently of the
	// raw retryLogFile — a distinct file/pattern that survives the JSONL
	// retry log being cleared or rotated. Guarded by throughputStatsMu
	// since request-handling goroutines update it concurrently.
	throughputStatsMu   sync.Mutex
	throughputStatsPath string
	throughputStats     []ThroughputStatsEntry

	// imageDescCache memoizes [vision_describe]'s one-shot image
	// descriptions by a hash of the image content (see
	// hashImageContent) — a real client resends the same image on every
	// later turn (full conversation history every request), so without
	// this every turn after the first would re-run a full VLM generation
	// for an image that's already been described. Unbounded for now (no
	// eviction) — acceptable for a single long-running local session, not
	// yet meant for a workload with many distinct images.
	imageDescCacheMu sync.Mutex
	imageDescCache   map[string]string
}

// NewProxyServer wires up the proxy. retryLogPath, if non-empty, is where
// every request that retried at least once gets its full adjustment
// trajectory appended as a JSON line (see logRetryTrajectory) — opened
// once here in append mode and written to under a mutex, since requests
// are handled concurrently. An empty path disables trajectory logging
// entirely (used by tests that don't want a file on disk).
func NewProxyServer(cfg *Config, provider *ProviderConfig, providerLabel string, events chan<- UIEvent, progressCh chan<- ProgressEvent, retryLogPath string) (*ProxyServer, error) {
	p := &ProxyServer{
		cfg:           cfg,
		provider:      provider,
		providerLabel: providerLabel,
		listenerName:  providerLabel,
		// Proxy explicitly nil, overriding http.DefaultTransport's
		// Proxy: ProxyFromEnvironment — this client's whole job is BEING
		// the proxy for HTTP_PROXY/HTTPS_PROXY-configured tools (forward-
		// proxy mode), so if this process itself happens to inherit those
		// same env vars (e.g. started from a shell that also exports them
		// for a client tool), honoring them here would route this
		// process's own outbound upstream calls back through itself —
		// confirmed in practice: it self-connects, hits its own MITM leaf
		// cert, and fails outbound TLS verification with "certificate
		// signed by unknown authority" since it never trusts its own CA
		// for outbound calls (nor should it — a real vendor's cert should
		// verify against the normal system trust store, not this proxy's
		// own leaf certs).
		client:                &http.Client{Transport: &http.Transport{Proxy: nil}},
		events:                events,
		progress:              progressCh,
		idleTimeout:           time.Duration(cfg.Server.RequestTimeoutSeconds) * time.Second,
		keepaliveGracePeriod:  60 * time.Second,
		keepaliveTickInterval: 15 * time.Second,
		imageDescCache:        make(map[string]string),
	}

	// Seed the live per-instance alert toggle from the global --alert flag.
	p.alertEnabled.Store(cfg.Server.AlertEnabled)
	p.visionDescribeEnabled.Store(cfg.VisionDescribe.Enabled)

	if retryLogPath != "" {
		f, err := os.OpenFile(retryLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("opening retry log: %w", err)
		}
		p.retryLogFile = f
	}

	p.throughputStatsPath = throughputStatsPathFor(retryLogPath, providerLabel)
	if p.throughputStatsPath != "" {
		entries, err := loadThroughputStats(p.throughputStatsPath)
		if err != nil {
			log.Printf("failed to load throughput stats from %s: %v", p.throughputStatsPath, err)
		} else {
			p.throughputStats = entries
		}
	}

	if provider != nil {
		p.upstreamBase = provider.BaseURL
		// Passthrough is deliberately unsupported for most remote providers:
		// each mounts its OpenAI-compatible surface at a different prefix
		// (see [provider.*] comments in config.ini), so naively joining an
		// incoming path onto base_url isn't reliable beyond the one endpoint
		// (/chat/completions) we've verified per provider. clinepass is the
		// documented exception (ProviderConfig.AllowPassthrough): it's
		// Cline's own account gateway, not a plain vendor, so Cline's
		// extension also calls auxiliary endpoints (token refresh,
		// recommended-models, remote-config) against this same base_url —
		// rejecting those with a chat/models-only allowlist breaks Cline's
		// own account/session handling even though model responses work.
		if provider.AllowPassthrough {
			rp, err := newPassthroughReverseProxy(provider.BaseURL)
			if err != nil {
				return nil, err
			}
			p.passthrough = rp
		}
		return p, nil
	}

	p.upstreamBase = fmt.Sprintf("http://%s:%d", cfg.Server.UpstreamHost, cfg.Server.UpstreamPort)
	rp, err := newPassthroughReverseProxy(p.upstreamBase)
	if err != nil {
		return nil, err
	}
	p.passthrough = rp

	return p, nil
}

// currentProvider returns a consistent snapshot of the currently active
// remote-provider target — label, config, resolved upstream base URL, and
// its passthrough reverse proxy — safe to call from any request-handling
// goroutine concurrently with SwitchProvider. The returned *ProviderConfig
// is never mutated in place after being handed out (SwitchProvider always
// allocates a fresh one and replaces the pointer under lock), so reading
// its fields after this call returns is safe without continuing to hold
// any lock. nil provider means local mode (never switches, so this is
// stable for the process lifetime in that case).
func (p *ProxyServer) currentProvider() (label string, provider *ProviderConfig, upstreamBase string, passthrough *httputil.ReverseProxy) {
	p.providerMu.RLock()
	defer p.providerMu.RUnlock()
	return p.providerLabel, p.provider, p.upstreamBase, p.passthrough
}

// SwitchProvider redirects this instance's outbound requests to a
// different configured remote provider, live — without restarting the
// proxy or changing which port/routes it listens on. This is what lets a
// single-provider agent (e.g. Codex, hardcoded to talk to one API) be
// pointed at a different real backend just by changing which provider is
// "active" here, the same way Cline's own provider dropdown works, for a
// tool that has no such dropdown of its own.
//
// Only meaningful when this instance was already started in provider mode
// (NewProxyServer's provider argument was non-nil) — Handler()'s route
// table (kindChat+/v1/models vs. local mode's full passthrough+kindCompletion)
// is fixed at construction time and never rebuilt, so switching into or out
// of local mode at runtime is not supported; callers (the dashboard) must
// only offer this control when applicable.
func (p *ProxyServer) SwitchProvider(label string, cfg ProviderConfig) error {
	rp, err := newPassthroughReverseProxy(cfg.BaseURL)
	if err != nil {
		return err
	}

	cfgCopy := cfg
	p.providerMu.Lock()
	defer p.providerMu.Unlock()
	p.provider = &cfgCopy
	p.providerLabel = label
	p.upstreamBase = cfg.BaseURL
	p.passthrough = rp
	return nil
}

// CurrentForcedBucket returns the dashboard-pinned classification bucket,
// if one is set (see SetForcedBucket) — ok is false when auto-detection
// (Classify, classify.go) is in effect, the default/original behavior.
func (p *ProxyServer) CurrentForcedBucket() (bucket TaskBucket, ok bool) {
	p.forcedBucketMu.RLock()
	defer p.forcedBucketMu.RUnlock()
	if p.forcedBucket == nil {
		return "", false
	}
	return *p.forcedBucket, true
}

// SetForcedBucket pins bucket for every subsequent request's classification
// — sticky until ClearForcedBucket is called, so a whole multi-turn task
// can be forced into (e.g.) architecture mode without re-forcing it on
// every message. Set from the dashboard when auto-detection doesn't pick
// the right bucket (see classify.go's substring-matching caveats).
func (p *ProxyServer) SetForcedBucket(bucket TaskBucket) {
	p.forcedBucketMu.Lock()
	defer p.forcedBucketMu.Unlock()
	p.forcedBucket = &bucket
}

// ClearForcedBucket returns to automatic classification (Classify).
func (p *ProxyServer) ClearForcedBucket() {
	p.forcedBucketMu.Lock()
	defer p.forcedBucketMu.Unlock()
	p.forcedBucket = nil
}

// SetBypassSampling toggles this instance between adaptive sampling (false,
// the default) and plain pass-through (true): when on, chat requests keep
// the client's own sampling params and get a single upstream call with no
// retry loop. Live and thread-safe (control-panel "pass through dynamic
// sampling" switch).
func (p *ProxyServer) SetBypassSampling(on bool) { p.bypassSampling.Store(on) }

// BypassSampling reports whether adaptive sampling is currently bypassed.
func (p *ProxyServer) BypassSampling() bool { return p.bypassSampling.Load() }

// SetModelOverride pins the outgoing model for this backend live (control
// panel model picker); "" clears it back to the configured/static model.
func (p *ProxyServer) SetModelOverride(model string) {
	p.modelMu.Lock()
	defer p.modelMu.Unlock()
	p.modelOverride = model
}

// ModelOverride returns the live model override, or "" if none.
func (p *ProxyServer) ModelOverride() string {
	p.modelMu.RLock()
	defer p.modelMu.RUnlock()
	return p.modelOverride
}

// SetAlertEnabled toggles alert-continuation for this backend live (control
// panel). AlertEnabled reports the current state.
func (p *ProxyServer) SetAlertEnabled(on bool) { p.alertEnabled.Store(on) }
func (p *ProxyServer) AlertEnabled() bool      { return p.alertEnabled.Load() }

func (p *ProxyServer) SetVisionDescribe(on bool) { p.visionDescribeEnabled.Store(on) }
func (p *ProxyServer) VisionDescribeEnabled() bool {
	return p.visionDescribeEnabled.Load()
}

// SetSystemPromptOverride selects a [system_prompt.*] entry by name for
// this backend live (web panel dropdown); "" clears it back to no
// injection at all. An unrecognized name is accepted here without
// validation (mirrors SetModelOverride) — injectSystemPrompt's map lookup
// simply no-ops if p.cfg.SystemPrompts has nothing under that name.
func (p *ProxyServer) SetSystemPromptOverride(name string) {
	p.systemPromptMu.Lock()
	defer p.systemPromptMu.Unlock()
	p.systemPromptOverride = name
}

// SystemPromptOverride returns the currently selected name, or "" if none.
func (p *ProxyServer) SystemPromptOverride() string {
	p.systemPromptMu.RLock()
	defer p.systemPromptMu.RUnlock()
	return p.systemPromptOverride
}

// SystemPromptNames returns every configured [system_prompt.*] name,
// sorted, for the web panel's dropdown options.
func (p *ProxyServer) SystemPromptNames() []string {
	names := make([]string, 0, len(p.cfg.SystemPrompts))
	for name := range p.cfg.SystemPrompts {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// EffectiveModel is what outgoing chat requests will actually carry: the live
// override if set, otherwise the current provider's configured Model ("" means
// the client's own model passes through). For panel display.
func (p *ProxyServer) EffectiveModel() string {
	if m := p.ModelOverride(); m != "" {
		return m
	}
	if _, provider, _, _ := p.currentProvider(); provider != nil {
		return provider.Model
	}
	return ""
}

// classifyOrForced returns the dashboard-pinned bucket if one is set,
// otherwise falls through to automatic classification — the single call
// site handleClassified uses instead of calling Classify directly.
func (p *ProxyServer) classifyOrForced(classificationText string) TaskBucket {
	if bucket, ok := p.CurrentForcedBucket(); ok {
		return bucket
	}
	return Classify(&p.cfg.Classification, classificationText)
}

// newPassthroughReverseProxy builds a reverse proxy that forwards requests
// to targetBaseURL unmodified, joining the incoming request path onto the
// target's own path (e.g. clinepass's base_url already ends in /api/v1, so
// an incoming /chat/completions request reaches /api/v1/chat/completions
// upstream) — used both for local mode's full passthrough and for any
// remote provider with AllowPassthrough set.
func newPassthroughReverseProxy(targetBaseURL string) (*httputil.ReverseProxy, error) {
	target, err := url.Parse(targetBaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid upstream address: %w", err)
	}

	rp := httputil.NewSingleHostReverseProxy(target)
	origErrHandler := rp.ErrorHandler
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("passthrough error for %s: %v", r.URL.Path, err)
		if origErrHandler != nil {
			origErrHandler(w, r, err)
			return
		}
		w.WriteHeader(http.StatusBadGateway)
	}
	return rp, nil
}

// tokensSecMultiplier returns the correction factor applied to the live
// chunk-count-based tok/s estimate (see ProviderConfig.TokensSecMultiplier).
// Local llama-server has no ProviderConfig at all (p.provider is nil) and
// its one-delta-per-token behavior is already verified against source, so
// it always gets the no-op 1.0 rather than needing an entry in config.ini.
// A zero value (e.g. a ProviderConfig built as a struct literal that
// doesn't set this field, as several tests do) is treated the same as
// "unset" rather than a real 0x multiplier — 0 would always report 0
// tokens, which is never a sane setting, only a missing one.
func (p *ProxyServer) tokensSecMultiplier() float64 {
	_, provider, _, _ := p.currentProvider()
	if provider == nil || provider.TokensSecMultiplier == 0 {
		return 1.0
	}
	return provider.TokensSecMultiplier
}

// Close releases the retry log file handle, if one was opened. Safe to
// call even when trajectory logging is disabled.
func (p *ProxyServer) Close() error {
	if p.retryLogFile == nil {
		return nil
	}
	return p.retryLogFile.Close()
}

func (p *ProxyServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", p.handleClassified(kindChat))
	mux.HandleFunc("/v1/models", p.handleModels)

	if p.provider != nil {
		// Some clients (Page Assist, unlike Cline) GET the bare base URL
		// as a reachability check before calling /v1/models. There's no
		// real upstream endpoint to proxy this to — it's not a documented
		// part of any provider's API — so answer it directly rather than
		// rejecting a harmless probe. Deliberately NOT a subtree pattern
		// ("/v1/"): any other /v1/* path we don't explicitly support
		// should still hit the clear 501 below (or passthrough, for a
		// provider with AllowPassthrough) rather than a misleading
		// fake-success empty response.
		mux.HandleFunc("/v1", p.handleRootProbe)
		// The catch-all re-checks AllowPassthrough/passthrough fresh on
		// every request rather than choosing once here at construction
		// time — SwitchProvider (see currentProvider) can change which
		// provider (and thus its AllowPassthrough) is active for the
		// lifetime of this mux, and a route baked in at Handler()-call
		// time would never notice.
		mux.HandleFunc("/completion", p.rejectRemoteOnly)
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			_, provider, _, passthrough := p.currentProvider()
			if provider != nil && provider.AllowPassthrough {
				// clinepass (see ProviderConfig.AllowPassthrough): Cline's
				// own account gateway needs more than chat/models — token
				// refresh, recommended-models, remote-config, etc. —
				// forwarded verbatim rather than rejected with 501.
				passthrough.ServeHTTP(w, r)
				return
			}
			p.rejectRemoteOnly(w, r)
		})
		return withCORS(mux)
	}

	mux.HandleFunc("/completion", p.handleClassified(kindCompletion))
	mux.HandleFunc("/", p.passthrough.ServeHTTP)
	return withCORS(mux)
}

// withCORS adds permissive CORS headers to every response and answers
// preflight OPTIONS requests directly. Cline runs in the VS Code extension
// host and is never subject to CORS, but browser-based clients (e.g. Page
// Assist) are: without Access-Control-Allow-Origin, the browser silently
// discards a perfectly successful response before the client's JS ever
// sees it — the request succeeds at the network level but looks like an
// empty/broken response to the client. This is a local, single-user dev
// proxy, so a permissive "*" origin carries no real multi-tenant risk.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		// Reflect whatever headers the browser's own preflight asked for
		// (Access-Control-Request-Headers), rather than a fixed allowlist —
		// a client is free to send vendor-specific extras (e.g. OpenRouter's
		// optional X-Title/HTTP-Referer attribution headers) that a static
		// list would otherwise need constant upkeep to keep up with. Falls
		// back to the two headers every client needs if this isn't a
		// preflight (i.e. Access-Control-Request-Headers is empty/absent).
		allowHeaders := r.Header.Get("Access-Control-Request-Headers")
		if allowHeaders == "" {
			allowHeaders = "Content-Type, Authorization"
		}
		w.Header().Set("Access-Control-Allow-Headers", allowHeaders)

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// handleRootProbe answers a bare GET /v1 (or /v1/) with 200 OK, for
// clients that ping the base URL itself before calling a real endpoint.
// Not a real provider endpoint — just enough to satisfy a reachability
// check without misrepresenting anything as a genuine API response.
func (p *ProxyServer) handleRootProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"object":"list","data":[]}`))
}

// handleModels proxies the client's model-list request (Cline calls this
// on startup to populate its model picker). In local mode this is just the
// existing passthrough. In remote-provider mode, GET <base_url>/models is
// a well-documented endpoint on OpenAI, OpenRouter, and Anthropic's
// OpenAI-compat layer; Gemini's compat layer is expected to support it too
// but wasn't independently verified the way the chat-completions path was
// — worth checking if model listing fails specifically on gemini.
func (p *ProxyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	providerLabel, provider, upstreamBaseDefault, passthrough := p.currentProvider()
	if provider == nil {
		passthrough.ServeHTTP(w, r)
		return
	}

	upstreamBase := upstreamBaseDefault
	modelsPath := "/models"
	authHeader := ""
	if fc, ok := forwardProxyOverrideFromCtx(r.Context()); ok {
		upstreamBase = fc.upstreamBase
		authHeader = fc.authHeader
		// Same reasoning as handleClassified: forward-proxy mode has no
		// registered vendor-specific path prefix anywhere, so the client's
		// own original path (whatever its real vendor actually expects) is
		// the only correct one to forward.
		modelsPath = r.URL.Path
	}

	// GET /models is a single blocking call with no incremental progress
	// signal to reset an idle timer against, unlike the streaming chat path
	// — p.idleTimeout is reused here as a plain absolute ceiling instead.
	ctx := r.Context()
	if p.idleTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.idleTimeout)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, r.Method, upstreamBase+modelsPath, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if authHeader != "" {
		req.Header.Set("Authorization", authHeader)
	} else if provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+provider.APIKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		log.Printf("models request to %q failed: %v", providerLabel, err)
		http.Error(w, fmt.Sprintf("upstream unreachable: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if resp.StatusCode >= 300 {
		log.Printf("models request to %q returned %d: %s", providerLabel, resp.StatusCode, truncateForLog(respBytes))
	}
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	w.Write(respBytes)
}

// rejectRemoteOnly responds clearly to any path other than
// /v1/chat/completions and /v1/models when running against a remote
// provider backend, rather than silently mis-routing (e.g. llama.cpp's
// native /completion has no equivalent on these APIs; generic passthrough
// can't be mapped reliably across providers with different URL prefixes).
// Logged to stdout so a rejected request is diagnosable even though it
// never reaches the dashboard's request log (that's reserved for the
// classify/inject/detect/retry cycle on the chat-completions route).
func (p *ProxyServer) rejectRemoteOnly(w http.ResponseWriter, r *http.Request) {
	providerLabel, _, _, _ := p.currentProvider()
	log.Printf("rejected %s %s: not available against remote provider %q", r.Method, r.URL.Path, providerLabel)
	http.Error(w, fmt.Sprintf(
		"path %q is not available when running against remote provider %q; only /v1/chat/completions and /v1/models are supported in this mode",
		r.URL.Path, providerLabel,
	), http.StatusNotImplemented)
}

func (p *ProxyServer) emit(ev UIEvent) {
	ev.Listener = p.listenerName
	select {
	case p.events <- ev:
	default:
		// dashboard is behind; drop rather than block request handling
	}
}

func (p *ProxyServer) emitProgress(ev ProgressEvent) {
	if p.progress == nil {
		return
	}
	ev.Listener = p.listenerName
	select {
	case p.progress <- ev:
	default:
		// dashboard is behind; drop rather than block the stream read loop
	}
}

// classifiedResult carries the outcome of handleClassified's
// classify/inject/detect/retry cycle from the goroutine that runs it back
// to the handler, which races that goroutine against
// keepaliveGracePeriod — see handleClassified and finishClassifiedRequest.
type classifiedResult struct {
	// clientGone mirrors the previous early `return` on
	// context.Canceled/context.DeadlineExceeded: the downstream client
	// disconnected, so there's nothing to send, regardless of whether
	// headers were already committed.
	clientGone bool
	// fatalErr/fatalStatus mirror the previous early http.Error(...) calls:
	// a transport-level failure talking to upstream (connection refused,
	// DNS failure, request-encoding failure, ...), not a response upstream
	// actually sent — fatalStatus preserves the exact status code each of
	// those cases used (502 for unreachable, 500 for encoding failure).
	fatalErr    error
	fatalStatus int

	bucket                TaskBucket
	dashboardHost         string
	respBytes             []byte
	respStatus            int
	finalIssue            Issue
	content               string
	reasoningContent      string
	finishReason          string
	toolCalls             []map[string]interface{}
	retryCount            int
	finalPromptTokens     int
	finalCompletionTokens int
	// finalGenerationElapsedMs is real decode time only (see
	// streamResult.generationElapsedMs), summed across alert rounds the same
	// way finalCompletionTokens is — this, not the request's total latency,
	// is what throughput tok/s must be divided by (see finishClassifiedRequest).
	finalGenerationElapsedMs int64
	streamable               bool
	upstreamErr              string
	adjustments              []RetryAdjustment
	// alertRounds is how many alert-continuation probes fired for this
	// request (see alert.go and the alertLoop in handleClassified) — 0 for
	// every request unless --alert is active and the model matched
	// alert_models. Reported on the dashboard so "the model needed a
	// nudge" is visible without having to dig through logs.
	alertRounds int
}

func (p *ProxyServer) handleClassified(kind endpointKind) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// dashboardHost is only ever non-empty in forward-proxy mode
		// (mitm.go) — every other mode has one fixed backend for the
		// whole run, already shown in the dashboard's static header.
		var dashboardHost string
		fc, isForwardProxy := forwardProxyOverrideFromCtx(r.Context())
		if isForwardProxy {
			dashboardHost = strings.TrimPrefix(fc.upstreamBase, "https://")
		}

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		r.Body.Close()

		// In every other mode, path is a fixed constant (the client's
		// incoming request always hits this proxy's own registered mux
		// pattern, decoupled from whatever the real vendor's path looks
		// like — p.upstreamBase already carries any vendor-specific
		// prefix). Forward-proxy mode has no such registration: the client
		// is talking to its real vendor's real base_url, which can mount
		// its OpenAI-compatible surface at any prefix (e.g. Cline's own
		// gateway at "/api/v1/chat/completions", not "/v1/..."). There's
		// no config anywhere recording that prefix in this mode — the
		// client's own original path IS the only correct one to forward.
		path := p.chatPath(kind)
		if isForwardProxy {
			path = r.URL.Path
		}

		var body map[string]interface{}
		if err := json.Unmarshal(rawBody, &body); err != nil {
			// Not a JSON body we understand; pass through unmodified.
			p.forwardRaw(w, r, rawBody, path)
			return
		}

		if p.cfg.ToolSchemaSanitizer.Enabled {
			sanitizeToolSchemas(body, p.cfg.ToolSchemaSanitizer)
		}

		if kind == kindChat {
			if h := systemPromptHash(body); h != "" {
				log.Printf("system_prompt_hash ingress: %s", h)
			}
		}

		// [system_prompt.*]: prepend the selected named prompt (if any) to
		// the client's own system message — deliberately AFTER the ingress
		// hash log above, so that log keeps reflecting the client's own,
		// pre-injection content (its whole purpose: a diff against it is
		// proof something altered the prompt) rather than baking this
		// intentional, configured modification in before the check.
		if kind == kindChat {
			if name := p.SystemPromptOverride(); name != "" {
				injectSystemPrompt(body, p.cfg.SystemPrompts[name])
			}
		}

		clientWantsStream, _ := body["stream"].(bool)

		// [vision_describe]: replace every image content part with a VLM
		// description BEFORE classification/injection — global (every
		// listener, no local-only gate) and before classify so the
		// description text itself can participate in bucket keyword
		// matching. Runs synchronously: a cache miss adds one VLM
		// round-trip of latency to this request, a cache hit adds
		// nothing. kindChat only — kindCompletion has no messages array.
		if kind == kindChat && p.VisionDescribeEnabled() {
			p.describeImagesInPlace(r.Context(), body)
		}

		// When sampling is bypassed (control-panel pass-through toggle),
		// classify only for the dashboard label but leave the client's own
		// sampling params untouched — no preset injection. The retry loop is
		// also short-circuited to a single attempt below (see maxRetries).
		bypass := p.BypassSampling()
		classificationText := extractClassificationText(body, kind)
		bucket := p.classifyOrForced(classificationText)
		if !bypass {
			applyPreset(body, p.cfg.Presets[bucket], p.cfg.Server.InjectThinkingBudget)
		}

		// A model name valid for the local llama-server almost certainly
		// isn't valid on a remote provider, so a configured provider model
		// always overrides whatever the client requested. If the provider
		// has no model configured, the client's request passes through
		// unchanged. Reads the CURRENT provider (not a construction-time
		// snapshot) so SwitchProvider takes effect on the very next request.
		if _, provider, _, _ := p.currentProvider(); provider != nil && provider.Model != "" {
			body["model"] = provider.Model
		}

		// A live model override (control-panel model picker) wins over the
		// static config.ini Model — this is what lets you switch a backend's
		// model (e.g. a clinepass model) without a restart.
		if m := p.ModelOverride(); m != "" {
			body["model"] = m
		}

		// For kindChat, upstream is always asked to stream internally now
		// (see postUpstreamChatStreaming), regardless of what the client
		// itself wants — this is what makes live progress tracking
		// possible. kindCompletion (llama.cpp's native endpoint, local-only
		// — remote providers never register this route) keeps the original
		// single blocking call; it's a low-traffic legacy path and local
		// llama.cpp already gives console-level progress visibility for it.
		if kind != kindChat {
			body["stream"] = false
		}

		// The classify/inject/detect/retry cycle runs in its own goroutine
		// so it can be raced against keepaliveGracePeriod below: a request
		// that resolves quickly (the overwhelming common case, including a
		// fast upstream rejection) is handled in finishClassifiedRequest
		// exactly as before, with zero behavior change. Only a request
		// still running past the grace period makes the handler commit to
		// SSE headers and start emitting keepalive ticks, so a downstream
		// client with its own hard idle-connection timeout (e.g. Cline's
		// ~5 minutes) sees a steady trickle of bytes instead of dead
		// silence for the entire cycle — which previously included every
		// retry attempt, since nothing was written to the client at all
		// until the whole cycle finished.
		resultCh := make(chan classifiedResult, 1)
		go func() {
			maxRetries := p.cfg.Server.MaxRetries
			if bypass {
				// Pass-through: one upstream call, no adaptive retries.
				maxRetries = 0
			}
			var (
				respBytes        []byte
				respStatus       int
				finalIssue       Issue
				content          string
				reasoningContent string
				finishReason     string
				toolCalls        []map[string]interface{}
				retryCount       int
				// finalPromptTokens/finalCompletionTokens are the real counts
				// from upstream's usage object (kindChat only), kept from
				// whichever attempt was last — reported on the completed-request
				// UIEvent, not the transient live indicator.
				finalPromptTokens     int
				finalCompletionTokens int
				// completionTokens is this round's own count (read after the
				// inner attempt-loop below to fold into totalCompletionTokens
				// across alert rounds) — hoisted out of the inner loop, which
				// used to declare it locally, since alert-continuation needs
				// its value to survive past that loop's closing brace.
				completionTokens int
				// generationElapsedMs mirrors completionTokens: this round's
				// own real decode time (see streamResult.generationElapsedMs),
				// folded into totalGenerationElapsedMs the same way.
				generationElapsedMs int64
				// streamable is only true once a 2xx response was successfully
				// parsed into usable content/finish_reason. If the upstream
				// response was an error or an unexpected shape, we must forward
				// it as-is rather than synthesizing a fake SSE stream around it.
				streamable  bool
				upstreamErr string
				adjustments []RetryAdjustment
				// reasoningLoopHit records whether THIS attempt's finishReason
				// was stream.go's internal sentinel (see
				// reasoningBudgetExceededFinishReason) before it gets
				// sanitized away below — checked further down to route into
				// IssueReasoningLoop instead of the normal DetectIssue path.
				reasoningLoopHit bool
			)

			// alertRound/accumulatedContent/accumulatedReasoning/totalCompletionTokens
			// carry alert-continuation state ACROSS the outer loop below (see the
			// end of this loop body) — separate from the per-attempt-loop retry
			// state above, which is reset fresh every alert round since each
			// round runs its own independent classify/inject/detect/retry cycle
			// against a conversation that's grown by one assistant+user turn pair.
			var (
				alertRound               int
				accumulatedContent       strings.Builder
				accumulatedReasoning     strings.Builder
				totalCompletionTokens    int
				totalGenerationElapsedMs int64
				// lastGoodXxx snapshot the most recently successful round's
				// own finishReason/toolCalls/respStatus/promptTokens. If a
				// LATER alert-continuation round itself fails (non-2xx,
				// unparseable response), the loop falls back to these
				// instead of surfacing that round's own failure — the user
				// already has a real, complete answer from the round(s)
				// before it, and a failed follow-up probe must never turn
				// an already-successful response into an error.
				lastGoodFinishReason string
				lastGoodToolCalls    []map[string]interface{}
				lastGoodRespStatus   int
				lastGoodPromptTokens int
			)

		alertLoop:
			for {
				for attempt := 0; ; attempt++ {
					completionTokens = 0

					if kind == kindChat {
						result, sErr := p.postUpstreamChatStreaming(r.Context(), path, r.URL.RawQuery, body, bucket, attempt)
						if sErr != nil {
							// r.Context().Err() (not errors.Is(sErr, ...)) is
							// the precise check here: p.client.Timeout applies
							// its deadline to a context *derived from*
							// r.Context(), so a plain client-side upstream
							// timeout also produces an error satisfying
							// errors.Is(sErr, context.DeadlineExceeded) —
							// identical to what a genuine downstream
							// disconnect looks like from that check alone.
							// Only r.Context().Err() being non-nil actually
							// means the incoming request's own context (tied
							// to the downstream connection) was canceled;
							// otherwise this is our own timeout talking to
							// upstream, which must still be reported as a
							// real error, not silently dropped as if the
							// client had gone away (confirmed regression:
							// it previously wasn't).
							if r.Context().Err() != nil {
								resultCh <- classifiedResult{clientGone: true} // downstream client disconnected; nothing to send
								return
							}
							p.emit(UIEvent{
								Timestamp:  time.Now(),
								Bucket:     bucket,
								RetryCount: attempt,
								Error:      sErr.Error(),
							})
							resultCh <- classifiedResult{
								fatalErr:    fmt.Errorf("upstream unreachable: %w", sErr),
								fatalStatus: http.StatusBadGateway,
							}
							return
						}

						respStatus = result.status
						if respStatus >= 300 {
							respBytes = result.rawErrorBody
							finalIssue = IssueNone
							upstreamErr = fmt.Sprintf("upstream returned %d: %s", respStatus, truncateForLog(respBytes))
							break
						}

						content = result.content
						reasoningContent = result.reasoningContent
						finishReason = result.finishReason
						reasoningLoopHit = finishReason == reasoningBudgetExceededFinishReason
						if reasoningLoopHit {
							// Sanitize the internal sentinel immediately —
							// must never leak into buildChatCompletionJSON
							// below or into lastGoodFinishReason if this
							// attempt ends up being accepted as final.
							finishReason = "length"
						}
						completionTokens = result.completionTokens
						generationElapsedMs = result.generationElapsedMs
						toolCalls = result.toolCalls
						streamable = true
						respBytes = buildChatCompletionJSON(content, reasoningContent, finishReason, result.promptTokens, completionTokens, toolCalls)

						finalPromptTokens = result.promptTokens
						finalCompletionTokens = result.completionTokens
					} else {
						reqBytes, err := json.Marshal(body)
						if err != nil {
							resultCh <- classifiedResult{
								fatalErr:    fmt.Errorf("failed to encode upstream request: %w", err),
								fatalStatus: http.StatusInternalServerError,
							}
							return
						}

						var lastErr error
						respStatus, respBytes, lastErr = p.postUpstream(r.Context(), path, r.URL.RawQuery, reqBytes)
						if lastErr != nil {
							p.emit(UIEvent{
								Timestamp:  time.Now(),
								Bucket:     bucket,
								RetryCount: attempt,
								Error:      lastErr.Error(),
							})
							resultCh <- classifiedResult{
								fatalErr:    fmt.Errorf("upstream unreachable: %w", lastErr),
								fatalStatus: http.StatusBadGateway,
							}
							return
						}

						if respStatus >= 300 {
							// Upstream itself rejected the request (bad params, etc.);
							// nothing to inspect or retry. Forward its error verbatim.
							finalIssue = IssueNone
							upstreamErr = fmt.Sprintf("upstream returned %d: %s", respStatus, truncateForLog(respBytes))
							break
						}

						var respBody map[string]interface{}
						if err := json.Unmarshal(respBytes, &respBody); err != nil {
							// Can't inspect it; pass through as-is.
							finalIssue = IssueNone
							upstreamErr = fmt.Sprintf("upstream response not valid JSON: %v", err)
							break
						}

						var ok bool
						content, reasoningContent, finishReason, ok = extractResponseContent(respBody, kind)
						if !ok {
							finalIssue = IssueNone
							upstreamErr = "upstream response missing expected choices/content shape"
							break
						}
						streamable = true
						completionTokens = extractCompletionTokens(respBody, kind)
					}

					var issueDetail string
					if reasoningLoopHit {
						// Detected live, mid-stream (stream.go), not by
						// analyzing finished text — skip DetectIssue's
						// normal (post-hoc) analysis entirely.
						finalIssue = IssueReasoningLoop
						issueDetail = fmt.Sprintf("reasoning ran past its budget with no real content yet (%d words of reasoning captured before abort)", len(strings.Fields(reasoningContent)))
					} else {
						finalIssue, issueDetail = DetectIssue(&p.cfg.Detection, content, finishReason, len(toolCalls) > 0)
					}
					if finalIssue == IssueNone || attempt >= maxRetries {
						break
					}
					if issueDetail != "" {
						// Goes to proxy_<backend>.log — the first thing to check
						// when retries seem too frequent: if the logged n-gram is
						// legitimate structure (XML tool tags, file paths) rather
						// than degenerate looping, the detector is false-positiving
						// and [detection] needs tuning, not the sampling params.
						if finalIssue == IssueReasoningLoop {
							log.Printf("detect (%s, attempt %d): %s triggered: %s", bucket, attempt, finalIssue, issueDetail)
						} else {
							log.Printf("detect (%s, attempt %d): %s triggered by repeated n-gram: %q", bucket, attempt, finalIssue, issueDetail)
						}
					}

					newAdjustments := adjustForIssue(body, finalIssue, p.cfg, kind, attempt, completionTokens)
					for i := range newAdjustments {
						newAdjustments[i].Detail = issueDetail
					}
					adjustments = append(adjustments, newAdjustments...)
					retryCount = attempt + 1
				}

				if !streamable || respStatus >= 300 {
					// Error/unparseable-response path (non-2xx status, bad
					// JSON, unexpected shape). On round 0 (the real first
					// attempt) there's nothing to fall back to — send
					// whatever the inner loop set, exactly as before alert-
					// continuation existed. On a LATER round (an alert-probe
					// follow-up that itself failed), the user already has a
					// real, complete answer from the round(s) before it — a
					// failed follow-up must never turn that into an error, so
					// fall back to the last successful round's state instead
					// of surfacing this round's failure.
					if alertRound > 0 {
						finishReason = lastGoodFinishReason
						toolCalls = lastGoodToolCalls
						respStatus = lastGoodRespStatus
						finalPromptTokens = lastGoodPromptTokens
						streamable = true
					}
					break alertLoop
				}

				// isConfirmationOnly only applies from round 1 onward — round 0
				// is the actual first attempt at the real task, never a reply
				// to alertProbeMessage, so it's never itself a "confirmation".
				isConfirmationOnly := alertRound > 0 && looksLikeConfirmation(content)
				if !isConfirmationOnly {
					accumulatedContent.WriteString(content)
					accumulatedReasoning.WriteString(reasoningContent)
					totalCompletionTokens += completionTokens
					totalGenerationElapsedMs += generationElapsedMs
				}
				lastGoodFinishReason = finishReason
				lastGoodToolCalls = toolCalls
				lastGoodRespStatus = respStatus
				lastGoodPromptTokens = finalPromptTokens

				model, _ := body["model"].(string)
				modelMatched := modelInAlertList(p.cfg.Server.AlertModels, model)
				alertOn := p.AlertEnabled()
				canAlertAgain := kind == kindChat &&
					alertOn &&
					finishReason == "stop" &&
					!isConfirmationOnly &&
					alertRound < p.cfg.Server.AlertMaxRounds &&
					modelMatched

				// Logged every round whenever --alert is active, regardless
				// of the outcome — this is the one place that answers "is
				// alert-continuation actually doing anything" concretely,
				// rather than only showing up as an eventual dashboard tally
				// (see UIEvent.AlertRounds) with no visibility into *why* a
				// given request didn't get a probe (wrong finish_reason,
				// model not listed, already at the round cap, or a detected
				// confirmation).
				if alertOn && kind == kindChat {
					decision := "stopping"
					if canAlertAgain {
						decision = "probing again"
					}
					log.Printf("alert (%s, round %d): model=%q finish_reason=%q confirmation=%v model_matched=%v -> %s",
						bucket, alertRound, model, finishReason, isConfirmationOnly, modelMatched, decision)
				}

				if !canAlertAgain {
					break alertLoop
				}

				// Extend the conversation with this round's reply plus the
				// probe, and let the outer loop run the classify/inject/
				// detect/retry cycle again from a fresh attempt 0 — the model
				// now sees its own prior (possibly incomplete) answer and is
				// asked directly whether more work remains.
				messages, ok := body["messages"].([]interface{})
				if !ok {
					break alertLoop // no messages array to extend against — shouldn't happen for kindChat, but bail safely rather than panic
				}
				messages = append(messages,
					map[string]interface{}{"role": "assistant", "content": content},
					map[string]interface{}{"role": "user", "content": p.cfg.Server.AlertProbeMessage},
				)
				body["messages"] = messages
				alertRound++
				retryCount = 0 // this is a fresh conversational turn, not a same-turn retry
			}

			content = accumulatedContent.String()
			reasoningContent = accumulatedReasoning.String()
			finalCompletionTokens = totalCompletionTokens
			if streamable {
				// Rebuilt from the accumulated multi-round content/reasoning —
				// the inner loop's own respBytes only reflects its LAST round,
				// which is exactly the delta this rebuild needs to fold in.
				respBytes = buildChatCompletionJSON(content, reasoningContent, finishReason, finalPromptTokens, finalCompletionTokens, toolCalls)
			}

			resultCh <- classifiedResult{
				bucket:                   bucket,
				dashboardHost:            dashboardHost,
				respBytes:                respBytes,
				respStatus:               respStatus,
				finalIssue:               finalIssue,
				content:                  content,
				reasoningContent:         reasoningContent,
				finishReason:             finishReason,
				toolCalls:                toolCalls,
				retryCount:               retryCount,
				finalPromptTokens:        finalPromptTokens,
				finalCompletionTokens:    finalCompletionTokens,
				finalGenerationElapsedMs: totalGenerationElapsedMs,
				streamable:               streamable,
				upstreamErr:              upstreamErr,
				adjustments:              adjustments,
				alertRounds:              alertRound,
			}
		}()

		select {
		case res := <-resultCh:
			p.finishClassifiedRequest(w, start, kind, clientWantsStream, false, nil, body, res)
			return
		case <-time.After(p.keepaliveGracePeriod):
			// Fell through: still running past the grace period. Only
			// kindChat clients that asked for streaming get eager headers
			// — there's no sane "keep it alive" filler for a plain JSON
			// response or the legacy completion endpoint, so those just
			// keep waiting synchronously on resultCh exactly as before.
		}

		var headersSent bool
		var flusher http.Flusher
		if kind == kindChat && clientWantsStream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.WriteHeader(http.StatusOK)
			flusher, _ = w.(http.Flusher)
			if flusher != nil {
				flusher.Flush()
			}
			headersSent = true
		}

		ticker := time.NewTicker(p.keepaliveTickInterval)
		defer ticker.Stop()
		for {
			select {
			case res := <-resultCh:
				p.finishClassifiedRequest(w, start, kind, clientWantsStream, headersSent, flusher, body, res)
				return
			case <-ticker.C:
				if !headersSent {
					continue
				}
				// A real, empty-delta chat-completion-chunk — NOT a bare
				// SSE comment line (the previous approach here). A comment
				// line is spec-legal SSE (ignorable by any compliant
				// parser), but real OpenAI-compatible APIs never actually
				// send one, so a client whose SSE handling only expects
				// "data: {...}" lines (many are a hand-rolled fetch reader,
				// not a full SDK) may never have been exercised against
				// that line shape at all — confirmed in practice: a client
				// reported "Stream decode error" specifically on a request
				// slow enough to reach this path. An empty delta is a
				// chunk shape every client here must already handle (it's
				// indistinguishable from any other content delta, just
				// with nothing new to say), so it keeps bytes flowing
				// without asking the client to support a line format nothing
				// else in this stream ever uses.
				if err := writeHeartbeatChunk(w, flusher); err != nil {
					return // client gone; nothing more to do
				}
			}
		}
	}
}

// finishClassifiedRequest writes the final response for a classified
// request once its classify/inject/detect/retry cycle (running in a
// separate goroutine — see handleClassified) has produced a result.
// headersSent is true only when handleClassified's keepalive path already
// committed SSE headers to the client (past keepaliveGracePeriod) — in
// that case, only streamChatSSE can be used for the actual write (headers
// can't change now, and writeResponse would try to send its own), and a
// fatal/upstream error necessarily surfaces as synthesized SSE content
// rather than a real status code, since none can be sent anymore. Below
// keepaliveGracePeriod (headersSent == false, by far the common case),
// behavior is byte-for-byte identical to before this restructuring,
// including forwarding a real non-2xx upstream status/body verbatim (see
// TestUpstreamErrorNeverWrappedInFakeStream).
func (p *ProxyServer) finishClassifiedRequest(w http.ResponseWriter, start time.Time, kind endpointKind, clientWantsStream, headersSent bool, flusher http.Flusher, body map[string]interface{}, res classifiedResult) {
	if res.clientGone {
		return
	}

	if res.fatalErr != nil {
		if !headersSent {
			http.Error(w, res.fatalErr.Error(), res.fatalStatus)
			return
		}
		streamChatSSE(w, flusher, "Error: "+res.fatalErr.Error(), "", "stop", nil, 0, 0)
		return
	}

	latency := time.Since(start).Milliseconds()

	if len(res.adjustments) > 0 {
		p.logRetryTrajectory(res.bucket, res.adjustments, res.streamable && res.finalIssue == IssueNone, res.retryCount, latency)
	}

	model, _ := body["model"].(string)

	// retryCount == 0 specifically: a retried request's latency includes
	// time spent on a discarded attempt, which would understate the real
	// tokens/sec of the generation that actually reached the client — see
	// logThroughput. kindCompletion is excluded implicitly: it never
	// populates real token counts (see UIEvent.PromptTokens's doc comment),
	// so logThroughput's completionTokens == 0 guard skips it anyway.
	//
	// generationMs (res.finalGenerationElapsedMs), NOT latency, is what
	// tok/s must be divided by: latency is total round-trip time, including
	// prompt processing (prefill) before the first token ever appeared —
	// confirmed to read as roughly HALF the live in-flight indicator's rate
	// for the same requests, since the live indicator already correctly
	// used generation-only time (ProgressEvent.GenerationElapsedMs) while
	// this persisted figure did not. A zero generationMs (kindCompletion,
	// which never populates it) falls back to latency so the guard in
	// logThroughput/updateThroughputStats still divides by something real
	// rather than silently reporting zero throughput.
	generationMs := res.finalGenerationElapsedMs
	if generationMs <= 0 {
		generationMs = latency
	}
	if kind == kindChat && res.streamable && res.retryCount == 0 {
		providerLabel, _, _, _ := p.currentProvider()
		p.logThroughput(providerLabel, model, res.bucket, res.finalPromptTokens, res.finalCompletionTokens, generationMs)
		p.updateThroughputStats(providerLabel, model, res.bucket, res.finalPromptTokens, res.finalCompletionTokens, generationMs)
	}

	temp, topP, topK, penalty, budget := effectiveDynamicValues(body)
	p.emit(UIEvent{
		Timestamp:            time.Now(),
		Bucket:               res.bucket,
		RetryCount:           res.retryCount,
		Issue:                res.finalIssue,
		LatencyMs:            latency,
		Temperature:          temp,
		TopP:                 topP,
		TopK:                 topK,
		RepeatPenalty:        penalty,
		ThinkingBudgetTokens: budget,
		PromptTokens:         res.finalPromptTokens,
		CompletionTokens:     res.finalCompletionTokens,
		Error:                res.upstreamErr,
		Host:                 res.dashboardHost,
		Model:                model,
		AlertRounds:          res.alertRounds,
	})

	if !headersSent {
		p.writeResponse(w, res.respStatus, res.respBytes, res.content, res.reasoningContent, res.finishReason, res.toolCalls, res.finalPromptTokens, res.finalCompletionTokens, clientWantsStream && res.streamable, kind)
		return
	}

	// Headers already committed to 200 SSE; writeResponse would try to
	// send its own headers (or write raw JSON) onto an already-committed
	// connection, so write directly instead.
	if !res.streamable {
		streamChatSSE(w, flusher, "Error: "+res.upstreamErr, "", "stop", nil, 0, 0)
		return
	}
	streamChatSSE(w, flusher, res.content, res.reasoningContent, res.finishReason, res.toolCalls, res.finalPromptTokens, res.finalCompletionTokens)
}

// chatPath returns the outbound request path for the given endpoint kind.
// In local mode this mirrors the incoming route exactly (llama-server
// mounts its OAI-compatible surface at /v1/chat/completions and its native
// completion endpoint at /completion, off a bare host:port base with no
// path of its own). In remote-provider mode, kind is always kindChat (the
// only route registered against a provider — see Handler), and the path is
// always exactly "/chat/completions": base_url already encodes each
// provider's own mount prefix up to that point, verified per provider in
// config.ini, so the incoming request's own path is irrelevant here.
func (p *ProxyServer) chatPath(kind endpointKind) string {
	if p.provider != nil {
		return "/chat/completions"
	}
	if kind == kindChat {
		return "/v1/chat/completions"
	}
	return "/completion"
}

// forwardRaw ships a request body upstream unmodified when it can't be
// parsed as JSON, still going through the same client/timeout plumbing.
func (p *ProxyServer) forwardRaw(w http.ResponseWriter, r *http.Request, rawBody []byte, path string) {
	status, respBytes, err := p.postUpstream(r.Context(), path, r.URL.RawQuery, rawBody)
	if err != nil {
		http.Error(w, fmt.Sprintf("upstream unreachable: %v", err), http.StatusBadGateway)
		return
	}
	w.WriteHeader(status)
	w.Write(respBytes)
}

func (p *ProxyServer) postUpstream(ctx context.Context, path string, rawQuery string, body []byte) (int, []byte, error) {
	_, provider, upstreamBase, _ := p.currentProvider()
	upstreamURL := upstreamBase + path
	if rawQuery != "" {
		upstreamURL += "?" + rawQuery
	}

	// This is a single blocking call (used by kindCompletion and
	// forwardRaw) with no incremental progress signal to reset an idle
	// timer against, unlike the streaming chat path (stream.go) —
	// p.idleTimeout is reused here as a plain absolute ceiling instead.
	if p.idleTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.idleTimeout)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if provider != nil && provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+provider.APIKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, err
	}
	return resp.StatusCode, respBytes, nil
}

// writeResponse sends the final validated response to the client, either
// as a plain JSON body or, if the original client request asked for
// streaming, re-chunked as SSE events.
func (p *ProxyServer) writeResponse(w http.ResponseWriter, status int, respBytes []byte, content, reasoningContent, finishReason string, toolCalls []map[string]interface{}, promptTokens, completionTokens int, clientWantsStream bool, kind endpointKind) {
	if !clientWantsStream {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(respBytes)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(status)
	flusher, _ := w.(http.Flusher)

	if kind == kindChat {
		streamChatSSE(w, flusher, content, reasoningContent, finishReason, toolCalls, promptTokens, completionTokens)
	} else {
		streamCompletionSSE(w, flusher, content, finishReason)
	}
}

// sseChunkRunes controls how many runes each synthesized SSE delta carries
// when re-chunking an already-complete response for a streaming client.
// Chosen to land in roughly the same delta size as the previous
// word-based chunking, without any of its data loss (see chunkText).
const sseChunkRunes = 20

// streamChatSSE re-chunks an already-complete chat response into an
// OpenAI-style SSE stream so streaming clients (e.g. Cline) see incremental
// deltas instead of a long silent wait followed by one huge event.
// reasoning_content (the model's thinking/reasoning trace, kept distinct
// from content by llama-server per the DeepSeek-style convention Cline
// already understands) is sent as its own delta chunk before the content
// deltas, mirroring generation order — dropping it here previously meant
// any UI that renders reasoning/plan output separately from the final
// answer (e.g. Cline's Plan Mode) silently lost that formatting whenever
// the client asked for streaming.
//
// toolCalls (already fully reassembled from upstream's fragmented deltas
// by postUpstreamChatStreaming) is sent as one complete delta chunk right
// before the finish_reason chunk. This isn't token-by-token incremental
// the way upstream's own stream was, but it's the same trade-off already
// made for content via chunkText: a client that accumulates delta pieces
// by index (which is what the tool_calls streaming format requires
// clients to do anyway) ends up with the identical final tool_calls array
// either way.
//
// A final usage-only chunk (empty choices, populated usage — the same
// shape upstream itself sends because every request forces
// stream_options.include_usage) is emitted right before [DONE]. Without
// it, a streaming client has no way to learn prompt_tokens/total_tokens at
// all: Cline reads usage.prompt_tokens off every response to track how
// full the model's context window is and decide when to auto-compact.
// Silently omitting it here (as this used to) reads as permanent 0%
// context utilization, so auto-compact never fires — the request just
// keeps growing until it fails outright once the real upstream context
// fills up.
// writeHeartbeatChunk writes a single well-formed, empty-delta
// chat-completion-chunk — see the keepalive ticker in handleClassified for
// why this replaced a bare SSE comment line: it's the same "data: {...}"
// shape every other chunk in the stream already uses, so any client that
// can parse the rest of the response already handles this too, with no new
// line format to support. finish_reason is explicitly null (mid-stream,
// nothing has finished yet) — a real client reading this must not mistake
// it for the terminal chunk.
func writeHeartbeatChunk(w io.Writer, flusher http.Flusher) error {
	chunk := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-proxy-%d", time.Now().UnixNano()),
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"choices": []map[string]interface{}{
			{"index": 0, "delta": map[string]interface{}{}, "finish_reason": nil},
		},
	}
	b, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

func streamChatSSE(w io.Writer, flusher http.Flusher, content, reasoningContent, finishReason string, toolCalls []map[string]interface{}, promptTokens, completionTokens int) {
	id := fmt.Sprintf("chatcmpl-proxy-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(delta map[string]interface{}, finish interface{}) {
		chunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"choices": []map[string]interface{}{
				{"index": 0, "delta": delta, "finish_reason": finish},
			},
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}

	writeChunk(map[string]interface{}{"role": "assistant", "content": ""}, nil)

	if reasoningContent != "" {
		writeChunk(map[string]interface{}{"reasoning_content": reasoningContent}, nil)
	}

	for _, piece := range chunkText(content, sseChunkRunes) {
		writeChunk(map[string]interface{}{"content": piece}, nil)
	}

	if len(toolCalls) > 0 {
		writeChunk(map[string]interface{}{"tool_calls": toolCalls}, nil)
	}

	if finishReason == "" {
		finishReason = "stop"
	}
	writeChunk(map[string]interface{}{}, finishReason)

	usageChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"choices": []interface{}{},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(usageChunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// streamCompletionSSE re-chunks a completed native /completion response
// into llama-server's own streaming event shape.
func streamCompletionSSE(w io.Writer, flusher http.Flusher, content, finishReason string) {
	for _, piece := range chunkText(content, sseChunkRunes) {
		event := map[string]interface{}{"content": piece, "stop": false}
		b, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	final := map[string]interface{}{"content": "", "stop": true, "truncated": finishReason == "length"}
	b, _ := json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}
}

// chunkText splits text into contiguous rune-count-sized pieces with zero
// loss: concatenating every returned piece reproduces the original string
// exactly, including all whitespace, indentation, and line breaks.
//
// The previous implementation (chunkWords) split on strings.Fields — any
// run of whitespace — and rejoined pieces with a single ASCII space. That
// silently destroyed every newline, every indentation level, and every
// multi-space run in the original text: a multi-line code block came back
// as one flattened line with no indentation, for any client that requests
// streaming (Cline does, by default). A non-streaming request to the same
// backend looked completely fine, which is what made this easy to miss —
// the corruption only showed up in the SSE re-chunking path, not in the
// content itself. Slicing by rune (not byte) index avoids splitting a
// multi-byte UTF-8 character across two chunks.
func chunkText(text string, n int) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var out []string
	for i := 0; i < len(runes); i += n {
		end := i + n
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}

// systemPromptHash returns a short, stable hash (12 hex chars — plenty to
// detect any change) of the first system-role message's content in
// body["messages"], or "" if there is none (e.g. kindCompletion, which has
// no message array at all). Logged at both ingress (handleClassified,
// before any proxy-side modification) and egress (postUpstreamChatStreaming,
// right before the request is actually sent, on every retry attempt) so a
// mismatch between the two — or across attempts — is instant, verifiable
// proof that something in the request path is altering the system prompt,
// rather than having to infer it indirectly from output quality.
func systemPromptHash(body map[string]interface{}) string {
	messages, ok := body["messages"].([]interface{})
	if !ok {
		return ""
	}
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); role != "system" {
			continue
		}
		content, _ := msg["content"].(string)
		sum := sha256.Sum256([]byte(content))
		return hex.EncodeToString(sum[:6])
	}
	return ""
}

// injectSystemPrompt merges promptText into the FIRST system-role
// message found anywhere in body["messages"] — never adds a second one
// — and moves it to index 0 if it isn't already there, inserting a new
// system message at the start only if none exists at all. For
// ProxyServer.systemPromptOverride.
//
// Both of those rules are load-bearing, not cosmetic: confirmed directly
// against llama-server (jinja chat template) that a SECOND system
// message anywhere, or a lone system message NOT at index 0, both make
// it reject the whole request with "Jinja Exception: System message must
// be at the beginning" — a real production failure hit with OpenHands
// (whose system message content is an array-of-parts, not a plain
// string; the previous version of this function skipped merging into
// that shape and inserted a second system message instead, which is
// exactly what triggered it).
//
// content shape is handled per Go type: a plain string gets promptText
// prepended with a blank-line separator; an array-of-parts gets a new
// leading {"type":"text","text":promptText} part (preserving whatever
// parts were already there); anything else (missing key, unexpected
// type) is simply overwritten with promptText as a plain string, since
// there's no known structure worth trying to preserve.
//
// Deliberately a pure, deterministic function of (promptText, the
// client's own original content) — no timestamps, no randomness —
// because prefix-cache reuse (llama-server's --cache-reuse) depends on
// the exact same text landing in the exact same place on every request
// that selects a given name; see Config.SystemPrompts' doc comment.
func injectSystemPrompt(body map[string]interface{}, promptText string) {
	if promptText == "" {
		return
	}
	rawMessages, ok := body["messages"]
	if !ok {
		return
	}
	messages, ok := rawMessages.([]interface{})
	if !ok {
		return
	}

	for i, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); role != "system" {
			continue
		}

		switch content := msg["content"].(type) {
		case string:
			if content != "" {
				msg["content"] = promptText + "\n\n" + content
			} else {
				msg["content"] = promptText
			}
		case []interface{}:
			newPart := map[string]interface{}{"type": "text", "text": promptText}
			msg["content"] = append([]interface{}{newPart}, content...)
		default:
			msg["content"] = promptText
		}

		if i != 0 {
			rest := make([]interface{}, 0, len(messages)-1)
			rest = append(rest, messages[:i]...)
			rest = append(rest, messages[i+1:]...)
			body["messages"] = append([]interface{}{msg}, rest...)
		}
		return
	}

	newSystemMsg := map[string]interface{}{"role": "system", "content": promptText}
	body["messages"] = append([]interface{}{newSystemMsg}, messages...)
}

func extractClassificationText(body map[string]interface{}, kind endpointKind) string {
	if kind == kindChat {
		return extractLastUserMessage(body)
	}
	if prompt, ok := body["prompt"].(string); ok {
		return prompt
	}
	return ""
}

// extractResponseContent pulls the generated text, reasoning/thinking
// trace, and finish reason out of an upstream response body, whether it
// came from the OAI-compatible chat endpoint or llama-server's native
// /completion endpoint. reasoningContent is always "" for /completion,
// which has no equivalent concept.
func extractResponseContent(respBody map[string]interface{}, kind endpointKind) (content, reasoningContent, finishReason string, ok bool) {
	if kind == kindChat {
		choices, has := respBody["choices"].([]interface{})
		if !has || len(choices) == 0 {
			return "", "", "", false
		}
		choice0, has := choices[0].(map[string]interface{})
		if !has {
			return "", "", "", false
		}
		finishReason, _ = choice0["finish_reason"].(string)
		message, has := choice0["message"].(map[string]interface{})
		if !has {
			return "", "", "", false
		}
		content, _ = message["content"].(string)
		reasoningContent, _ = message["reasoning_content"].(string)
		return content, reasoningContent, finishReason, true
	}

	content, has := respBody["content"].(string)
	if !has {
		return "", "", "", false
	}
	truncated, _ := respBody["truncated"].(bool)
	finishReason = "stop"
	if truncated {
		finishReason = "length"
	}
	return content, "", finishReason, true
}

// extractCompletionTokens reads how many tokens the model actually
// generated in this response — usage.completion_tokens for the OAI chat
// endpoint, tokens_predicted for llama.cpp's native /completion — so a
// truncation retry can escalate from the real length that got cut off
// instead of a guess. Returns 0 if the field is missing, so callers know
// to fall back to whatever's already in the request body.
func extractCompletionTokens(respBody map[string]interface{}, kind endpointKind) int {
	if kind == kindChat {
		usage, ok := respBody["usage"].(map[string]interface{})
		if !ok {
			return 0
		}
		return getInt(usage, "completion_tokens", 0)
	}
	return getInt(respBody, "tokens_predicted", 0)
}

// applyPreset fills in only the fields the client did not already set
// explicitly; an explicit client value always wins. repeat_penalty, DRY,
// and min_p are deliberately never injected here — they're either left to
// llama-server's own default (min_p) or reserved for the retry path only.
//
// injectThinkingBudget gates thinking_budget_tokens specifically: a
// llama-server launched with --reasoning off has no use for this field,
// and sending it anyway risks re-enabling a reasoning/thinking pass the
// operator deliberately disabled server-side — which silently eats into
// the shared max_tokens budget before the model ever produces its visible
// answer, and reads exactly like "didn't follow the full instructions"
// (the response is real, just truncated by a hidden phase the caller
// never asked for). Every other preset field is backend-agnostic and
// always applied.
func applyPreset(body map[string]interface{}, preset Preset, injectThinkingBudget bool) {
	setIfAbsent(body, "temperature", preset.Temperature)
	setIfAbsent(body, "top_p", preset.TopP)
	setIfAbsent(body, "top_k", preset.TopK)
	if injectThinkingBudget {
		setIfAbsent(body, "thinking_budget_tokens", preset.ThinkingBudgetTokens)
	}
}

func setIfAbsent[T any](body map[string]interface{}, key string, val *T) {
	if val == nil {
		return
	}
	if _, exists := body[key]; exists {
		return
	}
	body[key] = *val
}

// maxTokensKey returns the field name used for the generation length cap
// on each endpoint flavor.
func maxTokensKey(kind endpointKind) string {
	if kind == kindChat {
		return "max_tokens"
	}
	return "n_predict"
}

// RetryAdjustment records one parameter change made by adjustForIssue, for
// the retry trajectory log (see logRetryTrajectory). OldValue/NewValue are
// always numeric even for fields that are conceptually integers (top_k,
// max_tokens, dry_allowed_length), so a single log shape covers every
// adjustable param.
type RetryAdjustment struct {
	Attempt  int     `json:"attempt"`
	Issue    Issue   `json:"issue"`
	Param    string  `json:"param"`
	OldValue float64 `json:"old_value"`
	NewValue float64 `json:"new_value"`
	// Detail says what specifically triggered the issue — for repetition,
	// the exact n-gram that repeated. This is the field to read when
	// deciding whether a retry was a genuine catch or a false positive
	// (e.g. legitimately repeated XML tool tags or file paths in agentic
	// output tripping the detector on healthy content).
	Detail string `json:"detail,omitempty"`
}

// RetryLogEntry captures one request's full retry trajectory: every
// adjustment made, in order, and whether the request ultimately came back
// clean. This is the data retry_step_exponent should be tuned against —
// compare Resolved rates and TotalAttempts across requests logged at
// different exponent values to see which converges faster without
// overshooting.
//
// Type discriminates this log's two entry shapes (this one and
// ThroughputLogEntry, both written to the same file) — a consumer reading
// the file must switch on it before decoding the rest of the line, rather
// than assuming every line is a RetryLogEntry.
type RetryLogEntry struct {
	Type          string            `json:"type"`
	Timestamp     time.Time         `json:"timestamp"`
	Bucket        TaskBucket        `json:"bucket"`
	StepExponent  float64           `json:"step_exponent"`
	Adjustments   []RetryAdjustment `json:"adjustments"`
	Resolved      bool              `json:"resolved"`
	TotalAttempts int               `json:"total_attempts"`
	LatencyMs     int64             `json:"latency_ms"`
}

// logRetryTrajectory appends one JSON line to the retry log for a request
// that retried at least once. Requests that succeeded on the first attempt
// never call this — there's nothing to learn from a request that needed no
// adjustment. Safe for concurrent use; writes are serialized under a mutex
// so lines from different in-flight requests never interleave.
func (p *ProxyServer) logRetryTrajectory(bucket TaskBucket, adjustments []RetryAdjustment, resolved bool, totalAttempts int, latencyMs int64) {
	if p.retryLogFile == nil {
		return
	}

	entry := RetryLogEntry{
		Type:          "retry_trajectory",
		Timestamp:     time.Now(),
		Bucket:        bucket,
		StepExponent:  p.cfg.Retry.StepExponent,
		Adjustments:   adjustments,
		Resolved:      resolved,
		TotalAttempts: totalAttempts,
		LatencyMs:     latencyMs,
	}

	line, err := json.Marshal(entry)
	if err != nil {
		log.Printf("failed to marshal retry log entry: %v", err)
		return
	}

	p.retryLogMu.Lock()
	defer p.retryLogMu.Unlock()
	p.retryLogFile.Write(line)
	p.retryLogFile.Write([]byte("\n"))
}

// ThroughputLogEntry captures one clean chat completion's actual
// performance — provider/model/bucket alongside the real prompt/completion
// token counts and wall-clock latency the dashboard already tracks, plus
// the derived tokens/sec, so provider/model throughput can be compared
// from real traffic instead of guessed at from the live dashboard estimate
// alone. Written to the same file as RetryLogEntry (see its Type field).
//
// TokensPerSecond is derived from total request latency (connection +
// prompt-processing + generation), not generation-only time — this is
// deliberate: for comparing providers, queueing/connection overhead is
// part of the real-world throughput difference between them, not noise to
// exclude. Only logged for requests with TotalAttempts == 0 in mind (see
// logThroughput) so a retry's wasted regeneration time never contaminates
// the latency of what should be one clean generation.
type ThroughputLogEntry struct {
	Type             string     `json:"type"`
	Timestamp        time.Time  `json:"timestamp"`
	Provider         string     `json:"provider"`
	Model            string     `json:"model"`
	Bucket           TaskBucket `json:"bucket"`
	PromptTokens     int        `json:"prompt_tokens"`
	CompletionTokens int        `json:"completion_tokens"`
	LatencyMs        int64      `json:"latency_ms"`
	TokensPerSecond  float64    `json:"tokens_per_second"`
}

// logThroughput appends one JSON line recording a clean chat completion's
// real throughput. Only called for requests that needed no retry at all
// (see finishClassifiedRequest) — a request that retried spent part of its
// latency on a discarded attempt, which would understate the real tokens/
// sec of the generation that actually reached the client. completionTokens
// == 0 is skipped entirely (no meaningful rate to compute, and would only
// ever show as a division-by-zero 0 — see kindCompletion, which never
// populates real token counts at all).
//
// generationMs must be generation-only elapsed time (first token to last),
// not total request latency — prompt prefill time is excluded so this
// matches the live in-flight indicator's GenerationElapsedMs (see ui.go).
// The parameter is still stored under the JSON field name LatencyMs for
// on-disk/schema compatibility with existing throughput_stats_*.json and
// retry_log_*.jsonl files; only the semantics of the value changed.
func (p *ProxyServer) logThroughput(provider, model string, bucket TaskBucket, promptTokens, completionTokens int, generationMs int64) {
	if p.retryLogFile == nil || completionTokens == 0 {
		return
	}

	var tokensPerSecond float64
	if generationMs > 0 {
		tokensPerSecond = float64(completionTokens) / (float64(generationMs) / 1000.0)
	}

	entry := ThroughputLogEntry{
		Type:             "throughput",
		Timestamp:        time.Now(),
		Provider:         provider,
		Model:            model,
		Bucket:           bucket,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		LatencyMs:        generationMs,
		TokensPerSecond:  tokensPerSecond,
	}

	line, err := json.Marshal(entry)
	if err != nil {
		log.Printf("failed to marshal throughput log entry: %v", err)
		return
	}

	p.retryLogMu.Lock()
	defer p.retryLogMu.Unlock()
	p.retryLogFile.Write(line)
	p.retryLogFile.Write([]byte("\n"))
}

// updateThroughputStats accumulates one clean chat completion's data into
// the running per-(provider,model,bucket) totals and persists the whole
// set atomically — called alongside logThroughput (see
// finishClassifiedRequest) but kept in a separate file (throughput_stats.go)
// so this aggregated history survives independently of the raw per-request
// JSONL log being cleared or rotated. completionTokens == 0 or
// generationMs <= 0 skip entirely — no meaningful rate to accumulate.
//
// generationMs must be generation-only elapsed time, not total request
// latency — see logThroughput's doc comment for why, and note the running
// SumLatencyMs field keeps its on-disk name even though this fix changed
// what value flows into it: pre-existing accumulated entries (from before
// this fix) still carry old, total-latency-based sums, so their averages
// stay diluted until enough new, correctly-measured samples accumulate on
// top, or the stats file is reset.
func (p *ProxyServer) updateThroughputStats(provider, model string, bucket TaskBucket, promptTokens, completionTokens int, generationMs int64) {
	if p.throughputStatsPath == "" || completionTokens == 0 {
		return
	}
	// A near-instant response (a fast local test, or a genuinely tiny
	// completion) can measure 0ms latency — the sample is still real and
	// worth counting in Samples/sums, it just contributes 0 to min/max
	// rather than a division-by-zero, matching AverageTokensPerSecond's
	// own guard.
	var tps float64
	if generationMs > 0 {
		tps = float64(completionTokens) / (float64(generationMs) / 1000.0)
	}

	p.throughputStatsMu.Lock()
	defer p.throughputStatsMu.Unlock()

	var entry *ThroughputStatsEntry
	for i := range p.throughputStats {
		e := &p.throughputStats[i]
		if e.Provider == provider && e.Model == model && e.Bucket == bucket {
			entry = e
			break
		}
	}
	if entry == nil {
		p.throughputStats = append(p.throughputStats, ThroughputStatsEntry{
			Provider:           provider,
			Model:              model,
			Bucket:             bucket,
			MinTokensPerSecond: tps,
			MaxTokensPerSecond: tps,
		})
		entry = &p.throughputStats[len(p.throughputStats)-1]
	}
	entry.Samples++
	entry.SumPromptTokens += int64(promptTokens)
	entry.SumCompletionTokens += int64(completionTokens)
	entry.SumLatencyMs += generationMs
	if tps < entry.MinTokensPerSecond {
		entry.MinTokensPerSecond = tps
	}
	if tps > entry.MaxTokensPerSecond {
		entry.MaxTokensPerSecond = tps
	}
	entry.LastUpdated = time.Now()

	if err := saveThroughputStatsAtomic(p.throughputStatsPath, p.throughputStats); err != nil {
		log.Printf("failed to persist throughput stats to %s: %v", p.throughputStatsPath, err)
	}
}

// ThroughputStatsSnapshot returns a copy of the current in-memory
// aggregated stats — safe for concurrent use (e.g. the dashboard's
// periodic refresh tick, ui.go) without racing the request-handling
// goroutines that mutate the live slice via updateThroughputStats.
func (p *ProxyServer) ThroughputStatsSnapshot() []ThroughputStatsEntry {
	p.throughputStatsMu.Lock()
	defer p.throughputStatsMu.Unlock()
	out := make([]ThroughputStatsEntry, len(p.throughputStats))
	copy(out, p.throughputStats)
	return out
}

// adjustForIssue applies the reactive-only [retry] adjustments on top of
// whatever preset is already in the body. repeat_penalty and DRY only ever
// appear here, never on a clean first-pass request.
//
// attempt is the 0-indexed retry number (0 for the first retry, 1 for the
// second, ...), used to scale the adjustment magnitude by
// retry_step_exponent^attempt: with the default exponent of 1.0 every
// retry takes the same flat step (original behavior); a higher exponent
// makes later retries within the same request take bigger steps than
// earlier ones. max_tokens is deliberately excluded from this scaling —
// its multiplier already compounds naturally attempt over attempt, since
// each retry multiplies the previous attempt's value rather than a fixed
// base, so applying the exponent on top would compound two escalating
// factors together.
//
// actualCompletionTokens is how many tokens the model actually generated
// before hitting the cap that just got flagged as truncated (0 if
// unavailable). Since no preset ever injects max_tokens on the initial
// request, the request body itself usually has no prior value to read for
// the IssueTruncated case — using the real generated length here instead
// of a made-up fallback (previously a hardcoded 512, regardless of the
// backend's actual default cap, which is often much higher) means the
// escalated cap is grounded in what the response actually needed rather
// than a guess that could already be smaller than what was truncated.
func adjustForIssue(body map[string]interface{}, issue Issue, cfg *Config, kind endpointKind, attempt int, actualCompletionTokens int) []RetryAdjustment {
	det := &cfg.Detection
	rty := &cfg.Retry
	step := math.Pow(rty.StepExponent, float64(attempt))

	switch issue {
	case IssueRepetition, IssueReasoningLoop:
		// IssueReasoningLoop is the same underlying failure as
		// IssueRepetition (a degenerate repeating loop), just caught live
		// mid-stream instead of after the fact — same remedy applies.
		if rty.PreferDryOverRepeatPenalty {
			// Must escalate off the real accumulated old value, the same
			// way repeat_penalty does below — not recompute from
			// DryMultiplierOnRetry*step alone. At the shipped default
			// retry_step_exponent=1.0, step is 1^attempt=1 for every
			// attempt, so a formula that ignores old would produce the
			// exact same dry_multiplier on every retry (regression: found
			// in real retry_log_*.jsonl data showing two consecutive
			// repetition retries both landing on dry_multiplier=0.8 —
			// identical, no escalation at all).
			old := getFloat(body, "dry_multiplier", 0)
			next := old + rty.DryMultiplierOnRetry*step
			body["dry_multiplier"] = next
			body["dry_base"] = rty.DryBase
			body["dry_allowed_length"] = rty.DryAllowedLength
			return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "dry_multiplier", OldValue: old, NewValue: next}}
		}
		old := getFloat(body, "repeat_penalty", 1.0)
		next := old + rty.RepeatPenaltyIncrement*step
		body["repeat_penalty"] = next
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "repeat_penalty", OldValue: old, NewValue: next}}
	case IssueTruncated:
		key := maxTokensKey(kind)
		old := float64(actualCompletionTokens)
		if old <= 0 {
			old = getFloat(body, key, 512)
		}
		next := old * det.MaxTokensRetryMultiplier
		if next > float64(det.MaxTokensCeiling) {
			next = float64(det.MaxTokensCeiling)
		}
		nextInt := int(next)
		body[key] = nextInt
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: key, OldValue: old, NewValue: float64(nextInt)}}
	case IssueBadSyntax:
		// temperature_floor exists specifically so repeated bad-syntax
		// retries can never land on exactly 0 (greedy/degenerate decoding).
		old := getFloat(body, "temperature", 0.8)
		next := old - rty.TemperatureDecrementOnBadSyntax*step
		if next < rty.TemperatureFloor {
			next = rty.TemperatureFloor
		}
		body["temperature"] = next
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "temperature", OldValue: old, NewValue: next}}
	case IssueEmpty:
		// Opposite direction from IssueBadSyntax: a blank completion is more
		// often an unlucky early-EOS draw (too conservative a sampling
		// state) than an aversion fixed by determinism, so retries push
		// temperature UP toward more diverse sampling instead of down.
		// temperature_ceiling mirrors temperature_floor's role for
		// IssueBadSyntax — keeps repeated empty-retries from climbing
		// toward effectively-random output.
		old := getFloat(body, "temperature", 0.8)
		next := old + rty.TemperatureIncrementOnEmpty*step
		if next > rty.TemperatureCeiling {
			next = rty.TemperatureCeiling
		}
		body["temperature"] = next
		return []RetryAdjustment{{Attempt: attempt, Issue: issue, Param: "temperature", OldValue: old, NewValue: next}}
	}
	return nil
}

// truncateForLog keeps a raw upstream error body short enough to be
// readable in the dashboard's one-line log.
func truncateForLog(b []byte) string {
	const max = 200
	s := strings.TrimSpace(string(b))
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func getFloat(body map[string]interface{}, key string, def float64) float64 {
	v, ok := body[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	default:
		return def
	}
}

func getInt(body map[string]interface{}, key string, def int) int {
	return int(getFloat(body, key, float64(def)))
}

func getString(body map[string]interface{}, key string, def string) string {
	if s, ok := body[key].(string); ok {
		return s
	}
	return def
}

// effectiveDynamicValues reads back the sampling params currently in the
// (possibly preset-filled, possibly retry-adjusted) request body, for
// display on the dashboard. repeat_penalty naturally reads back as the
// neutral baseline (1.0) whenever no retry has injected it.
func effectiveDynamicValues(body map[string]interface{}) (temp, topP float64, topK int, penalty float64, budget int) {
	temp = getFloat(body, "temperature", 0)
	topP = getFloat(body, "top_p", 0)
	topK = getInt(body, "top_k", 0)
	penalty = getFloat(body, "repeat_penalty", 1.0)
	budget = getInt(body, "thinking_budget_tokens", 0)
	return
}
