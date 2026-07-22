package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func main() {
	exePath, err := os.Executable()
	if err != nil {
		exePath, _ = filepath.Abs(os.Args[0])
	}

	for _, a := range os.Args[1:] {
		if a == "--report" || a == "-report" {
			runReportMode(filepath.Dir(exePath))
			return
		}
	}

	// alertEnabled and the remaining positional args (provider selection)
	// are both derived from os.Args here, before either LoadConfig or
	// resolveBackend see it — --alert/-alert can appear in any position
	// (e.g. "clinepass --alert" or "--alert clinepass") and must never be
	// mistaken for a provider name by resolveBackend.
	remainingArgs, alertEnabled := filterAlertFlag(os.Args[1:])

	configPath := filepath.Join(filepath.Dir(exePath), "config.ini")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	cfg.Server.AlertEnabled = alertEnabled

	// --web launches the multi-listener control-panel mode (all backends
	// supervised, toggleable live from a browser) instead of the default
	// single-backend flow. --web-port sets the panel's port (default 9080).
	webEnabled, webPort, remainingArgs := filterWebFlags(remainingArgs)
	if webEnabled {
		runWebMode(cfg, exePath, webPort, alertEnabled)
		return
	}

	provider, providerLabel, backendDesc := resolveBackend(cfg, remainingArgs)

	// Running the clinepass backend directly: reuse the connector's stored
	// key, or prompt for one if none is configured yet (clinepass chat needs
	// an API key — see clinepass_auth.go).
	if provider != nil && providerLabel == string(ProviderClinepass) {
		provider.APIKey = ensureClinepassKey(provider.APIKey, provider.BaseURL)
	}

	listenPort := cfg.Server.ListenPort
	if provider != nil {
		listenPort = provider.ListenPort
	}
	listenAddr := fmt.Sprintf("127.0.0.1:%d", listenPort)

	logPath := filepath.Join(filepath.Dir(exePath), fmt.Sprintf("proxy_%s.log", providerLabel))
	fmt.Printf("llama-dyn-proxy\n  listening on http://%s\n  backend      %s (%s)\n  log          %s\n\n", listenAddr, providerLabel, backendDesc, logPath)
	if cfg.Server.AlertEnabled {
		if len(cfg.Server.AlertModels) == 0 {
			fmt.Println("  --alert enabled, but alert_models is empty in config.ini — no models will actually get alert-continuation until you list some")
		} else {
			fmt.Printf("  --alert enabled for: %s\n\n", strings.Join(cfg.Server.AlertModels, ", "))
		}
	}

	events := make(chan UIEvent, 32)
	progressCh := make(chan ProgressEvent, 32)

	retryLogPath := filepath.Join(filepath.Dir(exePath), fmt.Sprintf("retry_log_%s.jsonl", providerLabel))
	proxy, err := NewProxyServer(cfg, provider, providerLabel, events, progressCh, retryLogPath)
	if err != nil {
		log.Fatalf("failed to init proxy: %v", err)
	}
	defer proxy.Close()

	srv := &http.Server{
		Addr:    listenAddr,
		Handler: proxy.Handler(),
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("http server error: %v", err)
		}
	}()

	// Forward-proxy mode (mitm.go) is an alternative to the [provider.*]
	// reverse-proxy model above: instead of registering a base_url/model
	// per vendor and pointing an agent's own base_url at this proxy, the
	// agent's HTTP_PROXY/HTTPS_PROXY points at ForwardProxy.ListenPort
	// directly, keeping its own real base_url/API key/model unchanged. It
	// runs concurrently with whichever backend was selected above — both
	// share the same dashboard (events/progressCh), so forward-proxy
	// requests show up in the same live request log, tagged with the
	// intercepted host via UIEvent.Host.
	var forwardProxySrv *http.Server
	var forwardProxyInstance *ProxyServer
	if cfg.ForwardProxy.Enabled {
		// A relative ca_cert_path/ca_key_path (e.g. "cert/intermediate.crt")
		// resolves against the binary's own directory, exactly like
		// config.ini/logPath/retryLogPath above — not the process's current
		// working directory, which would only happen to work if the binary
		// is launched from inside its own directory. An absolute path is
		// left untouched.
		caCertPath := resolveRelativeToExeDir(exePath, cfg.ForwardProxy.CACertPath)
		caKeyPath := resolveRelativeToExeDir(exePath, cfg.ForwardProxy.CAKeyPath)
		ca, err := LoadCA(caCertPath, caKeyPath)
		if err != nil {
			log.Fatalf("failed to load forward-proxy CA: %v", err)
		}

		forwardRetryLogPath := filepath.Join(filepath.Dir(exePath), "retry_log_forward_proxy.jsonl")
		// provider is a non-nil-but-empty ProviderConfig, not the local
		// llama-server path: Handler() then registers /v1/chat/completions
		// and /v1/models only (matching what forward-proxy mode actually
		// supports), and the empty Model/APIKey fields mean the existing
		// "override model if configured" / "set Authorization from
		// provider.APIKey" logic naturally never fires — the per-request
		// forwardProxyOverride (mitm.go) supplies both dynamically instead.
		forwardProxyInstance, err = NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy", events, progressCh, forwardRetryLogPath)
		if err != nil {
			log.Fatalf("failed to init forward-proxy: %v", err)
		}

		fps := NewForwardProxyServer(ca, cfg.ForwardProxy.AllowedHosts, cfg.ForwardProxy.PassthroughHosts, newForwardProxyPipelineHandler(forwardProxyInstance))
		forwardListenAddr := fmt.Sprintf("127.0.0.1:%d", cfg.ForwardProxy.ListenPort)
		forwardProxySrv = &http.Server{
			Addr:    forwardListenAddr,
			Handler: fps,
		}
		fmt.Printf("  forward proxy http://%s (%d hosts allowed)\n", forwardListenAddr, len(cfg.ForwardProxy.AllowedHosts))

		go func() {
			if err := forwardProxySrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				log.Printf("forward-proxy http server error: %v", err)
			}
		}()
	}

	// The dashboard takes over the whole terminal via alt-screen, so any
	// log.Printf from this point on (diagnostics from postUpstreamChatStreaming,
	// passthrough errors, etc.) would otherwise be silently overwritten by
	// the next TUI redraw — never actually visible, not even briefly.
	// Redirecting to a file means it's always inspectable after the fact,
	// regardless of dashboard state. Startup errors above this line
	// (config/provider/proxy-init failures) intentionally still go to the
	// terminal directly, since the dashboard hasn't taken it over yet.
	if logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644); err != nil {
		log.Printf("warning: could not open %s for logging (%v); diagnostics will be lost once the dashboard starts", logPath, err)
	} else {
		defer logFile.Close()
		log.SetOutput(logFile)
	}

	// Combines this run's own throughput stats with forward-proxy mode's
	// (a separate ProxyServer instance, see above) into one snapshot for
	// the dashboard's summary panel — both share the same live dashboard
	// already (events/progressCh), so their throughput history is shown
	// together too.
	throughputSnapshot := func() []ThroughputStatsEntry {
		all := proxy.ThroughputStatsSnapshot()
		if forwardProxyInstance != nil {
			all = append(all, forwardProxyInstance.ThroughputStatsSnapshot()...)
		}
		return all
	}

	// Forced-bucket keybindings apply to every instance sharing this
	// dashboard (both the main proxy and forward-proxy mode's, if
	// enabled) so pinning a mode from the dashboard affects all traffic
	// through this process, not just whichever instance happens to
	// handle the next request.
	controls := DashboardControls{
		SetForcedBucket: func(b TaskBucket) {
			proxy.SetForcedBucket(b)
			if forwardProxyInstance != nil {
				forwardProxyInstance.SetForcedBucket(b)
			}
		},
		ClearForcedBucket: func() {
			proxy.ClearForcedBucket()
			if forwardProxyInstance != nil {
				forwardProxyInstance.ClearForcedBucket()
			}
		},
	}
	// Blocks until the dashboard quits (Ctrl+C).
	dashboardBackend := fmt.Sprintf("%s (%s)", providerLabel, backendDesc)
	if cfg.ForwardProxy.Enabled {
		dashboardBackend += fmt.Sprintf("  |  forward-proxy :%d (%d hosts)", cfg.ForwardProxy.ListenPort, len(cfg.ForwardProxy.AllowedHosts))
	}
	if err := runDashboard(events, progressCh, listenAddr, dashboardBackend, throughputSnapshot, controls); err != nil {
		log.Printf("dashboard error: %v", err)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("http server shutdown error: %v", err)
	}
	if forwardProxySrv != nil {
		if err := forwardProxySrv.Shutdown(shutdownCtx); err != nil {
			log.Printf("forward-proxy http server shutdown error: %v", err)
		}
		forwardProxyInstance.Close()
	}
}

// runReportMode reads every throughput_stats_*.json file next to the
// executable (one per provider label that's been run — see
// throughputStatsPathFor) and prints a combined, human-readable report,
// without starting the proxy itself. Lets you check accumulated
// performance data on demand (./llama-dyn-proxy --report), including while
// another instance of this proxy is already running normally — this reads
// the persisted stats files directly rather than talking to a running
// process.
func runReportMode(exeDir string) {
	matches, err := filepath.Glob(filepath.Join(exeDir, "throughput_stats_*.json"))
	if err != nil {
		log.Fatalf("failed to list throughput stats files: %v", err)
	}
	if len(matches) == 0 {
		fmt.Println("no throughput stats files found yet — run some requests through the proxy first")
		return
	}

	var all []ThroughputStatsEntry
	for _, path := range matches {
		entries, err := loadThroughputStats(path)
		if err != nil {
			log.Printf("warning: failed to read %s: %v", path, err)
			continue
		}
		all = append(all, entries...)
	}

	fmt.Print(formatThroughputReport(all))
}

// resolveRelativeToExeDir joins a relative path against the binary's own
// directory (the same base config.ini/logPath/retryLogPath already use),
// so it resolves consistently regardless of the process's current working
// directory. An empty or already-absolute path is returned unchanged —
// empty because ForwardProxyConfig's zero value must stay "unset" rather
// than resolving to the exe's directory itself.
func resolveRelativeToExeDir(exePath, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(filepath.Dir(exePath), path)
}

// filterAlertFlag strips --alert/-alert from args (wherever it appears)
// and reports whether it was present — kept separate from resolveBackend
// so a flag can be combined with a provider argument in either order
// (e.g. "clinepass --alert" or "--alert clinepass") without resolveBackend
// ever mistaking the flag for a provider name.
func filterAlertFlag(args []string) (remaining []string, alertEnabled bool) {
	remaining = make([]string, 0, len(args))
	for _, a := range args {
		if a == "--alert" || a == "-alert" {
			alertEnabled = true
			continue
		}
		remaining = append(remaining, a)
	}
	return remaining, alertEnabled
}

// filterWebFlags strips --web and --web-port <n> (or --web-port=<n>) from args,
// returning whether web mode was requested and the panel port (default 9080).
// Kept separate from resolveBackend for the same reason as filterAlertFlag: the
// flags can appear in any position without being mistaken for a provider name.
func filterWebFlags(args []string) (enabled bool, port int, remaining []string) {
	port = 9080
	remaining = make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--web" || a == "-web":
			enabled = true
		case a == "--web-port" || a == "-web-port":
			if i+1 < len(args) {
				if p, err := strconv.Atoi(args[i+1]); err == nil {
					port = p
				}
				i++
			}
		case strings.HasPrefix(a, "--web-port="):
			if p, err := strconv.Atoi(strings.TrimPrefix(a, "--web-port=")); err == nil {
				port = p
			}
		default:
			remaining = append(remaining, a)
		}
	}
	return enabled, port, remaining
}

// resolveBackend picks the upstream backend from an optional CLI argument:
// no argument means the local llama-server configured in [server]; a
// recognized provider name (e.g. "openrouter") switches to that remote
// backend for the lifetime of this process. An unrecognized name is a
// fatal, clearly-reported error rather than a silent fallback to local.
func resolveBackend(cfg *Config, args []string) (provider *ProviderConfig, label string, desc string) {
	if len(args) == 0 || args[0] == "" {
		return nil, "local", fmt.Sprintf("http://%s:%d", cfg.Server.UpstreamHost, cfg.Server.UpstreamPort)
	}

	name := ProviderName(strings.ToLower(args[0]))
	pc, ok := cfg.Providers[name]
	if !ok {
		names := make([]string, len(KnownProviders))
		for i, n := range KnownProviders {
			names[i] = string(n)
		}
		log.Fatalf("unknown provider %q; valid options: %s", args[0], strings.Join(names, ", "))
	}
	if pc.BaseURL == "" {
		log.Fatalf("provider %q has no base_url configured in config.ini ([provider.%s])", name, name)
	}
	if pc.APIKey == "" {
		log.Printf("warning: provider %q has no api_key (checked %s env var and config.ini); requests will likely be rejected", name, providerAPIKeyEnvVar(name))
	}

	return &pc, string(name), pc.BaseURL
}
