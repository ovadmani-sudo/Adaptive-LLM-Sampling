package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/unix"
)

// runWebMode is the multi-listener control-panel mode (--web). Unlike the
// default single-backend flow in main(), it builds a ProxyServer for every
// configured backend (local + each [provider.*]) plus the forward-proxy, hands
// them all to a Supervisor so each can be started/stopped live from the web
// panel, and serves the panel itself. The terminal dashboard still runs
// alongside, fed the same event streams through a Broker so neither starves
// the other. Blocks until the dashboard quits.
func runWebMode(cfg *Config, exePath string, webPort int, alertEnabled bool) {
	broker := NewBroker()
	sup := NewSupervisor()

	dir := filepath.Dir(exePath)
	retryLog := func(label string) string {
		return filepath.Join(dir, fmt.Sprintf("retry_log_%s.jsonl", label))
	}

	// track proxies for throughput aggregation + graceful close at exit.
	var proxies []*ProxyServer
	mkBackend := func(name, kind string, provider *ProviderConfig, port int) {
		p, err := NewProxyServer(cfg, provider, name, broker.EventsIn, broker.ProgressIn, retryLog(name))
		if err != nil {
			log.Printf("web: skipping backend %q: %v", name, err)
			return
		}
		// Alert-continuation is a local-only feature: only the local backend
		// starts from the --alert flag; every other backend has it off (and
		// no panel switch). The panel then toggles local live.
		p.SetAlertEnabled(name == "local" && cfg.Server.AlertEnabled)
		proxies = append(proxies, p)
		sup.Register(name, kind, port, p.Handler(), p)
	}

	// Local llama-server backend (always registered).
	mkBackend("local", "local", nil, cfg.Server.ListenPort)

	// Every configured remote provider becomes its own listener on its own
	// port — this is the multi-backend capability (all can run at once).
	for _, name := range KnownProviders {
		pc, ok := cfg.Providers[name]
		if !ok || pc.BaseURL == "" {
			continue // unconfigured — nothing to route to
		}
		pcCopy := pc
		// clinepass shares the standalone connector's credential: resolve it
		// (config.ini / env / connector login) and prompt on a terminal if
		// missing, so a fresh setup asks for the key on first start.
		if name == ProviderClinepass {
			pcCopy.APIKey = ensureClinepassKey(pc.APIKey, pc.BaseURL)
		}
		mkBackend(string(name), "provider", &pcCopy, pc.ListenPort)
	}

	// Forward-proxy (MITM) listener — registered only if its CA loads.
	var fwdProxy *ProxyServer
	if cfg.ForwardProxy.CACertPath != "" {
		caCertPath := resolveRelativeToExeDir(exePath, cfg.ForwardProxy.CACertPath)
		caKeyPath := resolveRelativeToExeDir(exePath, cfg.ForwardProxy.CAKeyPath)
		if ca, err := LoadCA(caCertPath, caKeyPath); err != nil {
			log.Printf("web: forward-proxy unavailable (CA load failed): %v", err)
		} else {
			fp, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy", broker.EventsIn, broker.ProgressIn, retryLog("forward_proxy"))
			if err != nil {
				log.Printf("web: forward-proxy init failed: %v", err)
			} else {
				fwdProxy = fp
				proxies = append(proxies, fp)
				fps := NewForwardProxyServer(ca, cfg.ForwardProxy.AllowedHosts, cfg.ForwardProxy.PassthroughHosts, newForwardProxyPipelineHandler(fp))
				sup.Register("forward-proxy", "forward-proxy", cfg.ForwardProxy.ListenPort, fps, fp)
			}
		}
	}

	// Restore per-listener switches (bypass, alert, model, forced bucket,
	// vision describe, system prompt, running) from the last time this
	// process ran, so a restart doesn't silently reset every agent's
	// controls back to config.ini's defaults. Must run after every
	// mkBackend/Register call above — a listener registered later would
	// never receive its saved state — and before the default auto-start
	// below, which is the fallback for names with no saved entry yet
	// (first run, or a newly-added backend).
	sup.EnablePersistence(filepath.Join(dir, "listener_state.json"))

	// Auto-start a sensible initial set: the local backend, plus forward-proxy
	// if it was enabled in config. Everything else starts stopped and is
	// toggled from the panel.
	if err := sup.Start("local"); err != nil {
		log.Printf("web: could not start local backend: %v", err)
	}
	if fwdProxy != nil && cfg.ForwardProxy.Enabled {
		if err := sup.Start("forward-proxy"); err != nil {
			log.Printf("web: could not start forward-proxy: %v", err)
		}
	}

	// Throughput snapshot aggregates every backend's stats.
	throughputSnapshot := func() []ThroughputStatsEntry {
		var all []ThroughputStatsEntry
		for _, p := range proxies {
			all = append(all, p.ThroughputStatsSnapshot()...)
		}
		return all
	}

	// Serve the control panel.
	panel := NewWebPanel(sup, broker, throughputSnapshot, cfg.Providers[ProviderClinepass].BaseURL)
	webAddr := fmt.Sprintf("127.0.0.1:%d", webPort)
	sup.Register("__web_panel__", "control-panel", webPort, panel.Handler(), nil)
	if err := sup.Start("__web_panel__"); err != nil {
		log.Fatalf("web: could not start control panel on %s: %v", webAddr, err)
	}

	fmt.Printf("llama-dyn-proxy — web control panel\n  open  http://%s\n\n", webAddr)
	fmt.Println("  backends registered:")
	for _, st := range sup.Status() {
		if st.Name == "__web_panel__" {
			continue
		}
		state := "off"
		if st.Running {
			state = "on"
		}
		fmt.Printf("    %-14s :%d  [%s]\n", st.Name, st.Port, state)
	}
	fmt.Println()

	// From here the TUI takes the terminal, so redirect diagnostics to a file.
	logPath := filepath.Join(dir, "proxy_web.log")
	if logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644); err == nil {
		defer logFile.Close()
		log.SetOutput(logFile)
	}

	if stdinIsTerminal() {
		// Interactive terminal: run the TUI alongside the web panel, fed via
		// broker subscriptions so it and the browser don't starve each other.
		// This TUI is a mirror of ALL backends combined (it has no single
		// listener of its own to target), so its keybinding scopes to
		// "local" specifically — the traditional default backend — rather
		// than every listener at once: forced bucket is per-listener now
		// (see Supervisor.SetForcedBucket), and silently reapplying it to
		// every agent sharing this process is exactly the cross-agent leak
		// that made it per-listener in the first place. Any other
		// listener's classification mode is controlled from the web panel
		// itself, which has a per-row control.
		controls := DashboardControls{
			SetForcedBucket:   func(b TaskBucket) { sup.SetForcedBucket("local", b) },
			ClearForcedBucket: func() { sup.SetForcedBucket("local", "") },
		}
		uiCh, cancelUI := broker.SubscribeUI()
		defer cancelUI()
		progCh, cancelProg := broker.SubscribeProgress()
		defer cancelProg()

		dashboardBackend := fmt.Sprintf("web panel :%d  ·  %d backends", webPort, len(proxies))
		if err := runDashboard(uiCh, progCh, webAddr, dashboardBackend, throughputSnapshot, controls); err != nil {
			log.Printf("dashboard error: %v", err)
		}
	} else {
		// Headless (no TTY): skip the TUI and just serve the panel until a
		// signal — so --web works over SSH / in a service, controlled purely
		// from the browser.
		fmt.Printf("  (no terminal detected — running headless; control via the web panel, Ctrl+C to quit)\n")
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		<-ctx.Done()
	}

	// Graceful shutdown.
	sup.StopAll()
	for _, p := range proxies {
		p.Close()
	}
}

// stdinIsTerminal reports whether stdin is a real interactive terminal, used
// to decide whether to run the TUI (vs. headless browser-only mode). A plain
// ModeCharDevice check is not enough — /dev/null is also a char device — so
// this uses a real isatty (TCGETS) check via x/sys/unix (already a dependency
// of the TUI library).
func stdinIsTerminal() bool {
	_, err := unix.IoctlGetTermios(int(os.Stdin.Fd()), unix.TCGETS)
	return err == nil
}
