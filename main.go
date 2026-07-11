package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	exePath, err := os.Executable()
	if err != nil {
		exePath, _ = filepath.Abs(os.Args[0])
	}
	configPath := filepath.Join(filepath.Dir(exePath), "config.ini")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	provider, providerLabel, backendDesc := resolveBackend(cfg, os.Args[1:])

	listenPort := cfg.Server.ListenPort
	if provider != nil {
		listenPort = provider.ListenPort
	}
	listenAddr := fmt.Sprintf("127.0.0.1:%d", listenPort)

	logPath := filepath.Join(filepath.Dir(exePath), fmt.Sprintf("proxy_%s.log", providerLabel))
	fmt.Printf("llama-dyn-proxy\n  listening on http://%s\n  backend      %s (%s)\n  log          %s\n\n", listenAddr, providerLabel, backendDesc, logPath)

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

	// Blocks until the dashboard quits (Ctrl+C).
	dashboardBackend := fmt.Sprintf("%s (%s)", providerLabel, backendDesc)
	if err := runDashboard(events, progressCh, listenAddr, dashboardBackend); err != nil {
		log.Printf("dashboard error: %v", err)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("http server shutdown error: %v", err)
	}
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
