package main

import "testing"

func testConfigWithProviders() *Config {
	cfg := testConfig()
	cfg.Server.UpstreamHost = "127.0.0.1"
	cfg.Server.UpstreamPort = 9091
	cfg.Providers = map[ProviderName]ProviderConfig{
		ProviderOpenRouter: {BaseURL: "https://openrouter.ai/api/v1", APIKey: "sk-or-test", ListenPort: 9095},
	}
	return cfg
}

func TestResolveBackendNoArgsUsesLocal(t *testing.T) {
	cfg := testConfigWithProviders()
	provider, label, desc := resolveBackend(cfg, nil)

	if provider != nil {
		t.Errorf("expected nil provider for local mode, got %+v", provider)
	}
	if label != "local" {
		t.Errorf("expected label 'local', got %q", label)
	}
	if desc != "http://127.0.0.1:9091" {
		t.Errorf("expected local desc to reflect [server] upstream, got %q", desc)
	}
}

func TestResolveBackendSelectsConfiguredProvider(t *testing.T) {
	cfg := testConfigWithProviders()
	provider, label, desc := resolveBackend(cfg, []string{"openrouter"})

	if provider == nil {
		t.Fatal("expected a non-nil provider")
	}
	if label != "openrouter" {
		t.Errorf("expected label 'openrouter', got %q", label)
	}
	if desc != "https://openrouter.ai/api/v1" {
		t.Errorf("expected desc to be the provider's base_url, got %q", desc)
	}
	if provider.APIKey != "sk-or-test" {
		t.Errorf("expected provider APIKey to come through, got %q", provider.APIKey)
	}
	if provider.ListenPort != 9095 {
		t.Errorf("expected provider's own ListenPort to come through, got %d", provider.ListenPort)
	}
}

// TestListenPortSelection mirrors main()'s port-selection logic: local mode
// uses [server].listen_port, provider mode uses that provider's own
// listen_port instead, so multiple backends can run concurrently on
// distinct ports.
func TestListenPortSelection(t *testing.T) {
	cfg := testConfigWithProviders()
	cfg.Server.ListenPort = 9090

	provider, _, _ := resolveBackend(cfg, nil)
	listenPort := cfg.Server.ListenPort
	if provider != nil {
		listenPort = provider.ListenPort
	}
	if listenPort != 9090 {
		t.Errorf("local mode: listenPort = %d, want [server].listen_port 9090", listenPort)
	}

	provider, _, _ = resolveBackend(cfg, []string{"openrouter"})
	listenPort = cfg.Server.ListenPort
	if provider != nil {
		listenPort = provider.ListenPort
	}
	if listenPort != 9095 {
		t.Errorf("openrouter mode: listenPort = %d, want provider's own ListenPort 9095", listenPort)
	}
}

func TestResolveBackendIsCaseInsensitive(t *testing.T) {
	cfg := testConfigWithProviders()
	provider, label, _ := resolveBackend(cfg, []string{"OpenRouter"})

	if provider == nil {
		t.Fatal("expected provider name matching to be case-insensitive")
	}
	if label != "openrouter" {
		t.Errorf("expected normalized label 'openrouter', got %q", label)
	}
}

// TestResolveRelativeToExeDirJoinsRelativePaths is the regression test for
// a real gap: config.ini/logPath/retryLogPath all resolve relative to the
// binary's own directory, but ca_cert_path/ca_key_path didn't — a relative
// path like "cert/intermediate.crt" only happened to work if the process
// was launched from inside its own directory, breaking for any other
// launch method (a symlink in PATH, a systemd service, a different cwd).
func TestResolveRelativeToExeDirJoinsRelativePaths(t *testing.T) {
	got := resolveRelativeToExeDir("/opt/llama-dyn-proxy/llama-dyn-proxy", "cert/intermediate.crt")
	want := "/opt/llama-dyn-proxy/cert/intermediate.crt"
	if got != want {
		t.Errorf("resolveRelativeToExeDir = %q, want %q", got, want)
	}
}

func TestResolveRelativeToExeDirLeavesAbsolutePathUnchanged(t *testing.T) {
	got := resolveRelativeToExeDir("/opt/llama-dyn-proxy/llama-dyn-proxy", "/home/user/cert/intermediate.crt")
	want := "/home/user/cert/intermediate.crt"
	if got != want {
		t.Errorf("resolveRelativeToExeDir = %q, want %q (absolute paths must pass through unchanged)", got, want)
	}
}

func TestResolveRelativeToExeDirLeavesEmptyPathUnchanged(t *testing.T) {
	got := resolveRelativeToExeDir("/opt/llama-dyn-proxy/llama-dyn-proxy", "")
	if got != "" {
		t.Errorf("resolveRelativeToExeDir(_, \"\") = %q, want empty string to stay unset rather than resolving to the exe's own directory", got)
	}
}
