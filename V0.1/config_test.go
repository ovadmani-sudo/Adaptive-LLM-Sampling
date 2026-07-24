package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProviderAPIKeyEnvVarNaming(t *testing.T) {
	cases := map[ProviderName]string{
		ProviderClaude:     "CLAUDE_API_KEY",
		ProviderGemini:     "GEMINI_API_KEY",
		ProviderOpenAI:     "OPENAI_API_KEY",
		ProviderOpenRouter: "OPENROUTER_API_KEY",
		ProviderClinepass:  "CLINEPASS_API_KEY",
	}
	for name, want := range cases {
		if got := providerAPIKeyEnvVar(name); got != want {
			t.Errorf("providerAPIKeyEnvVar(%q) = %q, want %q", name, got, want)
		}
	}
}

func writeTestConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.ini")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("writing test config: %v", err)
	}
	return path
}

const minimalProviderConfig = `[provider.openrouter]
base_url = https://openrouter.ai/api/v1
api_key = ini-key-value
model = openrouter/free
`

func TestLoadConfigUsesIniAPIKeyWhenNoEnvVarSet(t *testing.T) {
	path := writeTestConfig(t, minimalProviderConfig)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].APIKey; got != "ini-key-value" {
		t.Errorf("APIKey = %q, want ini-key-value", got)
	}
}

func TestLoadConfigEnvVarOverridesIniAPIKey(t *testing.T) {
	path := writeTestConfig(t, minimalProviderConfig)
	t.Setenv("OPENROUTER_API_KEY", "env-key-value")

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].APIKey; got != "env-key-value" {
		t.Errorf("APIKey = %q, want env-key-value (env must win over config.ini)", got)
	}
}

func TestLoadConfigEnvVarWorksWithoutIniProviderSection(t *testing.T) {
	// No [provider.claude] section at all in this file.
	path := writeTestConfig(t, minimalProviderConfig)
	t.Setenv("CLAUDE_API_KEY", "env-only-key")

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	pc := cfg.Providers[ProviderClaude]
	if pc.APIKey != "env-only-key" {
		t.Errorf("APIKey = %q, want env-only-key", pc.APIKey)
	}
	if pc.BaseURL != "" {
		t.Errorf("expected empty BaseURL when no ini section exists, got %q", pc.BaseURL)
	}
}

func TestDefaultProviderListenPortsAreSequentialFrom9092(t *testing.T) {
	want := map[ProviderName]int{
		ProviderClaude:     9092,
		ProviderGemini:     9093,
		ProviderOpenAI:     9094,
		ProviderOpenRouter: 9095,
		ProviderClinepass:  9096,
	}
	for name, port := range want {
		if got := defaultProviderListenPort(name); got != port {
			t.Errorf("defaultProviderListenPort(%q) = %d, want %d", name, got, port)
		}
	}
}

func TestLoadConfigAssignsDefaultListenPortsWithNoIniOverride(t *testing.T) {
	// minimalProviderConfig has no listen_port key for openrouter, and no
	// section at all for the other three providers.
	path := writeTestConfig(t, minimalProviderConfig)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].ListenPort; got != 9095 {
		t.Errorf("openrouter ListenPort = %d, want default 9095", got)
	}
	if got := cfg.Providers[ProviderClaude].ListenPort; got != 9092 {
		t.Errorf("claude ListenPort = %d, want default 9092 even with no [provider.claude] section", got)
	}
}

func TestLoadConfigListenPortOverride(t *testing.T) {
	path := writeTestConfig(t, `[provider.openrouter]
base_url = https://openrouter.ai/api/v1
listen_port = 7777
`)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].ListenPort; got != 7777 {
		t.Errorf("ListenPort = %d, want explicit override 7777", got)
	}
}

// TestLoadConfigTokensSecMultiplierDefaultsToOne covers both "section
// exists but key omitted" (openrouter here) and "no section at all"
// (claude here) — both must default to the no-op 1.0, not a Go zero
// value 0.0, which would make the live tok/s estimate always read 0.
func TestLoadConfigTokensSecMultiplierDefaultsToOne(t *testing.T) {
	path := writeTestConfig(t, minimalProviderConfig)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].TokensSecMultiplier; got != 1.0 {
		t.Errorf("openrouter TokensSecMultiplier = %v, want default 1.0", got)
	}
	if got := cfg.Providers[ProviderClaude].TokensSecMultiplier; got != 1.0 {
		t.Errorf("claude TokensSecMultiplier = %v, want default 1.0 even with no [provider.claude] section", got)
	}
}

// TestLoadConfigRepetitionWindowWordsDefault matters for existing
// config.ini files written before this key existed: they have no
// repetition_window_words entry, and must silently pick up the clustering
// default (96) rather than 0, which would mean "no clustering" and keep
// the false-positive behavior the window exists to fix.
func TestLoadConfigRepetitionWindowWordsDefault(t *testing.T) {
	path := writeTestConfig(t, minimalProviderConfig) // no [detection] section at all

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Detection.RepetitionWindowWords; got != 96 {
		t.Errorf("RepetitionWindowWords = %d, want default 96", got)
	}
}

func TestLoadConfigRepetitionWindowWordsOverride(t *testing.T) {
	path := writeTestConfig(t, `[detection]
repetition_window_words = 0
`)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Detection.RepetitionWindowWords; got != 0 {
		t.Errorf("RepetitionWindowWords = %d, want explicit 0 (clustering disabled)", got)
	}
}

// TestLoadConfigRepetitionRequiresLengthFinishDefault: existing config.ini
// files predate this key, and must pick up the gate (true) by default —
// false would keep scanning normally-completed responses, the confirmed
// false-positive class the gate exists to remove.
func TestLoadConfigRepetitionRequiresLengthFinishDefault(t *testing.T) {
	path := writeTestConfig(t, minimalProviderConfig) // no [detection] section at all

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if !cfg.Detection.RepetitionRequiresLengthFinish {
		t.Error("RepetitionRequiresLengthFinish = false, want default true")
	}
}

func TestLoadConfigRepetitionRequiresLengthFinishOverride(t *testing.T) {
	path := writeTestConfig(t, `[detection]
repetition_requires_length_finish = false
`)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if cfg.Detection.RepetitionRequiresLengthFinish {
		t.Error("RepetitionRequiresLengthFinish = true, want explicit false override")
	}
}

func TestLoadConfigTokensSecMultiplierOverride(t *testing.T) {
	path := writeTestConfig(t, `[provider.openrouter]
base_url = https://openrouter.ai/api/v1
tokens_sec_multiplier = 2.5
`)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderOpenRouter].TokensSecMultiplier; got != 2.5 {
		t.Errorf("TokensSecMultiplier = %v, want explicit override 2.5", got)
	}
}

// TestLoadConfigClinepassProviderResolvesFromIniAndEnv covers the
// clinepass provider end-to-end: it's Cline's own hosted gateway
// (api.cline.bot), not a model vendor's API directly, so its base_url
// looks nothing like the other four — worth a dedicated check that it
// parses correctly and that its env var key follows the same
// {NAME}_API_KEY convention as every other provider (CLINEPASS_API_KEY,
// not something bespoke).
func TestLoadConfigClinepassProviderResolvesFromIniAndEnv(t *testing.T) {
	path := writeTestConfig(t, `[provider.clinepass]
base_url = https://api.cline.bot/api/v1
`)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	pc := cfg.Providers[ProviderClinepass]
	if pc.BaseURL != "https://api.cline.bot/api/v1" {
		t.Errorf("clinepass BaseURL = %q, want https://api.cline.bot/api/v1", pc.BaseURL)
	}
	if pc.ListenPort != 9096 {
		t.Errorf("clinepass ListenPort = %d, want default 9096", pc.ListenPort)
	}
}

func TestLoadConfigClinepassEnvVarOverridesIniAPIKey(t *testing.T) {
	path := writeTestConfig(t, `[provider.clinepass]
base_url = https://api.cline.bot/api/v1
api_key = ini-key-value
`)
	t.Setenv("CLINEPASS_API_KEY", "env-key-value")

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if got := cfg.Providers[ProviderClinepass].APIKey; got != "env-key-value" {
		t.Errorf("APIKey = %q, want env-key-value (CLINEPASS_API_KEY must win over config.ini)", got)
	}
}
