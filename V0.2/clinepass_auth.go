package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// This file ties the sampling-proxy's clinepass backend to the same credential
// the standalone clinepass-connector uses, so you log in once and both tools
// share it. Resolution order for the clinepass API key:
//
//  1. [provider.clinepass] api_key in config.ini (explicit wins)
//  2. CLINEPASS_API_KEY environment variable
//  3. the connector's stored key (~/.config/clinepass-connector/config.json)
//  4. interactive prompt (terminal only) — saved back to the connector's
//     config so it persists and both tools pick it up next time
//
// clinepass chat only accepts an API key (the account/browser token is
// rejected by /chat/completions — see the connector's notes), so this is
// intentionally API-key only.

// connectorConfigPath returns the standalone connector's config location,
// honoring XDG_CONFIG_HOME (matching the connector itself).
func connectorConfigPath() string {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "clinepass-connector", "config.json")
}

// readConnectorAPIKey returns the api_key stored in the connector's config, or
// "" if none / unreadable.
func readConnectorAPIKey() string {
	path := connectorConfigPath()
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var cfg struct {
		APIKey string `json:"api_key"`
	}
	if json.Unmarshal(data, &cfg) != nil {
		return ""
	}
	return strings.TrimSpace(cfg.APIKey)
}

// saveConnectorAPIKey writes key into the connector's config, preserving any
// other fields already there, so the connector and the proxy stay in sync.
func saveConnectorAPIKey(key string) error {
	path := connectorConfigPath()
	if path == "" {
		return fmt.Errorf("cannot locate config dir")
	}
	m := map[string]any{}
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &m) // ignore parse errors, start fresh
	}
	m["auth_mode"] = "apikey"
	m["api_key"] = key
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	return os.WriteFile(path, out, 0600)
}

// resolveClinepassAPIKey applies the non-interactive resolution order (steps
// 1-3 above) and reports where it came from, for a startup message.
func resolveClinepassAPIKey(configKey string) (key, source string) {
	if k := strings.TrimSpace(configKey); k != "" {
		return k, "config.ini"
	}
	if k := strings.TrimSpace(os.Getenv("CLINEPASS_API_KEY")); k != "" {
		return k, "CLINEPASS_API_KEY env"
	}
	if k := readConnectorAPIKey(); k != "" {
		return k, "clinepass-connector login"
	}
	return "", ""
}

// ensureClinepassKey resolves the clinepass key and, if none is found and we're
// on an interactive terminal, prompts for one (validating and saving it to the
// shared connector config). Returns "" if unavailable and not prompted (e.g.
// headless) — the caller decides whether that's fatal. base is clinepass's
// base_url, used to validate a freshly entered key.
func ensureClinepassKey(configKey, base string) string {
	if key, source := resolveClinepassAPIKey(configKey); key != "" {
		fmt.Printf("  clinepass: using API key from %s\n", source)
		return key
	}
	if !stdinIsTerminal() {
		fmt.Println("  clinepass: no API key found (config.ini / CLINEPASS_API_KEY / connector login) — clinepass requests will 401 until one is set")
		return ""
	}

	fmt.Println("\nclinepass needs an API key (the account/browser login is not accepted for chat).")
	fmt.Println("Create one at: app.cline.bot → Settings → API Keys")
	key := strings.TrimSpace(promptLine("Paste your clinepass API key (or leave blank to skip): "))
	if key == "" {
		fmt.Println("  clinepass: skipped — requests will 401 until a key is set")
		return ""
	}
	if err := validateClinepassKey(base, key); err != nil {
		fmt.Printf("  clinepass: warning — key could not be verified (%v); saving anyway\n", err)
	} else {
		fmt.Println("  clinepass: key verified ✓")
	}
	if err := saveConnectorAPIKey(key); err != nil {
		fmt.Printf("  clinepass: warning — could not save key for reuse: %v\n", err)
	} else {
		fmt.Println("  clinepass: key saved (shared with clinepass-connector)")
	}
	return key
}

// validateClinepassKey does a minimal 1-token chat probe: a 401/403 means the
// key is bad; anything else means the credential was accepted.
func validateClinepassKey(base, key string) error {
	body := strings.NewReader(`{"model":"cline-pass/deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":1}`)
	req, _ := http.NewRequest(http.MethodPost, base+"/chat/completions", body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	client := &http.Client{Timeout: 20 * time.Second, Transport: &http.Transport{Proxy: nil}}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("unauthorized (status %d)", resp.StatusCode)
	}
	return nil
}

// promptLine reads a single trimmed line from stdin.
func promptLine(label string) string {
	fmt.Print(label)
	sc := bufio.NewScanner(os.Stdin)
	if sc.Scan() {
		return sc.Text()
	}
	return ""
}
