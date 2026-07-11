package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"gopkg.in/ini.v1"
)

const defaultConfigContent = `[server]
listen_port = 9099
upstream_host = 127.0.0.1
upstream_port = 9091
max_retries = 2
request_timeout_seconds = 600

# Remote backends, selected by passing the provider name as the first CLI
# argument, e.g. ./llama-dyn-proxy openrouter. With no argument the proxy
# talks to the local llama-server configured in [server] above, on
# [server]'s listen_port.
#
# Each provider has its own listen_port (9092+), separate from [server]'s.
# This means you can run multiple instances of this proxy side by side —
# one per backend — each on its own stable port, e.g. one Cline profile
# pointed at :9090 for local and another at :9092 for claude, without them
# fighting over the same port.
#
# base_url is the provider's OpenAI-compatible mount point, i.e. everything
# up to (but not including) "/chat/completions". api_key is sent as
# "Authorization: Bearer <api_key>". model is optional: if set, it
# overrides whatever model name the client requested, since a model name
# valid for your local llama-server almost certainly isn't valid on a
# remote provider; if left blank, the client's requested model is forwarded
# unchanged.
#
# api_key here is a fallback only: if the environment variable
# {PROVIDER}_API_KEY is set (e.g. OPENROUTER_API_KEY, OPENAI_API_KEY,
# CLAUDE_API_KEY, GEMINI_API_KEY), it always wins over whatever is written
# below, so a real key never has to sit in plaintext in this file. Prefer
# the env var and leave api_key blank here.
#
# These base URLs were verified against each provider's current docs, but
# are third-party services outside this proxy's control — double check
# them if a provider changes its API surface.
#
# Note: repeat_penalty/DRY/top_k/thinking_budget_tokens are llama.cpp
# sampler extensions. Remote providers generally ignore unrecognized
# fields rather than error, but the retry adjustments that set them won't
# have any real effect against a non-llama.cpp backend.
#
# tokens_sec_multiplier corrects the dashboard's live tok/s estimate for
# this provider. That estimate counts SSE delta events and assumes each
# delta is exactly one token, which is true for llama.cpp but not
# guaranteed for a remote provider that may batch multiple tokens per
# streamed delta — if so, the live number reads artificially low (e.g. a
# paid model appearing capped well below its real speed). There's no
# generic way to detect this automatically, so it's a manual correction:
# leave at 1.0 (no correction) unless you've compared the live estimate
# against the provider's own reported speed and found it consistently
# off by a stable factor.
[provider.claude]
listen_port = 9092
base_url = https://api.anthropic.com/v1
api_key =
model =
tokens_sec_multiplier = 1.0

[provider.gemini]
listen_port = 9093
base_url = https://generativelanguage.googleapis.com/v1beta/openai
api_key =
model =
tokens_sec_multiplier = 1.0

[provider.openai]
listen_port = 9094
base_url = https://api.openai.com/v1
api_key =
model =
tokens_sec_multiplier = 1.0

[provider.openrouter]
listen_port = 9095
base_url = https://openrouter.ai/api/v1
api_key =
model =
tokens_sec_multiplier = 1.0

[classification]
# comma-separated keywords, case-insensitive substring match against
# the latest user-role message content
strict_code_keywords = fix bug, refactor, implement, write function, edit file
exploratory_code_keywords = brainstorm, alternative approach, explore, sketch
explanation_keywords = explain, why, what does, how does
architecture_keywords = architecture, design, structure, plan
# match priority when multiple buckets match, highest first:
priority_order = strict_code, exploratory_code, explanation, architecture
# bucket used when nothing matches
default_bucket = strict_code

# Temperature is deliberately kept within a bounded, moderate range in every
# preset. Without Mirostat correcting for degeneration, pushing temperature
# high is riskier, so these presets stay conservative rather than chasing
# maximum variety.
# top_p / top_k are active (not ignored, since Mirostat is off) and are
# tuned per bucket to control variety directly.
# min_p is intentionally never set here, in any preset: it's left to
# llama-server's own default, which already scales its threshold relative to
# the top candidate's probability at each step.
# repeat_penalty starts at neutral (1.0) everywhere — see [retry] below for
# why it's a reactive fix, not a baseline setting.
[preset.strict_code]
temperature = 0.2
top_p = 0.9
top_k = 40
repeat_penalty = 1.0
thinking_budget_tokens = 512

[preset.exploratory_code]
temperature = 0.6
top_p = 0.95
top_k = 60
repeat_penalty = 1.0
thinking_budget_tokens = 2048

[preset.explanation]
temperature = 0.6
top_p = 0.95
top_k = 60
repeat_penalty = 1.0
thinking_budget_tokens = 1024

[preset.architecture]
temperature = 0.8
top_p = 0.97
top_k = 80
repeat_penalty = 1.0
thinking_budget_tokens = 4096

[detection]
repetition_ngram_size = 12
repetition_min_repeats = 3
# A repeated n-gram only counts as degeneration if all repetition_min_repeats
# occurrences fall within this many words of each other. Real degenerate
# loops emit the same n-gram clustered tightly (the model is stuck cycling);
# legitimate code repetition scatters — the same console.log/import/tag can
# appear several times across a whole file without being a loop. Found via
# real retry_log data: a normal JS debug-print line appearing 3x in a script
# tripped the detector, and the "fix" (DRY) then actively penalized the model
# for writing correct repeated code. 0 disables clustering (any spacing counts,
# the old behavior).
repetition_window_words = 96
# Only treat repetition as an issue when finish_reason is "length" — a real
# degenerate loop doesn't stop on its own, it cycles until it eats the token
# cap. A response that ended naturally and merely contains repeats is
# overwhelmingly legitimate repetitive code (confirmed twice in real retry
# logs: a debug console.log written 3x, and code-generator output
# concatenating NL/indent constants on every line — both completed normally,
# both got flagged, both wasted a full extra generation on a retry that then
# penalized correct code via DRY). Set to false to scan every response
# regardless of how it finished.
repetition_requires_length_finish = true
max_tokens_retry_multiplier = 1.5
max_tokens_ceiling = 8192

[retry]
# Reactive-only adjustments, applied on top of the preset in effect for the
# request. repeat_penalty and DRY are NEVER set in the base presets above —
# they only appear once a retry is triggered by detected repetition, so
# legitimate code repetition (variable names, imports, repeated syntax) is
# never penalized on a clean first pass.
#
# Prefer the DRY sampler over repeat_penalty if this llama-server build
# supports it (check --help / /props): DRY only fires on sufficiently long
# verbatim repeated sequences, so it targets genuine degenerate loops
# without touching ordinary token-frequency repetition the way
# repeat_penalty does.
prefer_dry_over_repeat_penalty = true
dry_multiplier_on_retry = 0.8
dry_base = 1.75
dry_allowed_length = 2
repeat_penalty_increment = 0.15
# repeat_last_n is left at llama-server's default (64) intentionally — a
# narrow window limits any penalty's reach to recent tokens, so it won't
# punish a variable name reused later in a long function.
temperature_floor = 0.1
temperature_decrement_on_bad_syntax = 0.15

# Each successive retry within the same request scales its adjustment
# magnitude by retry_step_exponent^attempt (attempt 0 = first retry, so the
# first retry is always the plain base increment/decrement above — this
# only affects the 2nd, 3rd, ... retry within one request, so it only
# matters if max_retries > 1). 1.0 = no escalation (every retry uses the
# same flat step, the original behavior). >1.0 = later retries take bigger
# steps than earlier ones, on the theory that if a small nudge didn't fix
# it, a bigger one is more likely to than repeating the same small nudge.
#
# There's no principled default here yet — retry_log.jsonl (written next
# to this file whenever a request retries at least once) records every
# adjustment made and whether the very next attempt came back clean, so
# you can compare outcomes across exponent values empirically rather than
# guessing. Start at 1.0, gather some data, then try raising it.
retry_step_exponent = 1.0
`

// TaskBucket identifies a classification bucket.
type TaskBucket string

const (
	BucketStrictCode      TaskBucket = "strict_code"
	BucketExploratoryCode TaskBucket = "exploratory_code"
	BucketExplanation     TaskBucket = "explanation"
	BucketArchitecture    TaskBucket = "architecture"
)

// Preset holds the sampling/reasoning parameters injected for a bucket on
// the initial request. repeat_penalty, DRY, and min_p are deliberately
// absent here — they're either left to llama-server's own default (min_p)
// or reserved for the retry path only (repeat_penalty / DRY fields), never
// injected on a clean first pass. Pointer fields distinguish "unset in
// config" from "set to zero value", since the client-wins merge logic only
// fills fields present here.
type Preset struct {
	Temperature          *float64
	TopP                 *float64
	TopK                 *int
	ThinkingBudgetTokens *int
}

type ServerConfig struct {
	ListenPort            int
	UpstreamHost          string
	UpstreamPort          int
	MaxRetries            int
	RequestTimeoutSeconds int
}

// ProviderName identifies a remote backend selectable via CLI argument.
type ProviderName string

const (
	ProviderClaude     ProviderName = "claude"
	ProviderGemini     ProviderName = "gemini"
	ProviderOpenAI     ProviderName = "openai"
	ProviderOpenRouter ProviderName = "openrouter"
)

// KnownProviders lists every provider name recognized as a CLI argument,
// in the order they're listed when reporting an invalid selection.
var KnownProviders = []ProviderName{ProviderClaude, ProviderGemini, ProviderOpenAI, ProviderOpenRouter}

// ProviderConfig holds a remote OpenAI-compatible backend's connection
// details. Model is optional: empty means "forward whatever the client
// requested unchanged". ListenPort is separate from [server]'s, so a
// dedicated proxy instance per provider can run concurrently without port
// conflicts.
type ProviderConfig struct {
	BaseURL    string
	APIKey     string
	Model      string
	ListenPort int
	// TokensSecMultiplier corrects the dashboard's live in-flight tok/s
	// estimate, which is derived from counting SSE delta events under the
	// assumption that each delta carries exactly one token — true for
	// llama.cpp (verified against its own source), but not guaranteed for
	// every remote provider. If a provider batches multiple tokens into
	// each streamed delta, the raw chunk count undercounts, and the live
	// estimate reads artificially low (e.g. a paid model appearing capped
	// well below its real throughput). Rather than guess at each
	// provider's actual batching behavior, this is a manual per-provider
	// correction factor: 1.0 (default) means "trust the chunk count as
	// is"; e.g. 2.0 means "each delta is actually ~2 tokens, double the
	// estimate". Defaults to 1.0 (no correction) when unset.
	TokensSecMultiplier float64
}

// defaultProviderListenPort returns the fallback listen_port for a
// provider when config.ini doesn't set one explicitly: 9092 for the first
// entry in KnownProviders, incrementing from there.
func defaultProviderListenPort(name ProviderName) int {
	for i, n := range KnownProviders {
		if n == name {
			return 9092 + i
		}
	}
	return 9092
}

// providerAPIKeyEnvVar returns the environment variable checked for a
// given provider's API key, e.g. "openrouter" -> "OPENROUTER_API_KEY".
// Note this is the proxy's own naming convention, not necessarily each
// provider's SDK-standard variable name (e.g. Anthropic's own tooling
// looks for ANTHROPIC_API_KEY, not CLAUDE_API_KEY).
func providerAPIKeyEnvVar(name ProviderName) string {
	return strings.ToUpper(string(name)) + "_API_KEY"
}

type ClassificationConfig struct {
	Keywords      map[TaskBucket][]string
	PriorityOrder []TaskBucket
	DefaultBucket TaskBucket
}

type DetectionConfig struct {
	RepetitionNgramSize  int
	RepetitionMinRepeats int
	// RepetitionWindowWords requires all RepetitionMinRepeats occurrences
	// of a repeated n-gram to fall within this many words of each other to
	// count as degeneration — real loops cluster their repeats tightly,
	// while legitimate code repetition (the same console.log/import/tag
	// several times across a file) scatters. 0 disables the clustering
	// requirement (any spacing counts).
	RepetitionWindowWords int
	// RepetitionRequiresLengthFinish gates repetition detection on
	// finish_reason == "length": a genuine loop runs until the token cap,
	// while a normally-completed response containing repeats is almost
	// always legitimate repetitive code. false scans every response.
	RepetitionRequiresLengthFinish bool
	MaxTokensRetryMultiplier       float64
	MaxTokensCeiling               int
}

// RetryConfig holds the reactive-only adjustments applied on top of the
// active preset once a retry is triggered by a detected issue. None of
// these values are ever sent on a clean first-pass request.
type RetryConfig struct {
	PreferDryOverRepeatPenalty      bool
	DryMultiplierOnRetry            float64
	DryBase                         float64
	DryAllowedLength                int
	RepeatPenaltyIncrement          float64
	TemperatureFloor                float64
	TemperatureDecrementOnBadSyntax float64
	StepExponent                    float64
}

type Config struct {
	Server         ServerConfig
	Providers      map[ProviderName]ProviderConfig
	Classification ClassificationConfig
	Presets        map[TaskBucket]Preset
	Detection      DetectionConfig
	Retry          RetryConfig
}

// LoadConfig loads config.ini next to the binary, creating it with
// defaults if it does not already exist.
func LoadConfig(path string) (*Config, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if werr := os.WriteFile(path, []byte(defaultConfigContent), 0644); werr != nil {
			return nil, fmt.Errorf("writing default config: %w", werr)
		}
		log.Printf("config file %s not found; created with defaults", path)
	}

	f, err := ini.Load(path)
	if err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	cfg := &Config{
		Presets:   make(map[TaskBucket]Preset),
		Providers: make(map[ProviderName]ProviderConfig),
	}

	srv := f.Section("server")
	cfg.Server = ServerConfig{
		ListenPort:            srv.Key("listen_port").MustInt(9090),
		UpstreamHost:          srv.Key("upstream_host").MustString("127.0.0.1"),
		UpstreamPort:          srv.Key("upstream_port").MustInt(9091),
		MaxRetries:            srv.Key("max_retries").MustInt(2),
		RequestTimeoutSeconds: srv.Key("request_timeout_seconds").MustInt(120),
	}

	for _, name := range KnownProviders {
		pc := ProviderConfig{ListenPort: defaultProviderListenPort(name), TokensSecMultiplier: 1.0}
		if sec, err := f.GetSection("provider." + string(name)); err == nil {
			pc = ProviderConfig{
				BaseURL:             strings.TrimRight(strings.TrimSpace(sec.Key("base_url").String()), "/"),
				APIKey:              strings.TrimSpace(sec.Key("api_key").String()),
				Model:               strings.TrimSpace(sec.Key("model").String()),
				ListenPort:          sec.Key("listen_port").MustInt(defaultProviderListenPort(name)),
				TokensSecMultiplier: sec.Key("tokens_sec_multiplier").MustFloat64(1.0),
			}
		}

		// {PROVIDER}_API_KEY in the environment always wins over api_key in
		// config.ini, so a real key never has to sit in plaintext on disk.
		if envKey := os.Getenv(providerAPIKeyEnvVar(name)); envKey != "" {
			pc.APIKey = envKey
		}

		cfg.Providers[name] = pc
	}

	cls := f.Section("classification")
	cfg.Classification = ClassificationConfig{
		Keywords: map[TaskBucket][]string{
			BucketStrictCode:      splitKeywords(cls.Key("strict_code_keywords").String()),
			BucketExploratoryCode: splitKeywords(cls.Key("exploratory_code_keywords").String()),
			BucketExplanation:     splitKeywords(cls.Key("explanation_keywords").String()),
			BucketArchitecture:    splitKeywords(cls.Key("architecture_keywords").String()),
		},
		PriorityOrder: parsePriorityOrder(cls.Key("priority_order").String()),
		DefaultBucket: TaskBucket(strings.TrimSpace(cls.Key("default_bucket").MustString(string(BucketStrictCode)))),
	}

	for _, bucket := range []TaskBucket{BucketStrictCode, BucketExploratoryCode, BucketExplanation, BucketArchitecture} {
		sec, err := f.GetSection("preset." + string(bucket))
		if err != nil {
			continue
		}
		cfg.Presets[bucket] = parsePreset(sec)
	}

	det := f.Section("detection")
	cfg.Detection = DetectionConfig{
		RepetitionNgramSize:            det.Key("repetition_ngram_size").MustInt(12),
		RepetitionMinRepeats:           det.Key("repetition_min_repeats").MustInt(3),
		RepetitionWindowWords:          det.Key("repetition_window_words").MustInt(96),
		RepetitionRequiresLengthFinish: det.Key("repetition_requires_length_finish").MustBool(true),
		MaxTokensRetryMultiplier:       det.Key("max_tokens_retry_multiplier").MustFloat64(1.5),
		MaxTokensCeiling:               det.Key("max_tokens_ceiling").MustInt(8192),
	}

	rty := f.Section("retry")
	cfg.Retry = RetryConfig{
		PreferDryOverRepeatPenalty:      rty.Key("prefer_dry_over_repeat_penalty").MustBool(true),
		DryMultiplierOnRetry:            rty.Key("dry_multiplier_on_retry").MustFloat64(0.8),
		DryBase:                         rty.Key("dry_base").MustFloat64(1.75),
		DryAllowedLength:                rty.Key("dry_allowed_length").MustInt(2),
		RepeatPenaltyIncrement:          rty.Key("repeat_penalty_increment").MustFloat64(0.15),
		TemperatureFloor:                rty.Key("temperature_floor").MustFloat64(0.1),
		TemperatureDecrementOnBadSyntax: rty.Key("temperature_decrement_on_bad_syntax").MustFloat64(0.15),
		StepExponent:                    rty.Key("retry_step_exponent").MustFloat64(1.0),
	}

	return cfg, nil
}

// parsePreset only reads the fields that are actually injected on the
// initial request: temperature, top_p, top_k, thinking_budget_tokens.
// repeat_penalty may appear in config.ini for documentation purposes (to
// show the neutral baseline), but is intentionally not read here — see
// [retry] for where it actually gets set.
func parsePreset(sec *ini.Section) Preset {
	var p Preset
	if sec.HasKey("temperature") {
		v := sec.Key("temperature").MustFloat64(0)
		p.Temperature = &v
	}
	if sec.HasKey("top_p") {
		v := sec.Key("top_p").MustFloat64(0)
		p.TopP = &v
	}
	if sec.HasKey("top_k") {
		v := sec.Key("top_k").MustInt(0)
		p.TopK = &v
	}
	if sec.HasKey("thinking_budget_tokens") {
		v := sec.Key("thinking_budget_tokens").MustInt(0)
		p.ThinkingBudgetTokens = &v
	}
	return p
}

func splitKeywords(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.ToLower(strings.TrimSpace(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parsePriorityOrder(raw string) []TaskBucket {
	parts := strings.Split(raw, ",")
	out := make([]TaskBucket, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, TaskBucket(p))
		}
	}
	return out
}
