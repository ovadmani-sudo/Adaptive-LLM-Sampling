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
# This is an IDLE (inactivity) timeout on the streaming chat path, not an
# absolute wall-clock ceiling: it only fires if no progress at all (no
# response headers, no SSE line) arrives from upstream for this long, and
# resets every time real progress is observed. A generation that's slow
# but steadily producing tokens can run indefinitely — only genuine
# silence this long gets cut off. The dashboard shows a live countdown to
# this value for the in-flight request. Also reused as a plain absolute
# ceiling for the two calls that have no per-byte progress signal to reset
# against: GET /v1/models and llama.cpp's native /completion endpoint.
request_timeout_seconds = 1800
# Set to false if llama-server is launched with --reasoning off (or the
# equivalent for your backend). thinking_budget_tokens is otherwise
# injected on every request per the classified bucket's preset (see
# [preset.*] below); sending it to a backend with reasoning disabled
# server-side risks silently re-enabling a reasoning/thinking pass that
# eats into the shared max_tokens budget before the model produces its
# visible answer — the response looks incomplete/truncated rather than
# erroring outright, since nothing actually failed.
inject_thinking_budget_tokens = true

# Some models occasionally end their turn early on a genuinely unfinished
# multi-step task (finish_reason "stop", not truncated) and need a nudge
# to continue — this only takes effect when the proxy is started with the
# --alert flag (a deliberate, easily-reversible kill switch: just omit the
# flag next run if this doesn't help). alert_models lists which model
# names this applies to (comma-separated, case-insensitive exact match
# against the request's "model" field) — leave empty to disable even with
# --alert passed, since this is opt-in per model observed to actually
# benefit, not a global behavior change. Use * or any to apply to every
# model instead of listing specific ones.
alert_models =
# Hard cap on how many times the proxy will ask "is this done, and if not
# please continue" for a single request, so a model that keeps confirming
# "yes" or keeps stopping short can't loop forever.
alert_max_rounds = 3
# The follow-up user turn sent when a configured model's turn ends
# naturally. Phrased to hedge two observed behaviors: a model that's
# genuinely done tends to answer this briefly and stop there (the proxy
# detects a short, confirmation-flavored reply and drops it rather than
# appending it to the response — see looksLikeConfirmation in alert.go);
# a model that stopped prematurely on an unfinished task typically just
# resumes the real work directly without literally answering at all,
# which the proxy appends and, if it also ends in "stop", asks again (up
# to alert_max_rounds).
alert_probe_message = Is the request fully completed? If not, please continue exactly where you left off.

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
# CLAUDE_API_KEY, GEMINI_API_KEY, CLINEPASS_API_KEY), it always wins over
# whatever is written below, so a real key never has to sit in plaintext in
# this file. Prefer the env var and leave api_key blank here.
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

# Cline's own hosted gateway — a Cline account credential (CLINEPASS_API_KEY),
# not a vendor API key, multiplexed across whichever models Cline's gateway
# backs. Only useful if you specifically want to route through Cline's
# account/billing rather than a model vendor directly.
#
# allow_passthrough is true here (and only here, of every provider) because
# clinepass is Cline's own backend, not a plain model vendor: Cline's
# extension also calls auxiliary endpoints against this same base_url
# (token refresh, recommended-models, remote-config, ...) beyond chat
# completions. Without this, those calls hit this proxy's chat/models-only
# allowlist and get rejected with a confusing 501 ("Token refresh failed:
# 501" is this exact rejection), breaking Cline's own account/session
# handling even though model responses work fine.
[provider.clinepass]
listen_port = 9096
base_url = https://api.cline.bot/api/v1
api_key =
model =
tokens_sec_multiplier = 1.0
allow_passthrough = true

[classification]
# comma-separated keywords, case-insensitive substring match against
# the latest user-role message content
#
# Keep entries to single words, not phrases: matching is plain substring
# containment with no word-boundary or word-order flexibility, so a
# phrase like "write function" only matches that exact contiguous text —
# "write the function" or "write this" (completely natural phrasing) will
# NOT match it at all. A single verb like "write" matches regardless of
# whatever comes after it. strict_code is checked first in priority_order,
# so a broader single-word match here is the safer failure mode: worst
# case it wins over a lower-priority bucket on an ambiguous message,
# rather than silently missing an actual code-change request and falling
# through to whatever bucket (e.g. architecture) matched instead —
# confirmed to happen in practice and produce visibly worse output (wrong
# temperature/thinking budget for the task actually being asked).
strict_code_keywords = fix, refactor, implement, write, edit
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

# agentic_loop is a FORCE-ONLY bucket: it has no classification keywords
# (and is not in priority_order), so it never auto-triggers on message
# content — activate it manually from the dashboard/web panel's
# classification-mode control when starting a long, iterative agent task
# (complex installs, many tool-call loops and failed attempts), and clear
# back to auto-detect when done. Keywords can't reliably identify this
# kind of task (the messages look like ordinary strict_code turns), which
# is why it's manual: strict_code-grade precision sampling, but with a
# much larger reasoning budget for working through repeated failures.
[preset.agentic_loop]
temperature = 0.2
top_p = 0.9
top_k = 40
repeat_penalty = 1.0
thinking_budget_tokens = 8192

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

# Alternative to [provider.*] above: instead of registering a base_url/model
# per vendor and pointing an agent's own base_url at this proxy's dedicated
# port, point the agent's HTTP_PROXY/HTTPS_PROXY at listen_port directly —
# the agent keeps its own real base_url/API key/model unchanged for
# whatever vendor it already talks to, and this proxy transparently
# intercepts the traffic.
#
# This only works for HTTPS destinations by terminating TLS (a plain
# forward proxy otherwise only sees an opaque CONNECT tunnel, useless for
# classify/inject/detect/retry, which all need the plaintext JSON body).
# ca_cert_path/ca_key_path must point at an EXISTING intermediate CA
# cert/key (e.g. one already set up for mitmproxy/Squid-style SSL bumping)
# — this proxy never generates its own CA. The CA's root certificate must
# already be trusted by whatever agent tool you're routing through this
# proxy (OS trust store, NODE_EXTRA_CA_CERTS, SSL_CERT_FILE, or similar,
# depending on the tool); some tools pin certificates and cannot be made to
# work this way regardless of trust configuration.
#
# allowed_hosts is a hard allowlist, not a suggestion: any CONNECT to a
# host NOT listed here gets a plain, unmodified byte-for-byte tunnel with
# zero TLS termination or inspection — only explicitly listed hostnames are
# ever decrypted. This bounds the security exposure of running this proxy:
# pointing other traffic (a browser, unrelated tools) through it stays
# safe, since nothing outside this list can ever be read by the proxy.
#
# passthrough_hosts is a subset of allowed_hosts (must already be
# TLS-terminated to matter) where any decrypted request path other than
# one ending in /chat/completions or /models is forwarded verbatim instead
# of being rejected with 501. Needed for api.cline.bot specifically: it's
# Cline's own account gateway, not a plain model vendor, so Cline's
# extension also calls auxiliary endpoints (token refresh,
# recommended-models, remote-config) against it through this same
# intercepted connection — rejecting those breaks Cline's own
# account/session handling even though model responses work fine. Leave
# every other host off this list: an unexpected path against a pure vendor
# API is more likely a real misconfiguration worth surfacing as an error.
[forward_proxy]
enabled = false
listen_port = 9100
ca_cert_path =
ca_key_path =
allowed_hosts = api.anthropic.com, generativelanguage.googleapis.com, api.openai.com, openrouter.ai, api.cline.bot
passthrough_hosts =

# Workaround for a real llama.cpp limitation: its JSON-schema-to-GBNF
# grammar converter turns string maxLength/minLength/pattern constraints
# into repetition rules (e.g. maxLength becomes a char{0,N} rule), and a
# large or deeply nested tool schema — many optional string fields,
# agentic clients like Cline registering many tools/MCP servers at once —
# can push the total generated grammar past llama.cpp's internal
# repetition-count sanity cap. When that happens, llama-server aborts the
# request outright with "failed to parse grammar" / "Failed to initialize
# samplers" before generation ever starts (see
# https://github.com/ggml-org/llama.cpp/issues/21228). Only relevant
# against a local llama.cpp backend using grammar-constrained tool calling;
# harmless no-op otherwise (nothing to strip if the request has no "tools"
# field).
#
# strip_max_length is the one to try first: it's the proven blowup vector
# (large maxLength values compound multiplicatively across many tool
# definitions). strip_min_length/strip_pattern are optional extras for
# schemas that still fail after stripping maxLength alone — enable them
# only if needed, since dropping a constraint means the model is no longer
# prevented from violating it (e.g. a string argument longer than the
# schema intended), a small risk traded for tool-calling working at all.
[tool_schema_sanitizer]
enabled = false
strip_max_length = true
strip_min_length = false
strip_pattern = false

# Some models (Gemma 4 26B confirmed via direct packet capture) can get
# stuck in a degenerate REASONING loop that never naturally ends — the same
# class of problem as [retry]'s repetition detection above, but happening
# DURING an still-in-progress generation instead of after one completes, so
# repetition_ngram_size/repetition_min_repeats above can never catch it
# (they only ever run on a finished response, which a model stuck looping
# forever in its reasoning phase may never produce). This guard watches
# reasoning_content as it streams and aborts+retries once it's clearly run
# away, applying the same dry_multiplier/repeat_penalty remedy as ordinary
# repetition, since it's the same underlying failure just caught earlier.
[reasoning_guard]
enabled = false
# Abort once accumulated reasoning content exceeds the request's own
# thinking_budget_tokens (see [preset.*] above) times this multiplier — a
# generous margin over the requested budget, not a tight one, since a model
# finishing its reasoning somewhat over budget is normal and not itself a
# bug.
budget_multiplier = 4.0
# Applies only when the request has no thinking_budget_tokens at all (falls
# back to this flat cap instead of no limit).
fallback_max_reasoning_tokens = 4096

# Describe-and-replace image pipeline: every image content part in an
# incoming chat request is described once by the VLM endpoint configured
# here and replaced inline with "[IMAGE DESCRIPTION: ...]" text, so the
# target model — whichever one the request was already going to, on ANY
# listener, local or remote — never sees an image at all. Descriptions
# are cached in-memory by image hash, so the same image resent on every
# later turn of a conversation (as all real clients do) is only ever
# described once. Also toggleable live from the --web control panel.
#
# Default VLM: Cline's gateway (clinepass) with a vision-capable model —
# api_key falls back to CLINEPASS_API_KEY / the clinepass connector's
# stored key when left empty here, same as [provider.clinepass]. Clear
# base_url to use the local llama-server from [server] instead (model
# must then be a locally served vision model, e.g. qwen3-vl).
[vision_describe]
enabled = false
# VLM model name sent on describe calls.
model = stepfun/step-3.7-flash
# The VLM's OpenAI-compatible mount point (up to but not including
# "/chat/completions"). Empty = the local llama-server from [server].
base_url = https://api.cline.bot/api/v1
api_key =

# Named, selectable system prompts — pick one live from the web panel's
# dropdown (global, applies to every listener). The selected prompt's
# text is PREPENDED to whatever system message the client itself sent
# (or becomes the system message if the client sent none) — it never
# replaces the client's own content, so an agent tool's own tool-definition
# system prompt survives underneath it. Loaded once at startup and never
# re-read per request: the same text is prepended the same way every
# time for a given selection, which is what lets llama-server's
# --cache-reuse prefix cache actually reuse it instead of missing on
# every request. Section name after "system_prompt." is the name shown
# in the dropdown; "text" is a triple-quoted multi-line value (INI
# syntax — leading/trailing blank lines are trimmed automatically). None
# selected by default (empty selection = no injection at all).
[system_prompt.research]
text = """
You are a deep research assistant. You can have natural multi-turn conversations with users on any topic.

## Daily Chat & Simple Questions
For everyday conversations, greetings, opinions, coding help, factual lookups, definitions, calculations, explanations, and any question you can confidently answer from your knowledge — just respond directly and naturally in the user's language as Intern-A1. Do NOT use any tools for these.

## Research & Search Questions
Only when the user's question requires up-to-date information, in-depth investigation, multi-source verification, or involves recent events, niche topics, or anything you are uncertain about, use the available tool **web_search_exa**.

Research strategy:
- Start with a focused search query to get an overview.
- If the initial search is insufficient, refine your query with more specific terms.
- Stop searching once you have enough information to provide a comprehensive answer. Do not over-research.
"""

[system_prompt.code]
text = """
## CODE QUALITY (all languages)
1. Before code that manipulates indices/pointers/offsets/cursors, state
   the convention in a comment (0- or 1-based, inclusive/exclusive,
   ownership if relevant) and keep it consistent.
2. For every loop, verify: first iteration, last iteration, empty input,
   single element.
3. When loop direction/order matters, state WHY it's correct before
   writing it.
4. After each function, trace one concrete 2–3 element example through it.
5. Prefer the standard textbook formulation over a clever variant;
   deviate only when required, and say so.
6. Every computed indexed/dereferenced access is preceded by reasoning
   for why it's valid (in bounds, non-null, initialized).
7. Match every acquisition with its release, every open with its close;
   in languages with manual memory or explicit errors, check every
   fallible call.
8. Use the language's idioms; name its task-relevant pitfalls before
   writing (overflow, float comparison, half-open ranges, aliasing,
   null vs undefined, truthiness).
9. Include self-checking tests native to the language: assertions with
   messages, a test main, or unit-test blocks.
10. Re-read the task's explicit constraints (forbidden libs, required
    APIs, output format) immediately before finalizing; confirm each.
11. When an operation in a sequence fails, state what happens to the
    cursor/iterator: stop, skip, or retry? A skipped failure must never
    advance a committed position past it.
"""
`

// TaskBucket identifies a classification bucket.
type TaskBucket string

const (
	BucketStrictCode      TaskBucket = "strict_code"
	BucketExploratoryCode TaskBucket = "exploratory_code"
	BucketExplanation     TaskBucket = "explanation"
	BucketArchitecture    TaskBucket = "architecture"
	// BucketAgenticLoop is force-only: it has no classification keywords
	// and is absent from the default priority_order, so Classify can
	// never pick it on its own — it's activated exclusively through the
	// forced-bucket control (dashboard keybinding / web panel), for
	// long, iterative agent tasks (complex installs, many tool-call
	// loops and retries) that message-content keywords can't reliably
	// identify. Its preset keeps strict_code-grade precision sampling
	// but with a much larger thinking budget.
	BucketAgenticLoop TaskBucket = "agentic_loop"
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
	// InjectThinkingBudget gates whether presets inject thinking_budget_tokens
	// (see applyPreset). Set to false when the backend is launched with
	// reasoning disabled (e.g. llama-server's --reasoning off) — sending
	// this field anyway risks silently re-enabling a reasoning/thinking
	// pass that eats into the shared max_tokens budget before the visible
	// answer, which reads as incomplete/truncated output rather than an
	// error.
	InjectThinkingBudget bool

	// AlertEnabled gates alert-continuation entirely (see alert.go and
	// handleClassified): some models occasionally end their turn early on
	// a genuinely unfinished multi-step task (finish_reason "stop", not
	// truncated) and need a nudge to continue. Deliberately NOT read from
	// config.ini — this is set only by the --alert CLI flag (main.go), so
	// it's trivial to stop using (just omit the flag next run) without
	// having to remember to also flip a config value back.
	AlertEnabled bool
	// AlertModels lists which model names alert-continuation applies to
	// (case-insensitive exact match against the request's "model" field)
	// — empty means no models get it even with --alert passed, since this
	// is opt-in per model observed to actually benefit, not a global
	// behavior change.
	AlertModels []string
	// AlertMaxRounds caps how many times the proxy will ask "is this done,
	// and if not please continue" for a single request, so a model that
	// keeps confirming "yes" or keeps stopping short can't loop forever.
	AlertMaxRounds int
	// AlertProbeMessage is the follow-up user turn sent when a configured
	// model's turn ends naturally (finish_reason "stop") — see
	// defaultAlertProbeMessage (alert.go) for the default wording and the
	// reasoning behind it.
	AlertProbeMessage string
}

// ToolSchemaSanitizerConfig controls a workaround for a real llama.cpp
// limitation: its JSON-schema-to-GBNF grammar converter turns string
// maxLength/minLength/pattern constraints into repetition rules (e.g.
// char{0,N} for maxLength), and a large or deeply nested tool schema (many
// optional string fields, agentic clients like Cline registering many
// tools/MCP servers at once) can push the total generated grammar past
// llama.cpp's internal repetition-count sanity cap, aborting the request
// with "failed to parse grammar" before generation ever starts (see
// https://github.com/ggml-org/llama.cpp/issues/21228). Each constraint kind
// is independently toggleable since maxLength is the proven blowup vector;
// minLength/pattern are optional extras for schemas that still fail after
// stripping maxLength alone. Every field is applied recursively across a
// tool's whole parameters schema (properties/items/anyOf/oneOf/allOf/
// $defs/definitions/additionalProperties), not just its top level, since a
// constraint buried under $defs is just as capable of blowing up the
// generated grammar as one at the root.
type ToolSchemaSanitizerConfig struct {
	Enabled        bool
	StripMaxLength bool
	StripMinLength bool
	StripPattern   bool
}

// ProviderName identifies a remote backend selectable via CLI argument.
type ProviderName string

const (
	ProviderClaude     ProviderName = "claude"
	ProviderGemini     ProviderName = "gemini"
	ProviderOpenAI     ProviderName = "openai"
	ProviderOpenRouter ProviderName = "openrouter"
	// ProviderClinepass is Cline's own hosted OpenAI-compatible gateway
	// (api.cline.bot), distinct from hitting a model vendor's API directly
	// — it multiplexes several backing models under one Cline account
	// credential rather than a vendor-issued key.
	ProviderClinepass ProviderName = "clinepass"
)

// KnownProviders lists every provider name recognized as a CLI argument,
// in the order they're listed when reporting an invalid selection. New
// entries are appended, not inserted, so existing providers keep their
// already-documented default listen_port.
var KnownProviders = []ProviderName{ProviderClaude, ProviderGemini, ProviderOpenAI, ProviderOpenRouter, ProviderClinepass}

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
	// AllowPassthrough forwards any request path other than /chat/completions
	// and /models straight through to BaseURL, unmodified — the same
	// behavior local mode always has via p.passthrough. Deliberately false
	// by default: for a pure model-vendor API (claude/gemini/openai/
	// openrouter), Cline only ever sends chat-completions-shaped traffic
	// through a custom base_url override, so anything else hitting this
	// proxy is unexpected and safer to reject outright than silently
	// forward somewhere unverified. clinepass is the one provider this
	// should be true for: it's Cline's own account gateway, not a plain
	// vendor, so Cline's extension also calls auxiliary endpoints against
	// the same base_url (token refresh, recommended-models, remote-config,
	// ...) that a chat-only allowlist would otherwise reject with a
	// confusing 501, breaking Cline's own account/session handling.
	AllowPassthrough bool
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
	// TemperatureIncrementOnEmpty/TemperatureCeiling are the IssueEmpty
	// counterpart to TemperatureDecrementOnBadSyntax/TemperatureFloor: a
	// blank completion is more often an unlucky early-EOS draw than bad
	// syntax, so retries push temperature UP (toward more diverse
	// sampling, away from whatever collapsed to an immediate stop) instead
	// of down.
	TemperatureIncrementOnEmpty float64
	TemperatureCeiling          float64
	StepExponent                float64
}

type Config struct {
	Server              ServerConfig
	Providers           map[ProviderName]ProviderConfig
	Classification      ClassificationConfig
	Presets             map[TaskBucket]Preset
	Detection           DetectionConfig
	Retry               RetryConfig
	ForwardProxy        ForwardProxyConfig
	ToolSchemaSanitizer ToolSchemaSanitizerConfig
	ReasoningGuard      ReasoningGuardConfig
	VisionDescribe      VisionDescribeConfig
	// SystemPrompts holds every named [system_prompt.<name>] preset,
	// keyed by name — an operator-chosen set of full system-prompt texts
	// selectable live (see ProxyServer.SystemPromptOverride /
	// injectSystemPrompt in proxy.go), e.g. from the web panel's dropdown.
	// Loaded once here and never re-read per request, which matters for
	// prefix-cache stability: the exact same text is prepended to the
	// exact same client-supplied system message on every request using a
	// given name, so llama-server's context/prefix cache (--cache-reuse)
	// can actually reuse it instead of missing every time.
	SystemPrompts map[string]string
}

// VisionDescribeConfig configures the describe-and-replace image
// pipeline (vision_describe.go): every image content part in an incoming
// chat request is described once by the VLM endpoint configured here and
// replaced inline with "[IMAGE DESCRIPTION: ...]" text, so the target
// model (whichever one the request was already going to — this applies
// GLOBALLY, to every listener, local and remote alike) never sees an
// image at all.
type VisionDescribeConfig struct {
	Enabled bool
	// Model is the VLM model name sent on describe calls.
	Model string
	// BaseURL is the VLM's OpenAI-compatible mount point (everything up
	// to but not including "/chat/completions"). Empty = the local
	// llama-server from [server] (http://upstream_host:upstream_port/v1).
	BaseURL string
	// APIKey, if set, is sent as "Authorization: Bearer <key>" on
	// describe calls — needed for a cloud VLM, unused for a local one.
	APIKey string
}

// ReasoningGuardConfig configures a live, mid-stream guard against a model
// stuck in a runaway reasoning loop (confirmed via packet capture: Gemma 4
// 26B can repeat short reasoning fragments indefinitely without ever
// producing real content). Every other detection mechanism in this proxy
// (DetectionConfig, RetryConfig) only ever inspects a COMPLETED response —
// useless against a model that never stops reasoning in the first place.
// This one watches reasoning_content as it streams (postUpstreamChatStreaming,
// stream.go) and aborts the in-progress generation once it clearly runs
// past whatever budget was requested, feeding IssueReasoningLoop
// (detect.go) into the same retry machinery as every other issue.
type ReasoningGuardConfig struct {
	Enabled bool
	// BudgetMultiplier: abort once the number of reasoning chunks received
	// exceeds the request's own thinking_budget_tokens times this
	// multiplier — a generous margin, not a tight one, since finishing
	// somewhat over budget is normal and not itself a bug.
	BudgetMultiplier float64
	// FallbackMaxReasoningTokens applies only when the request has no
	// thinking_budget_tokens at all (inject_thinking_budget_tokens=false,
	// or the client omitted it outright) — an absolute cap instead of no
	// limit, since a model looping with no requested budget at all is
	// exactly the case this guard exists to catch.
	FallbackMaxReasoningTokens int
}

// ForwardProxyConfig configures the alternative to the [provider.*]
// reverse-proxy model: instead of registering a base_url/model per vendor
// and pointing an agent's own base_url at this proxy's dedicated port, the
// agent's HTTP_PROXY/HTTPS_PROXY points at ListenPort directly, keeping its
// own real base_url/API key/model unchanged for whatever vendor it already
// talks to. CONNECT requests to a host in AllowedHosts are TLS-terminated
// (using the pre-existing intermediate CA at CACertPath/CAKeyPath) so the
// same classify/inject/detect/retry pipeline can run on the decrypted
// body; any other host gets an opaque byte-for-byte tunnel with no
// inspection, so routing arbitrary traffic through this proxy never risks
// decrypting more than the explicitly configured AI-vendor hostnames.
//
// CACertPath/CAKeyPath are expected to already exist (e.g. an intermediate
// CA cert/key generated by mitmproxy/Squid-style tooling) — this proxy
// never generates its own CA, since a locally-trusted CA hierarchy is a
// deliberate, security-sensitive setup step the operator owns, not
// something to silently create.
type ForwardProxyConfig struct {
	Enabled      bool
	ListenPort   int
	CACertPath   string
	CAKeyPath    string
	AllowedHosts []string
	// PassthroughHosts is a subset of AllowedHosts (a host not in
	// AllowedHosts is never TLS-terminated at all, so passthrough is moot
	// for it) where any decrypted request path other than one ending in
	// /chat/completions or /models is forwarded to that host verbatim
	// instead of being rejected with 501. Needed for api.cline.bot
	// specifically: it's Cline's own account gateway, not a plain model
	// vendor, so Cline's extension also calls auxiliary endpoints (token
	// refresh, recommended-models, remote-config) against it — rejecting
	// those breaks Cline's own account/session handling even though model
	// responses work fine. Every other allowed host is a pure vendor API
	// where an unexpected path is more likely a real misconfiguration
	// worth surfacing as an error, so this is opt-in per host, not blanket.
	PassthroughHosts []string
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
		RequestTimeoutSeconds: srv.Key("request_timeout_seconds").MustInt(1800),
		InjectThinkingBudget:  srv.Key("inject_thinking_budget_tokens").MustBool(true),
		AlertModels:           splitKeywords(srv.Key("alert_models").String()),
		AlertMaxRounds:        srv.Key("alert_max_rounds").MustInt(3),
		AlertProbeMessage:     srv.Key("alert_probe_message").MustString(defaultAlertProbeMessage),
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
				AllowPassthrough:    sec.Key("allow_passthrough").MustBool(false),
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

	for _, bucket := range []TaskBucket{BucketStrictCode, BucketExploratoryCode, BucketExplanation, BucketArchitecture, BucketAgenticLoop} {
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
		TemperatureIncrementOnEmpty:     rty.Key("temperature_increment_on_empty").MustFloat64(0.2),
		TemperatureCeiling:              rty.Key("temperature_ceiling").MustFloat64(1.5),
		StepExponent:                    rty.Key("retry_step_exponent").MustFloat64(1.0),
	}

	fwd := f.Section("forward_proxy")
	cfg.ForwardProxy = ForwardProxyConfig{
		Enabled:          fwd.Key("enabled").MustBool(false),
		ListenPort:       fwd.Key("listen_port").MustInt(9100),
		CACertPath:       strings.TrimSpace(fwd.Key("ca_cert_path").String()),
		CAKeyPath:        strings.TrimSpace(fwd.Key("ca_key_path").String()),
		AllowedHosts:     splitKeywords(fwd.Key("allowed_hosts").String()),
		PassthroughHosts: splitKeywords(fwd.Key("passthrough_hosts").String()),
	}

	tss := f.Section("tool_schema_sanitizer")
	cfg.ToolSchemaSanitizer = ToolSchemaSanitizerConfig{
		Enabled:        tss.Key("enabled").MustBool(false),
		StripMaxLength: tss.Key("strip_max_length").MustBool(true),
		StripMinLength: tss.Key("strip_min_length").MustBool(false),
		StripPattern:   tss.Key("strip_pattern").MustBool(false),
	}

	rg := f.Section("reasoning_guard")
	cfg.ReasoningGuard = ReasoningGuardConfig{
		Enabled:                    rg.Key("enabled").MustBool(false),
		BudgetMultiplier:           rg.Key("budget_multiplier").MustFloat64(4.0),
		FallbackMaxReasoningTokens: rg.Key("fallback_max_reasoning_tokens").MustInt(4096),
	}

	vd := f.Section("vision_describe")
	cfg.VisionDescribe = VisionDescribeConfig{
		Enabled: vd.Key("enabled").MustBool(false),
		// Defaults mirror the template: Cline's gateway with a
		// vision-capable model. base_url deliberately has NO parse-time
		// default — an explicitly empty base_url means "local
		// llama-server" (see visionDescribeBaseURL) and must stay
		// distinguishable from "key absent".
		Model:   vd.Key("model").MustString("stepfun/step-3.7-flash"),
		BaseURL: strings.TrimSpace(vd.Key("base_url").String()),
		APIKey:  strings.TrimSpace(vd.Key("api_key").String()),
	}

	// [system_prompt.*]: unlike buckets (a fixed enum), a prompt name is
	// an arbitrary string the operator picks, so these sections are
	// discovered by prefix rather than enumerated.
	cfg.SystemPrompts = make(map[string]string)
	for _, sec := range f.Sections() {
		name := sec.Name()
		if !strings.HasPrefix(name, "system_prompt.") {
			continue
		}
		promptName := strings.TrimPrefix(name, "system_prompt.")
		cfg.SystemPrompts[promptName] = strings.TrimSpace(sec.Key("text").String())
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
