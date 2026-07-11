# llama-dyn-proxy

A lightweight reverse proxy that sits between an OpenAI-compatible client
(e.g. Cline's "OpenAI Compatible" provider) and a local `llama-server`
instance. It classifies each chat completion request by task type,
injects sampling/reasoning parameters accordingly, detects bad outputs
(repetition, truncation, invalid JSON in fenced code blocks), retries with
adjusted parameters, and shows a live terminal dashboard.

**No Mirostat.** Mirostat was evaluated and rejected for this use case — it
requires a full-vocabulary sort/entropy calculation on every token, which is
CPU-bound and roughly halved generation speed in testing. All dynamic
behavior comes from bounded temperature + top-p/top-k + reasoning-budget
control instead, at full decode speed.

## Build

```sh
go build -o llama-dyn-proxy .
```

Produces a single static binary.

## Run

```sh
# start llama-server first, e.g.:
llama-server --port 9091 ...

# then start the proxy
./llama-dyn-proxy
```

On first run it creates `config.ini` next to the binary with sane defaults
and logs that it did so. Edit `config.ini` and restart to change
classification keywords, presets, or detection thresholds — no code
changes required.

Point Cline's "OpenAI Compatible" provider at `http://127.0.0.1:9090/v1`
(or whatever `listen_port` you configured).

`Ctrl+C` shuts down the HTTP server cleanly and exits the dashboard.

The dashboard runs in the terminal's alt-screen (like `vim`/`htop`) — your
shell's prior content is restored when it exits, and it redraws within a
proper full-screen viewport instead of relying on terminal scrollback.

**Diagnostics go to `proxy_<backend>.log`, not the terminal.** Once the
dashboard starts, it owns the entire display — any `log.Printf` output
from that point on (streaming/usage warnings, passthrough errors, HTTP
server errors, ...) would otherwise be silently overwritten by the next
redraw, never actually visible even briefly. The proxy prints this log
file's path once at startup, before the dashboard takes over:
```
llama-dyn-proxy
  listening on http://127.0.0.1:9090
  backend      local (http://127.0.0.1:9091)
  log          /path/to/proxy_local.log
```
Startup failures (bad config, unknown provider, can't bind the port) still
print directly to the terminal, since those happen before the dashboard
exists to steal the display.

### Switching backends

By default the proxy talks to the local `llama-server` configured in
`[server]`. Pass a provider name as the first CLI argument to redirect the
same Cline session to a remote OpenAI-compatible backend instead, without
touching Cline's own settings:

```sh
./llama-dyn-proxy openrouter
```

Valid names are `claude`, `gemini`, `openai`, `openrouter` — each configured
under its own `[provider.<name>]` section (`listen_port`, `base_url`,
`api_key`, optional `model`) right below `[server]` in `config.ini`. An
unrecognized name fails fast with a clear error listing the valid options,
rather than silently falling back to local.

**Each provider has its own `listen_port`** (9092, 9093, 9094, 9095 by
default), separate from `[server]`'s. That means you can run several
instances of this proxy side by side — one per backend — each on a stable,
predictable port, and point different Cline profiles at whichever one you
need without them fighting over the same port:

```sh
./llama-dyn-proxy &              # local llama-server, [server].listen_port
./llama-dyn-proxy openrouter &   # :9095
./llama-dyn-proxy openai &       # :9094
```

**API keys**: `{PROVIDER}_API_KEY` in the environment (e.g.
`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_API_KEY`, `GEMINI_API_KEY`)
always wins over `api_key` in `config.ini`, so a real key never has to sit
in plaintext on disk:

```sh
OPENROUTER_API_KEY=sk-or-v1-... ./llama-dyn-proxy openrouter
```

Note these are the proxy's own env var names, not necessarily each
provider's SDK-standard one (Anthropic's own tooling looks for
`ANTHROPIC_API_KEY`, not `CLAUDE_API_KEY`). If the resolved key is empty
from both sources, the proxy logs a warning and proceeds anyway (requests
will likely be rejected by the provider).

**CORS**: every response carries permissive `Access-Control-Allow-Origin: *`
headers, and `OPTIONS` preflight requests get a clean `204`. Cline runs in
the VS Code extension host and is never subject to CORS, but browser-based
clients (e.g. Page Assist) are — without this, the browser silently
discards an otherwise-successful response before the client's own JS ever
sees it, which looks like an empty model list even though the proxy did
everything right. This is a local, single-user dev proxy, so a permissive
`*` origin carries no real multi-tenant risk.

A bare `GET /v1` (some clients, e.g. Page Assist, ping the base URL itself
as a reachability check before calling `/v1/models`) gets a `200` rather
than the generic 501 below — there's no real provider endpoint to proxy it
to, so the proxy just answers it directly.

In remote-provider mode, **only `/v1/chat/completions` and `/v1/models`
are proxied** — `/v1/models` (Cline calls this on startup to populate its
model picker) forwards to `<base_url>/models` with auth; anything else,
including `/completion` (llama.cpp's native endpoint) and generic
passthrough (`/health`, `/props`, ...), returns a clear 501 instead of
being silently mis-routed, since each provider mounts its
OpenAI-compatible surface at a different URL prefix and passthrough can't
be mapped reliably across all of them from one code path. Rejected paths
are logged to stdout so a failure like "Cline can't connect" is
diagnosable even though it never reaches the dashboard's request log
(that panel is reserved for the classify/inject/detect/retry cycle on the
chat-completions route). Classification, preset injection, and
retry/detection all still run normally on chat completions — but
`repeat_penalty`/DRY/`top_k`/`thinking_budget_tokens` are llama.cpp sampler
extensions; real providers generally ignore fields they don't recognize
rather than error, so retries still classify and re-send correctly, but
those specific adjustments won't have any real effect against a
non-llama.cpp backend.

## How it works

1. Only `/v1/chat/completions` and `/completion` get special treatment;
   every other path (`/health`, `/props`, ...) is passed straight through
   to the upstream `llama-server`.
2. The latest user-role message is matched (case-insensitive substring)
   against the keyword lists in `[classification]` to pick a task bucket
   (`strict_code`, `exploratory_code`, `explanation`, `architecture`). If
   several buckets match, `priority_order` breaks the tie; if none match,
   `default_bucket` is used.
3. The matching `[preset.<bucket>]` values (`temperature`, `top_p`, `top_k`,
   `thinking_budget_tokens`) are merged into the outgoing request body, but
   only for fields the client did not already set explicitly — an explicit
   client value always wins. `repeat_penalty`, DRY, and `min_p` are never
   injected on this initial pass — `min_p` is left to `llama-server`'s own
   adaptive default, and `repeat_penalty`/DRY are reserved for the retry
   path only, so a clean first-pass request never carries them.
4. For `/v1/chat/completions`, the request is always sent upstream with
   `"stream": true` (plus `stream_options: {include_usage: true}`),
   **regardless of what the client itself asked for** — the inverse of
   this proxy's original "always force stream:false" behavior. This is
   what makes the dashboard's live activity indicator possible (see
   "Live activity indicator" below): the proxy observes generation as it
   happens and accumulates the full text internally, so detection/retry
   below works exactly as before. If the original client didn't ask for
   streaming, a complete `chat.completion` JSON object is synthesized from
   the accumulated deltas once the stream ends; if it did, the accumulated
   text is re-chunked back into SSE — including `reasoning_content` (the
   model's thinking/plan trace, kept distinct from the final answer per
   the DeepSeek-style convention `llama-server` and Cline both use) as its
   own delta chunk, so clients that render reasoning separately from the
   answer (e.g. Cline's Plan Mode) still see it. Re-chunking splits by
   **rune count**, not by words: an earlier version split on
   `strings.Fields` (any run of whitespace) and rejoined pieces with a
   single ASCII space, which silently destroyed every newline,
   indentation level, and multi-space run in the response — reported live
   as "any formatting is lost, including code formatting," since it only
   affected clients that request streaming (Cline does, by default). A
   non-streaming request to the same backend looked completely normal,
   which is what made it easy to miss — the corruption only happened in
   this re-chunking path, never in the underlying content. `chunkText`
   now slices the original string directly (by rune, so a multi-byte
   UTF-8 character never gets split across two chunks), so concatenating
   every delta reproduces the response exactly. `/completion`
   (llama.cpp's native endpoint, local-only) is unaffected by any of this
   — it keeps the original single blocking, non-streaming call, since it's
   a low-traffic legacy path and local `llama-server` already gives
   console-level progress visibility (GPU utilization, `tg = N t/s`, ...)
   that a remote provider never does.
5. The response is checked for repetition (sliding n-gram window),
   truncation (`finish_reason == "length"` plus unbalanced
   braces/brackets/parens or an unterminated fence/string), and — best
   effort — invalid JSON inside fenced ` ```json ` blocks.
6. On an issue, if retries remain, the proxy adjusts the relevant sampling
   parameter and resends the **identical** messages/prompt (only sampling
   params change, so `llama-server`'s per-slot KV cache is reused). See
   `[retry]` in `config.ini`:
   - repetition → if `prefer_dry_over_repeat_penalty` (default): sets
     `dry_multiplier`/`dry_base`/`dry_allowed_length` from config, since DRY
     only fires on genuine verbatim loops and leaves ordinary code
     repetition (variable names, imports) untouched. Otherwise falls back
     to `repeat_penalty += repeat_penalty_increment`.
   - truncated → `max_tokens *= max_tokens_retry_multiplier`, capped at
     `max_tokens_ceiling` — unaffected by `retry_step_exponent` below,
     since it already compounds naturally attempt over attempt (each retry
     multiplies the *previous* attempt's value, not a fixed base). The
     starting point for that multiplication is the real number of tokens
     the model actually generated (`usage.completion_tokens` for the OAI
     chat endpoint, `tokens_predicted` for llama.cpp's native
     `/completion`), not a guess — no preset ever injects `max_tokens`, so
     there's usually nothing already in the request body to read on the
     first truncation. Guessing a fixed fallback here previously meant the
     "escalated" cap could already be smaller than what the backend's own
     default limit had already produced, guaranteeing the retry got
     truncated again too — that fallback (512) is now only used if the
     response genuinely has no usage info to read.
   - bad_syntax → `temperature = max(temperature_floor, temperature -
     temperature_decrement_on_bad_syntax)` — the floor exists specifically
     so repeated bad-syntax retries can never land on exactly 0 (greedy/
     degenerate decoding).

   Within one request, each successive retry scales its repeat_penalty
   increment, DRY multiplier, or temperature decrement by
   `retry_step_exponent^attempt` (attempt 0 = first retry). The default,
   `1.0`, means every retry takes the same flat step — the original
   behavior. Raising it makes later retries within the same request take
   bigger steps than earlier ones, so a request that needs several retries
   escalates instead of repeating the same small nudge. See "Tuning
   `retry_step_exponent`" below for how to find a good value instead of
   guessing.
7. Every request/response cycle pushes an event to the dashboard over a
   channel — proxy handling never blocks on UI redraw.
8. When a remote provider is active, a configured `model` in its
   `[provider.<name>]` section always overrides whatever model name the
   client requested (a local model name is almost certainly invalid on a
   remote API); leaving `model` blank passes the client's request through
   unchanged. The outbound request always targets exactly
   `<base_url>/chat/completions` — never the client's own incoming path —
   since most providers' `base_url` already ends in `/v1` (or, for Gemini,
   `/v1beta/openai`), and naively appending the client's `/v1/...` path
   would double that prefix.

### Tuning `retry_step_exponent`

There's no principled default for this yet — it needs real data from your
model/workload, not a guess. Every request that retries at least once gets
its full trajectory appended as one JSON line to `retry_log_<backend>.jsonl`
next to the binary (e.g. `retry_log_local.jsonl`, `retry_log_openrouter.jsonl`
— one file per backend, since retry behavior isn't comparable across
different models). A clean first-pass request logs nothing; there's no
trajectory to record when nothing needed adjusting.

Each line looks like:

```json
{
  "timestamp": "2026-07-09T15:04:05Z",
  "bucket": "strict_code",
  "step_exponent": 1.0,
  "adjustments": [
    {"attempt": 0, "issue": "repetition", "param": "dry_multiplier", "old_value": 0, "new_value": 0.8, "detail": "the exact n-gram that repeated"}
  ],
  "resolved": true,
  "total_attempts": 1,
  "latency_ms": 842
}
```

`detail` says *what* triggered the detection — for repetition, the exact
n-gram that repeated (also logged to `proxy_<backend>.log` at detection
time). This is the first thing to read when retries seem too frequent:
agentic clients like Cline produce output full of legitimately repeated
structure (XML tool tags, file paths, boilerplate phrasing), which can
trip the n-gram detector on perfectly healthy content. A false-positive
retry is expensive — the client gets nothing until every attempt
finishes, so each one silently adds a full extra generation of latency
per step, which from the client's side looks like hanging. If the logged
n-grams are real degenerate loops, the retry was earned; if they're tool
tags or paths, tune `[detection]` (`repetition_ngram_size` up,
`repetition_min_repeats` up) rather than blaming the sampling params.

This exact scenario happened in practice and drove a detector change: the
first `detail`-logged entries showed a perfectly normal JS debug-print
line (`console.log(' - ' + c.id + ...)`) that legitimately appeared 3
times in a generated script — flagged as "repetition", costing a full
extra generation per request (121s in one case), with the DRY retry then
actively penalizing the model for writing correct repeated code. The fix:
`repetition_window_words` (default 96) requires all
`repetition_min_repeats` occurrences to fall within that many words of
each other before counting as degeneration. Real stuck loops emit their
repeats back-to-back (well inside the window); legitimate code repeats
scatter across the file (outside it). The check applies to the most
recent N occurrences, so earlier scattered legitimate repeats don't mask
a genuine loop that starts later in the output. Set it to 0 to restore
pure occurrence counting.

The window alone wasn't enough, though: a second confirmed false positive
showed a code-generator response building output like
`+ NL + I1 + '}' + NL + NL + I1` — string-concatenation of newline/indent
constants, which is *inherently* dense and clustered even when completely
legitimate (that's what code-formatting logic looks like). No spacing
heuristic can tell that apart from a real loop, because it isn't scattered
to begin with. What does tell them apart: a genuine degenerate loop
doesn't stop on its own — it cycles until it exhausts the token cap
(`finish_reason: "length"`) — while every confirmed false positive so far
came from a generation that completed normally (`"stop"`). So
`repetition_requires_length_finish` (default `true`) gates repetition
detection on `finish_reason == "length"` entirely: a normally-completed
response is never scanned for repetition at all, however dense its
legitimate repeats are. Set to `false` to go back to scanning every
response regardless of how it finished.

`resolved: false` entries (retries exhausted without fixing the issue)
matter as much as successes — they're the signal that an exponent was too
timid for that many retries to still not work, same as consistently
resolving in 1 attempt at a low exponent is the signal it's already fine
and doesn't need raising. Compare `resolved` rate and `total_attempts`
across runs at different `retry_step_exponent` values to find one that
converges faster without overshooting into values that hurt output
quality (e.g. too high a `repeat_penalty` suppressing legitimate repeated
tokens like variable names).

**Bug found via real trajectory data (fixed):** at the shipped default
`retry_step_exponent = 1.0` (and `prefer_dry_over_repeat_penalty = true`,
also the default), a second consecutive repetition retry within the same
request used to compute `dry_multiplier` from scratch each time
(`dry_multiplier_on_retry * step`) instead of building on the value the
previous attempt had already set. Since `step = step_exponent^attempt` is
`1^attempt = 1` for every attempt at the default exponent, this reproduced
the *exact same* `dry_multiplier` on every retry — e.g. two consecutive
entries both showing `"old_value": 0.8, "new_value": 0.8`. A retry that
resends identical sampling params has nothing but token-level randomness
to save it, which matches multi-minute latencies observed on requests that
needed 2 attempts to resolve. Fixed to escalate additively off the real
accumulated value (`old + dry_multiplier_on_retry*step`), the same pattern
`repeat_penalty`'s branch already used — so now attempt 1 genuinely
produces a higher `dry_multiplier` than attempt 0, even at the default
`step_exponent = 1.0`.

### Live activity indicator

Underneath the bars, the dashboard shows what's actively generating right
now:

```
generating (strict_code, attempt 1)... ~340 tokens (est) · 4.2s elapsed · ~62.4 tok/s (est)
```

or `idle` when nothing's in flight. This exists specifically for remote
providers: local `llama-server` already gives you GPU utilization and a
live `tg = N t/s` console log to tell whether it's working or stuck; a
remote provider gives you nothing until the whole response comes back,
which previously meant "thinking..." for anywhere from a few seconds to
several minutes with no way to tell the difference between normal latency
and a stalled/rate-limited request.

The elapsed timer starts ticking **before** upstream sends a single byte,
not just once content starts arriving — `p.client.Do` blocks until
response headers come back, and for a provider that queues a request
before generation starts (OpenRouter's free-tier rate-limiting is exactly
this), that wait can be most of the total latency. A background ticker
emits periodic "still waiting, N seconds elapsed, 0 tokens so far" updates
during that phase specifically so it doesn't look identical to no request
being in flight at all (regression: this originally only started tracking
once `Do` returned, so the whole pre-first-byte queueing wait showed
`idle` the entire time even with a message actively running in Cline).

That still leaves one gap the first fix didn't cover: `Do` only blocks
until response *headers* arrive, not until the *body* has anything in
it. Some upstreams flush headers immediately, then spend a long time
evaluating the prompt (prefill) before writing the first SSE line —
`scanner.Scan()` blocks synchronously for that whole window with nothing
else running, which would silently reproduce the exact same "looks idle
while a request is genuinely in flight" bug one layer deeper (real
example: local `llama-server` took 40+s on a large prompt before the
first token, entirely inside this window). The SSE reader now runs on
its own goroutine feeding a channel, so the main loop can `select`
between an incoming line and a periodic ticker tick — the same
"processing prompt" state as the pre-headers wait now also covers the
post-headers, pre-first-line gap, however long it turns out to be. The
reader goroutine always drains to a real EOF/error rather than being
cancelled the moment `[DONE]` is seen, so it can never block forever
trying to hand off a line nobody's reading (verified under `go test
-race`, including a misbehaving-upstream case that keeps sending lines
after `[DONE]`).

The `tok/s` rate is computed from a **separate clock that only starts once
the first real token arrives**, not from total elapsed time — total
elapsed includes that same connection/queueing/prompt-processing wait,
and dividing total tokens by total elapsed would silently understate the
real decode speed (a request that waits 5s then generates 50 tokens in 3s
of actual decode is a 16.7 tok/s model, not 6). Before the first token
arrives, the line reads `processing prompt (...) · N.Ns elapsed` instead
of showing a rate at all — that pre-generation wait is largely upstream
actually evaluating the prompt (prefill), not idle network time, so the
label says that rather than the vaguer "waiting for upstream" (confirmed
locally: a ~54k-token prompt took 40+s of prompt processing before the
first generated token). There's still no way to show a live prompt t/s
figure during this phase — no signal for prompt-eval progress exists
until either the first token or the final `usage` object arrives — but
naming the phase correctly at least tells you what's actually happening
instead of leaving it ambiguous.

The token count is an **estimate based on the number of SSE delta events
received, not an exact tokenizer count** — an exact count needs the
model's own tokenizer, which isn't available for arbitrary remote models.
For `llama.cpp`'s streaming loop (and most OpenAI-compatible streaming
implementations), each delta is exactly one generated token, so this
tracks the real count closely in practice — don't read it as ground truth
the way `llama-server`'s own `n_decoded` is, but it's a much better
estimate than it sounds. An earlier version counted whitespace-separated
words instead, which badly undercounts code (brackets, operators, and
identifiers each tokenize separately without being split by spaces) —
that version reported roughly 3-4x below the server's own logged rate for
code-heavy generations.

**`tokens_sec_multiplier` (per-provider, `config.ini`)**: the
one-delta-equals-one-token assumption above is verified against
`llama-server`'s own source, but there's no such guarantee for a remote
provider — some batch more than one token into each streamed delta, which
would make the live estimate undercount, and the reported tok/s read
artificially low (e.g. a paid model that never seems to top ~25 tok/s on
the dashboard even though it's genuinely faster). There's no generic way
to detect this per-provider automatically, so it's a manual correction
factor in each `[provider.*]` section:

```ini
[provider.openrouter]
...
tokens_sec_multiplier = 1.0
```

Defaults to `1.0` (no correction) everywhere, including local
`llama-server` (which has no `[provider.*]` section at all, so it always
gets the no-op). If you suspect a specific provider's live number is off
by a roughly constant factor, compare it against that provider's own
reported speed (dashboards, response headers, etc.) over a few requests
and set the multiplier accordingly — e.g. `2.0` if every delta actually
carries about two tokens. This only rescales the *live in-flight*
estimate; the "last request: N prompt + M completion" line stays exact,
since it comes from the real `usage` object, not the chunk count.

This only applies to `/v1/chat/completions` (see "How it works" above for
why `/completion` is out of scope) and updates at most 4x/second
(`progressEmitInterval` in `stream.go`) — fast enough to feel live,
throttled enough not to flood the terminal with redraws on a very fast
stream.

Once a request finishes, a second line shows its **exact** token counts:

```
last request: 727 prompt + 1437 completion = 2164 tokens
```

Unlike the live indicator's estimate, these come straight from upstream's
own `usage` object (`stream_options: {include_usage: true}` is set on
every outgoing streamed request specifically so this is available) — real
numbers, not a word count. The catch: `usage` only arrives in the *final*
SSE chunk, right before `[DONE]`, so there's no way to show live
prompt-processing progress the way generation gets one — prompt token
counts are only ever knowable once the whole response is already back.
This line is blank until the first `/v1/chat/completions` response
completes (kindCompletion and connection-error events never populate it).

**If it stays blank, or completion tokens show but prompt tokens read
`0`**: that means upstream isn't sending `prompt_tokens` in its usage
object, despite `stream_options.include_usage` being requested — this can
happen with some free-tier/proxied models (or particular `llama-server`
forks/builds) even when the general API is documented to support it. The
proxy logs this to `proxy_<backend>.log` (see above — not the terminal,
once the dashboard is running) so it's diagnosable instead of silently
wrong:
```
stream (strict_code, attempt 0): upstream never sent a usage object despite stream_options.include_usage being requested — prompt/completion token counts will show as 0
```
or, if usage arrived but was missing just the one field:
```
stream (strict_code, attempt 0): usage object present but has no prompt_tokens field; raw usage: map[completion_tokens:1437]
```
There's nothing the proxy can do to recover a number the upstream never
sent — no local tokenizer to fall back to — but at least it's visible
which case you're hitting rather than a silent zero.

## Project layout

```
main.go       // wiring: load config, resolve backend (local/provider), start HTTP server + dashboard, graceful shutdown
config.go     // load/parse config.ini, write defaults if missing
classify.go   // task classification
proxy.go      // reverse proxy handler, param injection, retry loop, remote provider routing, retry trajectory logging
stream.go     // always-stream-upstream for /v1/chat/completions, SSE parsing/accumulation, live progress events, response re-chunking/synthesis
detect.go     // repetition/truncation/syntax checks
ui.go         // bubbletea dashboard model, live activity indicator
config.ini    // generated with defaults on first run
retry_log_<backend>.jsonl  // one JSON line per request that retried at least once (see "Tuning retry_step_exponent" above)
proxy_<backend>.log        // all log.Printf diagnostics once the dashboard is running (see "Run" above — the dashboard's alt-screen would otherwise swallow them)
```

## Tests

```sh
go test ./...
```

Covers classification priority/defaults, repetition/truncation/syntax
detection, the retry-on-repetition flow against a fake upstream (both the
DRY-preferred path and the repeat_penalty fallback), the
client-value-always-wins merge rule, the temperature floor on repeated
bad-syntax retries, the "clean requests carry no reactive params" rule, the
connection-refused path (dashboard gets an error event, proxy returns 502,
nothing crashes), the "upstream error never wrapped in a fake SSE stream"
regression, remote-provider request routing/auth/model-override, the
`/v1/models` proxying, the `{PROVIDER}_API_KEY` env var overriding
`config.ini` (and working even with no `[provider.*]` section present at
all), per-provider default/overridden `listen_port` assignment, CORS
headers being present on every response and preflight `OPTIONS` requests
getting a clean 204, `reasoning_content` surviving SSE re-chunking as its
own delta chunk, the CLI provider-selection logic in `main.go`,
`retry_step_exponent` scaling repeat_penalty/DRY/temperature adjustments
correctly (and specifically *not* leaking into `max_tokens`'s own
multiplier), the retry trajectory log itself — written on retries that
resolve and ones that exhaust without resolving, skipped on a clean first
pass, and safe under concurrent requests (verified with `go test -race`) —
the dashboard's log line formatting always fitting one row regardless of
how long/verbose an upstream error is (regression: a long error body
word-wrapped a single log entry into several rows, and with only entry
*count* capped rather than rendered height, that pushed the bars panel
above it off the top of the terminal — bubbletea's inline renderer can't
reposition past whatever the terminal actually has room to show), and the
truncation retry escalating from the real `usage.completion_tokens` /
`tokens_predicted` instead of a hardcoded 512 guess (regression: found via
real `retry_log_*.jsonl` data showing the exact same truncation sequence
repeating twice, exhausting retries without ever resolving, because the
old guess could already be smaller than what the backend's own default
cap had already produced), and the always-stream-upstream mechanism for
`/v1/chat/completions` — content/reasoning_content/finish_reason/
completion_tokens correctly accumulated from a multi-chunk SSE response,
live progress events firing during a stream (and the final one marked
`Done`), a non-2xx status read as a raw error body rather than fed through
the SSE parser, `/completion` confirmed to still force `stream:false` and
never emit progress events (unaffected by any of this), the dashboard
model correctly clearing its in-flight indicator on a `Done` event, real
prompt/completion token counts from upstream's `usage` object surfacing
correctly on the completed-request UIEvent, and — the regression that
motivated it — a progress event firing with a ticking elapsed time (zero
tokens, not `Done`) while still waiting for upstream's first byte, so a
provider that queues a request before generation starts no longer looks
identical to `idle`, and `GenerationElapsedMs` correctly excluding that
same pre-first-byte wait (a slow-to-start-but-fast-to-decode upstream
reports a rate close to its real decode speed, not artificially deflated
by connection/queueing time it had nothing to do with). All of this is
verified clean under `go test -race` across repeated runs, including a
real synchronization bug the first version of the waiting-ticker fix had:
the background ticker goroutine wasn't guaranteed to have fully exited
before `postUpstreamChatStreaming` returned, so it could still be
mid-`emitProgress` after the caller had moved on — fixed by waiting on the
goroutine's own exit signal, not just closing its stop channel. Also
covers `ApproxTokens` tracking real per-token SSE delta count rather than
whitespace word count — regression test simulates 10 real token-level
deltas that concatenate into only 2 whitespace-separated words, and
asserts the reported count is 10, not 2 (the exact shape of the
undercounting bug that showed ~20 tok/s against a server logging ~70).
Also covers the truncation retry's logged `max_tokens` value matching the
integer actually sent upstream rather than the pre-truncation float
(regression: found in real `retry_log_*.jsonl` data logging
`new_value: 457.5` — `adjustForIssue` correctly sent `int(old *
multiplier)` but reported the un-truncated float in the trajectory log,
an impossible fractional token count that didn't match what upstream
actually received). Also covers `tokens_sec_multiplier`: config parsing
defaults it to `1.0` both when a `[provider.*]` section exists but omits
the key and when the section is absent entirely, an explicit override
value is read correctly, the live chunk-based `ApproxTokens` estimate is
scaled by the configured multiplier (verified with a mocked stream with
no `usage` object, forcing the chunks fallback rather than real
`completion_tokens`), and `tokensSecMultiplier()` treats a zero-value
`ProviderConfig` (as built by several other tests' struct literals, and
as Go's zero value would otherwise silently be) as "unset" and returns
the no-op `1.0` rather than a real `0x` multiplier that would always
report zero tokens. Also covers the post-headers, pre-first-line
progress gap: a progress event fires (elapsed ticking up, zero tokens,
not `Done`) while upstream has already sent response headers but hasn't
written the first SSE body line yet — simulated with a handler that
explicitly flushes a 200 response before sleeping, so `p.client.Do`
returns quickly and the old code would've blocked silently inside
`scanner.Scan()` with no ticker running. And a goroutine-leak regression:
a misbehaving upstream that keeps sending lines after `[DONE]` must not
hang `postUpstreamChatStreaming` — the reader goroutine has to keep
draining to a real EOF rather than being cut off the moment `[DONE]` is
seen, or it'd block forever handing off a line nobody's reading. Also
covers `dry_multiplier` escalating across two real consecutive attempts
sharing one body map (matching how the actual retry loop in
`handleClassified` mutates `body` in place), at the shipped default
`retry_step_exponent = 1.0` — the regression test for real
`retry_log_*.jsonl` data showing two attempts both landing on
`dry_multiplier: 0.8` (identical, no escalation at all). This is
deliberately a separate test from the existing exponent-scaling one,
which uses a fresh body per case and so can't see a bug that only exists
when attempts share state the way production actually does. Also covers
the repetition detail: `findRepeatedNgram` returns the actual offending
n-gram (not just a boolean), `DetectIssue` surfaces it, and the
end-to-end retry-log test asserts the `detail` field lands in the JSONL
entry containing the repeated text — the field false-positive diagnosis
depends on. Also covers the `repetition_window_words` clustering
requirement, using the exact confirmed false positive from real retry
logs as the regression case: the same legitimate `console.log` line
appearing 3 times separated by unrelated code must not be flagged, while
a back-to-back loop still is, a genuine loop late in the output is still
caught even after earlier scattered occurrences of the same n-gram, and
config parsing defaults the window to 96 for existing config.ini files
that predate the key (0 would silently mean "no clustering" and keep the
false-positive behavior the window exists to fix). Also covers
`repetition_requires_length_finish`, using the second confirmed false
positive as the regression case: dense, inherently-clustered code
(newline/indent string concatenation) is not flagged when the response
finished normally (`"stop"`), the identical content *is* flagged when
finish_reason is `"length"`, disabling the gate makes finish_reason stop
mattering again, and config parsing defaults the gate to `true` for
existing config.ini files that predate the key (`false` would silently
keep scanning normally-completed responses, the exact false-positive
class the gate exists to remove). Also covers the SSE re-chunking
formatting bug reported live as "any formatting is lost, including code
formatting": `chunkText` preserves the original text exactly (join every
returned piece and it must equal the input byte-for-byte, verified
against a multi-line, tab-indented, blank-line-containing code sample),
and an end-to-end test drives the real `/v1/chat/completions` streaming
path, reconstructs the message the way a real client would (concatenating
`delta.content` across every SSE chunk), and asserts the result matches
the original multi-line indented code exactly — the previous
`strings.Fields`-based implementation would have flattened it to one line
with no indentation.

