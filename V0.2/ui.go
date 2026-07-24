package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// UIEvent carries the outcome of one request/response cycle from the proxy
// goroutine to the dashboard. Sent over a channel; the proxy never blocks
// waiting for the UI to consume it.
type UIEvent struct {
	Timestamp            time.Time
	Bucket               TaskBucket
	RetryCount           int
	Issue                Issue
	LatencyMs            int64
	Temperature          float64
	TopP                 float64
	TopK                 int
	RepeatPenalty        float64
	ThinkingBudgetTokens int
	// PromptTokens/CompletionTokens are the real counts from the
	// upstream's usage object (kindChat only — kindCompletion keeps the
	// original blocking call and never populates these, left at 0), more
	// accurate than the live in-flight indicator's word-count estimate
	// since these come straight from the provider once the stream ends.
	PromptTokens     int
	CompletionTokens int
	Error            string
	// Host is the intercepted destination for a forward-proxy-mode
	// request (mitm.go) — empty for every other mode, where the backend
	// is fixed for the whole run and already shown in the dashboard's
	// static header instead.
	Host string
	// Model is the request's "model" field (post any provider-config
	// override) — populated for kindChat only, used to attribute
	// AlertRounds to the right model on the dashboard's alert-continuation
	// summary.
	Model string
	// AlertRounds is how many alert-continuation probes fired for this
	// request (see alert.go) — 0 unless --alert is active and Model
	// matched alert_models. The dashboard accumulates this per model
	// across the session (see dashboardModel.alertCounts).
	AlertRounds int
	// Listener is which supervised listener (Supervisor's registration
	// name, e.g. "local", "claude") this request came in on — stamped by
	// ProxyServer.emit, not set by callers. Lets the web panel attribute
	// each log row / in-flight tick to a specific backend when several
	// are running (and being used by different agents) at once.
	Listener string
}

// maxLogLines is deliberately small (debug-friendly setting): entries are
// no longer truncated to guarantee a single row each (see
// condenseForLogLine/formatLogLine below — full error text and full host
// are always shown, wrapping across multiple visual rows if needed), so
// capping the entry *count* is what keeps total rendered height bounded
// instead of a per-entry length budget.
const maxLogLines = 5

// logBoxWidth is the log panel's rendered width (border + padding
// included) — just a reasonable reading width; lines longer than this
// wrap onto additional rows rather than being cut short.
const logBoxWidth = 128

// ProgressEvent carries a live snapshot of an in-flight streaming request,
// so the dashboard can show whether generation is actively progressing or
// appears stalled — mainly useful against remote providers, where there's
// no local llama.cpp console/GPU utilization to watch for signs of life.
// ApproxTokens counts SSE delta events received (ChunksReceived), not an
// exact tokenizer count — but for llama.cpp's streaming loop (and most
// OpenAI-compatible streaming implementations) each delta is exactly one
// generated token, so this tracks real token count closely in practice.
// An earlier version estimated via whitespace word-count instead, which
// badly undercounts code (brackets/operators/identifiers tokenize far
// more densely than they split on spaces) — that showed roughly 3-4x
// below the server's own reported t/s for code-heavy generations.
type ProgressEvent struct {
	Bucket         TaskBucket
	Attempt        int
	ChunksReceived int
	ApproxTokens   int
	// ElapsedMs is total time since the request was first sent, including
	// any pre-first-byte connection/queueing wait — this is what should
	// keep ticking during that wait, so a stalled request doesn't look
	// identical to no request being in flight.
	ElapsedMs int64
	// GenerationElapsedMs is time since the *first* content/reasoning
	// token actually arrived (0 if generation hasn't started yet). Rate
	// (tokens/sec) must be computed from this, not ElapsedMs — dividing by
	// total elapsed time would silently fold connection/queueing/prompt-
	// processing dead time into the rate, understating it.
	GenerationElapsedMs int64
	// IdleTimeoutRemainingMs is how long until the idle/inactivity timeout
	// (postUpstreamChatStreaming's p.idleTimeout, stream.go) would fire if
	// no further progress arrives, recomputed on every tick — this is what
	// the dashboard's live countdown renders. -1 means no idle timeout is
	// configured, distinct from 0 ("about to fire").
	IdleTimeoutRemainingMs int64
	// Label, if non-empty, overrides the bucket/attempt/token rendering in
	// formatInFlightLine entirely with a plain "<Label>... Xs elapsed" line
	// — used for activity that has no classification bucket at all because
	// it bypasses classify/inject/detect/retry (forward-proxy passthrough
	// hosts, see ForwardProxyConfig.PassthroughHosts), so the dashboard
	// still shows something other than "idle" while it's in flight.
	Label string
	// Model is the request's actual "model" field (post any provider
	// config/live-override/vision_describe override) — shown alongside the
	// bucket/token line so it's visible in real time which model is doing
	// the work, not just after the fact in the completed-request log
	// (UIEvent.Model). Most useful when vision_describe (or a live model
	// override) can redirect an individual request to a different model
	// than whatever's "current" for the listener.
	Model string
	Done  bool
	// Listener is which supervised listener (Supervisor's registration
	// name, e.g. "local", "claude") this tick came from — stamped by
	// ProxyServer.emitProgress, not set by callers. Lets the web panel's
	// "In-flight" card attribute live progress to a specific backend, and
	// offer a session selector when several are running at once.
	Listener string
}

type uiEventMsg UIEvent
type progressMsg ProgressEvent

// listenForEvents returns a tea.Cmd that waits for the next event on the
// channel and wraps it as a tea.Msg. The model re-issues this command after
// every event so the listen loop keeps going.
func listenForEvents(events <-chan UIEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-events
		if !ok {
			return nil
		}
		return uiEventMsg(ev)
	}
}

// listenForProgress mirrors listenForEvents for the separate progress
// channel — kept as its own channel/command rather than multiplexed onto
// events, since progress updates fire far more often (throttled per
// streaming chunk) than the once-per-completed-request UIEvent.
func listenForProgress(progressCh <-chan ProgressEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-progressCh
		if !ok {
			return nil
		}
		return progressMsg(ev)
	}
}

type dashboardModel struct {
	events    <-chan UIEvent
	progress  <-chan ProgressEvent
	listenCfg string
	backend   string

	tempBar    progress.Model
	topPBar    progress.Model
	topKBar    progress.Model
	budgetBar  progress.Model
	penaltyBar progress.Model

	last     UIEvent
	log      []string
	inFlight *ProgressEvent

	// throughputSnapshot fetches the latest persisted per-(provider,model,
	// bucket) throughput aggregates (see throughput_stats.go) — re-read on
	// every throughputTickMsg rather than pushed over a channel, since this
	// data changes far less often than progress/log events and a simple
	// poll avoids adding yet another channel to plumb through main.go.
	throughputSnapshot func() []ThroughputStatsEntry
	throughputStats    []ThroughputStatsEntry
	// showThroughput toggles the throughput panel on/off (key "r") — kept
	// hidden by default so the dashboard isn't cluttered with an all-time
	// table most sessions don't need to see continuously; ./llama-dyn-proxy
	// --report covers the same data without a running dashboard at all.
	showThroughput bool

	// alertCounts accumulates, per model, how many alert-continuation
	// probes (see alert.go) have fired since this dashboard started —
	// pushed via UIEvent.AlertRounds rather than polled, since it's a
	// running total keyed to specific completed requests, not a
	// point-in-time snapshot like throughputStats.
	alertCounts map[string]int

	// controls bundles the keybinding callbacks for forcing a
	// classification bucket — see DashboardControls. forcedBucket is local
	// display state, updated immediately on keypress (the dashboard is
	// the only writer of it, so no polling needed).
	controls     DashboardControls
	forcedBucket *TaskBucket
}

// DashboardControls bundles the callbacks the dashboard uses to control
// live proxy behavior via keybindings (see dashboardModel.Update's
// tea.KeyMsg case). Any nil field disables its corresponding keybinding.
type DashboardControls struct {
	SetForcedBucket   func(TaskBucket)
	ClearForcedBucket func()
}

func newDashboardModel(events <-chan UIEvent, progressCh <-chan ProgressEvent, listenAddr, backendDesc string, throughputSnapshot func() []ThroughputStatsEntry, controls DashboardControls) dashboardModel {
	mk := func() progress.Model {
		p := progress.New(progress.WithDefaultGradient())
		p.Width = 40
		return p
	}
	return dashboardModel{
		events:             events,
		progress:           progressCh,
		listenCfg:          listenAddr,
		backend:            backendDesc,
		tempBar:            mk(),
		topPBar:            mk(),
		topKBar:            mk(),
		budgetBar:          mk(),
		penaltyBar:         mk(),
		throughputSnapshot: throughputSnapshot,
		alertCounts:        make(map[string]int),
		controls:           controls,
	}
}

type throughputTickMsg time.Time

// tickThroughput re-issues itself every 5s (see Update's throughputTickMsg
// case) — a slow poll is plenty since throughput aggregates change once
// per completed request at most, nowhere near as often as progress/log
// events.
func tickThroughput() tea.Cmd {
	return tea.Tick(5*time.Second, func(t time.Time) tea.Msg {
		return throughputTickMsg(t)
	})
}

func (m dashboardModel) Init() tea.Cmd {
	return tea.Batch(listenForEvents(m.events), listenForProgress(m.progress), tickThroughput())
}

// forceBucketKeys maps a keypress to the bucket it pins — checked in
// Update's tea.KeyMsg case. "0" clears back to auto-detect rather than
// mapping to another bucket. "5" is agentic_loop, the force-only bucket
// (no classification keywords — this keybinding and the web panel are
// the ONLY ways it ever activates).
var forceBucketKeys = map[string]TaskBucket{
	"1": BucketStrictCode,
	"2": BucketExploratoryCode,
	"3": BucketExplanation,
	"4": BucketArchitecture,
	"5": BucketAgenticLoop,
}

func (m dashboardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch key := msg.String(); key {
		case "ctrl+c":
			return m, tea.Quit
		case "0":
			m.forcedBucket = nil
			if m.controls.ClearForcedBucket != nil {
				m.controls.ClearForcedBucket()
			}
		case "r":
			m.showThroughput = !m.showThroughput
		default:
			if bucket, ok := forceBucketKeys[key]; ok {
				b := bucket
				m.forcedBucket = &b
				if m.controls.SetForcedBucket != nil {
					m.controls.SetForcedBucket(bucket)
				}
			}
		}
	case uiEventMsg:
		ev := UIEvent(msg)
		m.last = ev

		line := formatLogLine(ev)
		m.log = append(m.log, line)
		if len(m.log) > maxLogLines {
			m.log = m.log[len(m.log)-maxLogLines:]
		}

		if ev.AlertRounds > 0 && ev.Model != "" {
			m.alertCounts[ev.Model] += ev.AlertRounds
		}

		return m, listenForEvents(m.events)
	case progressMsg:
		ev := ProgressEvent(msg)
		if ev.Done {
			m.inFlight = nil
		} else {
			m.inFlight = &ev
		}
		return m, listenForProgress(m.progress)
	case throughputTickMsg:
		if m.throughputSnapshot != nil {
			m.throughputStats = m.throughputSnapshot()
		}
		return m, tickThroughput()
	}
	return m, nil
}

func formatLogLine(ev UIEvent) string {
	issue := string(ev.Issue)
	if issue == "" {
		issue = "clean"
	}
	// hostSuffix is only ever non-empty in forward-proxy mode, where the
	// backend varies per request instead of being fixed for the whole run
	// (and so shown in the dashboard's static header instead, for every
	// other mode). Shown in full (debug-friendly setting) — see maxLogLines
	// for how total rendered height stays bounded instead.
	hostSuffix := ""
	if ev.Host != "" {
		hostSuffix = fmt.Sprintf(" host=%s", ev.Host)
	}
	// modelSuffix shows which model actually handled this request — most
	// useful when it differs from whatever's "current" for the listener
	// (vision_describe or a live model override redirecting an individual
	// request), matching the web panel's log row (see addLog, web_page.go).
	modelSuffix := ""
	if ev.Model != "" {
		modelSuffix = fmt.Sprintf(" model=%s", ev.Model)
	}
	if ev.Error != "" {
		return fmt.Sprintf("%s  %-16s retry=%d  ERROR: %s%s%s",
			ev.Timestamp.Format("15:04:05"), "-", ev.RetryCount, condenseForLogLine(ev.Error), modelSuffix, hostSuffix)
	}
	return fmt.Sprintf("%s  %-16s retry=%d  issue=%-10s latency=%dms%s%s",
		ev.Timestamp.Format("15:04:05"), ev.Bucket, ev.RetryCount, issue, ev.LatencyMs, modelSuffix, hostSuffix)
}

// condenseForLogLine collapses an error message's embedded
// newlines/tabs/repeated whitespace onto a single logical line, so it
// reads as one entry rather than looking like several unrelated ones —
// the full text is preserved (debug-friendly setting: an entry may wrap
// across multiple visual rows if it's long, rather than being cut short).
// maxLogLines' small count is what keeps total rendered height bounded
// instead of a per-entry length budget.
func condenseForLogLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

var (
	titleStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	labelStyle    = lipgloss.NewStyle().Width(22)
	valueStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	boxStyle      = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1)
	errStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	activeStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	idleLineStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("220"))
)

// formatInFlightLine renders the live activity indicator: what's actively
// generating right now (if anything), so a stalled remote request is
// visibly distinguishable from one that's just quiet between updates —
// the throttle in postUpstreamChatStreaming means this only refreshes a
// few times a second, not per-token, but a frozen counter for many seconds
// still reads as "probably stuck" the same way a frozen llama.cpp console
// would.
func formatInFlightLine(ev *ProgressEvent) string {
	if ev == nil {
		return idleLineStyle.Render("idle")
	}
	elapsed := float64(ev.ElapsedMs) / 1000.0

	// modelPart surfaces which model is actually doing the work right now
	// — most useful when it differs from whatever's "current" for the
	// listener (vision_describe or a live model override redirecting this
	// one request), so that's visible in real time, not just after the
	// fact in the completed-request log (UIEvent.Model).
	modelPart := ""
	if ev.Model != "" {
		modelPart = fmt.Sprintf(" [%s]", ev.Model)
	}

	if ev.Label != "" {
		return activeStyle.Render(fmt.Sprintf("%s%s... %.1fs elapsed", ev.Label, modelPart, elapsed))
	}

	// Rate must come from generation-only elapsed time, not total elapsed
	// — total includes connection/queueing/prompt-processing wait, which
	// has nothing to do with decode speed and would otherwise silently
	// drag the reported rate down (a request that waited 5s then generated
	// 50 tokens in 3s of real decode time is a 16.7 tok/s model, not 6).
	//
	// This pre-generation phase isn't just network/queueing dead time —
	// for prompts of any real size, upstream is actively evaluating the
	// prompt (prefill) during this exact window (confirmed locally: a
	// ~54k-token prompt took 40+s of prompt processing before the first
	// generated token). "processing prompt" names that correctly instead
	// of the vaguer "waiting for upstream", even though we can't surface
	// a real prompt t/s figure (no signal for prompt-eval progress until
	// the first token or the final usage object arrives).
	countdown := formatIdleCountdownSuffix(ev.IdleTimeoutRemainingMs)

	if ev.GenerationElapsedMs == 0 {
		return activeStyle.Render(fmt.Sprintf(
			"processing prompt (%s, attempt %d)%s... %.1fs elapsed%s",
			ev.Bucket, ev.Attempt+1, modelPart, elapsed, countdown,
		))
	}
	genElapsed := float64(ev.GenerationElapsedMs) / 1000.0
	rate := float64(ev.ApproxTokens) / genElapsed
	return activeStyle.Render(fmt.Sprintf(
		"generating (%s, attempt %d)%s... ~%d tokens (est) · %.1fs elapsed · ~%.1f tok/s (est)%s",
		ev.Bucket, ev.Attempt+1, modelPart, ev.ApproxTokens, elapsed, rate, countdown,
	))
}

// formatIdleCountdownSuffix renders the live idle-timeout countdown as a
// " · idle timeout in Xs" suffix, or "" when no idle timeout is configured
// (remainingMs < 0 — see ProgressEvent.IdleTimeoutRemainingMs) so the
// in-flight line doesn't show a misleading countdown for a disabled
// timeout.
func formatIdleCountdownSuffix(remainingMs int64) string {
	if remainingMs < 0 {
		return ""
	}
	return fmt.Sprintf(" · idle timeout in %.0fs", float64(remainingMs)/1000.0)
}

// formatLastRequestTokensLine shows the exact prompt/completion token
// counts for the most recently completed request — unlike the in-flight
// line's word-count estimate, these come straight from the upstream's own
// usage object once the stream ends, so they're real numbers, not a guess.
// Only populated for kindChat (the streaming path); kindCompletion and
// connection-error events leave both at 0, in which case nothing renders,
// so the line doesn't show a misleading "0 + 0" before any real data
// exists.
// formatAlertCounts renders one "<model> was alerted <n> times" line per
// model that has ever needed an alert-continuation probe (see alert.go),
// sorted by model name for a stable, scannable order across redraws.
func formatAlertCounts(counts map[string]int) string {
	models := make([]string, 0, len(counts))
	for model := range counts {
		models = append(models, model)
	}
	sort.Strings(models)

	var b strings.Builder
	for _, model := range models {
		n := counts[model]
		times := "times"
		if n == 1 {
			times = "time"
		}
		fmt.Fprintf(&b, "%s was alerted %d %s\n", model, n, times)
	}
	return b.String()
}

func formatLastRequestTokensLine(ev UIEvent) string {
	if ev.PromptTokens == 0 && ev.CompletionTokens == 0 {
		return ""
	}
	total := ev.PromptTokens + ev.CompletionTokens
	return valueStyle.Render(fmt.Sprintf(
		"last request: %d prompt + %d completion = %d tokens",
		ev.PromptTokens, ev.CompletionTokens, total,
	))
}

func bar(m progress.Model, label string, value, min, max float64, valueFmt string) string {
	ratio := 0.0
	if max > min {
		ratio = (value - min) / (max - min)
	}
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return fmt.Sprintf("%s%s %s", labelStyle.Render(label), m.ViewAs(ratio), valueStyle.Render(fmt.Sprintf(valueFmt, value)))
}

func (m dashboardModel) View() string {
	var b strings.Builder

	b.WriteString(titleStyle.Render("llama-dyn-proxy") + "\n")
	b.WriteString(valueStyle.Render(fmt.Sprintf("listening on %s  →  backend %s", m.listenCfg, m.backend)) + "\n\n")

	bars := strings.Join([]string{
		bar(m.tempBar, "temperature", m.last.Temperature, 0, 1.2, "%.2f"),
		bar(m.topPBar, "top_p", m.last.TopP, 0, 1, "%.2f"),
		bar(m.topKBar, "top_k", float64(m.last.TopK), 0, 100, "%.0f"),
		bar(m.budgetBar, "thinking_budget", float64(m.last.ThinkingBudgetTokens), 0, 8192, "%.0f"),
		bar(m.penaltyBar, "repeat_penalty", m.last.RepeatPenalty, 1.0, 1.3, "%.2f"),
	}, "\n")
	b.WriteString(boxStyle.Render(bars) + "\n\n")

	b.WriteString(formatInFlightLine(m.inFlight) + "\n")
	if line := formatLastRequestTokensLine(m.last); line != "" {
		b.WriteString(line + "\n")
	}
	b.WriteString("\n")

	logContent := "(no requests yet)"
	if len(m.log) > 0 {
		styled := make([]string, len(m.log))
		for i, line := range m.log {
			if strings.Contains(line, "ERROR:") {
				styled[i] = errStyle.Render(line)
			} else {
				styled[i] = line
			}
		}
		logContent = strings.Join(styled, "\n")
	}
	b.WriteString(titleStyle.Render("recent requests") + "\n")
	b.WriteString(boxStyle.Width(logBoxWidth).Render(logContent) + "\n\n")

	if m.showThroughput && len(m.throughputStats) > 0 {
		b.WriteString(titleStyle.Render("throughput (all-time)") + "\n")
		b.WriteString(boxStyle.Width(logBoxWidth).Render(strings.TrimRight(formatThroughputReport(m.throughputStats), "\n")) + "\n\n")
	}

	if len(m.alertCounts) > 0 {
		b.WriteString(boxStyle.Render(strings.TrimRight(formatAlertCounts(m.alertCounts), "\n")) + "\n\n")
	}

	modeLine := "mode: auto-detect"
	if m.forcedBucket != nil {
		modeLine = fmt.Sprintf("mode: %s (forced)", *m.forcedBucket)
	}
	b.WriteString(valueStyle.Render(modeLine) + "\n")

	help := "1-5 force mode (5=agentic_loop) · 0 auto-detect · r toggle throughput · ctrl+c quit"
	b.WriteString(valueStyle.Render(help) + "\n")

	return b.String()
}

func runDashboard(events <-chan UIEvent, progressCh <-chan ProgressEvent, listenAddr, backendDesc string, throughputSnapshot func() []ThroughputStatsEntry, controls DashboardControls) error {
	// Alt-screen gives bubbletea a full, known-size viewport to redraw
	// within, instead of printing inline and relying on the terminal's own
	// scrollback — which is what let content taller than the visible
	// window desync the redraw and appear to "eat" earlier-printed rows.
	p := tea.NewProgram(newDashboardModel(events, progressCh, listenAddr, backendDesc, throughputSnapshot, controls), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
