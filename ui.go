package main

import (
	"fmt"
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
}

const maxLogLines = 12

// logBoxWidth is the log panel's rendered width (border + padding
// included). condenseForLogLine's truncation budget is sized against this,
// so if this ever changes, that budget needs revisiting too.
const logBoxWidth = 80

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
	Done                bool
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
}

func newDashboardModel(events <-chan UIEvent, progressCh <-chan ProgressEvent, listenAddr, backendDesc string) dashboardModel {
	mk := func() progress.Model {
		p := progress.New(progress.WithDefaultGradient())
		p.Width = 40
		return p
	}
	return dashboardModel{
		events:     events,
		progress:   progressCh,
		listenCfg:  listenAddr,
		backend:    backendDesc,
		tempBar:    mk(),
		topPBar:    mk(),
		topKBar:    mk(),
		budgetBar:  mk(),
		penaltyBar: mk(),
	}
}

func (m dashboardModel) Init() tea.Cmd {
	return tea.Batch(listenForEvents(m.events), listenForProgress(m.progress))
}

func (m dashboardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}
	case uiEventMsg:
		ev := UIEvent(msg)
		m.last = ev

		line := formatLogLine(ev)
		m.log = append(m.log, line)
		if len(m.log) > maxLogLines {
			m.log = m.log[len(m.log)-maxLogLines:]
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
	}
	return m, nil
}

func formatLogLine(ev UIEvent) string {
	issue := string(ev.Issue)
	if issue == "" {
		issue = "clean"
	}
	if ev.Error != "" {
		return fmt.Sprintf("%s  %-16s retry=%d  ERROR: %s",
			ev.Timestamp.Format("15:04:05"), "-", ev.RetryCount, condenseForLogLine(ev.Error))
	}
	return fmt.Sprintf("%s  %-16s retry=%d  issue=%-10s latency=%dms",
		ev.Timestamp.Format("15:04:05"), ev.Bucket, ev.RetryCount, issue, ev.LatencyMs)
}

// condenseForLogLine collapses an error message onto a single short line,
// so one log entry never word-wraps into multiple visual rows inside the
// fixed-width box. Long upstream error bodies (e.g. a verbose 429 message)
// previously wrapped into 4-5 rows each; with 12 entries capped only by
// count, not rendered height, that could push the bars panel above it off
// the top of the terminal — bubbletea's inline renderer can't reposition
// past whatever the terminal actually has room to show.
//
// max is deliberately conservative: the box is logBoxWidth (80) wide, minus
// border+padding (4 chars) leaves 76 for content, and the fixed
// "HH:MM:SS  %-16s retry=N  ERROR: " prefix already eats ~43 of those —
// so the error text itself gets a tight budget to guarantee the whole
// formatted line still fits on one row.
func condenseForLogLine(s string) string {
	s = strings.Join(strings.Fields(s), " ") // collapse any embedded newlines/whitespace
	const max = 28
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
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
	if ev.GenerationElapsedMs == 0 {
		return activeStyle.Render(fmt.Sprintf(
			"processing prompt (%s, attempt %d)... %.1fs elapsed",
			ev.Bucket, ev.Attempt+1, elapsed,
		))
	}
	genElapsed := float64(ev.GenerationElapsedMs) / 1000.0
	rate := float64(ev.ApproxTokens) / genElapsed
	return activeStyle.Render(fmt.Sprintf(
		"generating (%s, attempt %d)... ~%d tokens (est) · %.1fs elapsed · ~%.1f tok/s (est)",
		ev.Bucket, ev.Attempt+1, ev.ApproxTokens, elapsed, rate,
	))
}

// formatLastRequestTokensLine shows the exact prompt/completion token
// counts for the most recently completed request — unlike the in-flight
// line's word-count estimate, these come straight from the upstream's own
// usage object once the stream ends, so they're real numbers, not a guess.
// Only populated for kindChat (the streaming path); kindCompletion and
// connection-error events leave both at 0, in which case nothing renders,
// so the line doesn't show a misleading "0 + 0" before any real data
// exists.
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

	b.WriteString(valueStyle.Render("ctrl+c to quit") + "\n")

	return b.String()
}

func runDashboard(events <-chan UIEvent, progressCh <-chan ProgressEvent, listenAddr, backendDesc string) error {
	// Alt-screen gives bubbletea a full, known-size viewport to redraw
	// within, instead of printing inline and relying on the terminal's own
	// scrollback — which is what let content taller than the visible
	// window desync the redraw and appear to "eat" earlier-printed rows.
	p := tea.NewProgram(newDashboardModel(events, progressCh, listenAddr, backendDesc), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
