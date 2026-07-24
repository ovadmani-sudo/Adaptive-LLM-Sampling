package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// WebPanel is the local browser control panel: it mirrors the terminal
// dashboard's live state (request log, in-flight, sampling, throughput) via
// server-sent events and exposes toggle controls that drive the Supervisor
// (start/stop listeners, per-listener sampling pass-through, force mode). It's
// opt-in (--web) and runs alongside the TUI; both read the same event streams
// through the Broker, so neither starves the other.
type WebPanel struct {
	sup                *Supervisor
	broker             *Broker
	throughputSnapshot func() []ThroughputStatsEntry
	clinepassBase      string // clinepass base_url, for the model catalog fetch

	catalogMu   sync.Mutex
	catalog     []CatalogModel
	catalogTime time.Time
}

func NewWebPanel(sup *Supervisor, broker *Broker, throughputSnapshot func() []ThroughputStatsEntry, clinepassBase string) *WebPanel {
	return &WebPanel{sup: sup, broker: broker, throughputSnapshot: throughputSnapshot, clinepassBase: clinepassBase}
}

func (wp *WebPanel) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", wp.handleIndex)
	mux.HandleFunc("/api/status", wp.handleStatus)
	mux.HandleFunc("/api/events", wp.handleEvents)
	mux.HandleFunc("/api/listener", wp.handleListener)
	mux.HandleFunc("/api/bypass", wp.handleBypass)
	mux.HandleFunc("/api/alert", wp.handleAlert)
	mux.HandleFunc("/api/bucket", wp.handleBucket)
	mux.HandleFunc("/api/models", wp.handleModels)
	mux.HandleFunc("/api/model", wp.handleSetModel)
	mux.HandleFunc("/api/vision", wp.handleVision)
	mux.HandleFunc("/api/system_prompt", wp.handleSystemPrompt)
	return mux
}

// handleModels returns the clinepass model catalog (grouped by billing), for
// the panel's model picker. Cached for a minute so repeated panel loads don't
// re-hit the endpoint.
func (wp *WebPanel) handleModels(w http.ResponseWriter, r *http.Request) {
	wp.catalogMu.Lock()
	fresh := time.Since(wp.catalogTime) < time.Minute && len(wp.catalog) > 0
	cached := wp.catalog
	wp.catalogMu.Unlock()

	if !fresh {
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		models, err := fetchClinepassCatalog(ctx, wp.clinepassBase)
		if err != nil {
			// Serve stale cache if we have it; otherwise report the error.
			if len(cached) == 0 {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
		} else {
			wp.catalogMu.Lock()
			wp.catalog, wp.catalogTime = models, time.Now()
			cached = models
			wp.catalogMu.Unlock()
		}
	}
	writeJSON(w, map[string]any{"models": cached})
}

// handleSetModel pins (or clears) a backend's model live:
// POST /api/model?name=X&model=Y  (empty model clears the override)
func (wp *WebPanel) handleSetModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	model := r.URL.Query().Get("model")
	if err := wp.sup.SetModelOverride(name, model); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

func (wp *WebPanel) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(controlPanelHTML))
}

// handleStatus returns the current listener table (each row carrying its
// own forced_bucket/vision_describe/system_prompt — all per-listener, see
// ListenerStatus) and the throughput snapshot — everything the panel
// needs to render on load and on each poll after a control action.
// system_prompts is the one genuinely global piece: the menu of
// available [system_prompt.*] names, identical for every listener.
func (wp *WebPanel) handleStatus(w http.ResponseWriter, r *http.Request) {
	resp := map[string]any{
		"listeners":      wp.sup.Status(),
		"throughput":     wp.throughputSnapshot(),
		"system_prompts": wp.sup.SystemPromptNames(),
	}
	writeJSON(w, resp)
}

// handleEvents is the SSE stream: each completed request (UIEvent) and each
// in-flight progress tick (ProgressEvent) is pushed as a JSON line tagged with
// its type, so the browser can update the log and the live indicator.
func (wp *WebPanel) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	uiCh, cancelUI := wp.broker.SubscribeUI()
	defer cancelUI()
	progCh, cancelProg := wp.broker.SubscribeProgress()
	defer cancelProg()

	keepalive := time.NewTicker(20 * time.Second)
	defer keepalive.Stop()

	send := func(kind string, v any) bool {
		data, err := json.Marshal(map[string]any{"kind": kind, "data": v})
		if err != nil {
			return true
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	for {
		select {
		case <-r.Context().Done():
			return
		case ev := <-uiCh:
			if !send("event", ev) {
				return
			}
		case ev := <-progCh:
			if !send("progress", ev) {
				return
			}
		case <-keepalive.C:
			if _, err := fmt.Fprint(w, ": keepalive\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

// handleListener starts/stops a named listener: POST /api/listener?name=X&action=start|stop
func (wp *WebPanel) handleListener(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	var err error
	switch r.URL.Query().Get("action") {
	case "start":
		err = wp.sup.Start(name)
	case "stop":
		err = wp.sup.Stop(name)
	default:
		http.Error(w, "action must be start or stop", http.StatusBadRequest)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

// handleBypass toggles sampling pass-through: POST /api/bypass?name=X&on=true|false
func (wp *WebPanel) handleBypass(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	on, _ := strconv.ParseBool(r.URL.Query().Get("on"))
	if err := wp.sup.SetBypass(name, on); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

// handleAlert toggles alert-continuation on a backend:
// POST /api/alert?name=X&on=true|false
func (wp *WebPanel) handleAlert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	on, _ := strconv.ParseBool(r.URL.Query().Get("on"))
	if err := wp.sup.SetAlert(name, on); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

// handleBucket forces (or clears) the classification mode on ONE listener:
// POST /api/bucket?name=<listener>&bucket=strict_code|exploratory_code|explanation|architecture|agentic_loop|""
func (wp *WebPanel) handleBucket(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	b := r.URL.Query().Get("bucket")
	if err := wp.sup.SetForcedBucket(name, TaskBucket(b)); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

// handleVision toggles [vision_describe] on ONE listener:
// POST /api/vision?name=<listener>&on=true|false
func (wp *WebPanel) handleVision(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	on, _ := strconv.ParseBool(r.URL.Query().Get("on"))
	if err := wp.sup.SetVisionDescribe(name, on); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

// handleSystemPrompt selects a [system_prompt.*] entry on ONE listener:
// POST /api/system_prompt?name=<listener>&prompt=<name|empty>
// An empty (or omitted) prompt clears back to no injection.
func (wp *WebPanel) handleSystemPrompt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	name := r.URL.Query().Get("name")
	prompt := r.URL.Query().Get("prompt")
	if err := wp.sup.SetSystemPrompt(name, prompt); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"listeners": wp.sup.Status()})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
