package main

import "strings"

// defaultAlertProbeMessage is appended as a new user turn when a model's
// turn ends naturally (finish_reason "stop") on a request whose model is
// configured for alert-continuation (ServerConfig.AlertModels, gated by
// the --alert CLI flag — see main.go) — phrased to hedge both observed
// behaviors: a model that's actually done tends to answer briefly and
// stop there (see looksLikeConfirmation), while a model that stopped
// prematurely on an unfinished multi-step task typically just resumes the
// real work directly, without bothering to literally answer the question
// at all. Configurable via [server] alert_probe_message in config.ini
// (ServerConfig.AlertProbeMessage) — this default is only the fallback
// when that key is absent.
const defaultAlertProbeMessage = "Is the request fully completed? If not, please continue exactly where you left off."

// alertConfirmationMaxRunes bounds how long a reply can be and still be
// considered a bare confirmation rather than genuine continuation content
// — chosen generously short since actual continuation content (more code,
// more explanation) is virtually always longer than a one-line
// acknowledgment.
const alertConfirmationMaxRunes = 200

// alertConfirmationKeywords are checked case-insensitively against a short
// reply to decide whether it reads as "yes, already done" rather than
// substantive continuation content. This is a heuristic, not a guarantee
// — tune alongside alert_max_rounds if a specific model's phrasing
// routinely falls on the wrong side of it.
var alertConfirmationKeywords = []string{
	"yes", "already", "fulfilled", "complete", "completed", "finished",
	"done", "nothing more", "no further", "no additional",
}

// looksLikeConfirmation reports whether a reply to alertProbeMessage
// reads as a bare "yes, it's done" rather than genuine continuation
// content that should be appended to the delivered response. An empty
// reply counts as confirmation too — there's nothing left to append
// either way.
func looksLikeConfirmation(content string) bool {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return true
	}
	if len([]rune(trimmed)) > alertConfirmationMaxRunes {
		return false
	}
	lower := strings.ToLower(trimmed)
	for _, kw := range alertConfirmationKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// modelInAlertList reports whether model matches one of the configured
// alert_models entries — case-insensitive exact match (model names are
// well-defined identifiers, not natural language, so substring matching
// like classify.go's keyword matching would risk false positives, e.g.
// "gpt-4" matching "gpt-4o" unintentionally). A literal "*" or "any" entry
// (case-insensitive) is a wildcard matching every model, for anyone who
// wants alert-continuation applied broadly rather than to specific models
// observed to need it.
func modelInAlertList(alertModels []string, model string) bool {
	if model == "" {
		return false
	}
	lower := strings.ToLower(model)
	for _, m := range alertModels {
		trimmed := strings.ToLower(strings.TrimSpace(m))
		if trimmed == "*" || trimmed == "any" {
			return true
		}
		if trimmed == lower {
			return true
		}
	}
	return false
}
