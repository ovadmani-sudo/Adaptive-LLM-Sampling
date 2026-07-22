package main

import (
	"encoding/json"
	"regexp"
	"strings"
)

// Issue identifies a problem found in a completion's output.
type Issue string

const (
	IssueNone       Issue = ""
	IssueRepetition Issue = "repetition"
	IssueTruncated  Issue = "truncated"
	IssueBadSyntax  Issue = "bad_syntax"
	// IssueEmpty is a completion whose visible content is blank (or
	// whitespace-only) despite a "clean" finish_reason — e.g. "stop" — with
	// no tool calls to explain the absence of text. Confirmed real-world
	// failure: this previously passed through as IssueNone ("clean"), so
	// neither the adaptive retry loop nor alert-continuation (which only
	// nudges a model that already produced *some* reply) ever noticed or
	// fixed it — the client just received nothing and showed its own
	// generic "please resend" error.
	IssueEmpty Issue = "empty"
	// IssueReasoningLoop is the live, mid-stream counterpart to
	// IssueRepetition: some models (Gemma 4 26B confirmed via direct packet
	// capture) can get stuck in a degenerate reasoning loop that never
	// naturally finishes — repeating short fragments indefinitely without
	// ever producing real content. Every other Issue here is only ever
	// detected AFTER a response completes (DetectIssue below inspects
	// finished text), which never happens for a model looping forever in
	// its reasoning phase. This one is detected instead by
	// postUpstreamChatStreaming (stream.go) watching reasoning_content as
	// it streams, aborting once it clearly runs past whatever budget was
	// requested — see ReasoningGuardConfig. Handled directly in
	// handleClassified rather than through DetectIssue's normal analysis
	// path, since there's no finished text to analyze.
	IssueReasoningLoop Issue = "reasoning_loop"
)

// DetectIssue inspects the completion content and finish reason, returning
// the first issue found (empty > repetition > truncation > syntax, in that
// order) plus a human-readable detail of *what* triggered it — for
// repetition, the exact n-gram that repeated. The detail exists to make
// false positives diagnosable from the logs: without it, a retry log only
// says "repetition happened", with no way to tell genuine degeneration
// apart from legitimately repeated structure (XML tool tags, file paths,
// boilerplate) tripping the detector on healthy output.
//
// hasToolCalls must be true when the response carries tool/function calls
// — a tool-calls-only turn legitimately has empty visible content (the
// "response" is the tool call itself), so the empty check is skipped
// entirely in that case rather than flagging normal agentic behavior as a
// bug.
func DetectIssue(cfg *DetectionConfig, content string, finishReason string, hasToolCalls bool) (Issue, string) {
	if !hasToolCalls && strings.TrimSpace(content) == "" {
		return IssueEmpty, ""
	}
	// A genuine degenerate loop doesn't stop on its own — it cycles until
	// it eats the token cap, so it arrives with finish_reason "length". A
	// response that ended naturally and merely *contains* repeats is
	// overwhelmingly legitimate repetitive code: every confirmed false
	// positive in real retry_log data (a debug console.log written 3x, a
	// code-generator concatenating NL/indent constants on every emitted
	// line) came from a generation that completed normally. Requiring
	// "length" before even scanning for repetition removes that whole
	// false-positive class — clustered legit repeats included, which the
	// window check alone can't tell apart from a loop.
	checkRepetition := !cfg.RepetitionRequiresLengthFinish || finishReason == "length"
	if checkRepetition {
		if gram, found := findRepeatedNgram(content, cfg.RepetitionNgramSize, cfg.RepetitionMinRepeats, cfg.RepetitionWindowWords); found {
			return IssueRepetition, gram
		}
	}
	if isTruncated(content, finishReason) {
		return IssueTruncated, ""
	}
	if hasBadSyntax(content) {
		return IssueBadSyntax, ""
	}
	return IssueNone, ""
}

// findRepeatedNgram performs a sliding-window word n-gram scan: if any
// n-gram of the configured size repeats at least minRepeats times in the
// token stream, the output is considered stuck in a repetition loop, and
// the offending n-gram is returned so it can be logged.
//
// windowWords adds a clustering requirement: the minRepeats occurrences
// must all fall within that many words of each other (measured from the
// start of the earliest counted occurrence to the start of the latest).
// This is what separates degeneration from healthy code: a stuck model
// emits its loop back-to-back, so the repeats land close together, while
// a legitimate repeated line — the same console.log or import written
// several times across a file — scatters. Found via real retry_log data:
// a normal JS debug-print line appearing 3x in generated code tripped
// the pure-count detector, and the resulting DRY retry then actively
// penalized the model for writing correct repeated code. windowWords <= 0
// disables clustering (pure occurrence counting, the old behavior).
func findRepeatedNgram(content string, ngramSize int, minRepeats int, windowWords int) (string, bool) {
	if ngramSize <= 0 || minRepeats <= 1 {
		return "", false
	}
	words := strings.Fields(content)
	if len(words) < ngramSize*minRepeats {
		return "", false
	}

	positions := make(map[string][]int)
	for i := 0; i+ngramSize <= len(words); i++ {
		gram := strings.Join(words[i:i+ngramSize], " ")
		positions[gram] = append(positions[gram], i)
		occ := positions[gram]
		if len(occ) < minRepeats {
			continue
		}
		if windowWords <= 0 {
			return gram, true
		}
		// Check whether the most recent minRepeats occurrences fit inside
		// the window — earlier scattered occurrences don't disqualify a
		// genuine loop that starts later in the output.
		if occ[len(occ)-1]-occ[len(occ)-minRepeats] <= windowWords {
			return gram, true
		}
	}
	return "", false
}

// isTruncated flags output that was cut off mid-statement: finish_reason
// is "length" and the content has unbalanced brackets/braces/parens or an
// unterminated code fence/string.
func isTruncated(content string, finishReason string) bool {
	if finishReason != "length" {
		return false
	}
	return !isBalanced(content)
}

func isBalanced(content string) bool {
	var stack []rune
	pairs := map[rune]rune{')': '(', ']': '[', '}': '{'}
	openers := map[rune]bool{'(': true, '[': true, '{': true}

	inString := false
	var stringQuote rune
	escaped := false

	for _, r := range content {
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if r == '\\' {
				escaped = true
				continue
			}
			if r == stringQuote {
				inString = false
			}
			continue
		}

		switch {
		case r == '"' || r == '\'':
			inString = true
			stringQuote = r
		case openers[r]:
			stack = append(stack, r)
		case r == ')' || r == ']' || r == '}':
			if len(stack) == 0 || stack[len(stack)-1] != pairs[r] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}

	if inString {
		return false
	}
	if len(stack) != 0 {
		return false
	}

	// unterminated fenced code block (odd number of ``` occurrences)
	if strings.Count(content, "```")%2 != 0 {
		return false
	}

	return true
}

var fencedJSONBlock = regexp.MustCompile("(?s)```json\\s*\\n(.*?)```")

// hasBadSyntax is a best-effort, cheap-only syntax check: it validates
// fenced ```json blocks with encoding/json. Other languages are skipped
// silently since a real check would require shelling out to a toolchain.
func hasBadSyntax(content string) bool {
	matches := fencedJSONBlock.FindAllStringSubmatch(content, -1)
	for _, m := range matches {
		var v interface{}
		if err := json.Unmarshal([]byte(m[1]), &v); err != nil {
			return true
		}
	}
	return false
}
