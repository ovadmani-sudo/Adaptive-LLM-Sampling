package main

import (
	"fmt"
	"strings"
	"testing"
)

func TestFindRepeatedNgramDetectsLoop(t *testing.T) {
	loop := strings.Repeat("the quick brown fox jumps over ", 5)
	gram, found := findRepeatedNgram(loop, 6, 3, 96)
	if !found {
		t.Fatal("expected repetition to be detected")
	}
	if gram != "the quick brown fox jumps over" {
		t.Errorf("repeated n-gram = %q, want the actual offending 6-gram — this is what gets logged for false-positive diagnosis, so it must be the real trigger, not empty or garbage", gram)
	}
}

func TestFindRepeatedNgramCleanText(t *testing.T) {
	clean := "this is a normal sentence with no looping pattern in it at all really"
	if gram, found := findRepeatedNgram(clean, 6, 3, 96); found {
		t.Errorf("did not expect repetition in clean text, got n-gram %q", gram)
	}
}

// TestFindRepeatedNgramScatteredCodeRepeatsNotFlagged is the regression
// test for the confirmed false positive from real retry_log data: a
// legitimate JS debug-print line ("{ console.log(' - ' + c.id + ...")
// appearing 3 times spread across a generated script tripped the
// pure-count detector, and the resulting DRY retry actively penalized the
// model for writing correct repeated code — each false positive costing a
// full extra generation (121s in one logged case). Scattered occurrences
// separated by more than repetition_window_words of other content must
// not count as degeneration.
func TestFindRepeatedNgramScatteredCodeRepeatsNotFlagged(t *testing.T) {
	line := "console.log(' - ' + c.id + ': ' + c.getAssignments().join(', ')); "
	content := line + uniqueWords(60, "a") + line + uniqueWords(60, "b") + line
	if gram, found := findRepeatedNgram(content, 6, 3, 40); found {
		t.Errorf("scattered legitimate code repeats must not be flagged as degeneration, got n-gram %q", gram)
	}
}

// uniqueWords builds filler of n distinct words (prefixed to stay distinct
// across calls) — unlike strings.Repeat, it can't itself trip the
// repetition detector the way a repeated filler word would.
func uniqueWords(n int, prefix string) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "%s%d ", prefix, i)
	}
	return b.String()
}

// TestFindRepeatedNgramClusteredLoopStillFlaggedWithWindow verifies the
// clustering requirement doesn't break genuine degeneration detection: a
// real stuck loop emits its repeats back-to-back, well inside the window.
func TestFindRepeatedNgramClusteredLoopStillFlaggedWithWindow(t *testing.T) {
	loop := strings.Repeat("the quick brown fox jumps over ", 5)
	if _, found := findRepeatedNgram(loop, 6, 3, 40); !found {
		t.Error("back-to-back repetition must still be detected with the clustering window enabled")
	}
}

// TestFindRepeatedNgramLateLoopAfterScatteredOccurrences verifies earlier
// scattered (legitimate) occurrences don't mask a genuine loop that starts
// later: the window check applies to the most recent minRepeats
// occurrences, not the first ones seen.
func TestFindRepeatedNgramLateLoopAfterScatteredOccurrences(t *testing.T) {
	phrase := "alpha beta gamma delta epsilon zeta "
	content := phrase + uniqueWords(80, "c") + phrase + uniqueWords(80, "d") + strings.Repeat(phrase, 3) // scattered twice, then a real loop
	if _, found := findRepeatedNgram(content, 6, 3, 40); !found {
		t.Error("a genuine loop late in the output must be detected even when earlier occurrences were scattered")
	}
}

func TestDetectIssueReturnsRepetitionDetail(t *testing.T) {
	cfg := &DetectionConfig{RepetitionNgramSize: 6, RepetitionMinRepeats: 3, RepetitionWindowWords: 96, RepetitionRequiresLengthFinish: true}
	loop := strings.Repeat("the quick brown fox jumps over ", 5)
	issue, detail := DetectIssue(cfg, loop, "length")
	if issue != IssueRepetition {
		t.Fatalf("issue = %q, want repetition", issue)
	}
	if detail == "" {
		t.Error("expected the repeated n-gram as detail — without it, retry logs can't distinguish genuine degeneration from legitimate repeated structure (XML tags, paths) tripping the detector")
	}
}

// TestDetectIssueRepetitionRequiresLengthFinish covers the finish-reason
// gate added after the clustering window alone proved insufficient: a
// code-generator response concatenating NL/indent constants on every
// emitted line legitimately clusters the same 12-gram within any window,
// so no spacing heuristic can tell it from a loop. What does tell them
// apart: a genuine degenerate loop doesn't stop on its own — it runs until
// the token cap (finish_reason "length"), while every confirmed false
// positive came from a generation that completed normally ("stop").
func TestDetectIssueRepetitionRequiresLengthFinish(t *testing.T) {
	cfg := &DetectionConfig{RepetitionNgramSize: 6, RepetitionMinRepeats: 3, RepetitionWindowWords: 96, RepetitionRequiresLengthFinish: true}
	loop := strings.Repeat("out += NL + I1 + '}' ", 5) // dense legit-looking repeats, clustered

	if issue, _ := DetectIssue(cfg, loop, "stop"); issue == IssueRepetition {
		t.Error("a normally-completed response must not be flagged for repetition when the length-finish gate is on — repeats in output that ended on its own are almost always legitimate repetitive code")
	}
	if issue, _ := DetectIssue(cfg, loop, "length"); issue != IssueRepetition {
		t.Errorf("a response that ran to the token cap with clustered repeats must still be flagged, got %q", issue)
	}

	cfg.RepetitionRequiresLengthFinish = false
	if issue, _ := DetectIssue(cfg, loop, "stop"); issue != IssueRepetition {
		t.Errorf("with the gate disabled, finish_reason must not matter, got %q", issue)
	}
}

func TestIsTruncatedUnbalanced(t *testing.T) {
	content := "func foo() {\n  if (x {\n    return"
	if !isTruncated(content, "length") {
		t.Error("expected truncation to be detected for unbalanced braces")
	}
}

func TestIsTruncatedNotFlaggedWhenStopped(t *testing.T) {
	content := "func foo() {\n  return\n}"
	if isTruncated(content, "stop") {
		t.Error("finish_reason=stop should never be flagged as truncated")
	}
}

func TestIsTruncatedBalancedContentNotFlagged(t *testing.T) {
	content := "func foo() {\n  return 1\n}"
	if isTruncated(content, "length") {
		t.Error("balanced content should not be flagged even with finish_reason=length")
	}
}

func TestIsBalancedUnterminatedFence(t *testing.T) {
	content := "```go\nfunc main() {}\n"
	if isBalanced(content) {
		t.Error("expected unterminated code fence to be unbalanced")
	}
}

func TestHasBadSyntaxInvalidJSON(t *testing.T) {
	content := "here you go:\n```json\n{\"a\": 1,}\n```\n"
	if !hasBadSyntax(content) {
		t.Error("expected invalid JSON in fenced block to be flagged")
	}
}

func TestHasBadSyntaxValidJSON(t *testing.T) {
	content := "here you go:\n```json\n{\"a\": 1}\n```\n"
	if hasBadSyntax(content) {
		t.Error("did not expect valid JSON to be flagged")
	}
}
