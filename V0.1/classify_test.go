package main

import "testing"

func testClassificationConfig() *ClassificationConfig {
	return &ClassificationConfig{
		Keywords: map[TaskBucket][]string{
			BucketStrictCode:      {"fix bug", "refactor", "implement", "write function", "edit file"},
			BucketExploratoryCode: {"brainstorm", "alternative approach", "explore", "sketch"},
			BucketExplanation:     {"explain", "why", "what does", "how does"},
			BucketArchitecture:    {"architecture", "design", "structure", "plan"},
		},
		PriorityOrder: []TaskBucket{BucketStrictCode, BucketExploratoryCode, BucketExplanation, BucketArchitecture},
		DefaultBucket: BucketStrictCode,
	}
}

func TestClassifySingleMatch(t *testing.T) {
	cfg := testClassificationConfig()
	got := Classify(cfg, "Can you explain why this function fails?")
	if got != BucketExplanation {
		t.Errorf("got %s, want %s", got, BucketExplanation)
	}
}

func TestClassifyPriorityOrder(t *testing.T) {
	cfg := testClassificationConfig()
	// matches both strict_code ("fix bug") and explanation ("why") -> strict_code wins
	got := Classify(cfg, "please fix bug, and explain why it happened")
	if got != BucketStrictCode {
		t.Errorf("got %s, want %s", got, BucketStrictCode)
	}
}

func TestClassifyDefaultBucket(t *testing.T) {
	cfg := testClassificationConfig()
	got := Classify(cfg, "hello there, how are you today")
	if got != BucketStrictCode {
		t.Errorf("got %s, want default %s", got, BucketStrictCode)
	}
}

func TestClassifyCaseInsensitive(t *testing.T) {
	cfg := testClassificationConfig()
	got := Classify(cfg, "Let's BRAINSTORM some ideas")
	if got != BucketExploratoryCode {
		t.Errorf("got %s, want %s", got, BucketExploratoryCode)
	}
}

func TestExtractLastUserMessage(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "sys"},
			map[string]interface{}{"role": "user", "content": "first"},
			map[string]interface{}{"role": "assistant", "content": "reply"},
			map[string]interface{}{"role": "user", "content": "explain why this crashes"},
		},
	}
	got := extractLastUserMessage(body)
	want := "explain why this crashes"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestExtractLastUserMessageMultimodal(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": "please refactor"},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "x"}},
				},
			},
		},
	}
	got := extractLastUserMessage(body)
	if got != "please refactor " {
		t.Errorf("got %q", got)
	}
}
