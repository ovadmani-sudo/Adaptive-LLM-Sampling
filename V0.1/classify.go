package main

import "strings"

// Classify inspects the latest user-role message content and returns the
// matching task bucket per the configured keyword lists and priority order.
func Classify(cfg *ClassificationConfig, lastUserMessage string) TaskBucket {
	content := strings.ToLower(lastUserMessage)

	matched := make(map[TaskBucket]bool)
	for bucket, keywords := range cfg.Keywords {
		for _, kw := range keywords {
			if strings.Contains(content, kw) {
				matched[bucket] = true
				break
			}
		}
	}

	for _, bucket := range cfg.PriorityOrder {
		if matched[bucket] {
			return bucket
		}
	}

	return cfg.DefaultBucket
}

// extractLastUserMessage pulls the text content of the last role:"user"
// message out of a generic OpenAI-style chat completion request body.
func extractLastUserMessage(body map[string]interface{}) string {
	rawMessages, ok := body["messages"]
	if !ok {
		return ""
	}
	messages, ok := rawMessages.([]interface{})
	if !ok {
		return ""
	}

	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		return extractMessageText(msg["content"])
	}
	return ""
}

// extractMessageText handles both plain-string content and the
// OpenAI multimodal array-of-parts content shape.
func extractMessageText(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, part := range c {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if partMap["type"] == "text" {
				if text, ok := partMap["text"].(string); ok {
					sb.WriteString(text)
					sb.WriteString(" ")
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}
