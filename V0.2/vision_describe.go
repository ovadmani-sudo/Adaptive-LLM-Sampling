package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// [vision_describe] turns an image-bearing chat request into an all-text
// one before it ever reaches the target model: every image content part
// in the conversation is described once by a configured VLM endpoint and
// replaced inline with "[IMAGE DESCRIPTION: ...]" text. Works for ANY
// listener — local or remote provider — since the target model never
// sees an image at all, and the VLM itself can be local (base_url empty
// → the local llama-server) or a cloud endpoint (base_url + optional
// api_key).
//
// Descriptions are cached per image hash (see ProxyServer.imageDescCache):
// a real client resends the same image on every later turn of the same
// conversation, and without the cache every one of those turns would
// re-run a full VLM generation for an image already described.

// visionDescribeTimeout caps one VLM describe call. Deliberately its own
// constant rather than reusing p.idleTimeout (often 30min for slow local
// generations): a single-image description is a short bounded task, and a
// hung VLM here would otherwise stall the real request behind it for the
// full idle timeout.
const visionDescribeTimeout = 2 * time.Minute

// visionDescribePrompt is the fixed instruction sent alongside each image.
const visionDescribePrompt = "Describe this image in detail. Include any visible text verbatim, layout, UI elements, diagrams, and anything else needed for someone who cannot see the image to fully understand it."

// hashImageRef returns the cache key for one image content part — a
// sha256 over the image_url's url string itself. For the data: URI case
// (inline base64, what agent clients actually send) this is equivalent to
// hashing the image bytes without paying for a base64 decode; for a
// plain http(s) URL it keys on the URL, which is the only stable identity
// available without fetching it.
func hashImageRef(url string) string {
	sum := sha256.Sum256([]byte(url))
	return hex.EncodeToString(sum[:])
}

// imageURLFromPart digs the url string out of one OpenAI-style image
// content part: {"type":"image_url","image_url":{"url":"..."}}. Returns
// "" when the part isn't shaped like that (including the bare
// {"type":"image"} variant, which carries no addressable payload to
// describe).
func imageURLFromPart(p map[string]interface{}) string {
	iu, ok := p["image_url"].(map[string]interface{})
	if !ok {
		return ""
	}
	url, _ := iu["url"].(string)
	return url
}

// describeImagesInPlace walks every message and replaces each image
// content part with a text part carrying the VLM's description of it,
// collapsing each affected message's content back to a plain string —
// the shape a text-only chat template expects; an array-of-parts content
// reaching a text-only template is itself a failure mode, regardless of
// what the parts are (confirmed: gemma/ornith chokes on leftover image
// parts in resent history even with a correct model choice).
//
// A describe failure must never fail the real request riding behind it:
// the part is replaced with a fixed "[IMAGE: description unavailable]"
// marker instead, and the error is logged. Leaving the raw image in
// place would be worse than losing the description — the text-only
// target model chokes on it, per the above.
func (p *ProxyServer) describeImagesInPlace(ctx context.Context, body map[string]interface{}) {
	rawMessages, ok := body["messages"]
	if !ok {
		return
	}
	messages, ok := rawMessages.([]interface{})
	if !ok {
		return
	}

	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := msg["content"].([]interface{})
		if !ok {
			continue // plain string content (or absent) — nothing to describe
		}

		var sb strings.Builder
		appendText := func(text string) {
			if sb.Len() > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(text)
		}
		for _, part := range parts {
			pm, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch t, _ := pm["type"].(string); t {
			case "text":
				if text, ok := pm["text"].(string); ok {
					appendText(text)
				}
			case "image_url", "image":
				url := imageURLFromPart(pm)
				if url == "" {
					appendText("[IMAGE: description unavailable]")
					continue
				}
				desc, err := p.describeImageCached(ctx, url)
				if err != nil {
					log.Printf("vision-describe: describing image failed (request continues with a placeholder): %v", err)
					appendText("[IMAGE: description unavailable]")
					continue
				}
				appendText("[IMAGE DESCRIPTION: " + desc + "]")
			}
		}
		msg["content"] = sb.String()
	}
}

// describeImageCached returns the description for one image url, from
// cache when available, otherwise via one VLM call whose result is then
// cached. The lock is NOT held across the VLM call — a slow description
// must not block a concurrent request's cache hit on a different image.
// The tradeoff is that two concurrent requests carrying the SAME new
// image can each run their own describe call and both store (last write
// wins, identical content) — wasteful once, but never wrong.
func (p *ProxyServer) describeImageCached(ctx context.Context, url string) (string, error) {
	key := hashImageRef(url)

	p.imageDescCacheMu.Lock()
	if desc, ok := p.imageDescCache[key]; ok {
		p.imageDescCacheMu.Unlock()
		return desc, nil
	}
	p.imageDescCacheMu.Unlock()

	desc, err := p.callVisionModel(ctx, url)
	if err != nil {
		return "", err
	}

	p.imageDescCacheMu.Lock()
	if p.imageDescCache == nil {
		p.imageDescCache = make(map[string]string)
	}
	p.imageDescCache[key] = desc
	p.imageDescCacheMu.Unlock()
	return desc, nil
}

// visionDescribeBaseURL resolves where the VLM lives: the configured
// base_url if set, otherwise the local llama-server from [server] — the
// same default target every other local call uses. Note this default is
// deliberately NOT the current listener's own upstream: a provider
// listener's upstream is a remote text API that may not host any vision
// model at all, while the local llama-server demonstrably does (it's the
// whole reason this feature exists).
func (p *ProxyServer) visionDescribeBaseURL() string {
	if u := strings.TrimSpace(p.cfg.VisionDescribe.BaseURL); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return fmt.Sprintf("http://%s:%d/v1", p.cfg.Server.UpstreamHost, p.cfg.Server.UpstreamPort)
}

// visionDescribeAPIKey resolves the credential for describe calls: the
// configured api_key if set; otherwise, when the endpoint is Cline's own
// gateway, the same non-interactive clinepass chain every other clinepass
// use in this codebase resolves through (CLINEPASS_API_KEY env, then the
// connector's stored login — see resolveClinepassAPIKey), so the
// default clinepass-backed setup works without duplicating the key into
// [vision_describe]. Empty for any other keyless endpoint (e.g. local
// llama-server, which wants no Authorization at all).
func (p *ProxyServer) visionDescribeAPIKey() string {
	if key := strings.TrimSpace(p.cfg.VisionDescribe.APIKey); key != "" {
		return key
	}
	if strings.Contains(p.visionDescribeBaseURL(), "cline.bot") {
		key, _ := resolveClinepassAPIKey("")
		return key
	}
	return ""
}

// callVisionModel runs one STREAMING chat completion against the VLM
// endpoint with the image inline — llama-server's multimodal API (and
// every OpenAI-compatible cloud VLM) accepts the base64 data: URI
// directly in image_url.url, so no temp file is ever involved.
//
// stream:true is required, not an optimization: Cline's gateway (the
// default endpoint) returns {"error":"empty response content"} for any
// non-streaming request — confirmed against the live API — so the
// description is accumulated from delta.content chunks here instead of
// read from a single blocking response. Reasoning deltas (delta.reasoning
// on this gateway, delta.reasoning_content on llama.cpp) are deliberately
// ignored: only the final answer text is a description worth injecting.
func (p *ProxyServer) callVisionModel(ctx context.Context, imageURL string) (string, error) {
	reqBody := map[string]interface{}{
		"model":  p.cfg.VisionDescribe.Model,
		"stream": true,
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": visionDescribePrompt},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL}},
				},
			},
		},
	}
	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("encoding describe request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, visionDescribeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.visionDescribeBaseURL()+"/chat/completions", bytes.NewReader(reqBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if key := p.visionDescribeAPIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("vision endpoint unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		errBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("vision endpoint returned %d: %s", resp.StatusCode, truncateForLog(errBody))
	}

	var contentB strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // skip a malformed line rather than aborting the whole stream
		}
		choices, ok := chunk["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}
		choice0, _ := choices[0].(map[string]interface{})
		delta, ok := choice0["delta"].(map[string]interface{})
		if !ok {
			continue
		}
		if c, ok := delta["content"].(string); ok && c != "" {
			contentB.WriteString(c)
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("reading vision stream: %w", err)
	}

	content := strings.TrimSpace(contentB.String())
	if content == "" {
		return "", fmt.Errorf("vision response missing content")
	}
	return content, nil
}
