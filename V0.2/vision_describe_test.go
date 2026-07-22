package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// newFakeVLM returns a server implementing just enough of the STREAMING
// chat completions shape for callVisionModel (stream:true is required —
// Cline's gateway returns "empty response content" for non-streaming
// requests, so the real code always streams), counting how many describe
// calls actually hit it — the cache assertions below hinge on that count.
func newFakeVLM(t *testing.T, description string, calls *int32) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(calls, 1)
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		if stream, _ := body["stream"].(bool); !stream {
			t.Error("describe call must be streaming (stream=true) — Cline's gateway rejects non-streaming with 'empty response content'")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Split the description across two deltas to prove accumulation,
		// with a reasoning delta mixed in that must be ignored.
		half := len(description) / 2
		for _, delta := range []map[string]interface{}{
			{"reasoning": "thinking about the image"},
			{"content": description[:half]},
			{"content": description[half:]},
		} {
			chunk := map[string]interface{}{
				"choices": []map[string]interface{}{
					{"index": 0, "delta": delta, "finish_reason": nil},
				},
			}
			data, _ := json.Marshal(chunk)
			w.Write([]byte("data: "))
			w.Write(data)
			w.Write([]byte("\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}))
}

func imageMsg(text, url string) map[string]interface{} {
	parts := []interface{}{}
	if text != "" {
		parts = append(parts, map[string]interface{}{"type": "text", "text": text})
	}
	parts = append(parts, map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": url}})
	return map[string]interface{}{"role": "user", "content": parts}
}

// TestVisionDescribeReplacesImageWithDescription verifies the core
// describe-and-replace flow end-to-end through a listener: the image part
// is gone from what reaches the target model, replaced by the VLM's
// description inline, and the request stays on the client's own model.
func TestVisionDescribeReplacesImageWithDescription(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "a red square on white background", &vlmCalls)
	defer vlm.Close()

	var sawRawBody []byte
	var sawModel string
	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sawRawBody = raw
		var body map[string]interface{}
		json.Unmarshal(raw, &body)
		sawModel, _ = body["model"].(string)
		w.Write(sseChatStream("understood", "", "stop", 5))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{imageMsg("please fix bug in this", "data:image/png;base64,abc")},
	})

	if got := atomic.LoadInt32(&vlmCalls); got != 1 {
		t.Errorf("VLM describe calls = %d, want 1", got)
	}
	if sawModel != "ornith-35b" {
		t.Errorf("target model = %q, want the client's own %q (describe must not change routing)", sawModel, "ornith-35b")
	}
	body := string(sawRawBody)
	if strings.Contains(body, "image_url") {
		t.Errorf("image_url part still present in the forwarded request: %s", body)
	}
	if !strings.Contains(body, "[IMAGE DESCRIPTION: a red square on white background]") {
		t.Errorf("expected the inline description in the forwarded request, got: %s", body)
	}
	if !strings.Contains(body, "please fix bug in this") {
		t.Errorf("original text part must be preserved alongside the description, got: %s", body)
	}
}

// TestVisionDescribeCachesByImageHash verifies the same image resent on a
// later turn (full history, as every real client does) does NOT trigger a
// second VLM call — the cached description is reused.
func TestVisionDescribeCachesByImageHash(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "a diagram", &vlmCalls)
	defer vlm.Close()

	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("ok", "", "stop", 5))
	})
	defer closeFn()

	turn1 := imageMsg("what is this?", "data:image/png;base64,samebytes")
	postChat(proxy, map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{turn1},
	})
	// Turn 2 resends turn 1's image plus a new text turn.
	postChat(proxy, map[string]interface{}{
		"model": "ornith-35b",
		"messages": []interface{}{
			imageMsg("what is this?", "data:image/png;base64,samebytes"),
			map[string]interface{}{"role": "assistant", "content": "a diagram"},
			map[string]interface{}{"role": "user", "content": "explain more"},
		},
	})

	if got := atomic.LoadInt32(&vlmCalls); got != 1 {
		t.Errorf("VLM describe calls = %d, want 1 (second turn must hit the cache)", got)
	}
}

// TestVisionDescribeDistinctImagesEachDescribed verifies two different
// images each get their own describe call — the cache keys on content
// hash, not "any image seen before".
func TestVisionDescribeDistinctImagesEachDescribed(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "some image", &vlmCalls)
	defer vlm.Close()

	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		w.Write(sseChatStream("ok", "", "stop", 5))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model": "ornith-35b",
		"messages": []interface{}{
			imageMsg("first", "data:image/png;base64,AAA"),
			imageMsg("second", "data:image/png;base64,BBB"),
		},
	})

	if got := atomic.LoadInt32(&vlmCalls); got != 2 {
		t.Errorf("VLM describe calls = %d, want 2 (distinct images)", got)
	}
}

// TestVisionDescribeFailureFallsBackToPlaceholder verifies a failing VLM
// endpoint never fails the real request: the image is replaced with a
// placeholder marker (NOT left in place, which would choke a text-only
// template) and the request continues to the target model.
func TestVisionDescribeFailureFallsBackToPlaceholder(t *testing.T) {
	vlm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer vlm.Close()

	var sawRawBody []byte
	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sawRawBody = raw
		w.Write(sseChatStream("ok", "", "stop", 5))
	})
	defer closeFn()

	rec := postChat(proxy, map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{imageMsg("what is this?", "data:image/png;base64,abc")},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("request status = %d, want 200 (a describe failure must not fail the request)", rec.Code)
	}
	body := string(sawRawBody)
	if strings.Contains(body, "image_url") {
		t.Errorf("image_url still present after describe failure — must be stripped to the placeholder: %s", body)
	}
	if !strings.Contains(body, "[IMAGE: description unavailable]") {
		t.Errorf("expected the failure placeholder in the forwarded request, got: %s", body)
	}
}

// TestVisionDescribeDisabledLeavesRequestUntouched verifies the global
// switch: disabled means the request body passes through with its image
// intact and the VLM is never called.
func TestVisionDescribeDisabledLeavesRequestUntouched(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "unused", &vlmCalls)
	defer vlm.Close()

	var sawRawBody []byte
	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: false, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sawRawBody = raw
		w.Write(sseChatStream("ok", "", "stop", 5))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"model":    "ornith-35b",
		"messages": []interface{}{imageMsg("what is this?", "data:image/png;base64,abc")},
	})

	if got := atomic.LoadInt32(&vlmCalls); got != 0 {
		t.Errorf("VLM describe calls = %d, want 0 when disabled", got)
	}
	if !strings.Contains(string(sawRawBody), "image_url") {
		t.Errorf("expected the image left untouched when disabled, got: %s", sawRawBody)
	}
}

// TestVisionDescribeAppliesOnProviderListenerToo verifies the GLOBAL
// scope: the feature is global by design: it runs on
// a remote-provider-backed listener as well.
func TestVisionDescribeAppliesOnProviderListenerToo(t *testing.T) {
	var vlmCalls int32
	vlm := newFakeVLM(t, "a chart", &vlmCalls)
	defer vlm.Close()

	var sawRawBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sawRawBody = raw
		w.Write(sseChatStream("ok", "", "stop", 5))
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "test-vlm", BaseURL: vlm.URL}

	proxy, err := NewProxyServer(cfg, &ProviderConfig{BaseURL: upstream.URL, Model: "provider-model"}, "openrouter", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	postChat(proxy, map[string]interface{}{
		"model":    "anything",
		"messages": []interface{}{imageMsg("what is this?", "data:image/png;base64,abc")},
	})

	if got := atomic.LoadInt32(&vlmCalls); got != 1 {
		t.Errorf("VLM describe calls = %d, want 1 on a provider listener (feature is global)", got)
	}
	body := string(sawRawBody)
	if strings.Contains(body, "image_url") {
		t.Errorf("image_url still present in the provider-bound request: %s", body)
	}
	if !strings.Contains(body, "[IMAGE DESCRIPTION: a chart]") {
		t.Errorf("expected the inline description in the provider-bound request, got: %s", body)
	}
}

// TestVisionDescribeBaseURLDefaultsToLocalUpstream verifies an empty
// base_url resolves to the local llama-server from [server].
func TestVisionDescribeBaseURLDefaultsToLocalUpstream(t *testing.T) {
	cfg := testConfig()
	cfg.Server.UpstreamHost = "127.0.0.1"
	cfg.Server.UpstreamPort = 9091
	cfg.VisionDescribe = VisionDescribeConfig{Enabled: true, Model: "qwen3-vl", BaseURL: ""}

	proxy, err := NewProxyServer(cfg, nil, "local", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	defer proxy.Close()

	if got := proxy.visionDescribeBaseURL(); got != "http://127.0.0.1:9091/v1" {
		t.Errorf("visionDescribeBaseURL() = %q, want %q", got, "http://127.0.0.1:9091/v1")
	}
}

func TestHashImageRefStableAndDistinct(t *testing.T) {
	a1 := hashImageRef("data:image/png;base64,AAA")
	a2 := hashImageRef("data:image/png;base64,AAA")
	b := hashImageRef("data:image/png;base64,BBB")
	if a1 != a2 {
		t.Error("same input must hash identically")
	}
	if a1 == b {
		t.Error("different inputs must hash differently")
	}
}
