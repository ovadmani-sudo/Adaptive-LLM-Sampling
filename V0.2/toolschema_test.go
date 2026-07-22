package main

import (
	"encoding/json"
	"net/http"
	"testing"
)

// TestSanitizeToolSchemasStripsMaxLengthAtTopLevel verifies the basic case:
// a single tool's top-level parameters schema has maxLength removed when
// StripMaxLength is enabled.
func TestSanitizeToolSchemasStripsMaxLengthAtTopLevel(t *testing.T) {
	body := map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": "write_file",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"content": map[string]interface{}{
								"type":      "string",
								"maxLength": float64(1000000),
							},
						},
					},
				},
			},
		},
	}

	sanitizeToolSchemas(body, ToolSchemaSanitizerConfig{StripMaxLength: true})

	props := body["tools"].([]interface{})[0].(map[string]interface{})["function"].(map[string]interface{})["parameters"].(map[string]interface{})["properties"].(map[string]interface{})
	content := props["content"].(map[string]interface{})
	if _, has := content["maxLength"]; has {
		t.Error("expected maxLength to be stripped from the nested property schema")
	}
	if content["type"] != "string" {
		t.Errorf("expected unrelated schema fields (type) left untouched, got %v", content["type"])
	}
}

// TestSanitizeToolSchemasRecursesThroughNestedShapes verifies every JSON
// Schema nesting construct that can hide a constraint gets walked: items
// (arrays), anyOf/oneOf/allOf (unions), $defs/definitions (referenced sub-
// schemas), and additionalProperties (maps) — a constraint buried in any of
// these is just as capable of blowing up the generated grammar as one at
// the schema's top level.
func TestSanitizeToolSchemasRecursesThroughNestedShapes(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"tags": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type":      "string",
					"maxLength": float64(64),
				},
			},
			"flavor": map[string]interface{}{
				"anyOf": []interface{}{
					map[string]interface{}{"type": "string", "maxLength": float64(32)},
					map[string]interface{}{"type": "null"},
				},
			},
			"extra": map[string]interface{}{
				"additionalProperties": map[string]interface{}{
					"type":      "string",
					"maxLength": float64(16),
				},
			},
		},
		"$defs": map[string]interface{}{
			"SubThing": map[string]interface{}{
				"type":      "string",
				"maxLength": float64(8),
			},
		},
	}
	body := map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"function": map[string]interface{}{"parameters": schema},
			},
		},
	}

	sanitizeToolSchemas(body, ToolSchemaSanitizerConfig{StripMaxLength: true})

	props := schema["properties"].(map[string]interface{})
	tagsItems := props["tags"].(map[string]interface{})["items"].(map[string]interface{})
	if _, has := tagsItems["maxLength"]; has {
		t.Error("expected maxLength stripped inside array items schema")
	}
	flavorAnyOf := props["flavor"].(map[string]interface{})["anyOf"].([]interface{})
	if _, has := flavorAnyOf[0].(map[string]interface{})["maxLength"]; has {
		t.Error("expected maxLength stripped inside anyOf branch")
	}
	extraAdditional := props["extra"].(map[string]interface{})["additionalProperties"].(map[string]interface{})
	if _, has := extraAdditional["maxLength"]; has {
		t.Error("expected maxLength stripped inside additionalProperties schema")
	}
	subThing := schema["$defs"].(map[string]interface{})["SubThing"].(map[string]interface{})
	if _, has := subThing["maxLength"]; has {
		t.Error("expected maxLength stripped inside $defs entry")
	}
}

// TestSanitizeToolSchemasRespectsIndividualToggles verifies
// minLength/pattern are only stripped when their own config flag is set,
// independent of StripMaxLength — these are optional extras, not bundled
// together.
func TestSanitizeToolSchemasRespectsIndividualToggles(t *testing.T) {
	body := map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"function": map[string]interface{}{
					"parameters": map[string]interface{}{
						"type":      "string",
						"maxLength": float64(100),
						"minLength": float64(1),
						"pattern":   "^[a-z]+$",
					},
				},
			},
		},
	}

	sanitizeToolSchemas(body, ToolSchemaSanitizerConfig{StripMaxLength: true}) // minLength/pattern left at zero value (false)

	params := body["tools"].([]interface{})[0].(map[string]interface{})["function"].(map[string]interface{})["parameters"].(map[string]interface{})
	if _, has := params["maxLength"]; has {
		t.Error("expected maxLength stripped (StripMaxLength was true)")
	}
	if _, has := params["minLength"]; !has {
		t.Error("expected minLength left untouched (StripMinLength was false)")
	}
	if _, has := params["pattern"]; !has {
		t.Error("expected pattern left untouched (StripPattern was false)")
	}
}

// TestSanitizeToolSchemasNoopWithoutTools verifies a request with no
// "tools" field at all (the common case for most chat requests) is left
// completely untouched — no panic, no spurious mutation.
func TestSanitizeToolSchemasNoopWithoutTools(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
	}
	sanitizeToolSchemas(body, ToolSchemaSanitizerConfig{StripMaxLength: true})
	if _, has := body["tools"]; has {
		t.Error("expected no tools field to be introduced")
	}
}

// TestToolSchemaSanitizerAppliedWhenEnabled is the full request-cycle
// integration test: with [tool_schema_sanitizer].enabled = true, upstream
// must receive the tools schema with maxLength already stripped — this is
// the actual mechanism that prevents llama.cpp's grammar converter from
// choking on Cline's large tool schemas.
func TestToolSchemaSanitizerAppliedWhenEnabled(t *testing.T) {
	var sawParams map[string]interface{}
	cfg := testConfig()
	cfg.ToolSchemaSanitizer = ToolSchemaSanitizerConfig{Enabled: true, StripMaxLength: true}
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		tools := reqBody["tools"].([]interface{})
		sawParams = tools[0].(map[string]interface{})["function"].(map[string]interface{})["parameters"].(map[string]interface{})
		w.Write(sseChatStream("ok", "", "stop", 1))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": "write_file",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"content": map[string]interface{}{"type": "string", "maxLength": float64(500000)},
						},
					},
				},
			},
		},
	})

	if sawParams == nil {
		t.Fatal("upstream never received a tools[].function.parameters object")
	}
	content := sawParams["properties"].(map[string]interface{})["content"].(map[string]interface{})
	if _, has := content["maxLength"]; has {
		t.Error("expected upstream to receive the tool schema with maxLength already stripped")
	}
}

// TestToolSchemaSanitizerNoopWhenDisabled verifies the default
// (enabled=false) leaves tool schemas completely untouched, so nothing
// changes for anyone not hitting the grammar-parsing bug this works around.
func TestToolSchemaSanitizerNoopWhenDisabled(t *testing.T) {
	var sawParams map[string]interface{}
	cfg := testConfig() // ToolSchemaSanitizer left at zero value: Enabled=false
	proxy, _, closeFn := newFixtureProxy(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		tools := reqBody["tools"].([]interface{})
		sawParams = tools[0].(map[string]interface{})["function"].(map[string]interface{})["parameters"].(map[string]interface{})
		w.Write(sseChatStream("ok", "", "stop", 1))
	})
	defer closeFn()

	postChat(proxy, map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": "write_file",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"content": map[string]interface{}{"type": "string", "maxLength": float64(500000)},
						},
					},
				},
			},
		},
	})

	if sawParams == nil {
		t.Fatal("upstream never received a tools[].function.parameters object")
	}
	content := sawParams["properties"].(map[string]interface{})["content"].(map[string]interface{})
	if got, has := content["maxLength"]; !has || got != float64(500000) {
		t.Errorf("expected maxLength left untouched when sanitizer disabled, got %v (has=%v)", got, has)
	}
}
