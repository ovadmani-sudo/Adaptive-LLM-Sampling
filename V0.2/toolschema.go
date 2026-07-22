package main

// sanitizeToolSchemas strips string-length/pattern constraints from every
// tool's JSON Schema `parameters` object in body["tools"], in place — a
// workaround for a real llama.cpp limitation where these constraints
// compile into GBNF repetition rules that can exceed its internal sanity
// cap on complex schemas (see ToolSchemaSanitizerConfig for the full
// rationale). A no-op if body has no "tools" field (any non-chat request,
// or a chat request that doesn't register tools).
func sanitizeToolSchemas(body map[string]interface{}, cfg ToolSchemaSanitizerConfig) {
	tools, ok := body["tools"].([]interface{})
	if !ok {
		return
	}
	for _, t := range tools {
		tool, ok := t.(map[string]interface{})
		if !ok {
			continue
		}
		fn, ok := tool["function"].(map[string]interface{})
		if !ok {
			continue
		}
		params, ok := fn["parameters"].(map[string]interface{})
		if !ok {
			continue
		}
		stripSchemaConstraints(params, cfg)
	}
}

// stripSchemaConstraints recursively walks a JSON Schema object, deleting
// the constraint keywords enabled in cfg wherever they appear. A schema
// nests through properties/items/anyOf/oneOf/allOf/$defs/definitions/
// additionalProperties, and a constraint buried in any of these is just as
// capable of blowing up the generated grammar as one at the schema's top
// level, so every one of these is walked. Safe against unbounded recursion:
// this walks the actual parsed map/slice structure produced by
// json.Unmarshal, which is a tree (JSON has no cycles), not a $ref graph —
// $ref pointers themselves are left untouched and never followed.
func stripSchemaConstraints(schema map[string]interface{}, cfg ToolSchemaSanitizerConfig) {
	if cfg.StripMaxLength {
		delete(schema, "maxLength")
	}
	if cfg.StripMinLength {
		delete(schema, "minLength")
	}
	if cfg.StripPattern {
		delete(schema, "pattern")
	}

	if props, ok := schema["properties"].(map[string]interface{}); ok {
		for _, v := range props {
			if sub, ok := v.(map[string]interface{}); ok {
				stripSchemaConstraints(sub, cfg)
			}
		}
	}
	if items, ok := schema["items"].(map[string]interface{}); ok {
		stripSchemaConstraints(items, cfg)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		list, ok := schema[key].([]interface{})
		if !ok {
			continue
		}
		for _, v := range list {
			if sub, ok := v.(map[string]interface{}); ok {
				stripSchemaConstraints(sub, cfg)
			}
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		defs, ok := schema[key].(map[string]interface{})
		if !ok {
			continue
		}
		for _, v := range defs {
			if sub, ok := v.(map[string]interface{}); ok {
				stripSchemaConstraints(sub, cfg)
			}
		}
	}
	if additionalProps, ok := schema["additionalProperties"].(map[string]interface{}); ok {
		stripSchemaConstraints(additionalProps, cfg)
	}
}
