package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CatalogModel is one selectable model from clinepass's public catalog, tagged
// with its billing group so the control panel can show what each choice costs.
type CatalogModel struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Group       string `json:"group"` // "subscription" | "pay-as-you-go" | "free"
}

// fetchClinepassCatalog returns clinepass's full model catalog from the public
// recommended-models endpoint (no auth needed — it's a catalog, not account
// data). base is the clinepass base_url from config (e.g.
// https://api.cline.bot/api/v1). The three id-namespace arrays map to billing
// groups: clinePass -> subscription, recommended -> pay-as-you-go, free ->
// free. Proxy env vars are ignored on this call (see the connector's note).
func fetchClinepassCatalog(ctx context.Context, base string) ([]CatalogModel, error) {
	client := &http.Client{Timeout: 15 * time.Second, Transport: &http.Transport{Proxy: nil}}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, base+"/ai/cline/recommended-models", nil)
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching clinepass catalog: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("clinepass catalog returned %d", resp.StatusCode)
	}

	var parsed struct {
		ClinePass   []CatalogModel `json:"clinePass"`
		Recommended []CatalogModel `json:"recommended"`
		Free        []CatalogModel `json:"free"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parsing clinepass catalog: %w", err)
	}

	out := make([]CatalogModel, 0, len(parsed.ClinePass)+len(parsed.Recommended)+len(parsed.Free))
	for _, g := range []struct {
		models []CatalogModel
		group  string
	}{
		{parsed.ClinePass, "subscription"},
		{parsed.Recommended, "pay-as-you-go"},
		{parsed.Free, "free"},
	} {
		for _, m := range g.models {
			m.Group = g.group
			out = append(out, m)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("clinepass catalog was empty")
	}
	return out, nil
}
