package main

import (
	"path/filepath"
	"testing"
)

func TestSaveLoadListenerStatesRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "listener_state.json")
	want := map[string]ListenerState{
		"local": {
			Running: true, BypassSampling: true, Alert: false,
			Model: "gpt-oss-120b", ForcedBucket: string(BucketArchitecture),
			VisionDescribe: true, SystemPrompt: "research",
		},
		"claude": {Running: false},
	}
	if err := saveListenerStatesAtomic(path, want); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := loadListenerStates(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d entries, want %d", len(got), len(want))
	}
	for name, w := range want {
		if got[name] != w {
			t.Errorf("state[%q] = %+v, want %+v", name, got[name], w)
		}
	}
}

func TestLoadListenerStatesMissingFileIsNotError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does_not_exist.json")
	states, err := loadListenerStates(path)
	if err != nil {
		t.Fatalf("expected no error for a missing file, got: %v", err)
	}
	if states != nil {
		t.Errorf("expected nil states for a missing file, got: %+v", states)
	}
}

func TestLoadSaveListenerStatesEmptyPathIsNoop(t *testing.T) {
	if err := saveListenerStatesAtomic("", map[string]ListenerState{"x": {}}); err != nil {
		t.Fatalf("save with empty path should no-op, got: %v", err)
	}
	states, err := loadListenerStates("")
	if err != nil || states != nil {
		t.Fatalf("load with empty path should return nil, nil, got: %+v, %v", states, err)
	}
}
