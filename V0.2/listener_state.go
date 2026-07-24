package main

import (
	"encoding/json"
	"os"
)

// ListenerState is one listener's persisted control-panel switches — the
// same live toggles ListenerStatus reports, minus the read-only fields
// (LastErr, Kind, Port, SupportsBypass) that aren't user-set state. Saved
// on every change (see Supervisor.persist) and reapplied on the next
// startup (see Supervisor.EnablePersistence) so restarting the process
// doesn't silently reset every agent's mode/vision/system-prompt/running
// state back to config.ini's defaults.
type ListenerState struct {
	Running        bool   `json:"running"`
	BypassSampling bool   `json:"bypass_sampling"`
	Alert          bool   `json:"alert"`
	Model          string `json:"model"`
	ForcedBucket   string `json:"forced_bucket"`
	VisionDescribe bool   `json:"vision_describe"`
	SystemPrompt   string `json:"system_prompt"`
}

// loadListenerStates reads a previously-persisted state file, keyed by
// listener name. A missing file is not an error (first run, or
// persistence disabled) — returns a nil map.
func loadListenerStates(path string) (map[string]ListenerState, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	var states map[string]ListenerState
	if err := json.Unmarshal(data, &states); err != nil {
		return nil, err
	}
	return states, nil
}

// saveListenerStatesAtomic writes the full state map to path via a
// temp-file-then-rename, so a process killed mid-write can never leave a
// half-written, corrupt state file behind for the next load to choke on.
func saveListenerStatesAtomic(path string, states map[string]ListenerState) error {
	if path == "" {
		return nil
	}
	data, err := json.MarshalIndent(states, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
