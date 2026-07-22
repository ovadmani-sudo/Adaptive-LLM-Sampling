package main

import "sync"

// Broker fans out the single UIEvent/ProgressEvent streams the ProxyServers
// produce to multiple independent consumers (the terminal dashboard AND every
// connected web-panel SSE client). A raw channel has exactly one receiver, so
// without this the TUI and the web panel would steal events from each other.
//
// ProxyServers write to EventsIn/ProgressIn (as their chan<- params); the
// broker's pump goroutines read those and copy each event to every current
// subscriber. Sends to subscribers are non-blocking — a slow/stalled consumer
// drops events rather than back-pressuring the proxy hot path.
type Broker struct {
	EventsIn   chan UIEvent
	ProgressIn chan ProgressEvent

	mu       sync.Mutex
	uiSubs   map[chan UIEvent]struct{}
	progSubs map[chan ProgressEvent]struct{}
}

func NewBroker() *Broker {
	b := &Broker{
		EventsIn:   make(chan UIEvent, 64),
		ProgressIn: make(chan ProgressEvent, 64),
		uiSubs:     make(map[chan UIEvent]struct{}),
		progSubs:   make(map[chan ProgressEvent]struct{}),
	}
	go b.pumpUI()
	go b.pumpProgress()
	return b
}

func (b *Broker) pumpUI() {
	for ev := range b.EventsIn {
		b.mu.Lock()
		for ch := range b.uiSubs {
			select {
			case ch <- ev:
			default:
			}
		}
		b.mu.Unlock()
	}
}

func (b *Broker) pumpProgress() {
	for ev := range b.ProgressIn {
		b.mu.Lock()
		for ch := range b.progSubs {
			select {
			case ch <- ev:
			default:
			}
		}
		b.mu.Unlock()
	}
}

// SubscribeUI returns a new UIEvent channel plus a cancel func that removes and
// closes it. Buffered so brief consumer stalls don't drop everything.
func (b *Broker) SubscribeUI() (<-chan UIEvent, func()) {
	ch := make(chan UIEvent, 64)
	b.mu.Lock()
	b.uiSubs[ch] = struct{}{}
	b.mu.Unlock()
	return ch, func() {
		b.mu.Lock()
		if _, ok := b.uiSubs[ch]; ok {
			delete(b.uiSubs, ch)
			close(ch)
		}
		b.mu.Unlock()
	}
}

// SubscribeProgress mirrors SubscribeUI for the progress stream.
func (b *Broker) SubscribeProgress() (<-chan ProgressEvent, func()) {
	ch := make(chan ProgressEvent, 64)
	b.mu.Lock()
	b.progSubs[ch] = struct{}{}
	b.mu.Unlock()
	return ch, func() {
		b.mu.Lock()
		if _, ok := b.progSubs[ch]; ok {
			delete(b.progSubs, ch)
			close(ch)
		}
		b.mu.Unlock()
	}
}
