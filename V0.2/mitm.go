package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"
)

// CertAuthority signs per-host leaf certificates on the fly using a
// pre-existing intermediate CA cert/key (see [forward_proxy].ca_cert_path /
// ca_key_path in config.ini). It never generates its own CA — a locally
// trusted CA hierarchy is a deliberate, security-sensitive setup step the
// operator owns (e.g. the same intermediate already used for mitmproxy or
// Squid SSL-bump), not something this proxy should silently create.
type CertAuthority struct {
	cert    *x509.Certificate
	certDER []byte
	key     any // concrete type satisfies crypto.Signer, per tls.LoadX509KeyPair

	mu    sync.Mutex
	cache map[string]*tls.Certificate
}

// LoadCA loads an existing intermediate CA certificate and private key.
// Both files must already exist; this deliberately does not fall back to
// generating a new CA on a missing path; the CA's root must already be
// trusted by whatever agent tool is routed through this proxy, and
// silently substituting a fresh untrusted CA would just produce a
// confusing new class of TLS errors instead of a clear one.
func LoadCA(certPath, keyPath string) (*CertAuthority, error) {
	pair, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("loading CA cert/key (%s, %s): %w", certPath, keyPath, err)
	}
	leaf, err := x509.ParseCertificate(pair.Certificate[0])
	if err != nil {
		return nil, fmt.Errorf("parsing CA certificate %s: %w", certPath, err)
	}
	if !leaf.IsCA {
		return nil, fmt.Errorf("%s is not a CA certificate (basic constraints CA=false)", certPath)
	}
	return &CertAuthority{
		cert:    leaf,
		certDER: pair.Certificate[0],
		key:     pair.PrivateKey,
		cache:   make(map[string]*tls.Certificate),
	}, nil
}

// LeafCertFor returns a short-lived leaf certificate for host, signed by
// the loaded intermediate CA, generating and caching one on first use so
// repeat connections to the same vendor don't re-sign every time. The
// chain returned is [leaf, intermediate] — the root is deliberately
// omitted, since it's expected to already be trusted directly by the
// client (standard TLS chain-building only needs the intermediate here).
func (ca *CertAuthority) LeafCertFor(host string) (*tls.Certificate, error) {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	if cert, ok := ca.cache[host]; ok {
		return cert, nil
	}

	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generating leaf key for %s: %w", host, err)
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("generating serial for %s: %w", host, err)
	}

	template := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: host},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(90 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	if ip := net.ParseIP(host); ip != nil {
		template.IPAddresses = []net.IP{ip}
	} else {
		template.DNSNames = []string{host}
	}

	leafDER, err := x509.CreateCertificate(rand.Reader, template, ca.cert, &leafKey.PublicKey, ca.key)
	if err != nil {
		return nil, fmt.Errorf("signing leaf cert for %s: %w", host, err)
	}

	cert := &tls.Certificate{
		Certificate: [][]byte{leafDER, ca.certDER},
		PrivateKey:  leafKey,
	}
	ca.cache[host] = cert
	return cert, nil
}

// forwardProxyOverride carries the per-connection destination for a
// request that arrived through forward-proxy mode, where the upstream
// isn't a single fixed ProxyServer.upstreamBase the way every other mode
// uses, but whatever host the client's CONNECT actually targeted.
// authHeader is the client's own incoming Authorization header, forwarded
// as-is: forward-proxy mode has no configured ProviderConfig.APIKey for an
// arbitrary host, so unlike every other mode, the client's own credential
// (already correct, since it believes it's talking directly to the real
// vendor) is what must reach upstream.
type forwardProxyOverride struct {
	upstreamBase string
	authHeader   string
	// allowPassthrough mirrors ForwardProxyConfig.PassthroughHosts for this
	// specific intercepted host, decided once per CONNECT (see
	// ForwardProxyServer.ServeHTTP) and threaded down onto every request's
	// context so newForwardProxyPipelineHandler's default branch knows
	// whether to forward an unrecognized path verbatim instead of
	// rejecting it with 501.
	allowPassthrough bool
}

type forwardProxyCtxKey struct{}

func withForwardProxyOverride(r *http.Request, o forwardProxyOverride) *http.Request {
	return r.WithContext(context.WithValue(r.Context(), forwardProxyCtxKey{}, o))
}

// forwardProxyOverrideFromCtx is called from postUpstream/postUpstreamChatStreaming/
// handleModels, which only ever see a context.Context (already carrying
// whatever value withForwardProxyOverride attached to the originating
// request), not the *http.Request itself.
func forwardProxyOverrideFromCtx(ctx context.Context) (forwardProxyOverride, bool) {
	o, ok := ctx.Value(forwardProxyCtxKey{}).(forwardProxyOverride)
	return o, ok
}

// ForwardProxyServer handles CONNECT requests for forward-proxy mode: a
// host in allowedHosts gets TLS-terminated (decrypted, then served through
// proxyHandler — the same classify/inject/detect/retry pipeline every
// other mode uses) and re-forwarded to the real vendor; any other host
// gets an opaque byte-for-byte tunnel with no inspection, so pointing
// arbitrary traffic through this proxy never risks decrypting anything
// beyond the explicitly configured AI-vendor hostnames.
type ForwardProxyServer struct {
	ca               *CertAuthority
	allowedHosts     map[string]bool
	passthroughHosts map[string]bool
	proxyHandler     http.Handler
}

func NewForwardProxyServer(ca *CertAuthority, allowedHosts []string, passthroughHosts []string, proxyHandler http.Handler) *ForwardProxyServer {
	allowed := make(map[string]bool, len(allowedHosts))
	for _, h := range allowedHosts {
		allowed[strings.ToLower(strings.TrimSpace(h))] = true
	}
	passthrough := make(map[string]bool, len(passthroughHosts))
	for _, h := range passthroughHosts {
		passthrough[strings.ToLower(strings.TrimSpace(h))] = true
	}
	return &ForwardProxyServer{ca: ca, allowedHosts: allowed, passthroughHosts: passthrough, proxyHandler: proxyHandler}
}

// newForwardProxyPipelineHandler builds proxyHandler for forward-proxy
// mode — deliberately NOT p.Handler(), which every other mode uses.
// Handler()'s mux matches exact literal paths ("/v1/chat/completions",
// "/v1/models") because in every other mode the client is configured to
// point its base_url at THIS proxy's own address, so its outgoing request
// always hits exactly that registered pattern regardless of the real
// vendor's actual URL shape (p.upstreamBase, built from config.ini,
// separately carries whatever vendor-specific prefix is needed).
//
// Forward-proxy mode has no such registration: the client is configured
// with its REAL vendor's base_url and constructs its own request path
// relative to that — which can mount its OpenAI-compatible surface at any
// prefix (e.g. Cline's own gateway serves "/api/v1/chat/completions", not
// "/v1/..."). An exact-match mux would reject that outright. Matching by
// suffix instead, combined with handleClassified/handleModels forwarding
// the client's original r.URL.Path verbatim (see proxy.go) rather than a
// fixed constant, means any vendor's real path shape survives unchanged.
func newForwardProxyPipelineHandler(p *ProxyServer) http.Handler {
	chatHandler := p.handleClassified(kindChat)
	passthrough := p.wrapPassthroughWithActivity(newForwardProxyPassthroughProxy(p.client.Transport))
	return withCORS(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			chatHandler(w, r)
		case strings.HasSuffix(r.URL.Path, "/models"):
			p.handleModels(w, r)
		default:
			fc, _ := forwardProxyOverrideFromCtx(r.Context())
			if fc.allowPassthrough {
				passthrough.ServeHTTP(w, r)
				return
			}
			http.Error(w, fmt.Sprintf("forward-proxy: unsupported path %q (only paths ending in /chat/completions or /models are supported)", r.URL.Path), http.StatusNotImplemented)
		}
	}))
}

// wrapPassthroughWithActivity wraps a passthrough handler so a request that
// completely bypasses classify/inject/detect/retry (forward-proxy
// PassthroughHosts — e.g. Gemini's native API, which uses a different
// request/response shape this proxy can't parse at all) still logs a clear
// message explaining why, and still shows a live in-flight indicator on
// the dashboard instead of looking identical to "idle" for however long
// the request runs — there's no classification bucket or token count to
// report here, so the indicator is just a label and an elapsed timer (see
// ProgressEvent.Label), but that's enough to distinguish "genuinely
// working" from "stuck".
func (p *ProxyServer) wrapPassthroughWithActivity(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fc, _ := forwardProxyOverrideFromCtx(r.Context())
		host := strings.TrimPrefix(strings.TrimPrefix(fc.upstreamBase, "https://"), "http://")
		label := fmt.Sprintf("passthrough (%s)", host)
		log.Printf("forward-proxy passthrough: %s bypassed adaptive sampling (%s %s)", host, r.Method, r.URL.Path)

		start := time.Now()
		waitDone := make(chan struct{})
		tickerStopped := make(chan struct{})
		go func() {
			defer close(tickerStopped)
			ticker := time.NewTicker(progressEmitInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					p.emitProgress(ProgressEvent{
						Label:                  label,
						ElapsedMs:              time.Since(start).Milliseconds(),
						IdleTimeoutRemainingMs: -1,
					})
				case <-waitDone:
					return
				}
			}
		}()

		next.ServeHTTP(w, r)

		close(waitDone)
		<-tickerStopped // don't proceed until the ticker goroutine has fully exited, matching the same pattern in stream.go

		p.emitProgress(ProgressEvent{
			Label:                  label,
			ElapsedMs:              time.Since(start).Milliseconds(),
			IdleTimeoutRemainingMs: -1,
			Done:                   true,
		})
	})
}

// newForwardProxyPassthroughProxy builds a reverse proxy that forwards any
// request whose path didn't match /chat/completions or /models straight to
// the real intercepted host (ForwardProxyConfig.PassthroughHosts),
// unmodified — needed for api.cline.bot, whose Cline's-own-account-gateway
// auxiliary endpoints (token refresh, recommended-models, remote-config)
// would otherwise hit the 501 above and break Cline's own account/session
// handling. Unlike p.passthrough (a single fixed target, used by local
// mode), this reads the actual intercepted host per request from the
// forwardProxyOverride already attached to the request's context —
// forward-proxy mode serves many different intercepted hosts through one
// shared handler, so a fixed-target reverse proxy won't work here.
//
// transport is p.client.Transport, reused rather than left at
// httputil.ReverseProxy's http.DefaultTransport fallback, so this respects
// whatever TLS trust configuration the rest of the proxy already uses
// (e.g. a test's self-signed CA pool) instead of silently using a
// different, untrusted default.
func newForwardProxyPassthroughProxy(transport http.RoundTripper) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Transport: transport,
		Director: func(req *http.Request) {
			fc, _ := forwardProxyOverrideFromCtx(req.Context())
			target, err := url.Parse(fc.upstreamBase)
			if err != nil {
				log.Printf("forward-proxy passthrough: invalid upstream base %q: %v", fc.upstreamBase, err)
				return // req.URL left unchanged; the RoundTrip below fails cleanly
			}
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			// req.URL.Path is left untouched: it already carries the
			// client's own real request path, which forward-proxy mode
			// never rewrites (see newForwardProxyPipelineHandler above).
		},
	}
}

// ServeHTTP dispatches on two entirely different ways a client can use this
// proxy: a real HTTP CONNECT tunnel (system/OS-level proxy config — what
// Cline, Goose, Codex etc. use, handled below), or the "CORS-anywhere" style
// some browser-based tools use instead, where a single tab's fetch() can't be
// routed through a CONNECT proxy at all — the real target URL is embedded
// directly in the request path instead (see serveCORSProxy).
func (f *ForwardProxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodConnect {
		f.serveCORSProxy(w, r)
		return
	}

	host, _, err := net.SplitHostPort(r.Host)
	if err != nil {
		host = r.Host // no explicit port in the CONNECT target
	}
	host = strings.ToLower(host)

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijacking not supported", http.StatusInternalServerError)
		return
	}
	clientConn, _, err := hijacker.Hijack()
	if err != nil {
		log.Printf("forward-proxy: hijack failed for %s: %v", r.Host, err)
		return
	}
	defer clientConn.Close()

	if !f.allowedHosts[host] {
		f.tunnelPassthrough(clientConn, r.Host)
		return
	}

	if _, err := clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		log.Printf("forward-proxy: writing 200 to client failed for %s: %v", r.Host, err)
		return
	}

	tlsConn := tls.Server(clientConn, &tls.Config{
		GetCertificate: func(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return f.ca.LeafCertFor(hello.ServerName)
		},
		// serveDecrypted below only ever speaks HTTP/1.1 (net/http's
		// standard server machinery, no HTTP/2 framing implemented) — a
		// client offering ALPN ["h2", "http/1.1"] (real api.cline.bot
		// supports h2, and Node's fetch/undici negotiates it eagerly) needs
		// an explicit answer here, or Go's tls.Server leaves ALPN
		// unnegotiated entirely and such clients abort the connection
		// outright (observed as "connection reset by peer" mid-handshake)
		// rather than falling back to http/1.1 on their own.
		NextProtos: []string{"http/1.1"},
	})
	defer tlsConn.Close()

	if err := tlsConn.Handshake(); err != nil {
		log.Printf("forward-proxy: TLS handshake failed for %s: %v", host, err)
		return
	}

	// r.Host (not the port-stripped host) is what feeds the outbound
	// upstreamBase, so a non-standard port on the CONNECT target survives
	// (most HTTP client libraries send an explicit ":443" anyway, which is
	// harmless here). host (bare, no port) is only used for the allowlist
	// check above and SNI-based leaf cert lookup, both hostname-only by
	// nature — also used here for the passthrough-hosts lookup, same
	// reasoning (hostname-only, port-independent).
	f.serveDecrypted(tlsConn, r.Host, f.passthroughHosts[host])
}

// serveCORSProxy handles the "CORS-anywhere" request pattern: instead of a
// system/OS-level CONNECT proxy (which a single browser tab's fetch() calls
// can't be routed through), the real target URL is embedded directly in the
// request path — e.g. a browser page pointed at this proxy fetches
// "http://127.0.0.1:9100/https://openrouter.ai/api/v1/models" expecting the
// prefix stripped, the request forwarded, and CORS headers added to the
// response.
//
// This reuses f.proxyHandler — the exact same classify/inject/detect/retry
// pipeline (and its withCORS wrap, see newForwardProxyPipelineHandler) the
// CONNECT path uses — by rewriting the request to look like it arrived
// through serveDecrypted: r.URL becomes the embedded target and a
// forwardProxyOverride carries the real upstream base. No new CORS-header
// code is needed here at all; OPTIONS preflight and Access-Control-* headers
// on the response are already handled by that existing wrap.
//
// The embedded host is checked against the same allowedHosts list CONNECT
// uses — this must never become an open arbitrary-URL relay, even though
// it's bound to localhost, since that would silently widen forward-proxy
// mode's whole security model (only ever decrypting/forwarding to the
// explicitly configured AI-vendor hosts).
func (f *ForwardProxyServer) serveCORSProxy(w http.ResponseWriter, r *http.Request) {
	target, host, ok := parseEmbeddedTargetURL(r.URL)
	if !ok {
		http.Error(w, "forward-proxy: expected either an HTTP CONNECT request, or a CORS-proxy-style path embedding the full target URL (e.g. /https://host/path)", http.StatusBadRequest)
		return
	}
	if !f.allowedHosts[host] {
		http.Error(w, fmt.Sprintf("forward-proxy: host %q is not in allowed_hosts", host), http.StatusForbidden)
		return
	}

	r.URL = target
	r.Host = target.Host
	r = withForwardProxyOverride(r, forwardProxyOverride{
		upstreamBase:     target.Scheme + "://" + target.Host,
		authHeader:       r.Header.Get("Authorization"),
		allowPassthrough: f.passthroughHosts[host],
	})
	f.proxyHandler.ServeHTTP(w, r)
}

// parseEmbeddedTargetURL extracts an absolute URL embedded directly in a
// request path with its leading slash stripped — e.g. a request path of
// "/https://openrouter.ai/api/v1/models" yields the parsed URL
// https://openrouter.ai/api/v1/models and lowercased host "openrouter.ai".
// ok is false if the path doesn't start with a recognizable http(s)://
// scheme (i.e. this wasn't a CORS-proxy-style request at all).
func parseEmbeddedTargetURL(u *url.URL) (target *url.URL, host string, ok bool) {
	raw := strings.TrimPrefix(u.Path, "/")
	if u.RawQuery != "" {
		raw += "?" + u.RawQuery
	}
	if !strings.HasPrefix(raw, "http://") && !strings.HasPrefix(raw, "https://") {
		return nil, "", false
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" {
		return nil, "", false
	}
	return parsed, strings.ToLower(parsed.Hostname()), true
}

// tunnelPassthrough relays raw bytes both directions with zero
// inspection — used for any CONNECT target not in the configured
// allowlist, so traffic to hosts outside the explicit AI-vendor list this
// proxy is meant to intercept is never decrypted.
func (f *ForwardProxyServer) tunnelPassthrough(clientConn net.Conn, hostport string) {
	upstreamConn, err := net.DialTimeout("tcp", hostport, 10*time.Second)
	if err != nil {
		log.Printf("forward-proxy: dial %s failed: %v", hostport, err)
		return
	}
	defer upstreamConn.Close()

	if _, err := clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		return
	}

	done := make(chan struct{}, 2)
	go func() { io.Copy(upstreamConn, clientConn); done <- struct{}{} }()
	go func() { io.Copy(clientConn, upstreamConn); done <- struct{}{} }()
	<-done
}

// serveDecrypted runs the standard library's HTTP/1.1 server machinery
// (keep-alive, chunked transfer encoding, http.Flusher for SSE — all
// already correctly implemented there) against the now-decrypted
// connection, via a listener that yields exactly this one already-
// established net.Conn. Each request is tagged with a forwardProxyOverride
// pointing at the real vendor host before being handed to the shared
// proxyHandler (the same mux every other mode uses), so
// postUpstream/postUpstreamChatStreaming resolve the dynamic destination
// and forward the client's own Authorization header instead of a
// configured ProviderConfig.APIKey.
func (f *ForwardProxyServer) serveDecrypted(conn net.Conn, hostport string, allowPassthrough bool) {
	listener := newSingleConnListener()
	wrapped := &connCloseNotifier{Conn: conn, onClose: listener.Close}
	listener.conn = wrapped

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = withForwardProxyOverride(r, forwardProxyOverride{
			upstreamBase:     "https://" + hostport,
			authHeader:       r.Header.Get("Authorization"),
			allowPassthrough: allowPassthrough,
		})
		f.proxyHandler.ServeHTTP(w, r)
	})

	srv := &http.Server{Handler: handler}
	if err := srv.Serve(listener); err != nil && err != http.ErrServerClosed {
		log.Printf("forward-proxy: serving decrypted connection to %s: %v", hostport, err)
	}
}

// singleConnListener yields exactly one already-accepted net.Conn to an
// http.Server's Serve loop, then blocks until Close is called — letting
// http.Server's own request/response handling run against a connection
// this proxy already terminated TLS on via CONNECT, instead of hand-
// rolling HTTP/1.1 parsing and (critically) chunked-encoding/streaming
// response writing, which net/http already implements correctly.
type singleConnListener struct {
	conn   net.Conn
	once   sync.Once
	closed chan struct{}
}

func newSingleConnListener() *singleConnListener {
	return &singleConnListener{closed: make(chan struct{})}
}

func (l *singleConnListener) Accept() (net.Conn, error) {
	if l.conn != nil {
		c := l.conn
		l.conn = nil
		return c, nil
	}
	<-l.closed
	return nil, http.ErrServerClosed
}

func (l *singleConnListener) Close() error {
	l.once.Do(func() { close(l.closed) })
	return nil
}

func (l *singleConnListener) Addr() net.Addr {
	return &net.TCPAddr{}
}

// connCloseNotifier calls onClose exactly once when the wrapped
// connection is closed — used to tie singleConnListener's lifetime to the
// one real connection it serves: once http.Server is done with this
// connection (client disconnected, keep-alive timeout, ...) it closes the
// conn, which in turn unblocks the listener's second Accept() call so
// Serve() returns instead of leaking a goroutine parked forever.
type connCloseNotifier struct {
	net.Conn
	onClose func() error
	once    sync.Once
}

func (c *connCloseNotifier) Close() error {
	err := c.Conn.Close()
	c.once.Do(func() { c.onClose() })
	return err
}
