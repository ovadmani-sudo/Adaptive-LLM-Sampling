package main

import (
	"bufio"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// generateTestCA creates a throwaway self-signed CA — standing in for the
// pre-existing intermediate CA cert/key LoadCA expects in production (e.g.
// one already set up for mitmproxy/Squid-style SSL bumping) — writes it to
// PEM files in a temp dir, and returns the paths plus a CertPool
// containing it for verifying certs it signs.
func generateTestCA(t *testing.T) (certPath, keyPath string, pool *x509.CertPool) {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generating test CA key: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "Test Local Proxy CA"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("creating test CA certificate: %v", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("marshaling test CA key: %v", err)
	}

	dir := t.TempDir()
	certPath = filepath.Join(dir, "ca.crt")
	keyPath = filepath.Join(dir, "ca.key")
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER}), 0644); err != nil {
		t.Fatalf("writing test CA cert: %v", err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), 0600); err != nil {
		t.Fatalf("writing test CA key: %v", err)
	}

	caCert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("parsing test CA certificate: %v", err)
	}
	pool = x509.NewCertPool()
	pool.AddCert(caCert)
	return certPath, keyPath, pool
}

// TestLoadCASignsValidLeafCert is the core correctness check for the
// CA/cert-signing machinery in isolation, no network involved: a leaf
// cert LeafCertFor issues for a given host must chain to (and be
// verifiable against) the CA that signed it, for that exact hostname.
func TestLoadCASignsValidLeafCert(t *testing.T) {
	certPath, keyPath, pool := generateTestCA(t)

	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	leaf, err := ca.LeafCertFor("example.com")
	if err != nil {
		t.Fatalf("LeafCertFor: %v", err)
	}

	leafCert, err := x509.ParseCertificate(leaf.Certificate[0])
	if err != nil {
		t.Fatalf("parsing signed leaf cert: %v", err)
	}

	if _, err := leafCert.Verify(x509.VerifyOptions{DNSName: "example.com", Roots: pool}); err != nil {
		t.Errorf("signed leaf cert does not verify against the CA that signed it: %v", err)
	}
}

// TestLoadCALeafCertCachedPerHost verifies repeat calls for the same host
// return the same cached certificate rather than re-signing every time —
// re-signing per connection would be wasteful and defeats the point of
// the cache documented on LeafCertFor.
func TestLoadCALeafCertCachedPerHost(t *testing.T) {
	certPath, keyPath, _ := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	first, err := ca.LeafCertFor("example.com")
	if err != nil {
		t.Fatalf("LeafCertFor (first): %v", err)
	}
	second, err := ca.LeafCertFor("example.com")
	if err != nil {
		t.Fatalf("LeafCertFor (second): %v", err)
	}
	if string(first.Certificate[0]) != string(second.Certificate[0]) {
		t.Error("expected the same cached leaf certificate on repeat calls for the same host, got two different certs")
	}
}

func TestLoadCARejectsNonCACertificate(t *testing.T) {
	// A leaf (non-CA) cert/key pair, self-signed for this test only.
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generating key: %v", err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "not-a-ca"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		IsCA:         false,
	}
	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("creating certificate: %v", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("marshaling key: %v", err)
	}

	dir := t.TempDir()
	certPath := filepath.Join(dir, "leaf.crt")
	keyPath := filepath.Join(dir, "leaf.key")
	os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER}), 0644)
	os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), 0600)

	if _, err := LoadCA(certPath, keyPath); err == nil {
		t.Error("expected LoadCA to reject a non-CA certificate, got nil error")
	}
}

// TestForwardProxyTunnelsNonAllowlistedHostWithoutInspection is the
// regression test for the security boundary the plan hinges on: a CONNECT
// target not in the configured allowlist must get a plain, opaque
// byte-for-byte tunnel — no TLS termination, no inspection — proven here
// by echoing arbitrary bytes through it unmodified.
func TestForwardProxyTunnelsNonAllowlistedHostWithoutInspection(t *testing.T) {
	certPath, keyPath, _ := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	echoLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for echo server: %v", err)
	}
	defer echoLn.Close()
	go func() {
		conn, err := echoLn.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		io.Copy(conn, conn)
	}()

	// api.anthropic.com is allowed; the echo server's own address is not —
	// this is the point of the test.
	fps := NewForwardProxyServer(ca, []string{"api.anthropic.com"}, nil, http.NotFoundHandler())
	proxyLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for forward proxy: %v", err)
	}
	defer proxyLn.Close()
	go http.Serve(proxyLn, fps)

	conn, err := net.Dial("tcp", proxyLn.Addr().String())
	if err != nil {
		t.Fatalf("dialing forward proxy: %v", err)
	}
	defer conn.Close()

	echoAddr := echoLn.Addr().String()
	fmt.Fprintf(conn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", echoAddr, echoAddr)

	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatalf("reading CONNECT response: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response status = %d, want 200", resp.StatusCode)
	}

	payload := "hello through the tunnel\n"
	if _, err := conn.Write([]byte(payload)); err != nil {
		t.Fatalf("writing payload: %v", err)
	}

	buf := make([]byte, len(payload))
	if _, err := io.ReadFull(br, buf); err != nil {
		t.Fatalf("reading echoed payload: %v", err)
	}
	if string(buf) != payload {
		t.Errorf("echoed payload = %q, want %q — a non-allowlisted tunnel must pass bytes through completely unmodified", buf, payload)
	}
}

// TestForwardProxyDecryptsAllowlistedHostAndForwardsRequest is the
// end-to-end vertical slice from the plan: CONNECT to an allowlisted
// host, TLS-terminate using the loaded CA, run the request through the
// exact same classify/inject/detect/retry pipeline every other mode uses
// (via proxy.Handler()), forward to a fake "vendor" whose cert chains to
// the same test CA, and confirm the response makes it back through the
// tunnel — while also confirming the client's own Authorization header
// (not any configured ProviderConfig.APIKey, since none exists for an
// arbitrary intercepted host) is what reaches the fake vendor.
func TestForwardProxyDecryptsAllowlistedHostAndForwardsRequest(t *testing.T) {
	certPath, keyPath, pool := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	// "localhost" (not a bare IP) so Go's TLS client actually sends SNI —
	// crypto/tls omits the SNI extension entirely for literal IP
	// ServerNames, which would break GetCertificate's per-host lookup.
	// "localhost" also reliably resolves to 127.0.0.1 via the OS resolver
	// with no /etc/hosts changes needed, so the proxy's own outbound
	// dial (to whatever port the fake vendor is actually listening on)
	// lands in the right place.
	const vendorHost = "localhost"

	var receivedAuth string
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(sseChatStream("hello from fake vendor", "", "stop", 3))
	}))
	leafCert, err := ca.LeafCertFor(vendorHost)
	if err != nil {
		t.Fatalf("LeafCertFor for fake vendor: %v", err)
	}
	upstream.TLS = &tls.Config{Certificates: []tls.Certificate{*leafCert}}
	upstream.StartTLS()
	defer upstream.Close()

	upstreamURL, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parsing upstream URL: %v", err)
	}
	hostport := vendorHost + ":" + upstreamURL.Port()

	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy-test", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	// Same-package test: directly trust our test CA for the proxy's own
	// outbound connection to the fake vendor, exactly the way it would
	// trust a real vendor's cert chaining to a real public CA — avoids
	// relying on process-global, version-sensitive mechanisms like
	// SSL_CERT_FILE for something this test-local.
	proxy.client = &http.Client{
		Timeout:   time.Duration(cfg.Server.RequestTimeoutSeconds) * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}},
	}

	fps := NewForwardProxyServer(ca, []string{vendorHost}, nil, proxy.Handler())
	proxyLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for forward proxy: %v", err)
	}
	defer proxyLn.Close()
	go http.Serve(proxyLn, fps)

	// --- test client: CONNECT, then its own TLS handshake trusting the
	// same test CA (standing in for the operator's OS/tool trust store) ---
	rawConn, err := net.Dial("tcp", proxyLn.Addr().String())
	if err != nil {
		t.Fatalf("dialing forward proxy: %v", err)
	}
	defer rawConn.Close()

	fmt.Fprintf(rawConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", hostport, hostport)
	br := bufio.NewReader(rawConn)
	connectResp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatalf("reading CONNECT response: %v", err)
	}
	if connectResp.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response status = %d, want 200", connectResp.StatusCode)
	}

	tlsConn := tls.Client(rawConn, &tls.Config{RootCAs: pool, ServerName: vendorHost})
	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("client TLS handshake: %v", err)
	}
	defer tlsConn.Close()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})
	req, err := http.NewRequest(http.MethodPost, "https://"+hostport+"/v1/chat/completions", strings.NewReader(string(reqBody)))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer real-client-key")
	if err := req.Write(tlsConn); err != nil {
		t.Fatalf("writing request over TLS tunnel: %v", err)
	}

	finalResp, err := http.ReadResponse(bufio.NewReader(tlsConn), req)
	if err != nil {
		t.Fatalf("reading response over TLS tunnel: %v", err)
	}
	defer finalResp.Body.Close()

	respBytes, err := io.ReadAll(finalResp.Body)
	if err != nil {
		t.Fatalf("reading response body: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		t.Fatalf("response is not valid JSON: %v, body: %s", err, respBytes)
	}
	choices, _ := parsed["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected exactly 1 choice, got: %v", parsed["choices"])
	}
	choice0 := choices[0].(map[string]interface{})
	message, _ := choice0["message"].(map[string]interface{})
	if message["content"] != "hello from fake vendor" {
		t.Errorf("content = %v, want %q — request must survive CONNECT+TLS-terminate+classify/detect/retry+re-forward intact", message["content"], "hello from fake vendor")
	}

	if receivedAuth != "Bearer real-client-key" {
		t.Errorf("upstream received Authorization = %q, want %q — the client's own credential must be forwarded unchanged in forward-proxy mode, since there's no configured ProviderConfig.APIKey for an arbitrary intercepted host", receivedAuth, "Bearer real-client-key")
	}
}

// TestForwardProxyNegotiatesHTTP1WhenClientOffersH2 is the regression test
// for a confirmed real-world failure: real api.cline.bot supports HTTP/2,
// and Node's fetch/undici (what Cline's own CLI uses) offers ALPN
// ["h2", "http/1.1"] accordingly. serveDecrypted only ever speaks HTTP/1.1
// (net/http's standard server, no HTTP/2 framing implemented) — without an
// explicit NextProtos on the MITM TLS server, Go leaves ALPN entirely
// unnegotiated, and such a client aborts the connection outright (observed
// as "connection reset by peer" mid-handshake) instead of falling back to
// http/1.1 on its own. This proves the server now answers with "http/1.1"
// even when the client offers "h2" first.
func TestForwardProxyNegotiatesHTTP1WhenClientOffersH2(t *testing.T) {
	certPath, keyPath, pool := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	const vendorHost = "localhost"
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy-test", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}

	fps := NewForwardProxyServer(ca, []string{vendorHost}, nil, proxy.Handler())
	proxyLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for forward proxy: %v", err)
	}
	defer proxyLn.Close()
	go http.Serve(proxyLn, fps)

	rawConn, err := net.Dial("tcp", proxyLn.Addr().String())
	if err != nil {
		t.Fatalf("dialing forward proxy: %v", err)
	}
	defer rawConn.Close()

	hostport := vendorHost + ":443"
	fmt.Fprintf(rawConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", hostport, hostport)
	br := bufio.NewReader(rawConn)
	connectResp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatalf("reading CONNECT response: %v", err)
	}
	if connectResp.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response status = %d, want 200", connectResp.StatusCode)
	}

	// Mirrors what Node's fetch/undici actually sends: h2 offered first,
	// http/1.1 as fallback.
	tlsConn := tls.Client(rawConn, &tls.Config{
		RootCAs:    pool,
		ServerName: vendorHost,
		NextProtos: []string{"h2", "http/1.1"},
	})
	defer tlsConn.Close()

	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("client TLS handshake failed (this is the exact failure mode seen in production — a client offering h2 got no ALPN answer and the handshake was aborted): %v", err)
	}

	if got := tlsConn.ConnectionState().NegotiatedProtocol; got != "http/1.1" {
		t.Errorf("negotiated ALPN protocol = %q, want %q — serveDecrypted can only speak HTTP/1.1, so the server must not silently agree to (or fail to negotiate) anything else", got, "http/1.1")
	}
}

// TestForwardProxyPreservesVendorSpecificPathPrefix is the regression test
// for a confirmed real-world failure: Cline's own gateway (api.cline.bot)
// mounts its OpenAI-compatible surface at "/api/v1/chat/completions", not
// "/v1/chat/completions" — unlike every vendor the reverse-proxy
// [provider.*] mode was verified against. Forward-proxy mode has no
// per-vendor base_url/path configuration at all (that's the point), so
// the only correct source for the real path is the client's own original
// request — this test proves that prefix survives end-to-end instead of
// being silently replaced by a fixed "/v1/..." assumption baked in for
// the other modes.
func TestForwardProxyPreservesVendorSpecificPathPrefix(t *testing.T) {
	certPath, keyPath, pool := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	const vendorHost = "localhost"
	const vendorPath = "/api/v1/chat/completions" // Cline-gateway-shaped, not "/v1/..."

	var receivedPath string
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(sseChatStream("hello from vendor-specific path", "", "stop", 3))
	}))
	leafCert, err := ca.LeafCertFor(vendorHost)
	if err != nil {
		t.Fatalf("LeafCertFor: %v", err)
	}
	upstream.TLS = &tls.Config{Certificates: []tls.Certificate{*leafCert}}
	upstream.StartTLS()
	defer upstream.Close()

	upstreamURL, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parsing upstream URL: %v", err)
	}
	hostport := vendorHost + ":" + upstreamURL.Port()

	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy-test", make(chan UIEvent, 8), nil, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.client = &http.Client{
		Timeout:   time.Duration(cfg.Server.RequestTimeoutSeconds) * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}},
	}

	fps := NewForwardProxyServer(ca, []string{vendorHost}, nil, newForwardProxyPipelineHandler(proxy))
	proxyLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for forward proxy: %v", err)
	}
	defer proxyLn.Close()
	go http.Serve(proxyLn, fps)

	rawConn, err := net.Dial("tcp", proxyLn.Addr().String())
	if err != nil {
		t.Fatalf("dialing forward proxy: %v", err)
	}
	defer rawConn.Close()

	fmt.Fprintf(rawConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", hostport, hostport)
	br := bufio.NewReader(rawConn)
	connectResp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatalf("reading CONNECT response: %v", err)
	}
	if connectResp.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response status = %d, want 200", connectResp.StatusCode)
	}

	tlsConn := tls.Client(rawConn, &tls.Config{RootCAs: pool, ServerName: vendorHost})
	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("client TLS handshake: %v", err)
	}
	defer tlsConn.Close()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})
	req, err := http.NewRequest(http.MethodPost, "https://"+hostport+vendorPath, strings.NewReader(string(reqBody)))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if err := req.Write(tlsConn); err != nil {
		t.Fatalf("writing request over TLS tunnel: %v", err)
	}

	finalResp, err := http.ReadResponse(bufio.NewReader(tlsConn), req)
	if err != nil {
		t.Fatalf("reading response over TLS tunnel: %v", err)
	}
	defer finalResp.Body.Close()
	io.Copy(io.Discard, finalResp.Body)

	if finalResp.StatusCode != http.StatusOK {
		t.Fatalf("response status = %d, want 200 (got a %d likely means the vendor-specific path was rejected instead of forwarded)", finalResp.StatusCode, finalResp.StatusCode)
	}
	if receivedPath != vendorPath {
		t.Errorf("vendor received path = %q, want %q — the client's real vendor path must survive unchanged, not get replaced with a fixed /v1/chat/completions assumption", receivedPath, vendorPath)
	}
}

// forwardProxyPassthroughHarness sets up a CONNECT-and-TLS-terminate
// forward-proxy tunnel to a fake vendor server, mirroring
// TestForwardProxyPreservesVendorSpecificPathPrefix's setup exactly, and
// sends one GET request for auxiliaryPath over it — used by both the
// clinepass-style "aux endpoint gets forwarded" test and its
// passthrough-disabled counterpart below.
func forwardProxyPassthroughHarness(t *testing.T, passthroughHosts []string, auxiliaryPath string) (status int, receivedPath string, progressEvents []ProgressEvent) {
	t.Helper()
	certPath, keyPath, pool := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}

	const vendorHost = "localhost"

	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))
	leafCert, err := ca.LeafCertFor(vendorHost)
	if err != nil {
		t.Fatalf("LeafCertFor: %v", err)
	}
	upstream.TLS = &tls.Config{Certificates: []tls.Certificate{*leafCert}}
	upstream.StartTLS()
	defer upstream.Close()

	upstreamURL, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parsing upstream URL: %v", err)
	}
	hostport := vendorHost + ":" + upstreamURL.Port()

	cfg := testConfig()
	progressCh := make(chan ProgressEvent, 32)
	proxy, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy-test", make(chan UIEvent, 8), progressCh, "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	proxy.client = &http.Client{
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}},
	}

	fps := NewForwardProxyServer(ca, []string{vendorHost}, passthroughHosts, newForwardProxyPipelineHandler(proxy))
	proxyLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listening for forward proxy: %v", err)
	}
	defer proxyLn.Close()
	go http.Serve(proxyLn, fps)

	rawConn, err := net.Dial("tcp", proxyLn.Addr().String())
	if err != nil {
		t.Fatalf("dialing forward proxy: %v", err)
	}
	defer rawConn.Close()

	fmt.Fprintf(rawConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", hostport, hostport)
	br := bufio.NewReader(rawConn)
	connectResp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatalf("reading CONNECT response: %v", err)
	}
	if connectResp.StatusCode != http.StatusOK {
		t.Fatalf("CONNECT response status = %d, want 200", connectResp.StatusCode)
	}

	tlsConn := tls.Client(rawConn, &tls.Config{RootCAs: pool, ServerName: vendorHost})
	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("client TLS handshake: %v", err)
	}
	defer tlsConn.Close()

	req, err := http.NewRequest(http.MethodGet, "https://"+hostport+auxiliaryPath, nil)
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	if err := req.Write(tlsConn); err != nil {
		t.Fatalf("writing request over TLS tunnel: %v", err)
	}

	finalResp, err := http.ReadResponse(bufio.NewReader(tlsConn), req)
	if err != nil {
		t.Fatalf("reading response over TLS tunnel: %v", err)
	}
	defer finalResp.Body.Close()
	io.Copy(io.Discard, finalResp.Body)

	// Collect whatever progress events arrive, up to and including the
	// final Done:true one — the client having fully read the response
	// doesn't guarantee the server-side handler has finished its own
	// post-ServeHTTP cleanup (the Done emission), so this waits rather
	// than assuming everything is already buffered.
collecting:
	for {
		select {
		case ev := <-progressCh:
			progressEvents = append(progressEvents, ev)
			if ev.Done {
				break collecting
			}
		case <-time.After(300 * time.Millisecond):
			break collecting
		}
	}

	return finalResp.StatusCode, receivedPath, progressEvents
}

// TestForwardProxyPassthroughForwardsUnrecognizedPath is the regression
// test for the confirmed clinepass bug: "Error: Token refresh failed: 501"
// happened because forward-proxy mode's dispatcher only recognizes paths
// ending in /chat/completions or /models — every other path (token
// refresh, recommended-models, remote-config, all real endpoints Cline's
// own account gateway calls) got rejected with 501, even though the
// earlier reverse-proxy-mode fix (ProviderConfig.AllowPassthrough) doesn't
// apply here at all, since forward-proxy mode never goes through
// p.Handler(). With the intercepted host in passthroughHosts, an
// unrecognized path must reach the real vendor verbatim instead.
func TestForwardProxyPassthroughForwardsUnrecognizedPath(t *testing.T) {
	const auxiliaryPath = "/api/v1/users/me/token/refresh"
	status, receivedPath, _ := forwardProxyPassthroughHarness(t, []string{"localhost"}, auxiliaryPath)

	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200 (path should be forwarded to the real vendor, not rejected with 501)", status)
	}
	if receivedPath != auxiliaryPath {
		t.Errorf("vendor received path = %q, want %q", receivedPath, auxiliaryPath)
	}
}

// TestForwardProxyPassthroughEmitsActivityIndicator verifies a passthrough
// request still produces a progress event carrying a non-empty Label (so
// the dashboard shows something other than "idle" while it's in flight,
// even though the request completely bypasses classify/inject/detect/
// retry) and a final Done:true event once it completes — the same
// mechanism postUpstreamChatStreaming's Done-event fix relies on, applied
// here so a passthrough request can't leave the in-flight indicator stuck.
func TestForwardProxyPassthroughEmitsActivityIndicator(t *testing.T) {
	_, _, events := forwardProxyPassthroughHarness(t, []string{"localhost"}, "/api/v1/users/me/token/refresh")

	if len(events) == 0 {
		t.Fatal("expected at least one progress event for a passthrough request, got none")
	}
	last := events[len(events)-1]
	if !last.Done {
		t.Error("expected the final progress event to have Done=true")
	}
	for _, ev := range events {
		if ev.Label == "" {
			t.Errorf("expected every passthrough progress event to carry a non-empty Label, got: %+v", ev)
		}
	}
}

// TestForwardProxyWithoutPassthroughStillRejectsUnrecognizedPath verifies
// the default (host not in passthroughHosts) is completely unaffected by
// this feature — every other allowed host must keep rejecting unrecognized
// paths with 501 exactly as before, since an unexpected path against a
// pure vendor API is more likely a real misconfiguration worth surfacing.
func TestForwardProxyWithoutPassthroughStillRejectsUnrecognizedPath(t *testing.T) {
	status, _, _ := forwardProxyPassthroughHarness(t, nil, "/api/v1/users/me/token/refresh")

	if status != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501 (passthroughHosts is empty; unrecognized paths must still be rejected)", status)
	}
}

// newCORSProxyTestServer wires up a ForwardProxyServer (served over plain
// HTTP — no CONNECT/TLS involved in this request style at all) plus a fake
// vendor whose host is in allowedHosts, for testing the "CORS-anywhere"
// request pattern (see ForwardProxyServer.serveCORSProxy): the real target
// URL embedded directly in the request path instead of a CONNECT tunnel.
func newCORSProxyTestServer(t *testing.T, vendor *httptest.Server, allowedHosts []string) *httptest.Server {
	t.Helper()
	certPath, keyPath, _ := generateTestCA(t)
	ca, err := LoadCA(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadCA: %v", err)
	}
	cfg := testConfig()
	proxy, err := NewProxyServer(cfg, &ProviderConfig{}, "forward-proxy-test", make(chan UIEvent, 8), make(chan ProgressEvent, 32), "")
	if err != nil {
		t.Fatalf("NewProxyServer: %v", err)
	}
	fps := NewForwardProxyServer(ca, allowedHosts, nil, newForwardProxyPipelineHandler(proxy))
	srv := httptest.NewServer(fps)
	t.Cleanup(srv.Close)
	return srv
}

// TestForwardProxyCORSStyleForwardsModelsRequest verifies a plain GET whose
// path embeds the real target URL (e.g. /https://host/v1/models) is forwarded
// to that host and the response carries the CORS headers a browser's
// preflight check requires — reusing f.proxyHandler's existing withCORS
// wrap, the same one the CONNECT path already relies on.
func TestForwardProxyCORSStyleForwardsModelsRequest(t *testing.T) {
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[{"id":"test-model"}]}`))
	}))
	defer vendor.Close()
	vendorURL, _ := url.Parse(vendor.URL)

	fps := newCORSProxyTestServer(t, vendor, []string{vendorURL.Hostname()})

	resp, err := http.Get(fps.URL + "/" + vendor.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Access-Control-Allow-Origin = %q, want \"*\"", got)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "test-model") {
		t.Errorf("expected the vendor's real response forwarded, got: %s", body)
	}
}

// TestForwardProxyCORSStyleHandlesPreflight verifies an OPTIONS preflight
// (what a browser sends before the real cross-origin request) gets a 204
// with CORS headers and never reaches the vendor at all.
func TestForwardProxyCORSStyleHandlesPreflight(t *testing.T) {
	var vendorHit bool
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vendorHit = true
	}))
	defer vendor.Close()
	vendorURL, _ := url.Parse(vendor.URL)

	fps := newCORSProxyTestServer(t, vendor, []string{vendorURL.Hostname()})

	req, _ := http.NewRequest(http.MethodOptions, fps.URL+"/"+vendor.URL+"/v1/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("OPTIONS: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("status = %d, want 204", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Access-Control-Allow-Origin = %q, want \"*\"", got)
	}
	if vendorHit {
		t.Error("preflight OPTIONS must never reach the vendor")
	}
}

// TestForwardProxyCORSStyleReflectsRequestedHeaders is the regression test
// for a confirmed real-world failure: a client sent a vendor-specific header
// (OpenRouter's optional X-Title attribution header) that a fixed
// Access-Control-Allow-Headers allowlist didn't include, so the browser
// blocked the real request even though the preflight itself got a 204 —
// "Request header field x-title is not allowed by
// Access-Control-Allow-Headers in preflight response". withCORS must
// reflect whatever the browser's own Access-Control-Request-Headers asked
// for, not a static list, so any client-chosen header is automatically
// permitted.
func TestForwardProxyCORSStyleReflectsRequestedHeaders(t *testing.T) {
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer vendor.Close()
	vendorURL, _ := url.Parse(vendor.URL)

	fps := newCORSProxyTestServer(t, vendor, []string{vendorURL.Hostname()})

	req, _ := http.NewRequest(http.MethodOptions, fps.URL+"/"+vendor.URL+"/v1/chat/completions", nil)
	req.Header.Set("Access-Control-Request-Headers", "content-type, authorization, x-title, http-referer")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("OPTIONS: %v", err)
	}
	defer resp.Body.Close()

	got := resp.Header.Get("Access-Control-Allow-Headers")
	for _, want := range []string{"x-title", "http-referer", "authorization"} {
		if !strings.Contains(strings.ToLower(got), want) {
			t.Errorf("Access-Control-Allow-Headers = %q, missing requested header %q", got, want)
		}
	}
}

// TestForwardProxyCORSStyleForwardsChatCompletion verifies a POST to a
// /chat/completions-suffixed embedded URL goes through the full
// classify/inject/detect/retry pipeline (chatHandler) exactly like the
// CONNECT path, including forwarding the client's own Authorization header
// (forward-proxy mode has no configured ProviderConfig.APIKey for an
// arbitrary embedded host).
func TestForwardProxyCORSStyleForwardsChatCompletion(t *testing.T) {
	var receivedAuth string
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(sseChatStream("hello from vendor", "", "stop", 3))
	}))
	defer vendor.Close()
	vendorURL, _ := url.Parse(vendor.URL)

	fps := newCORSProxyTestServer(t, vendor, []string{vendorURL.Hostname()})

	reqBody, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": "fix bug"}},
	})
	req, err := http.NewRequest(http.MethodPost, fps.URL+"/"+vendor.URL+"/v1/chat/completions", strings.NewReader(string(reqBody)))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer browser-side-key")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Access-Control-Allow-Origin = %q, want \"*\"", got)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "hello from vendor") {
		t.Errorf("expected the vendor's content in the response, got: %s", body)
	}
	if receivedAuth != "Bearer browser-side-key" {
		t.Errorf("vendor received Authorization = %q, want the client's own header forwarded", receivedAuth)
	}
}

// TestForwardProxyCORSStyleRejectsDisallowedHost verifies the embedded-URL
// host is checked against the same allowedHosts list CONNECT uses — this
// must never become an open arbitrary-URL relay, even bound to localhost.
func TestForwardProxyCORSStyleRejectsDisallowedHost(t *testing.T) {
	var vendorHit bool
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vendorHit = true
	}))
	defer vendor.Close()

	// allowedHosts deliberately does NOT include the vendor's host.
	fps := newCORSProxyTestServer(t, vendor, []string{"some-other-allowed-host.example"})

	resp, err := http.Get(fps.URL + "/" + vendor.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("status = %d, want 403 (host not in allowed_hosts)", resp.StatusCode)
	}
	if vendorHit {
		t.Error("a disallowed host must never be reached")
	}
}

// TestForwardProxyCORSStyleRejectsNonURLPath verifies a plain non-CONNECT
// request whose path doesn't embed a recognizable http(s):// target URL gets
// a clear 400 rather than being silently misrouted.
func TestForwardProxyCORSStyleRejectsNonURLPath(t *testing.T) {
	fps := newCORSProxyTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})), []string{"example.com"})

	resp, err := http.Get(fps.URL + "/just/a/plain/path")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 (no embedded target URL and not a CONNECT request)", resp.StatusCode)
	}
}
