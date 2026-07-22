# Local HTTPS Interception CA

This directory contains the recipe for a local CA used by `llama-dyn-proxy`'s
forward-proxy mode (see `[forward_proxy]` in `config.ini`) to terminate TLS
for allowlisted hosts. **Only `generate_certs.sh` and this README are
tracked in git** — everything else here (`openssl.cnf`, `root.key`,
`root.crt`, `intermediate.key`, `intermediate.crt`, `ca-chain.pem`,
`root.srl`) is generated output specific to your machine, is gitignored,
and must never be committed — `root.key`/`intermediate.key` are real
private keys.

## 1. Generate your own local CA

Run this once, from inside this directory:

```bash
cd cert
./generate_certs.sh
```

This produces `intermediate.key`/`intermediate.crt` (what `config.ini`'s
`ca_cert_path`/`ca_key_path` should point at) plus `root.key`/`root.crt`
(the offline root that signed the intermediate — its private key is only
needed if you ever need to re-sign a new intermediate later, not for
day-to-day use).

## 2. Trust the Root CA on this host

For your browser or CLI tools (`curl`, `wget`, etc.) to trust the proxy's
intercepted traffic, add the Root CA (`root.crt`) to the system's trust
store. Node-based tools (Cline's CLI, VS Code extensions, etc.) usually
need `NODE_EXTRA_CA_CERTS` pointed at it instead/as well — see the main
README's forward-proxy section for that caveat.

### Linux (Ubuntu/Debian)
```bash
sudo cp root.crt /usr/local/share/ca-certificates/local-proxy-root.crt
sudo update-ca-certificates
```

### Linux (Fedora/RHEL/CentOS)
```bash
sudo cp root.crt /etc/pki/ca-trust/source/anchors/local-proxy-root.crt
sudo update-ca-trust
```

### Browsers (Firefox / Chrome)
Browsers often maintain their own certificate store.
- **Firefox:** Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import... → select `root.crt` → check "Trust this CA to identify websites".
- **Chrome/Chromium:** Settings → Privacy & Security → Security → Manage certificates → Authorities → Import → select `root.crt` → check "Trust this CA to identify websites".

## 3. Configuring other HTTPS interception proxies

`llama-dyn-proxy` itself only needs `ca_cert_path`/`ca_key_path` pointed at
`intermediate.crt`/`intermediate.key` (see `config.ini`). If you also want
to point a *different* interception tool at this same CA:

### mitmproxy
```bash
mitmproxy --certs *=ca-chain.pem --set confdir=.
```
Or combine the intermediate key and chain into the single PEM file
`mitmproxy` expects:
```bash
cat intermediate.key ca-chain.pem > mitmproxy-ca.pem
mitmproxy --certs *=mitmproxy-ca.pem
```

### Squid Proxy (with SSL Bump)
```conf
http_port 3128 ssl-bump cert=/absolute/path/to/ca-chain.pem key=/absolute/path/to/intermediate.key generate-host-certificates=on dynamic_cert_mem_cache_size=4MB
ssl_bump server-first all
```
(Squid needs absolute paths in `squid.conf`, and the `squid` system user
needs read access to both files.)
