#!/usr/bin/env bash
set -euo pipefail

# Configuration file name
CONFIG_FILE="openssl.cnf"
ROOT_KEY="root.key"
ROOT_CRT="root.crt"
INT_KEY="intermediate.key"
INT_CRT="intermediate.crt"
INT_CSR="intermediate.csr"
CHAIN_PEM="ca-chain.pem"

echo "==> 0. Creating OpenSSL configuration file (${CONFIG_FILE})..."
cat << 'EOF' > "$CONFIG_FILE"
[ req ]
default_bits        = 4096
default_md          = sha256
distinguished_name  = req_distinguished_name
string_mask         = utf8only
prompt              = no

[ req_distinguished_name ]
C            = US
ST           = California
L            = San Francisco
O            = Local Proxy Organization
OU           = Interception Authority
CN           = Local Proxy Root CA

# Extensions for Root CA
[ v3_root_ca ]
subjectKeyIdentifier    = hash
authorityKeyIdentifier  = keyid:always,issuer
basicConstraints        = critical, CA:true
keyUsage                = critical, digitalSignature, cRLSign, keyCertSign

# Extensions for Intermediate CA
[ v3_intermediate_ca ]
subjectKeyIdentifier    = hash
authorityKeyIdentifier  = keyid:always,issuer
basicConstraints        = critical, CA:true, pathlen:0
keyUsage                = critical, digitalSignature, cRLSign, keyCertSign
EOF

echo "==> 1. Generating Root CA Private Key..."
openssl genrsa -out "$ROOT_KEY" 4096

echo "==> 2. Generating Self-signed Root CA Certificate..."
openssl req -new -x509 \
  -key "$ROOT_KEY" \
  -sha256 \
  -days 3650 \
  -config "$CONFIG_FILE" \
  -extensions v3_root_ca \
  -subj "/C=US/ST=California/L=San Francisco/O=Local Proxy Org/OU=Root CA/CN=Local Proxy Root CA" \
  -out "$ROOT_CRT"

echo "==> 3. Generating Intermediate CA Private Key..."
openssl genrsa -out "$INT_KEY" 4096

echo "==> 4. Generating Certificate Signing Request (CSR) for Intermediate CA..."
openssl req -new \
  -key "$INT_KEY" \
  -sha256 \
  -config "$CONFIG_FILE" \
  -subj "/C=US/ST=California/L=San Francisco/O=Local Proxy Org/OU=Intermediate CA/CN=Local Proxy Intermediate CA" \
  -out "$INT_CSR"

echo "==> 5. Signing Intermediate CA with Root CA..."
openssl x509 -req \
  -in "$INT_CSR" \
  -CA "$ROOT_CRT" \
  -CAkey "$ROOT_KEY" \
  -CAcreateserial \
  -days 1825 \
  -sha256 \
  -extfile "$CONFIG_FILE" \
  -extensions v3_intermediate_ca \
  -out "$INT_CRT"

echo "==> 6. Creating Certificate Chain (Intermediate + Root CA)..."
cat "$INT_CRT" "$ROOT_CRT" > "$CHAIN_PEM"

echo "==> 7. Securing private keys (chmod 600)..."
chmod 600 "$ROOT_KEY" "$INT_KEY"

echo "==> 8. Verifying the Intermediate Certificate..."
openssl verify -CAfile "$ROOT_CRT" "$INT_CRT"

echo "==> 9. Cleaning up temporary files..."
rm -f "$INT_CSR"

echo ""
echo "=========================================================================="
echo " Certificate Generation Completed Successfully!"
echo "=========================================================================="
echo "Files generated in $(pwd):"
echo "  - $ROOT_KEY       : Root CA Private Key (Secure, chmod 600)"
echo "  - $ROOT_CRT       : Root CA Public Certificate"
echo "  - $INT_KEY        : Intermediate CA Private Key (Secure, chmod 600)"
echo "  - $INT_CRT        : Intermediate CA Certificate"
echo "  - $CHAIN_PEM      : Certificate Chain (Intermediate + Root CA)"
echo "=========================================================================="
echo "Next Steps to trust this certificate on your local host:"
echo ""
echo "For Debian/Ubuntu:"
echo "  sudo cp $ROOT_CRT /usr/local/share/ca-certificates/local-proxy-root.crt"
echo "  sudo update-ca-certificates"
echo ""
echo "For Fedora/RHEL/CentOS:"
echo "  sudo cp $ROOT_CRT /etc/pki/ca-trust/source/anchors/local-proxy-root.crt"
echo "  sudo update-ca-trust"
echo "=========================================================================="
