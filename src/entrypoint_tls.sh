#!/bin/sh
# Generate a fresh self-signed certificate on every container start, then run
# uvicorn with TLS. The key never leaves the container's filesystem and is
# regenerated on every task launch — nothing to rotate or store.
#
# TPAI HIPAA M1.2 (§164.312(e)): this encrypts the ALB→container hop. The ALB
# does not validate target certificates by design, so self-signed is the
# standard pattern for this hop.
set -eu

CERT_DIR=/tmp/tls
mkdir -p "$CERT_DIR"

openssl req -x509 -newkey rsa:2048 -nodes \
  -days 3650 \
  -keyout "$CERT_DIR/tls.key" \
  -out "$CERT_DIR/tls.crt" \
  -subj "/CN=bedrock-gateway.local" \
  >/dev/null 2>&1

chmod 600 "$CERT_DIR/tls.key"
echo "bedrock-gateway: self-signed certificate generated, starting uvicorn with TLS on port ${PORT:-443}"

exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-443}" \
  --ssl-keyfile "$CERT_DIR/tls.key" \
  --ssl-certfile "$CERT_DIR/tls.crt"
