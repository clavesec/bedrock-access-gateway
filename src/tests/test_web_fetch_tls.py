"""TLS-pinning tests for the gateway→connector session (TPAI#365).

Real handshakes against a local TLS server (trustme-issued certs) — the
load-bearing assertions cannot be proven with mocks:

- the pinned session trusts EXACTLY the egress-local CA and asserts the
  fixed contract SAN (``CONNECTOR_TLS_SERVER_NAME``) instead of the URL
  hostname (the URL carries a per-endpoint VPCE DNS name the egress account
  cannot know at signing time);
- requests>=2.32 must not widen the pin by pushing the public certifi
  bundle into the pinned ssl_context (the ``cert_store_stats`` assertion
  after a live request);
- the default (unpinned) session still performs public-CA verification, so
  an egress-local cert fails closed without the pin.
"""

import base64
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
import trustme

from api import setting
from api.tools import web_fetch


CONTRACT_NAME = web_fetch.CONNECTOR_TLS_SERVER_NAME


class _ConnectorHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the connector's /v1/web/fetch route."""

    def do_POST(self):
        body = json.dumps({"content": "typed output", "truncated": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        # Keep handshake-failure noise out of the pytest output.
        pass


@pytest.fixture
def fresh_session(monkeypatch):
    """Reset the module-level session singleton around each test so every
    test builds (and tears down) its own pinned/unpinned session."""
    monkeypatch.setattr(web_fetch, "_http_session", None)
    yield
    web_fetch._http_session = None


@pytest.fixture
def tls_server():
    """Start local HTTPS servers serving a given trustme cert; yields a
    ``start(server_cert) -> base_url`` factory."""
    servers = []

    def start(server_cert: trustme.LeafCert) -> str:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_cert.configure_cert(ctx)
        httpd = HTTPServer(("127.0.0.1", 0), _ConnectorHandler)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append((httpd, thread))
        return f"https://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for httpd, thread in servers:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _pin(monkeypatch, ca: trustme.CA, url: str) -> None:
    monkeypatch.setattr(
        setting,
        "TPAI_CONNECTOR_CA_B64",
        base64.b64encode(ca.cert_pem.bytes()).decode("ascii"),
    )
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", url)


# ------------------------------------------------------------ positive path


def test_pinned_session_accepts_the_ca_signed_contract_cert(
    monkeypatch, fresh_session, tls_server
):
    """Chain verifies against the pinned CA and the contract SAN is asserted
    even though the URL host (127.0.0.1, standing in for the VPCE DNS name)
    does not appear in the cert — the hostname-pinning contract itself."""
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    _pin(monkeypatch, ca, url)
    result = web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    assert result.text == "typed output"


def test_der_encoded_ca_is_accepted(monkeypatch, fresh_session, tls_server):
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    der = ssl.PEM_cert_to_DER_cert(ca.cert_pem.bytes().decode("ascii"))
    monkeypatch.setattr(
        setting, "TPAI_CONNECTOR_CA_B64", base64.b64encode(der).decode("ascii")
    )
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", url)
    assert web_fetch.execute_web_fetch("https://a.example/1", "jwt").text == "typed output"


def test_trust_store_holds_exactly_the_pinned_ca_after_a_live_request(
    monkeypatch, fresh_session, tls_server
):
    """requests>=2.32 has two paths that push the public certifi bundle into
    a custom ssl_context (per-request pool attributes + cert_verify). Both
    are overridden in _ConnectorTlsAdapter; a regression on either shows up
    here as the certifi CAs joining the pinned CA in the trust store."""
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    _pin(monkeypatch, ca, url)
    web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    adapter = web_fetch._session().get_adapter("https://x.example/")
    assert isinstance(adapter, web_fetch._ConnectorTlsAdapter)
    assert adapter._ssl_context.cert_store_stats()["x509_ca"] == 1


# ------------------------------------------------------------ negative paths


def test_pinned_session_rejects_a_cert_from_another_ca(
    monkeypatch, fresh_session, tls_server
):
    pinned_ca = trustme.CA()
    other_ca = trustme.CA()
    url = tls_server(other_ca.issue_cert(CONTRACT_NAME))
    _pin(monkeypatch, pinned_ca, url)
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_pinned_session_rejects_a_ca_signed_cert_without_the_contract_san(
    monkeypatch, fresh_session, tls_server
):
    ca = trustme.CA()
    url = tls_server(ca.issue_cert("imposter.internal"))
    _pin(monkeypatch, ca, url)
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_unpinned_session_rejects_the_egress_local_cert(
    monkeypatch, fresh_session, tls_server
):
    """Without the pin the session keeps default public-CA verification, so
    the private-CA cert fails closed (the pre-#365 SSLError, now the
    misconfiguration backstop rather than the steady state)."""
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", url)
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_unpinned_session_mounts_no_pinned_adapter(monkeypatch, fresh_session):
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "")
    adapter = web_fetch._session().get_adapter("https://x.example/")
    assert type(adapter) is requests.adapters.HTTPAdapter


# ------------------------------------------------------- malformed CA config


def test_malformed_base64_fails_closed(monkeypatch, fresh_session):
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "not!!valid@@base64")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://vpce.connector.test")
    with pytest.raises(web_fetch.WebFetchError, match="not valid base64"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_base64_of_non_certificate_fails_closed(monkeypatch, fresh_session):
    monkeypatch.setattr(
        setting,
        "TPAI_CONNECTOR_CA_B64",
        base64.b64encode(b"this is not a certificate").decode("ascii"),
    )
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://vpce.connector.test")
    with pytest.raises(web_fetch.WebFetchError, match="loadable CA certificate"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_session_construction_failure_does_not_poison_the_singleton(
    monkeypatch, fresh_session
):
    """A malformed CA must fail every call (not just the first) AND recover
    once the config is fixed — the singleton stays unset on failure."""
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "%%%")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://vpce.connector.test")
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    assert web_fetch._http_session is None
