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
- empty/whitespace CA config must fail closed — falsy ``cadata`` is the one
  shape ``ssl.create_default_context`` answers by loading the ENTIRE public
  trust store instead of raising;
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
    """Reset the module-level session singleton so each test builds its own
    pinned/unpinned session (monkeypatch restores the prior value after)."""
    monkeypatch.setattr(web_fetch, "_http_session", None)


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


def _pin(monkeypatch, ca: trustme.CA, url: str, ca_b64: str | None = None) -> None:
    """Point the settings at ``url`` pinning ``ca`` (or an explicit b64)."""
    if ca_b64 is None:
        ca_b64 = base64.b64encode(ca.cert_pem.bytes()).decode("ascii")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", ca_b64)
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", url)


def _pinned_server(monkeypatch, tls_server, san: str = CONTRACT_NAME) -> trustme.CA:
    """One pinned CA + one server presenting a cert it signed for ``san``."""
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(san))
    _pin(monkeypatch, ca, url)
    return ca


# ------------------------------------------------------------ positive path


def test_pinned_session_accepts_the_ca_signed_contract_cert(
    monkeypatch, fresh_session, tls_server
):
    """Chain verifies against the pinned CA and the contract SAN is asserted
    even though the URL host (127.0.0.1, standing in for the VPCE DNS name)
    does not appear in the cert — the hostname-pinning contract itself."""
    _pinned_server(monkeypatch, tls_server)
    result = web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    assert result.text == "typed output"


def test_der_encoded_ca_is_accepted(monkeypatch, fresh_session, tls_server):
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    der = ssl.PEM_cert_to_DER_cert(ca.cert_pem.bytes().decode("ascii"))
    _pin(monkeypatch, ca, url, ca_b64=base64.b64encode(der).decode("ascii"))
    assert web_fetch.execute_web_fetch("https://a.example/1", "jwt").text == "typed output"


def test_line_wrapped_base64_is_accepted(monkeypatch, fresh_session, tls_server):
    """76-column wrapping is the ``base64``/``openssl base64`` CLI default;
    an operator-regenerated bundle must not brick the fetch path over it."""
    ca = trustme.CA()
    url = tls_server(ca.issue_cert(CONTRACT_NAME))
    flat = base64.b64encode(ca.cert_pem.bytes()).decode("ascii")
    wrapped = "\n".join(flat[i : i + 76] for i in range(0, len(flat), 76))
    _pin(monkeypatch, ca, url, ca_b64=wrapped)
    assert web_fetch.execute_web_fetch("https://a.example/1", "jwt").text == "typed output"


def test_trust_store_holds_exactly_the_pinned_ca_after_a_live_request(
    monkeypatch, fresh_session, tls_server
):
    """requests>=2.32 has two paths that push the public certifi bundle into
    a custom ssl_context (per-request pool attributes + cert_verify). Both
    are overridden in _ConnectorTlsAdapter; a regression on either shows up
    here as the certifi CAs joining the pinned CA in the trust store."""
    _pinned_server(monkeypatch, tls_server)
    web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    adapter = web_fetch._session().get_adapter(setting.TPAI_CONNECTOR_URL + "/v1/web/fetch")
    assert isinstance(adapter, web_fetch._ConnectorTlsAdapter)
    assert adapter._ssl_context.cert_store_stats()["x509_ca"] == 1


def test_pin_is_scoped_to_the_connector_url_prefix(
    monkeypatch, fresh_session, tls_server
):
    """The pinned adapter mounts on the connector URL only; every other
    https prefix stays on the default public-CA adapter."""
    _pinned_server(monkeypatch, tls_server)
    session = web_fetch._session()
    assert isinstance(
        session.get_adapter(setting.TPAI_CONNECTOR_URL + "/v1/web/fetch"),
        web_fetch._ConnectorTlsAdapter,
    )
    assert type(session.get_adapter("https://other.example/")) is requests.adapters.HTTPAdapter


# ------------------------------------------------------------ negative paths


def test_pinned_session_rejects_a_cert_from_another_ca(
    monkeypatch, fresh_session, tls_server
):
    other_ca = trustme.CA()
    url = tls_server(other_ca.issue_cert(CONTRACT_NAME))
    _pin(monkeypatch, trustme.CA(), url)
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_pinned_session_rejects_a_ca_signed_cert_without_the_contract_san(
    monkeypatch, fresh_session, tls_server
):
    _pinned_server(monkeypatch, tls_server, san="imposter.internal")
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
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://vpce.connector.test")
    adapter = web_fetch._session().get_adapter("https://vpce.connector.test/v1/web/fetch")
    assert type(adapter) is requests.adapters.HTTPAdapter


def test_http_connector_url_is_refused(monkeypatch, fresh_session):
    """An http:// connector URL would route past the pinned adapter and put
    the connector JWT on the wire in cleartext — refuse before any request."""
    _pin_url = "http://vpce.connector.test"
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", _pin_url)
    with pytest.raises(web_fetch.WebFetchError, match="https"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_hostless_connector_url_is_refused(monkeypatch, fresh_session):
    """'https://' with no host raises a urllib3 ValueError deep in the pool
    layer that would bypass the RequestException mapping — refuse it first."""
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", "")
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://")
    with pytest.raises(web_fetch.WebFetchError, match="host"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


# ------------------------------------------------------- malformed CA config


def test_malformed_base64_fails_closed(monkeypatch, fresh_session):
    _pin_settings(monkeypatch, "not!!valid@@base64")
    with pytest.raises(web_fetch.WebFetchError, match="not valid base64"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_whitespace_only_ca_fails_closed_not_open(monkeypatch, fresh_session):
    """Whitespace decodes to b'' — the exact input ssl.create_default_context
    would answer by loading ALL public CAs (silently un-pinning the session).
    It must raise instead."""
    _pin_settings(monkeypatch, "   ")
    with pytest.raises(web_fetch.WebFetchError, match="decodes to nothing"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_base64_of_non_certificate_fails_closed(monkeypatch, fresh_session):
    _pin_settings(monkeypatch, base64.b64encode(b"this is not a certificate").decode("ascii"))
    with pytest.raises(web_fetch.WebFetchError, match="loadable CA certificate"):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")


def test_session_construction_failure_does_not_poison_the_singleton(
    monkeypatch, fresh_session
):
    """A malformed CA must fail every call (not just the first) AND recover
    once the config is fixed — the singleton stays unset on failure."""
    _pin_settings(monkeypatch, "%%%")
    with pytest.raises(web_fetch.WebFetchError):
        web_fetch.execute_web_fetch("https://a.example/1", "jwt")
    assert web_fetch._http_session is None


def _pin_settings(monkeypatch, ca_b64: str) -> None:
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_CA_B64", ca_b64)
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", "https://vpce.connector.test")
