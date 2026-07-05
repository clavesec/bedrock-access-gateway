"""Tests for the /connectors/gmail passthrough surface (m3 G3, R14).

The surface must be dark by default (404), require the gateway API key AND
a forwarded identity, mint per request with one re-mint on a connector
401, relay only the allowlisted status/body set, and treat a connector
403 as the revocation signal for the gmail_search DEK cache.
"""

import json

import pytest
from fastapi.testclient import TestClient

from api import identity, mint, setting
from api.app import app
from api.tools import gmail, web_fetch
from tests.conftest import AUTH

CONNECTOR_URL = "https://vpce-test.example"
USER_ID = "b" * 64
IDENTITY_HEADERS = {**AUTH, identity.OWUI_USER_ID_HEADER: USER_ID}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = json.dumps(body).encode("utf-8")
        self.closed = False

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeSession:
    def __init__(self):
        self.calls = []
        self.queued = []

    def queue(self, response):
        self.queued.append(response)

    def request(self, method, url, json=None, headers=None, timeout=None, stream=None, allow_redirects=None):
        self.calls.append(
            {"method": method, "url": url, "json": json, "auth": headers.get("Authorization")}
        )
        return self.queued.pop(0)


@pytest.fixture
def session(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(web_fetch, "connector_session", lambda: fake)
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", CONNECTOR_URL)
    return fake


@pytest.fixture
def enabled(monkeypatch, session):
    monkeypatch.setattr(setting, "ENABLE_GMAIL_TOOLS", True)
    minted = []

    def fake_mint(identity_value, binding, subject_id):
        minted.append((identity_value, binding, subject_id))
        return mint.MintedToken(token=f"jwt-{len(minted)}", expires_at=2**31, subject_id=subject_id)

    monkeypatch.setattr(mint, "get_connector_token", fake_mint)
    monkeypatch.setattr(mint, "invalidate", lambda identity_value: None)
    return minted


@pytest.fixture
def client():
    return TestClient(app)


def test_surface_is_dark_by_default(client, session):
    for method, path in [
        ("GET", "/api/v1/connectors/gmail/status"),
        ("POST", "/api/v1/connectors/gmail/consent-session"),
        ("POST", "/api/v1/connectors/gmail/confirm"),
        ("POST", "/api/v1/connectors/gmail/disconnect"),
    ]:
        response = client.request(method, path, headers=IDENTITY_HEADERS)
        assert response.status_code == 404
    assert session.calls == []


def test_requires_the_gateway_api_key(client, enabled):
    response = client.get("/api/v1/connectors/gmail/status")
    assert response.status_code in (401, 403)


def test_requires_a_forwarded_identity(client, enabled, session):
    response = client.get("/api/v1/connectors/gmail/status", headers=AUTH)
    assert response.status_code in (401, 403)
    assert session.calls == []


def test_status_relays_connector_body_and_mints_owui_binding(client, enabled, session):
    session.queue(FakeResponse(200, {"schema": "tpai.connector.gmail-status.v1", "status": "connected"}))
    response = client.get("/api/v1/connectors/gmail/status", headers=IDENTITY_HEADERS)
    assert response.status_code == 200
    assert response.json()["status"] == "connected"
    call = session.calls[0]
    assert call["url"] == CONNECTOR_URL + "/v1/gmail/status"
    assert call["auth"] == "Bearer jwt-1"
    assert enabled[0][1] == mint.BINDING_OWUI_SESSION


def test_consent_session_relays_201_and_consent_url(client, enabled, session):
    session.queue(
        FakeResponse(201, {"consent_url": "https://connect.example/oauth/gmail/start?cs=x", "expires_in": 900})
    )
    response = client.post("/api/v1/connectors/gmail/consent-session", headers=IDENTITY_HEADERS)
    assert response.status_code == 201
    assert "consent_url" in response.json()


def test_confirm_validates_and_relays_the_nonce(client, enabled, session):
    session.queue(FakeResponse(200, {"confirmed": True}))
    response = client.post(
        "/api/v1/connectors/gmail/confirm",
        headers=IDENTITY_HEADERS,
        json={"nonce": "n" * 43},
    )
    assert response.status_code == 200
    assert session.calls[0]["json"] == {"nonce": "n" * 43}

    for bad in ["short", "bad nonce!" + "x" * 20]:
        response = client.post(
            "/api/v1/connectors/gmail/confirm", headers=IDENTITY_HEADERS, json={"nonce": bad}
        )
        assert response.status_code in (400, 422)
    assert len(session.calls) == 1


def test_connector_403_is_relayed_and_drops_the_dek_cache(client, enabled, session):
    # Any 403 doubles as the revocation signal (S15 R-5): the DEK cache for
    # this identity must be dropped.
    from tests.conftest import expected_hmac

    computed_identity = expected_hmac("owui-user-id", USER_ID)
    gmail._dek_cache[computed_identity] = (b"\x11" * 32, 2**31)
    session.queue(FakeResponse(403, {"reason": "not-connected"}))
    response = client.get("/api/v1/connectors/gmail/status", headers=IDENTITY_HEADERS)
    assert response.status_code == 403
    assert response.json()["reason"] == "not-connected"
    assert computed_identity not in gmail._dek_cache


def test_connector_401_triggers_exactly_one_remint(client, enabled, session):
    session.queue(FakeResponse(401, {"detail": "invalid_token:expired"}))
    session.queue(FakeResponse(200, {"status": "connected"}))
    response = client.get("/api/v1/connectors/gmail/status", headers=IDENTITY_HEADERS)
    assert response.status_code == 200
    assert [c["auth"] for c in session.calls] == ["Bearer jwt-1", "Bearer jwt-2"]


def test_unexpected_connector_status_maps_to_502(client, enabled, session):
    session.queue(FakeResponse(500, {"detail": "boom"}))
    response = client.get("/api/v1/connectors/gmail/status", headers=IDENTITY_HEADERS)
    assert response.status_code == 502
    assert response.json()["detail"] == "connector-error"


def test_mint_refused_maps_to_403_no_live_session(client, enabled, session, monkeypatch):
    def refuse(identity_value, binding, subject_id):
        raise mint.MintRefusedError("no_active_session")

    monkeypatch.setattr(mint, "get_connector_token", refuse)
    response = client.get("/api/v1/connectors/gmail/status", headers=IDENTITY_HEADERS)
    assert response.status_code == 403
    assert response.json()["detail"] == "no-live-session"
    assert session.calls == []
