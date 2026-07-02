"""Identity middleware tests (external-content m1 Phase A / decisions D7, E2).

Covers: HMAC derivation + domain separation, reject-missing on
identity-required routes, api-proxy identity mapping, conflicting-header
rejection, enforcement-disabled (pre-flip / rollback) behavior, and the
raw-email-never-logged log-capture assertion.
"""

import hashlib
import hmac
import logging

import api.identity as identity
from tests.conftest import AUTH, CHAT_BODY

TEST_KEY = "test-identity-hmac-key"
TEST_EMAIL = "Alice.Example@Example.COM"


def expected_hmac(domain: str, value: str) -> str:
    return hmac.new(TEST_KEY.encode(), f"{domain}:{value}".encode(), hashlib.sha256).hexdigest()


# --- Enforcement on identity-required routes ---------------------------------


def test_chat_missing_identity_rejected(client):
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing user identity"


def test_embeddings_missing_identity_rejected(client):
    resp = client.post(
        "/api/v1/embeddings",
        json={"model": "cohere.embed-multilingual-v3", "input": ["hi"]},
        headers=AUTH,
    )
    assert resp.status_code == 401


def test_chat_with_owui_email_accepted(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_EMAIL_HEADER: TEST_EMAIL},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello."


def test_chat_with_api_proxy_identity_accepted(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.API_PROXY_USER_HEADER: "a" * 64},
    )
    assert resp.status_code == 200


def test_conflicting_identity_headers_rejected(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
            identity.API_PROXY_USER_HEADER: "a" * 64,
        },
    )
    assert resp.status_code == 400


def test_empty_identity_header_treated_as_missing(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_EMAIL_HEADER: "   "},
    )
    assert resp.status_code == 401


def test_identity_still_requires_api_key(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            "Authorization": "Bearer wrong-key",
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
        },
    )
    assert resp.status_code == 401


# --- Routes that must NOT require identity -----------------------------------


def test_health_requires_nothing(client):
    assert client.get("/health").status_code == 200


def test_models_requires_key_but_not_identity(client, monkeypatch):
    monkeypatch.setattr(
        "api.routers.model.chat_model.list_models",
        lambda: ["anthropic.claude-3-sonnet-20240229-v1:0"],
    )
    resp = client.get("/api/v1/models", headers=AUTH)
    assert resp.status_code == 200


# --- HMAC derivation ----------------------------------------------------------


def _resolve(headers: dict[str, str]) -> tuple[str, "Request"]:
    """Run the dependency directly against a synthetic request."""
    import asyncio

    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    request = Request(scope)
    result = asyncio.run(identity.require_identity(request))
    return result, request


def test_email_identity_is_domain_separated_hmac():
    """The resolved identity is HMAC-SHA256('owui-email:' + lowercased email)."""
    result, request = _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert result == expected_hmac("owui-email", TEST_EMAIL.lower())
    assert request.state.tpai_identity == result


def test_api_proxy_identity_uses_distinct_domain():
    value = "ABCDEF" + "0" * 58
    result, _ = _resolve({identity.API_PROXY_USER_HEADER: value})
    assert result == expected_hmac("api-key-user", value.lower())
    # Domain separation: same input string under the email domain differs.
    assert result != expected_hmac("owui-email", value.lower())


# --- Enforcement disabled (pre-flip / rolled-back deployment) ------------------


def test_no_key_disables_enforcement(client, monkeypatch):
    monkeypatch.setattr(identity, "IDENTITY_HMAC_KEY", "")
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
    assert resp.status_code == 200


# --- The raw email never appears in logs (plan.md Phase A verification) --------


def test_raw_email_never_logged(client, caplog):
    """Log-capture proof: a request carrying a raw email produces no log line
    containing it — the email's only line of existence is the HMAC call."""
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_EMAIL_HEADER: TEST_EMAIL},
    )
    assert resp.status_code == 200

    email_variants = {TEST_EMAIL, TEST_EMAIL.lower(), TEST_EMAIL.upper()}
    for record in caplog.records:
        rendered = record.getMessage()
        for variant in email_variants:
            assert variant not in rendered, (
                f"raw email leaked into log record from {record.name}:{record.lineno}"
            )
    # And it never leaks into the response body either.
    for variant in email_variants:
        assert variant not in resp.text


def test_raw_email_never_logged_on_rejection_paths(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
            identity.API_PROXY_USER_HEADER: "a" * 64,
        },
    )
    assert TEST_EMAIL not in caplog.text
    assert TEST_EMAIL.lower() not in caplog.text
