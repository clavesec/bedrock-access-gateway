"""Identity middleware tests (external-content m1 Phase A / decisions D7, E2).

Covers: HMAC derivation + domain separation, reject-missing on
identity-required routes, api-proxy identity mapping, conflicting-header
rejection, enforcement-disabled (pre-flip / rollback) behavior, and the
raw-email-never-logged log-capture assertion.
"""

import logging

import pytest
from fastapi import HTTPException

import api.identity as identity
from tests.conftest import AUTH, CHAT_BODY, expected_hmac

TEST_EMAIL = "Alice.Example@Example.COM"


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


def test_chat_with_owui_user_id_accepted(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_USER_ID_HEADER: "b" * 64},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello."


def test_chat_with_email_only_rejected(client):
    """Email-only requests are anomalous (every supported OWUI sends the
    User-Id header in the same block) and must fail closed rather than
    open a second, unlinked identity space (SH1 D7 adjustment)."""
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_EMAIL_HEADER: TEST_EMAIL},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing user identity"


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
            identity.OWUI_USER_ID_HEADER: "b" * 64,
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


def test_email_only_resolution_rejected():
    """The email header is never an identity input (SH1 D7 adjustment):
    without a User-Id header the dependency raises 401 fail-closed."""
    with pytest.raises(HTTPException) as exc:
        _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert exc.value.status_code == 401


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


def test_disabled_enforcement_still_sets_request_state(monkeypatch):
    """Downstream audit/budget consumers read request.state.tpai_identity
    unconditionally — it must exist (as None) in the disabled state too."""
    monkeypatch.setattr(identity, "IDENTITY_HMAC_KEY", "")
    result, request = _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert result is None
    assert request.state.tpai_identity is None


def test_enforce_flag_without_key_refuses_startup():
    """Fail closed (TPAI_IDENTITY_ENFORCE): a deployment that declares
    enforcement but lost the HMAC key must crash, not boot fail-open."""
    import pytest

    with pytest.raises(RuntimeError, match="TPAI_IDENTITY_ENFORCE"):
        identity._require_key_when_enforced(True, "")
    # All other combinations start normally.
    identity._require_key_when_enforced(True, "some-key")
    identity._require_key_when_enforced(False, "")
    identity._require_key_when_enforced(False, "some-key")


# --- The raw email never appears in logs (plan.md Phase A verification) --------


def _assert_email_never_leaks(caplog, resp=None):
    """One canonical leak scan for every path a raw email header can take:
    per-record (with source diagnostics), all case variants, and — when a
    response is given — the response body too. Strengthen it HERE, not in
    per-test copies."""
    email_variants = {TEST_EMAIL, TEST_EMAIL.lower(), TEST_EMAIL.upper()}
    for record in caplog.records:
        rendered = record.getMessage()
        for variant in email_variants:
            assert variant not in rendered, (
                f"raw email leaked into log record from {record.name}:{record.lineno}"
            )
    if resp is not None:
        for variant in email_variants:
            assert variant not in resp.text


def test_raw_email_never_logged_on_accept_path(client, caplog):
    """A served request carrying a raw email header (alongside the User-Id
    it derives identity from) leaks the email nowhere — it is read for the
    conflict check and never used."""
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
            identity.OWUI_USER_ID_HEADER: "b" * 64,
        },
    )
    assert resp.status_code == 200
    _assert_email_never_leaks(caplog, resp)


def test_raw_email_never_logged_on_rejection_paths(client, caplog):
    caplog.set_level(logging.DEBUG)
    # Conflict rejection (400): OWUI + api-proxy identities together.
    resp_conflict = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
            identity.API_PROXY_USER_HEADER: "a" * 64,
        },
    )
    assert resp_conflict.status_code == 400
    _assert_email_never_leaks(caplog, resp_conflict)
    # Missing-identity rejection (401): email-only request.
    resp_missing = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={**AUTH, identity.OWUI_EMAIL_HEADER: TEST_EMAIL},
    )
    assert resp_missing.status_code == 401
    _assert_email_never_leaks(caplog, resp_missing)


# --- Mint-path subject capture (m1 Phase E(i), D8/R12) --------------------------


def test_owui_identity_captures_session_binding_subject():
    """OWUI traffic: the mint subject is the enrollment-space user id from
    X-OpenWebUI-User-Id, with the owui-session (live login) binding."""
    subject = "b" * 64
    _, request = _resolve(
        {identity.OWUI_EMAIL_HEADER: TEST_EMAIL, identity.OWUI_USER_ID_HEADER: subject}
    )
    assert request.state.tpai_mint_binding == "owui-session"
    assert request.state.tpai_mint_subject_id == subject


def test_api_proxy_identity_captures_api_key_binding_subject():
    """api-proxy traffic: the asserted per-user key id is the mint subject,
    with the api-key (live credential, R12) binding."""
    value = "c" * 64
    _, request = _resolve({identity.API_PROXY_USER_HEADER: value})
    assert request.state.tpai_mint_binding == "api-key"
    assert request.state.tpai_mint_subject_id == value


def test_api_key_subject_preserves_case_while_identity_lowercases():
    """The subject is an exact DynamoDB key in tpai-api-keys: the HMAC's
    .lower() normalization must not leak into it, or legacy mixed-case
    userIds could never pass the mint Lambda's live-credential cross-check."""
    value = "Legacy-User-ABC123"
    result, request = _resolve({identity.API_PROXY_USER_HEADER: f"  {value} "})
    assert request.state.tpai_mint_subject_id == value
    assert result == expected_hmac("api-key-user", value.lower())


def test_disabled_enforcement_sets_mint_state_to_none(monkeypatch):
    """The disabled state must define the mint attributes too, so the m2
    choke point can read them unconditionally."""
    monkeypatch.setattr(identity, "IDENTITY_HMAC_KEY", "")
    _, request = _resolve(
        {identity.OWUI_EMAIL_HEADER: TEST_EMAIL, identity.OWUI_USER_ID_HEADER: "b" * 64}
    )
    assert request.state.tpai_mint_binding is None
    assert request.state.tpai_mint_subject_id is None


# --- User-Id identity (SH1 flag-day fix — TPAI#346, D7 adjustment) -------------

USER_ID = "d" * 64
PLACEHOLDER_EMAIL = f"{'d' * 16}@placeholder.tpai.internal"


def test_user_id_identity_is_domain_separated_hmac():
    """Primary OWUI path: identity = HMAC-SHA256('owui-user-id:' + user id)."""
    result, request = _resolve({identity.OWUI_USER_ID_HEADER: USER_ID})
    assert result == expected_hmac("owui-user-id", USER_ID)
    assert request.state.tpai_identity == result
    assert request.state.tpai_mint_binding == "owui-session"
    assert request.state.tpai_mint_subject_id == USER_ID
    # Domain separation from both legacy spaces.
    assert result != expected_hmac("owui-email", USER_ID)
    assert result != expected_hmac("api-key-user", USER_ID)


@pytest.mark.parametrize(
    "email_header",
    [None, "", TEST_EMAIL, PLACEHOLDER_EMAIL],
    ids=["absent", "empty", "real-email", "reconciled-placeholder"],
)
def test_identity_derives_from_user_id_regardless_of_email(email_header):
    """The email header — absent, empty (billing-enrolled users), real, or
    rewritten by provisioning name-reconciliation — never perturbs the
    identity. Budget counters and the 7-year audit trail hang off identity
    stability, and email is a mutable field."""
    headers = {identity.OWUI_USER_ID_HEADER: USER_ID}
    if email_header is not None:
        headers[identity.OWUI_EMAIL_HEADER] = email_header
    result, _ = _resolve(headers)
    assert result == expected_hmac("owui-user-id", USER_ID)


def test_billing_enrolled_header_shape_accepted(client):
    """The post-fix fork shape for billing-enrolled users (email=None ->
    coalesced to an empty header) resolves identity and serves the request."""
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: "",
            identity.OWUI_USER_ID_HEADER: USER_ID,
        },
    )
    assert resp.status_code == 200


def test_user_id_case_and_whitespace_handling():
    """The user id is an exact session-table key: strip whitespace, preserve
    case, and feed the exact value to both the HMAC and the mint subject."""
    result, request = _resolve({identity.OWUI_USER_ID_HEADER: "  MixedCase-Id-01 "})
    assert request.state.tpai_mint_subject_id == "MixedCase-Id-01"
    assert result == expected_hmac("owui-user-id", "MixedCase-Id-01")


def test_user_id_conflicts_with_api_proxy_header(client):
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_USER_ID_HEADER: USER_ID,
            identity.API_PROXY_USER_HEADER: "a" * 64,
        },
    )
    assert resp.status_code == 400


