"""Identity middleware tests (external-content m1 Phase A / decisions D7, E2).

Covers: HMAC derivation + domain separation, reject-missing on
identity-required routes, api-proxy identity mapping, conflicting-header
rejection, enforcement-disabled (pre-flip / rollback) behavior, and the
raw-email-never-logged log-capture assertion.
"""

import logging

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


def test_owui_identity_without_user_id_header_yields_no_subject():
    """No User-Id header → no mint subject (api.mint refuses to mint an
    unbound token). Ingestion itself stays permissive: identity resolves."""
    result, request = _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert result is not None
    assert request.state.tpai_mint_binding == "owui-session"
    assert request.state.tpai_mint_subject_id is None


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


def test_subject_and_identity_derive_from_the_same_user_id():
    """SH1 D7 adjustment (TPAI#346): the User-Id header is now BOTH the
    identity-HMAC input and the mint cross-check subject on the primary OWUI
    path — one immutable key, two derived views. (Inverts the pre-SH1
    property that the subject never perturbed an email-derived identity;
    email-derived identity survives only on the User-Id-less fallback.)"""
    with_subject, request = _resolve(
        {identity.OWUI_EMAIL_HEADER: TEST_EMAIL, identity.OWUI_USER_ID_HEADER: "b" * 64}
    )
    assert with_subject == expected_hmac("owui-user-id", "b" * 64)
    assert request.state.tpai_mint_subject_id == "b" * 64
    without_subject, _ = _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert without_subject == expected_hmac("owui-email", TEST_EMAIL.lower())
    assert with_subject != without_subject


# --- User-Id-primary identity (SH1 flag-day fix — TPAI#346, D7 adjustment) -----

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


def test_user_id_takes_precedence_over_email():
    """When both OWUI headers are present the email is deliberately unused —
    email is mutable (reconciliation placeholder) and None for
    billing-enrolled users; the user id is immutable."""
    result, _ = _resolve(
        {identity.OWUI_USER_ID_HEADER: USER_ID, identity.OWUI_EMAIL_HEADER: TEST_EMAIL}
    )
    assert result == expected_hmac("owui-user-id", USER_ID)


def test_identity_stable_across_email_mutation():
    """Provisioning name-reconciliation can rewrite a user's email
    (None -> @placeholder.tpai.internal). The identity must not re-key —
    budget counters and the 7-year audit trail hang off it."""
    empty, _ = _resolve(
        {identity.OWUI_USER_ID_HEADER: USER_ID, identity.OWUI_EMAIL_HEADER: ""}
    )
    placeholder, _ = _resolve(
        {
            identity.OWUI_USER_ID_HEADER: USER_ID,
            identity.OWUI_EMAIL_HEADER: PLACEHOLDER_EMAIL,
        }
    )
    assert empty == placeholder == expected_hmac("owui-user-id", USER_ID)


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


def test_email_only_fallback_still_resolves_email_identity():
    """Legacy fallback (pre-fix fork / non-fork OWUI): email-only requests
    keep the original owui-email derivation and carry no mint subject."""
    result, request = _resolve({identity.OWUI_EMAIL_HEADER: TEST_EMAIL})
    assert result == expected_hmac("owui-email", TEST_EMAIL.lower())
    assert request.state.tpai_mint_subject_id is None


def test_raw_email_never_logged_on_primary_path(client, caplog):
    """The one-line-of-existence property must hold on the primary path too,
    where the email header is read but never used."""
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            **AUTH,
            identity.OWUI_EMAIL_HEADER: TEST_EMAIL,
            identity.OWUI_USER_ID_HEADER: USER_ID,
        },
    )
    assert resp.status_code == 200
    for variant in (TEST_EMAIL, TEST_EMAIL.lower(), TEST_EMAIL.upper()):
        assert variant not in caplog.text
        assert variant not in resp.text
