"""Tests for the connector-JWT mint client (api.mint, m1 Phase E(i) — D8/R12).

The Lambda transport is faked at the ``mint._lambda`` seam (the same
injection pattern as ``test_audit.py``'s FakeS3 / ``test_taint.py``'s
FakeDynamoDB), so no test can ever invoke a real function.
"""

import io
import json
import logging
import time

import pytest

from api import mint

# Captured at import time, BEFORE any fixture can reset the module global:
# the autouse `configured` fixture nulls mint._lambda_client per test (order
# hygiene), which would otherwise make the import-laziness assertion below
# unfalsifiable.
_CLIENT_WAS_NONE_AT_IMPORT = mint._lambda_client is None

IDENTITY = "a" * 64
SUBJECT = "b" * 64
FAKE_ARN = "arn:aws:lambda:us-east-1:000000000000:function:tpai-connector-mint-test"
FAKE_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.c2lnbmF0dXJl"


class FakeLambda:
    """Minimal stand-in for boto3's Lambda client."""

    def __init__(self, responses=None, error=None):
        self.invocations = []
        self._responses = list(responses or [])
        self._error = error

    def invoke(self, **kwargs):
        self.invocations.append(kwargs)
        if self._error is not None:
            raise self._error
        body = self._responses.pop(0)
        result = {"StatusCode": 200}
        if isinstance(body, dict) and body.pop("__function_error__", None):
            result["FunctionError"] = "Unhandled"
        result["Payload"] = io.BytesIO(json.dumps(body).encode())
        return result


def ok_response(expires_in=900):
    return {
        "ok": True,
        "token": FAKE_JWT,
        "expires_at": int(time.time()) + expires_in,
    }


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    """Point the module at a fake ARN and pristine module state for every
    test (the client singleton too, so no test depends on execution order)."""
    monkeypatch.setattr(mint, "MINT_FUNCTION_ARN", FAKE_ARN)
    monkeypatch.setattr(mint, "_token_cache", {})
    monkeypatch.setattr(mint, "_lambda_client", None)
    yield


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(mint, "_lambda", lambda: fake)
    return fake


# --- Dark / configuration state -------------------------------------------------


def test_unconfigured_arn_fails_closed(monkeypatch):
    monkeypatch.setattr(mint, "MINT_FUNCTION_ARN", "")
    with pytest.raises(mint.MintError, match="not configured"):
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)


def test_importing_the_module_never_builds_a_client():
    """The dark path must not resolve AWS credentials at import (the same
    lazy-singleton invariant as api.audit / api.ddb). Asserted against the
    import-time snapshot — the autouse fixture resets the global per test,
    so checking the live attribute here would always pass."""
    assert _CLIENT_WAS_NONE_AT_IMPORT


# --- Input validation ------------------------------------------------------------


def test_empty_identity_rejected():
    with pytest.raises(ValueError):
        mint.get_connector_token("", mint.BINDING_OWUI_SESSION, SUBJECT)


def test_unknown_binding_rejected():
    with pytest.raises(ValueError):
        mint.get_connector_token(IDENTITY, "gmail", SUBJECT)


def test_missing_subject_is_a_clean_refusal(monkeypatch):
    """No cross-check subject (e.g. OWUI omitted the User-Id header) means
    no live-ness check is possible — refuse without invoking the Lambda."""
    fake = use_fake(monkeypatch, FakeLambda())
    with pytest.raises(mint.MintRefusedError, match="missing-subject"):
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, None)
    assert fake.invocations == []


# --- Happy path + caching ---------------------------------------------------------


def test_mint_returns_token_and_caches_it(monkeypatch):
    fake = use_fake(monkeypatch, FakeLambda([ok_response()]))
    first = mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    second = mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    assert first.token == FAKE_JWT
    assert second is first
    assert len(fake.invocations) == 1

    request = json.loads(fake.invocations[0]["Payload"])
    assert request == {
        "schema": "tpai.connector-mint.request.v1",
        "identity": IDENTITY,
        "binding": "owui-session",
        "subject_id": SUBJECT,
    }
    assert fake.invocations[0]["FunctionName"] == FAKE_ARN


def test_cache_is_per_identity(monkeypatch):
    fake = use_fake(monkeypatch, FakeLambda([ok_response(), ok_response()]))
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    mint.get_connector_token("d" * 64, mint.BINDING_API_KEY, "e" * 64)
    assert len(fake.invocations) == 2


def test_token_within_refresh_margin_is_reminted(monkeypatch):
    """A token expiring inside the refresh margin must not be served: an
    in-flight connector call could outlive it."""
    fake = use_fake(
        monkeypatch,
        FakeLambda([ok_response(expires_in=mint.REFRESH_MARGIN_SECONDS - 5), ok_response()]),
    )
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    assert len(fake.invocations) == 2


def test_invalidate_drops_the_cached_token(monkeypatch):
    fake = use_fake(monkeypatch, FakeLambda([ok_response(), ok_response()]))
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    mint.invalidate(IDENTITY)
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    assert len(fake.invocations) == 2


def test_changed_subject_bypasses_the_cache(monkeypatch):
    """A cached token was live-ness-checked against its subject; a request
    asserting a different subject under the same identity must re-mint (and
    re-cross-check), never be served the stale binding."""
    fake = use_fake(monkeypatch, FakeLambda([ok_response(), ok_response()]))
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    other = "f" * 64
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, other)
    assert len(fake.invocations) == 2
    assert json.loads(fake.invocations[1]["Payload"])["subject_id"] == other


# --- Refusals (clean deny) ---------------------------------------------------------


def test_refusal_raises_and_is_not_cached(monkeypatch):
    fake = use_fake(
        monkeypatch,
        FakeLambda([{"ok": False, "reason": "no_active_session"}, ok_response()]),
    )
    with pytest.raises(mint.MintRefusedError) as excinfo:
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    assert excinfo.value.reason == "no_active_session"
    # A refusal is not sticky: the user can log in and retry immediately.
    token = mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)
    assert token.token == FAKE_JWT
    assert len(fake.invocations) == 2


def test_revoked_api_key_refusal_reason_is_surfaced(monkeypatch):
    use_fake(monkeypatch, FakeLambda([{"ok": False, "reason": "revoked_api_key"}]))
    with pytest.raises(mint.MintRefusedError) as excinfo:
        mint.get_connector_token(IDENTITY, mint.BINDING_API_KEY, SUBJECT)
    assert excinfo.value.reason == "revoked_api_key"


# --- Failure modes (fail closed) ----------------------------------------------------


def test_transport_error_fails_closed(monkeypatch):
    use_fake(monkeypatch, FakeLambda(error=ConnectionError("boom")))
    with pytest.raises(mint.MintError):
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)


def test_function_error_fails_closed(monkeypatch):
    use_fake(monkeypatch, FakeLambda([{"__function_error__": True, "errorMessage": "x"}]))
    with pytest.raises(mint.MintError, match="FunctionError"):
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)


def test_malformed_payloads_fail_closed(monkeypatch):
    for body in [
        {"ok": True},  # no token
        {"ok": True, "token": "", "expires_at": int(time.time()) + 900},
        {"ok": True, "token": FAKE_JWT},  # no expires_at
        {"ok": True, "token": FAKE_JWT, "expires_at": int(time.time()) - 10},
        ["not", "an", "object"],
    ]:
        use_fake(monkeypatch, FakeLambda([body]))
        mint.invalidate(IDENTITY)
        with pytest.raises(mint.MintError):
            mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)


# --- The token and the enrollment pseudonym never touch logs ------------------------


def test_token_and_subject_never_logged(monkeypatch, caplog):
    """E2 linkage + credential hygiene: neither the minted JWT nor the
    enrollment-space subject id may appear in any log line, on the success,
    refusal, or failure paths."""
    caplog.set_level(logging.DEBUG)

    use_fake(monkeypatch, FakeLambda([ok_response()]))
    mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)

    use_fake(monkeypatch, FakeLambda([{"ok": False, "reason": "no_active_session"}]))
    mint.invalidate(IDENTITY)
    with pytest.raises(mint.MintRefusedError):
        mint.get_connector_token(IDENTITY, mint.BINDING_OWUI_SESSION, SUBJECT)

    use_fake(monkeypatch, FakeLambda(error=ConnectionError("boom")))
    with pytest.raises(mint.MintError):
        mint.get_connector_token("f" * 64, mint.BINDING_OWUI_SESSION, SUBJECT)

    assert FAKE_JWT not in caplog.text
    assert SUBJECT not in caplog.text
