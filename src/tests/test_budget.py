"""Budget-counter tests (external-content m1 Phase D, R12).

Three properties:

1. **Atomic limit enforcement** — counters saturate exactly at the limit
   under the conditional-increment semantics; exhaustion is a clean deny
   (``BudgetExceededError`` with the tripped window).
2. **Both windows, right keys** — user-day first, then the taint-scope
   counter; api-proxy/session callers share the same key space (R12).
3. **Fail closed** — unconfigured table or store failure raises
   ``BudgetStoreError``; a tool call with an unenforceable budget must not
   execute.
"""

import time

import pytest

import api.budget as budget
import api.taint as taint
from tests.conftest import expected_hmac
from tests.ddb_fake import FakeDynamoDB

IDENTITY = expected_hmac("owui-email", "alice@example.com")
API_IDENTITY = expected_hmac("api-key-user", "svc-key-7")
CHAT_SCOPE = taint.resolve_taint_scope(IDENTITY, "chat-1234")
SESSION_SCOPE = taint.resolve_taint_scope(API_IDENTITY, None)

BUDGET_TABLE = "TPAI-TEST-gateway-budget"


@pytest.fixture
def fake_ddb(monkeypatch):
    fake = FakeDynamoDB()
    monkeypatch.setattr(budget, "_ddb", lambda: fake)
    monkeypatch.setattr(budget, "BUDGET_TABLE", BUDGET_TABLE)
    return fake


def _counters(fake):
    return fake.tables.get(BUDGET_TABLE, {})


def _day_key(identity):
    import datetime

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    return f"day#{identity}#{today}"


# --- Happy path -------------------------------------------------------------------


def test_consume_increments_both_windows(fake_ddb):
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)

    counters = _counters(fake_ddb)
    assert int(counters[_day_key(IDENTITY)]["n"]["N"]) == 2
    assert int(counters[f"scope#{CHAT_SCOPE}"]["n"]["N"]) == 2


def test_identities_and_scopes_count_separately(fake_ddb):
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    budget.check_and_consume(API_IDENTITY, SESSION_SCOPE)

    counters = _counters(fake_ddb)
    assert int(counters[_day_key(IDENTITY)]["n"]["N"]) == 1
    assert int(counters[_day_key(API_IDENTITY)]["n"]["N"]) == 1
    assert int(counters[f"scope#{CHAT_SCOPE}"]["n"]["N"]) == 1
    assert int(counters[f"scope#{SESSION_SCOPE}"]["n"]["N"]) == 1


# --- Limit enforcement ------------------------------------------------------------


def test_scope_window_saturates_at_limit(fake_ddb, monkeypatch):
    monkeypatch.setattr(budget, "SCOPE_LIMIT", 3)
    for _ in range(3):
        budget.check_and_consume(IDENTITY, CHAT_SCOPE)

    with pytest.raises(budget.BudgetExceededError) as exc_info:
        budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    assert exc_info.value.window == budget.WINDOW_SCOPE
    assert exc_info.value.limit == 3

    counters = _counters(fake_ddb)
    # The scope counter never exceeds its limit...
    assert int(counters[f"scope#{CHAT_SCOPE}"]["n"]["N"]) == 3
    # ...and the denied call burned one user-day unit (documented, bounded).
    assert int(counters[_day_key(IDENTITY)]["n"]["N"]) == 4


def test_user_day_window_saturates_at_limit(fake_ddb, monkeypatch):
    monkeypatch.setattr(budget, "USER_DAILY_LIMIT", 2)
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    budget.check_and_consume(IDENTITY, taint.resolve_taint_scope(IDENTITY, "chat-other"))

    # A fresh scope does not evade the daily cap.
    with pytest.raises(budget.BudgetExceededError) as exc_info:
        budget.check_and_consume(IDENTITY, taint.resolve_taint_scope(IDENTITY, "chat-fresh"))
    assert exc_info.value.window == budget.WINDOW_USER_DAY

    counters = _counters(fake_ddb)
    assert int(counters[_day_key(IDENTITY)]["n"]["N"]) == 2
    # The daily deny happens before the scope counter is touched.
    assert f"scope#chat#{IDENTITY}#chat-fresh" not in counters


# --- TTLs -------------------------------------------------------------------------


def test_counter_ttls(fake_ddb, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 2_000_000.0)
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    budget.check_and_consume(API_IDENTITY, SESSION_SCOPE)

    counters = _counters(fake_ddb)
    assert int(counters[_day_key(IDENTITY)]["ttl"]["N"]) == 2_000_000 + 2 * 86400
    assert (
        int(counters[f"scope#{CHAT_SCOPE}"]["ttl"]["N"])
        == 2_000_000 + taint.CONVERSATION_TAINT_TTL_DAYS * 86400
    )
    assert (
        int(counters[f"scope#{SESSION_SCOPE}"]["ttl"]["N"])
        == 2_000_000 + taint.SESSION_TAINT_TTL_SECONDS
    )


def test_scope_counter_ttl_tracks_taint_scope_ttl(fake_ddb, monkeypatch):
    # Drift guard: the scope counter's lifetime comes from
    # taint.scope_ttl_epoch, not a private copy — patching taint's TTL
    # must move the budget counter's TTL with it.
    monkeypatch.setattr(time, "time", lambda: 2_000_000.0)
    monkeypatch.setattr(taint, "CONVERSATION_TAINT_TTL_DAYS", 1)
    budget.check_and_consume(IDENTITY, CHAT_SCOPE)
    counters = _counters(fake_ddb)
    assert int(counters[f"scope#{CHAT_SCOPE}"]["ttl"]["N"]) == 2_000_000 + 86400


# --- Fail closed -------------------------------------------------------------------


def test_unconfigured_table_fails_closed(monkeypatch):
    monkeypatch.setattr(budget, "BUDGET_TABLE", "")
    with pytest.raises(budget.BudgetStoreError):
        budget.check_and_consume(IDENTITY, CHAT_SCOPE)


def test_store_failure_fails_closed(monkeypatch):
    fake = FakeDynamoDB(fail_with=RuntimeError("dynamodb unavailable"))
    monkeypatch.setattr(budget, "_ddb", lambda: fake)
    monkeypatch.setattr(budget, "BUDGET_TABLE", BUDGET_TABLE)
    with pytest.raises(budget.BudgetStoreError):
        budget.check_and_consume(IDENTITY, CHAT_SCOPE)


def test_requires_identity_and_scope(fake_ddb):
    with pytest.raises(ValueError):
        budget.check_and_consume("", CHAT_SCOPE)
    with pytest.raises(ValueError):
        budget.check_and_consume(IDENTITY, "")
    assert _counters(fake_ddb) == {}


# --- Defaults are generous (abuse backstops, not rate limits) -----------------------


def test_default_limits():
    assert budget.USER_DAILY_LIMIT == 512
    assert budget.SCOPE_LIMIT == 128
