"""Tool-call budget counters (TPAI external-content program, m1 Phase D).

Per-turn caps (bytes, timeouts) bound a single fetch; budgets bound the
*aggregate* — exfiltration bandwidth and cost DoS — per pseudonymous HMAC
identity. Two windows, both consumed before every server-side tool
execution (m2/m3; client tools are never budgeted):

    user-day   all tool calls by one identity in a UTC calendar day —
               the durable backstop
    scope      tool calls within one taint scope (conversation via
               X-OpenWebUI-Chat-Id, or the session-keyed fallback — the
               same scope key as api.taint, so api-proxy callers are
               covered from v1, R12). For session scopes this is a cap per
               sliding 15-minute window, not a cumulative cap: the counter
               expires with its scope. Conversation-scope counters live as
               long as the conversation's taint state.

Limits are config-driven with generous defaults — these are abuse
backstops, not rate limits:

    TPAI_BUDGET_USER_DAILY_LIMIT   default 512
    TPAI_BUDGET_SCOPE_LIMIT        default 128

Counters live in DynamoDB (``TPAI_BUDGET_TABLE``): multiple gateway tasks
run behind the ALB, so process-local counting undercounts. Each counter is
an atomic conditional increment — ``ADD n 1`` guarded by ``n < limit`` —
so concurrent calls cannot push a counter past its limit. (``ADD`` is not
idempotent: a retried write whose first attempt committed can count one
call twice. The limit invariant still holds — the over-count only ever errs
toward denying sooner.)

Accounting semantics (accepted for the simplicity of never needing rollback
writes): the user-day counter is consumed first, then the scope counter, so
a call denied by the scope check burns one user-day unit; and a call denied
afterwards by the taint rule has already consumed both windows. Both burns
are self-inflicted (scope keys embed the caller's own identity) and bounded
— and the executor contract has the loop check its already-known taint
state *before* consuming budget, which removes the taint-deny burn for
every case except a genuine write race.

Operating contract (same posture as ``api.audit`` / ``api.taint``):

- **Fail closed.** Store errors — including an unconfigured table — raise
  ``BudgetStoreError``; a tool call whose budget cannot be checked must not
  execute.
- **Exhaustion is a clean deny**, not an error: ``BudgetExceededError``
  carries the tripped window; the m2 executor turns it into a denied tool
  call (audit ``policy_decision=deny``, ``policy_reason=budget-exhausted``)
  and a clean toolResult error to the model.
- **Blocking I/O** — wrap in ``run_in_threadpool`` from async code, like
  ``api.audit``.
- **Dark today.** Nothing calls into this module until m2 Phase 0 lands.
"""

import datetime
import logging
import os
import time

from api import ddb
from api import taint

logger = logging.getLogger(__name__)

BUDGET_TABLE = os.environ.get("TPAI_BUDGET_TABLE", "")

USER_DAILY_LIMIT = int(os.environ.get("TPAI_BUDGET_USER_DAILY_LIMIT", "512"))
SCOPE_LIMIT = int(os.environ.get("TPAI_BUDGET_SCOPE_LIMIT", "128"))

WINDOW_USER_DAY = "user-day"
WINDOW_SCOPE = "scope"


class BudgetStoreError(RuntimeError):
    """Budget state could not be read or recorded. Callers must fail the
    tool call — never execute with an unenforced budget."""


class BudgetExceededError(RuntimeError):
    """The identity has exhausted a budget window; the tool call is denied
    (clean deny, not a failure)."""

    def __init__(self, window: str, limit: int):
        self.window = window
        self.limit = limit
        super().__init__(f"budget exhausted: {window} window (limit {limit})")


def _ddb():
    """Indirection over the shared client so tests can stub this module's
    store without touching the other consumers."""
    return ddb.client()


def _consume(counter_key: str, limit: int, window: str, ttl_epoch: int) -> None:
    """Atomically consume one unit, or raise.

    The condition allows the write when the counter is absent (first call)
    or strictly below the limit — the increment lands at most at ``limit``.
    """
    try:
        _ddb().update_item(
            TableName=BUDGET_TABLE,
            Key={"pk": {"S": counter_key}},
            UpdateExpression="ADD n :one SET #ttl = :ttl",
            ConditionExpression="attribute_not_exists(n) OR n < :limit",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":one": {"N": "1"},
                ":limit": {"N": str(limit)},
                ":ttl": {"N": str(ttl_epoch)},
            },
        )
    except Exception as exc:
        if ddb.is_conditional_check_failure(exc):
            raise BudgetExceededError(window, limit)
        logger.error("budget write failed (%s)", type(exc).__name__)
        raise BudgetStoreError("failed to consume budget") from exc


def check_and_consume(identity: str, scope: str) -> None:
    """Consume one tool call from both budget windows, before execution.

    Raises ``BudgetExceededError`` (deny) when either window is exhausted,
    ``BudgetStoreError`` (fail) when the store is unreachable or
    unconfigured, ``ValueError`` for calls without an identity.
    """
    if not identity:
        raise ValueError("budget enforcement requires a non-empty HMAC identity")
    if not scope:
        raise ValueError("budget enforcement requires a non-empty taint scope")
    if not BUDGET_TABLE:
        raise BudgetStoreError(
            "TPAI_BUDGET_TABLE is not configured - refusing to execute an unbudgeted tool call"
        )

    now = time.time()
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    # User-day counter: expires two days after creation — comfortably past
    # the UTC day it counts, without a midnight-boundary TTL computation.
    _consume(
        counter_key=f"day#{identity}#{today}",
        limit=USER_DAILY_LIMIT,
        window=WINDOW_USER_DAY,
        ttl_epoch=int(now) + 2 * 86400,
    )

    # Scope counter: keyed by the taint-scope key (which embeds the
    # identity), with the lifetime taken from the scope itself
    # (taint.scope_ttl_epoch) so the counter and the scope's taint item can
    # never expire on different schedules. Rotating chat ids to mint fresh
    # scope budgets is bounded by the user-day counter above.
    _consume(
        counter_key=f"scope#{scope}",
        limit=SCOPE_LIMIT,
        window=WINDOW_SCOPE,
        ttl_epoch=taint.scope_ttl_epoch(scope, now),
    )
