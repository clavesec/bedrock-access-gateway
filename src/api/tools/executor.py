"""Single choke point for server-side tool execution (m2 Phase 0).

Implements the binding executor contract in ``docs/TaintAndBudgets.md`` and
the audit requirement in ``docs/AuditRecords.md``: every server-side tool
call — success, denied, error, or timeout — flows through
``run_server_tool``, which sequences the controls in the ratified order and
emits exactly one WORM audit record per call. The Converse loop in
``models/bedrock.py`` never calls taint/budget/audit/mint directly; its
catch-all for unexpected executor failures goes through
``unexpected_failure`` so even a crashed call leaves a best-effort record.

Planning (``plan_server_tools``) decides whether this request gets server
tools at all. The injection list is built from the gateway's own registry
(``taint.TOOL_CLASSES`` keys gated by their feature flags) — never from
names appearing in the request (a client tool that happens to be named
``web_fetch`` is still a client tool and bypasses this module entirely,
because the presence of *any* client tool disables injection).

web_fetch is the only registered tool today; ``run_server_tool`` guards its
dispatch loudly (`ValueError`) so an m3 session that registers gmail tools
in ``taint.TOOL_CLASSES``/``registry_tools`` without adding an execution
handler here fails at the enforcement point instead of being misrouted
through the web_fetch pipeline.

Everything here is blocking I/O (DynamoDB, S3, Lambda, HTTP) — the async
Converse loop must call ``plan_server_tools``, ``run_server_tool``,
``deny_iteration_cap``, and ``unexpected_failure`` via
``starlette.concurrency.run_in_threadpool``.

E3: no log line in this module carries URLs, content, or tokens.

Audit pairing note (extends docs/AuditRecords.md for the fail-closed rows):
``policy_decision=deny`` + ``outcome=denied`` is a clean policy deny;
``policy_decision=deny`` + ``outcome=error|timeout`` is a fail-closed
control failure (store/mint/transport — the call was refused because a
control could not run, not by a policy choice); ``policy_decision=allow``
+ ``outcome=error|timeout`` is an execution failure after the policy
allowed the fetch. ``policy_reason`` names the specific control.
"""

import logging
import time
from dataclasses import dataclass

from api import audit, budget, mint, setting, taint
from api.tools import web_fetch

logger = logging.getLogger(__name__)


def internal_error_text(tool_name: str) -> str:
    """Model-facing text when a control or the transport fails closed.
    Deliberately detail-free: failure specifics belong in metadata logs and
    the audit trail, not in the conversation."""
    return f"{tool_name} failed: internal error. Answer with the information you already have."


@dataclass(frozen=True)
class ServerToolContext:
    """Request-scoped identity/scope material, built by the chat router from
    ``request.state`` (populated by require_identity + capture_conversation)."""

    identity: str
    binding: str
    subject_id: str | None
    chat_id: str | None


@dataclass(frozen=True)
class ServerToolPlan:
    """The per-request decision to inject server tools, plus everything the
    loop and the choke point need to execute them."""

    tool_config: dict
    urls: list[str]
    scope: str
    injected: list[str]
    dropped: list[str]


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call produced: the toolResult text for the model, and
    the audit-aligned outcome for status reporting."""

    result_text: str
    outcome: str  # "success" | "denied" | "error" | "timeout"

    @property
    def ok(self) -> bool:
        return self.outcome == "success"


def loggable_tool_name(tool_name: str) -> str:
    """A safe label for the E3 access-log line: the raw toolUse name is
    model-emitted (and, via fetched content, attacker-influenceable), so
    anything outside the registry is collapsed to a constant."""
    return tool_name if tool_name in taint.TOOL_CLASSES else "unregistered-tool"


def registry_tools(model: str) -> list[str]:
    """Server tools this deployment offers for this model — the gateway's
    own registry, flag-gated. m3 sessions extend this alongside
    ``taint.TOOL_CLASSES`` AND must give ``run_server_tool`` an execution
    handler for every name added here (it fails loudly otherwise)."""
    tools: list[str] = []
    if setting.ENABLE_WEB_FETCH_TOOL and (
        setting.WEB_FETCH_MODELS_ALL or "anthropic" in model.lower()
    ):
        tools.append(web_fetch.TOOL_NAME)
    return tools


def server_tools_enabled(chat_request) -> bool:
    """Cheap, I/O-free pre-gate so the dark/default path never pays a
    threadpool hop: flags off, client tools present, reasoning requests, or
    nothing fetchable all short-circuit here (URL extraction is pure CPU —
    regex over the request — so it belongs on this side of the hop).
    ``plan_server_tools`` re-checks everything."""
    if chat_request.tools:
        # Loop invariant: any client tool present -> the response is returned
        # untouched, so server tools are never injected alongside them.
        return False
    if chat_request.reasoning_effort or chat_request.extra_body:
        # Reasoning turns require round-tripping signed thinking blocks
        # through the continuation conversation — out of scope for v1.
        return False
    tools = registry_tools(chat_request.model)
    if not tools:
        return False
    if tools == [web_fetch.TOOL_NAME] and not web_fetch.extract_human_urls(chat_request.messages):
        # web_fetch is the only candidate and there is nothing fetchable in
        # the human turns (D3) — skip the threadpool hop and the taint read.
        return False
    return True


def plan_server_tools(chat_request, ctx: ServerToolContext | None) -> ServerToolPlan | None:
    """Decide injection for this request. Blocking (reads taint state).

    Returns None — request proceeds exactly as before this feature — when
    the flag family is off, the client declared tools, there is no
    authenticated identity, no human-turn URL exists, taint state is
    unreadable (fail toward fewer capabilities), or the advisory filter
    dropped everything.
    """
    if ctx is None or not ctx.identity or not ctx.binding:
        return None
    if not server_tools_enabled(chat_request):
        return None
    tools = registry_tools(chat_request.model)

    urls = web_fetch.extract_human_urls(chat_request.messages)
    if not urls and web_fetch.TOOL_NAME in tools:
        # Nothing fetchable in the human turns -> nothing to offer (D3).
        tools.remove(web_fetch.TOOL_NAME)
    if not tools:
        return None

    scope = taint.resolve_taint_scope(ctx.identity, ctx.chat_id)
    try:
        tainted = taint.get_taint(scope)
    except taint.TaintStoreError:
        # Advisory read failed: inject nothing. record_tool_use is the
        # authoritative gate, but with no injected tools it is never reached
        # — strictly fewer capabilities, never more.
        logger.warning("taint state unreadable - not injecting server tools (fail closed)")
        return None
    allowed, dropped = taint.filter_tools(tainted, tools)
    if web_fetch.TOOL_NAME not in allowed:
        # web_fetch is the only tool with a spec builder today.
        return None
    return ServerToolPlan(
        tool_config={
            "tools": [web_fetch.build_tool_spec(urls)],
            "toolChoice": {"auto": {}},
        },
        urls=urls,
        scope=scope,
        injected=allowed,
        dropped=dropped,
    )


def _emit_record(
    ctx: ServerToolContext,
    tool_name: str,
    *,
    target: str,
    decision: str,
    reason: str,
    outcome: str,
    bytes_returned: int | None,
    latency_ms: int,
) -> None:
    """The one place audit records are shaped — every emitter in this module
    goes through it so the field contract cannot drift between branches."""
    audit.emit_audit_record(
        identity=ctx.identity,
        tool=tool_name,
        target=target,
        policy_decision=decision,
        policy_reason=reason,
        outcome=outcome,
        bytes_returned=bytes_returned,
        latency_ms=latency_ms,
        conversation_id=ctx.chat_id,
    )


def run_server_tool(plan: ServerToolPlan, ctx: ServerToolContext, tool_name: str, tool_input) -> ToolOutcome:
    """Execute one server tool call through the full control sequence.

    Order (docs/TaintAndBudgets.md): known-deny checks that need no store
    round-trip (dropped-by-filter, unresolvable URL index, gateway
    allowlist) -> budget -> taint record -> mint+execute -> audit at
    completion. Deny branches emit their audit record immediately; the
    execute branch emits exactly one record once the outcome
    (success/error/timeout), byte count, and latency are known — and if
    that record cannot be written, the fetched content is discarded and the
    call fails (a retrieval that cannot be audited must not reach the
    model).
    """
    started = time.monotonic()

    def latency_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    def deny(target: str, reason: str, message: str) -> ToolOutcome:
        _emit_record(
            ctx, tool_name, target=target, decision="deny", reason=reason,
            outcome="denied", bytes_returned=None, latency_ms=latency_ms(),
        )
        return ToolOutcome(result_text=message, outcome="denied")

    def fail(target: str, reason: str) -> ToolOutcome:
        # Fail-closed control failure: decision=deny + outcome=error (see
        # the module-docstring pairing note).
        _emit_record(
            ctx, tool_name, target=target, decision="deny", reason=reason,
            outcome="error", bytes_returned=None, latency_ms=latency_ms(),
        )
        return ToolOutcome(result_text=internal_error_text(tool_name), outcome="error")

    if tool_name not in taint.TOOL_CLASSES:
        # The loop only dispatches names it injected, so this is a bug —
        # fail closed at the enforcement point (TaintAndBudgets contract).
        raise ValueError(f"tool {tool_name!r} is not a registered server-side tool")

    # D3: resolve the input to a human-turn URL first — pure in-memory, and
    # every subsequent record (including denies) then carries the full URL
    # (R11) rather than a placeholder.
    url = web_fetch.resolve_url(plan.urls, tool_input)

    if tool_name in plan.dropped:
        # 2a: the advisory filter already removed this class for the scope —
        # the taint outcome is known, deny without consuming budget.
        return deny(
            url or "invalid:url_index",
            "taint-blocked",
            f"{tool_name} is not available in this conversation (external-content policy).",
        )
    if tool_name not in plan.injected:
        raise ValueError(f"tool {tool_name!r} was not planned for this request")
    if tool_name != web_fetch.TOOL_NAME:
        # Registered and planned but no execution handler in this build —
        # fail loudly rather than misroute through the web_fetch pipeline.
        raise ValueError(f"no executor handler for tool {tool_name!r}")

    if url is None:
        return deny(
            "invalid:url_index",
            "invalid-url-index",
            "web_fetch error: url_index must be one of the listed indices; "
            "URLs cannot be passed directly.",
        )

    try:
        gateway_policy_reason = web_fetch.policy_reason_for(url)
    except web_fetch.WebFetchDenied as exc:
        return deny(url, exc.reason, "web_fetch denied: this destination is not allowed.")

    try:
        budget.check_and_consume(ctx.identity, plan.scope)
    except budget.BudgetExceededError as exc:
        return deny(
            url,
            "budget-exhausted",
            f"{tool_name} denied: the {exc.window} tool-call budget is exhausted. "
            "Answer with the information you already have.",
        )
    except budget.BudgetStoreError:
        return fail(url, "budget-store-error")

    try:
        taint.record_tool_use(plan.scope, tool_name)
    except taint.TaintConflictError:
        return deny(
            url,
            "taint-blocked",
            f"{tool_name} is not available in this conversation (external-content policy).",
        )
    except taint.TaintStoreError:
        return fail(url, "taint-store-error")

    try:
        result = _mint_and_fetch(ctx, url)
    except mint.MintRefusedError as exc:
        return deny(
            url,
            f"mint-refused:{exc.reason}",
            f"{tool_name} denied: no live session or credential for this user.",
        )
    except (mint.MintError, ValueError):
        return fail(url, "mint-error")
    except web_fetch.WebFetchDenied as exc:
        return deny(url, exc.reason, "web_fetch denied: this destination is not allowed.")
    except web_fetch.WebFetchAuthError:
        return fail(url, "connector-auth")
    except web_fetch.WebFetchError as exc:
        _emit_record(
            ctx, tool_name, target=url, decision="allow", reason=gateway_policy_reason,
            outcome=exc.outcome, bytes_returned=None, latency_ms=latency_ms(),
        )
        message = (
            "web_fetch error: the fetch timed out."
            if exc.outcome == "timeout"
            else "web_fetch error: the fetch failed."
        )
        return ToolOutcome(result_text=message, outcome=exc.outcome)

    fenced = web_fetch.fence_external_content(result.text, url)
    if result.truncated:
        fenced += "\n[Content truncated at the configured cap.]"
    try:
        _emit_record(
            ctx, tool_name, target=url, decision="allow", reason=gateway_policy_reason,
            outcome="success", bytes_returned=result.bytes_returned, latency_ms=latency_ms(),
        )
    except audit.AuditEmitError:
        # Retrieval happened but cannot be audited: discard the content so
        # nothing unaudited ever reaches the model (fail closed).
        return ToolOutcome(result_text=internal_error_text(tool_name), outcome="error")
    return ToolOutcome(result_text=fenced, outcome="success")


def deny_iteration_cap(plan: ServerToolPlan, ctx: ServerToolContext, tool_name: str, tool_input) -> ToolOutcome:
    """Deny a tool call because the per-request fetch-round cap is reached.

    Still audited (one deny record per denied call) — the WORM trail records
    attempts, not just executions. No budget/taint consumption: the outcome
    is known without any store round-trip.
    """
    url = web_fetch.resolve_url(plan.urls, tool_input) or "invalid:url_index"
    _emit_record(
        ctx, tool_name, target=url, decision="deny", reason="iteration-cap",
        outcome="denied", bytes_returned=None, latency_ms=0,
    )
    return ToolOutcome(
        result_text=(
            f"{tool_name} denied: the fetch limit for this request is reached. "
            "Answer with the information you already have."
        ),
        outcome="denied",
    )


def unexpected_failure(plan: ServerToolPlan, ctx: ServerToolContext, tool_name: str, tool_input) -> ToolOutcome:
    """Outcome for a tool call whose executor crashed (loop catch-all).

    Best-effort audit so even a crashed call leaves a record ("emits on all
    branches" — docs/AuditRecords.md); if the record itself cannot be
    written there is nothing left to fail closed over, so the emit error is
    swallowed (exception class logged, never fields).
    """
    target = web_fetch.resolve_url(plan.urls, tool_input) if isinstance(tool_input, dict) else None
    try:
        _emit_record(
            ctx, tool_name, target=target or "invalid:url_index", decision="deny",
            reason="executor-error", outcome="error", bytes_returned=None, latency_ms=0,
        )
    except Exception as exc:
        logger.error("audit record for crashed tool call failed (%s)", type(exc).__name__)
    return ToolOutcome(result_text=internal_error_text(tool_name), outcome="error")


def _mint_and_fetch(ctx: ServerToolContext, url: str) -> web_fetch.FetchResult:
    """Mint (cached) and execute the fetch, re-minting exactly once if the
    connector rejects a possibly cached-stale token."""
    token = mint.get_connector_token(ctx.identity, ctx.binding, ctx.subject_id)
    try:
        return web_fetch.execute_web_fetch(url, token.token)
    except web_fetch.WebFetchAuthError:
        mint.invalidate(ctx.identity)
        fresh = mint.get_connector_token(ctx.identity, ctx.binding, ctx.subject_id)
        return web_fetch.execute_web_fetch(url, fresh.token)
