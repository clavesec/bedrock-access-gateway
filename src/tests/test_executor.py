"""Tests for api.tools.executor — planning gates and the execution choke
point (m2 Phase 0).

Planning must decline for every gate in docs/WebFetch.md (leaving the
request byte-identical to the pre-m2 path), and run_server_tool must apply
the docs/TaintAndBudgets.md control order with exactly one audit record on
every branch. Control internals (conditional-write semantics, HMAC, S3
puts) are covered by their own suites; here they are stubbed and their
call order recorded.
"""

import logging

import pytest

from api import audit, budget, mint, setting, taint
from api.schema import ChatRequest, Function, Tool, UserMessage
from api.tools import executor, web_fetch

MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

CTX = executor.ServerToolContext(
    identity="hmac-identity",
    binding=mint.BINDING_OWUI_SESSION,
    subject_id="user-1",
    chat_id="chat-1",
)


def chat_request(**overrides) -> ChatRequest:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Fetch https://a.example/1 please"}],
    }
    body.update(overrides)
    return ChatRequest(**body)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(setting, "ENABLE_WEB_FETCH_TOOL", True)
    monkeypatch.setattr(setting, "WEB_FETCH_ALLOWED_DOMAINS", "")


@pytest.fixture
def taint_clean(monkeypatch):
    monkeypatch.setattr(taint, "get_taint", lambda scope: None)


# ------------------------------------------------------------------ planning


def test_flag_off_means_no_plan_and_no_store_read(monkeypatch):
    def explode(scope):
        raise AssertionError("taint store must not be read when the flag is off")

    monkeypatch.setattr(taint, "get_taint", explode)
    assert executor.server_tools_enabled(chat_request()) is False
    assert executor.plan_server_tools(chat_request(), CTX) is None


def test_client_tools_disable_injection(enabled, taint_clean):
    request = chat_request(
        tools=[Tool(type="function", function=Function(name="my_tool", parameters={}))]
    )
    assert executor.server_tools_enabled(request) is False
    assert executor.plan_server_tools(request, CTX) is None


def test_reasoning_requests_are_excluded(enabled, taint_clean):
    assert executor.plan_server_tools(chat_request(reasoning_effort="low"), CTX) is None
    assert executor.plan_server_tools(chat_request(extra_body={"x": 1}), CTX) is None


def test_non_claude_models_excluded_unless_models_all(enabled, taint_clean, monkeypatch):
    request = chat_request(model="meta.llama3-1-70b-instruct-v1:0")
    assert executor.plan_server_tools(request, CTX) is None
    monkeypatch.setattr(setting, "WEB_FETCH_MODELS_ALL", True)
    assert executor.plan_server_tools(request, CTX) is not None


def test_missing_identity_context_means_no_plan(enabled, taint_clean):
    assert executor.plan_server_tools(chat_request(), None) is None


def test_no_human_turn_urls_means_no_plan(enabled, taint_clean):
    request = chat_request(messages=[{"role": "user", "content": "no links here"}])
    assert executor.plan_server_tools(request, CTX) is None


def test_unreadable_taint_state_fails_toward_no_injection(enabled, monkeypatch):
    def broken(scope):
        raise taint.TaintStoreError("down")

    monkeypatch.setattr(taint, "get_taint", broken)
    assert executor.plan_server_tools(chat_request(), CTX) is None


def test_inbound_private_taint_drops_web_fetch(enabled, monkeypatch):
    monkeypatch.setattr(taint, "get_taint", lambda scope: taint.INBOUND_PRIVATE)
    assert executor.plan_server_tools(chat_request(), CTX) is None


def test_plan_carries_scope_urls_and_tool_config(enabled, taint_clean):
    plan = executor.plan_server_tools(chat_request(), CTX)
    assert plan is not None
    assert plan.scope == "chat#hmac-identity#chat-1"
    assert plan.urls == ["https://a.example/1"]
    assert plan.injected == ["web_fetch"]
    assert plan.dropped == []
    assert plan.tool_config["toolChoice"] == {"auto": {}}
    assert plan.tool_config["tools"][0]["toolSpec"]["name"] == "web_fetch"


def test_plan_uses_session_scope_without_chat_id(enabled, taint_clean):
    ctx = executor.ServerToolContext(
        identity="hmac-identity", binding=mint.BINDING_API_KEY, subject_id="u", chat_id=None
    )
    plan = executor.plan_server_tools(chat_request(), ctx)
    assert plan.scope == "session#hmac-identity"


# ---------------------------------------------------------- the choke point


class Recorder:
    """Stubs every control run_server_tool sequences, recording call order."""

    def __init__(self, monkeypatch):
        self.order: list[str] = []
        self.records: list[dict] = []
        self.fetches: list[str] = []
        self.invalidated: list[str] = []
        monkeypatch.setattr(budget, "check_and_consume", self._budget)
        monkeypatch.setattr(taint, "record_tool_use", self._taint)
        monkeypatch.setattr(mint, "get_connector_token", self._mint)
        monkeypatch.setattr(mint, "invalidate", self.invalidated.append)
        monkeypatch.setattr(audit, "emit_audit_record", self._audit)
        monkeypatch.setattr(web_fetch, "execute_web_fetch", self._fetch)

    budget_exc = None
    taint_exc = None
    mint_exc = None
    fetch_exc = None
    audit_exc = None

    def _budget(self, identity, scope):
        self.order.append("budget")
        if self.budget_exc:
            raise self.budget_exc

    def _taint(self, scope, tool_name):
        self.order.append("taint")
        if self.taint_exc:
            raise self.taint_exc
        return taint.OUTBOUND_FETCH

    def _mint(self, identity, binding, subject_id):
        self.order.append("mint")
        if self.mint_exc:
            raise self.mint_exc
        return mint.MintedToken(token="jwt", expires_at=2**31, subject_id=subject_id)

    def _fetch(self, url, token):
        self.order.append("fetch")
        self.fetches.append(url)
        if self.fetch_exc:
            exc = self.fetch_exc
            # One-shot so the 401-retry test can succeed on the second call.
            self.fetch_exc = None
            raise exc
        return web_fetch.FetchResult(text="PAGE", url=url, bytes_returned=4, truncated=False)

    def _audit(self, **kwargs):
        self.order.append("audit")
        self.records.append(kwargs)
        if self.audit_exc:
            raise self.audit_exc


@pytest.fixture
def recorder(monkeypatch, enabled, taint_clean):
    return Recorder(monkeypatch)


@pytest.fixture
def plan(enabled, taint_clean):
    return executor.plan_server_tools(chat_request(), CTX)


def test_success_path_order_and_audit_record(recorder, plan):
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.ok
    assert recorder.order == ["budget", "taint", "mint", "fetch", "audit"]
    assert "PAGE" in outcome.result_text
    assert "TPAI-EXTERNAL-CONTENT" in outcome.result_text
    record = recorder.records[0]
    assert record["policy_decision"] == "allow"
    assert record["policy_reason"] == "beta-allow-all"
    assert record["outcome"] == "success"
    assert record["target"] == "https://a.example/1"
    assert record["identity"] == "hmac-identity"
    assert record["conversation_id"] == "chat-1"
    assert record["bytes_returned"] == 4


def test_invalid_url_index_denies_before_any_store_call(recorder, plan):
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url": "https://evil.example"})
    assert outcome.outcome == "denied"
    assert recorder.order == ["audit"]  # deny record only — no budget, no taint, no fetch
    assert recorder.records[0]["policy_reason"] == "invalid-url-index"
    assert recorder.fetches == []


def test_gateway_allowlist_miss_denies_before_budget(recorder, plan, monkeypatch):
    monkeypatch.setattr(setting, "WEB_FETCH_ALLOWED_DOMAINS", "docs.example.org")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.order == ["audit"]
    assert recorder.records[0]["policy_reason"] == "allowlist-miss"


def test_budget_exhausted_is_a_clean_deny(recorder, plan):
    recorder.budget_exc = budget.BudgetExceededError("user-day", 512)
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.order == ["budget", "audit"]
    assert recorder.records[0]["policy_reason"] == "budget-exhausted"
    assert recorder.fetches == []


def test_taint_conflict_is_a_clean_deny_after_budget(recorder, plan):
    recorder.taint_exc = taint.TaintConflictError(plan.scope, "web_fetch", taint.INBOUND_PRIVATE)
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.order == ["budget", "taint", "audit"]
    assert recorder.records[0]["policy_reason"] == "taint-blocked"
    assert recorder.fetches == []


def test_store_errors_fail_closed(recorder, plan):
    recorder.budget_exc = budget.BudgetStoreError("down")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "error"
    assert recorder.records[0]["policy_reason"] == "budget-store-error"
    assert recorder.fetches == []

    recorder.budget_exc = None
    recorder.taint_exc = taint.TaintStoreError("down")
    recorder.order.clear()
    recorder.records.clear()
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "error"
    assert recorder.records[0]["policy_reason"] == "taint-store-error"
    assert recorder.fetches == []


def test_mint_refusal_is_a_clean_deny(recorder, plan):
    recorder.mint_exc = mint.MintRefusedError("no-active-session")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.records[0]["policy_reason"] == "mint-refused:no-active-session"
    assert recorder.fetches == []


def test_connector_401_reauths_exactly_once(recorder, plan):
    recorder.fetch_exc = web_fetch.WebFetchAuthError("stale")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.ok
    assert recorder.invalidated == ["hmac-identity"]
    assert recorder.order == ["budget", "taint", "mint", "fetch", "mint", "fetch", "audit"]


def test_connector_denial_is_audited_with_the_connector_reason(recorder, plan):
    recorder.fetch_exc = web_fetch.WebFetchDenied("ssrf-blocked")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.records[0]["policy_decision"] == "deny"
    assert recorder.records[0]["policy_reason"] == "ssrf-blocked"


def test_fetch_timeout_is_audited_as_allow_timeout(recorder, plan):
    recorder.fetch_exc = web_fetch.WebFetchError("slow", outcome="timeout")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "timeout"
    record = recorder.records[0]
    assert record["policy_decision"] == "allow"
    assert record["outcome"] == "timeout"
    assert record["bytes_returned"] is None


def test_unauditable_success_discards_the_fetched_content(recorder, plan):
    recorder.audit_exc = audit.AuditEmitError("sink down")
    outcome = executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "error"
    assert "PAGE" not in outcome.result_text
    assert "TPAI-EXTERNAL-CONTENT" not in outcome.result_text


def test_unregistered_tool_name_fails_closed(recorder, plan):
    with pytest.raises(ValueError):
        executor.run_server_tool(plan, CTX, "not_a_tool", {"url_index": 0})
    assert recorder.fetches == []


def test_registered_tool_without_a_handler_fails_loudly(recorder, plan, monkeypatch):
    """A taint-classified tool with no execution handler in this build must
    raise at the enforcement point, never be misrouted through another
    tool's pipeline as an invalid-input deny. (gmail gained its handler in
    m3 G3, so the seam is exercised with a hypothetical future tool.)"""
    monkeypatch.setitem(taint.TOOL_CLASSES, "future_tool", taint.INBOUND_PRIVATE)
    future_plan = executor.ServerToolPlan(
        tool_config=plan.tool_config,
        urls=plan.urls,
        scope=plan.scope,
        injected=["future_tool"],
        dropped=[],
    )
    with pytest.raises(ValueError, match="no executor handler"):
        executor.run_server_tool(future_plan, CTX, "future_tool", {"query": "x"})
    assert recorder.fetches == []


def test_every_handler_key_is_taint_classified():
    """The dispatch table must stay a subset of taint.TOOL_CLASSES — a
    handler for an unclassified tool would bypass the taint rule."""
    assert set(executor._HANDLERS) <= set(taint.TOOL_CLASSES)


def test_taint_dropped_deny_records_the_full_url(recorder, plan):
    """R11: even the step-2a deny (advisory filter already removed the
    class) must carry the attempted full URL — resolution is pure and needs
    no store round-trip."""
    dropped_plan = executor.ServerToolPlan(
        tool_config=plan.tool_config,
        urls=plan.urls,
        scope=plan.scope,
        injected=[],
        dropped=["web_fetch"],
    )
    outcome = executor.run_server_tool(dropped_plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.order == ["audit"]  # no budget burn
    record = recorder.records[0]
    assert record["policy_reason"] == "taint-blocked"
    assert record["target"] == "https://a.example/1"


def test_unexpected_failure_writes_a_best_effort_record(recorder, plan):
    outcome = executor.unexpected_failure(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "error"
    record = recorder.records[0]
    assert record["policy_reason"] == "executor-error"
    assert record["target"] == "https://a.example/1"
    # And it swallows audit failures — there is nothing left to fail closed
    # over when the record itself cannot be written.
    recorder.audit_exc = audit.AuditEmitError("sink down")
    outcome = executor.unexpected_failure(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "error"


def test_loggable_tool_name_collapses_unregistered_names():
    assert executor.loggable_tool_name("web_fetch") == "web_fetch"
    assert executor.loggable_tool_name("exfil<data>") == "unregistered-tool"


def test_iteration_cap_denial_is_audited_without_budget_burn(recorder, plan):
    outcome = executor.deny_iteration_cap(plan, CTX, "web_fetch", {"url_index": 0})
    assert outcome.outcome == "denied"
    assert recorder.order == ["audit"]
    record = recorder.records[0]
    assert record["policy_reason"] == "iteration-cap"
    assert record["target"] == "https://a.example/1"


def test_no_urls_or_content_in_executor_logs(recorder, plan, caplog):
    """E3: the choke point never logs the URL, the page text, or the token."""
    with caplog.at_level(logging.DEBUG):
        executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
        recorder.fetch_exc = web_fetch.WebFetchError("boom")
        executor.run_server_tool(plan, CTX, "web_fetch", {"url_index": 0})
    assert "a.example" not in caplog.text
    assert "PAGE" not in caplog.text
    assert "jwt" not in caplog.text
