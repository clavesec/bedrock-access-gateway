"""Log-hygiene tests (external-content m1 Phase B(i) / decision E3).

Two properties, both required by ADR-1 condition 2:

1. **No content in logs** — neither request nor response message content
   appears in any log record, on any path (non-stream, stream, error), and
   the deleted DIAG marker is gone from the source tree (drift guard).
2. **The always-on metadata line exists** — exactly one structured JSON
   line per completed request carrying status, latency, token counts, tool,
   outcome, and the pseudonymous HMAC identity.

The behavioral log-capture tests are the primary guard; the source-scan
drift guards below are a belt-and-braces backstop, not the acceptance
property itself.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import HTTPException

import api.identity as identity
from api.routers.chat import _stream_with_access_log
from tests.conftest import AUTH, CHAT_BODY, StubBedrockModel, expected_hmac

TEST_EMAIL = "alice@example.com"
EXPECTED_IDENTITY = expected_hmac("owui-email", TEST_EMAIL)
IDENTITY_HEADERS = {**AUTH, identity.OWUI_EMAIL_HEADER: TEST_EMAIL}

# Sentinel content that must never surface in a log record.
SENTINEL = "PHI-SENTINEL patient John Doe, MRN 0012345"
SENTINEL_BODY = {
    "model": CHAT_BODY["model"],
    "messages": [{"role": "user", "content": SENTINEL}],
}


def access_lines(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "tpai.access"]


# --- The metadata line (schema + values) --------------------------------------


def test_non_stream_emits_one_metadata_line(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=IDENTITY_HEADERS)
    assert resp.status_code == 200

    lines = access_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert isinstance(line.pop("latency_ms"), int)
    assert line == {
        "event": "chat_completion",
        "identity": EXPECTED_IDENTITY,
        "model": CHAT_BODY["model"],
        "stream": False,
        "status": 200,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "tool": None,
        "outcome": "success",
    }


def test_stream_emits_metadata_line_with_usage(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/chat/completions",
        json={**CHAT_BODY, "stream": True},
        headers=IDENTITY_HEADERS,
    )
    assert resp.status_code == 200
    assert "[DONE]" in resp.text

    lines = access_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert line["stream"] is True
    assert line["status"] == 200
    assert line["outcome"] == "success"
    assert line["prompt_tokens"] == 1
    assert line["completion_tokens"] == 1
    assert line["identity"] == EXPECTED_IDENTITY


def test_error_path_emits_metadata_line(client, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)

    async def throttled(self, chat_request):
        raise HTTPException(status_code=429, detail="Too many requests")

    monkeypatch.setattr(StubBedrockModel, "chat", throttled)
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=IDENTITY_HEADERS)
    assert resp.status_code == 429

    lines = access_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["status"] == 429
    assert lines[0]["outcome"] == "error"
    assert lines[0]["prompt_tokens"] is None


def test_error_path_stream_field_is_bool(client, caplog, monkeypatch):
    """ChatRequest.stream is `bool | None`; the error path must coerce None
    to False so the line's stream field never drifts from the schema."""
    caplog.set_level(logging.DEBUG)

    async def broken(self, chat_request):
        raise HTTPException(status_code=500, detail="boom")

    monkeypatch.setattr(StubBedrockModel, "chat", broken)
    resp = client.post(
        "/api/v1/chat/completions",
        json={**CHAT_BODY, "stream": None},
        headers=IDENTITY_HEADERS,
    )
    assert resp.status_code == 500

    lines = access_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["stream"] is False


def test_stream_error_recorded_in_outcome(client, caplog, monkeypatch):
    """chat_stream converts internal failures into an SSE error event on a
    wire-status-200 response; the metadata line still says outcome=error."""
    caplog.set_level(logging.DEBUG)

    async def broken_stream(self, chat_request):
        self.stream_usage = None
        self.stream_error = True
        yield b'data: {"error": {"message": "stream failed"}}\n\n'

    monkeypatch.setattr(StubBedrockModel, "chat_stream", broken_stream)
    resp = client.post(
        "/api/v1/chat/completions",
        json={**CHAT_BODY, "stream": True},
        headers=IDENTITY_HEADERS,
    )
    assert resp.status_code == 200

    lines = access_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["outcome"] == "error"


def test_client_disconnect_recorded_as_aborted():
    """A client dropping mid-stream closes the generator (GeneratorExit);
    the line must say aborted, not success."""
    emitted = []

    async def scenario():
        model = StubBedrockModel()
        gen = _stream_with_access_log(
            model,
            type("Req", (), {"model": "m"})(),
            lambda **kw: emitted.append(kw),
        )
        await anext(gen)  # first chunk delivered, then the client vanishes
        await gen.aclose()

    asyncio.run(scenario())
    assert len(emitted) == 1
    assert emitted[0]["outcome"] == "aborted"
    assert emitted[0]["status"] == 200


def test_metadata_line_emitted_when_enforcement_disabled(client, caplog, monkeypatch):
    """Pre-flip / rolled-back deployments still get the line, identity=null."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(identity, "IDENTITY_HMAC_KEY", "")
    resp = client.post("/api/v1/chat/completions", json=CHAT_BODY, headers=AUTH)
    assert resp.status_code == 200

    lines = access_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["identity"] is None


def test_embeddings_emits_metadata_line(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/embeddings",
        json={"model": "cohere.embed-multilingual-v3", "input": ["hi"]},
        headers=IDENTITY_HEADERS,
    )
    assert resp.status_code == 200

    lines = access_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert isinstance(line.pop("latency_ms"), int)
    assert line == {
        "event": "embeddings",
        "identity": EXPECTED_IDENTITY,
        "model": "cohere.embed-multilingual-v3",
        "stream": None,
        "status": 200,
        "prompt_tokens": 3,
        "completion_tokens": None,
        "tool": None,
        "outcome": "success",
    }


# --- No content in logs, ever (the E3 acceptance property) ---------------------


def assert_no_content_logged(caplog, resp_content: str = "Hello."):
    for record in caplog.records:
        rendered = record.getMessage()
        assert SENTINEL not in rendered, (
            f"request content leaked into log record from {record.name}:{record.lineno}"
        )
        # Also catch truncated leaks that drop the head of the sentinel.
        assert "John Doe" not in rendered
        assert resp_content not in rendered, (
            f"response content leaked into log record from {record.name}:{record.lineno}"
        )


def test_no_content_in_logs_non_stream(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post("/api/v1/chat/completions", json=SENTINEL_BODY, headers=IDENTITY_HEADERS)
    assert resp.status_code == 200
    # The response itself carries the assistant content — logs must not.
    assert "Hello." in resp.text
    assert_no_content_logged(caplog)


def test_no_content_in_logs_stream(client, caplog):
    caplog.set_level(logging.DEBUG)
    resp = client.post(
        "/api/v1/chat/completions",
        json={**SENTINEL_BODY, "stream": True},
        headers=IDENTITY_HEADERS,
    )
    assert resp.status_code == 200
    assert "Hello." in resp.text
    assert_no_content_logged(caplog)


def test_no_content_in_logs_on_error_path(client, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)

    async def broken(self, chat_request):
        raise HTTPException(status_code=500, detail="upstream failure")

    monkeypatch.setattr(StubBedrockModel, "chat", broken)
    resp = client.post("/api/v1/chat/completions", json=SENTINEL_BODY, headers=IDENTITY_HEADERS)
    assert resp.status_code == 500
    assert_no_content_logged(caplog)


# --- Drift guards ---------------------------------------------------------------
#
# Backstops only: the behavioral tests above are the acceptance property.

DIAG_MARKER = "TPAI-" + "DIAG"  # split so this file never matches itself


def source_files(root: Path):
    return (p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_diag_markers_in_source():
    """The DIAG-line deletion is permanent — nothing may reintroduce the marker."""
    src_root = Path(__file__).resolve().parents[1]
    offenders = [str(p) for p in source_files(src_root) if DIAG_MARKER in p.read_text()]
    assert offenders == []


def test_no_body_dump_logging_in_source():
    """No logger call may serialize a request/response object — the historic
    content sinks were `model_dump_json()` and body/chunk str() dumps fed to
    logger.info under DEBUG."""
    src_root = Path(__file__).resolve().parents[1] / "api"
    offenders = []
    for path in source_files(src_root):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "logger." in line and (
                "model_dump_json" in line
                or "str(response_body)" in line
                or "str(chunk)" in line
            ):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert offenders == []
