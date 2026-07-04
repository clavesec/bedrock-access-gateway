"""Always-on, metadata-only structured access log (TPAI external-content
program, m1 Phase B(i) / decision E3).

Replaces the deleted DIAG-tagged content-dump lines. Exactly one JSON line
is emitted per request that reaches a completion/embeddings handler: status,
latency, token counts, tool name, outcome, and the pseudonymous HMAC
identity (``request.state.tpai_identity``) — never message content, never
headers, never the raw email.

Scope and semantics:

- Requests rejected before the handler runs (401/400 from ``api_key_auth``,
  ``require_identity``, or FastAPI request validation) produce no line —
  there is no completion to describe; uvicorn's access log still records
  method/path/status for them.
- Streaming responses have already sent wire status 200 when a failure
  occurs, so ``status`` stays 200 and the failure surfaces in ``outcome``
  (``error``, or ``aborted`` when the client disconnected mid-stream).
  Consumers computing error rates must key on ``outcome``, not ``status``.

There is deliberately no debug flag and no break-glass: production
content-debugging moves to test accounts. The Phase F acceptance check
asserts that no content fields exist in gateway logs; nothing interpolating
request or response data may be added here.
"""

import json
import logging

logger = logging.getLogger("tpai.access")


def emit_access_log(
    *,
    event: str,
    identity: str | None,
    model: str,
    stream: bool | None,
    status: int,
    latency_ms: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    outcome: str,
    tool: str | None = None,
) -> None:
    """Emit the per-request metadata line.

    ``event`` is ``chat_completion`` or ``embeddings`` (``stream`` is None
    for the latter — not applicable). ``tool`` is the server-side tool name
    when the request attempted one (m2 ``web_fetch``), else ``None`` — the
    name only, never the target URL (full URLs live in the WORM audit
    trail, ``api.audit``).
    """
    logger.info(
        json.dumps(
            {
                "event": event,
                "identity": identity,
                "model": model,
                "stream": stream,
                "status": status,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tool": tool,
                "outcome": outcome,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
