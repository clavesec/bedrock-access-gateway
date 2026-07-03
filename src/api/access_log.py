"""Always-on, metadata-only structured access log (TPAI external-content
program, m1 Phase B(i) / decision E3).

Replaces the deleted DIAG-tagged content-dump lines. Exactly one JSON line
is emitted per chat completion: status, latency, token counts, tool name,
outcome, and the pseudonymous HMAC identity (``request.state.tpai_identity``)
— never message content, never headers, never the raw email.

There is deliberately no debug flag and no break-glass: production
content-debugging moves to test accounts. The Phase F acceptance check
asserts that no content fields exist in gateway logs; nothing interpolating
request or response data may be added here.
"""

import json
import logging

logger = logging.getLogger("tpai.access")


def emit_chat_access_log(
    *,
    identity: str | None,
    model: str,
    stream: bool,
    status: int,
    latency_ms: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    outcome: str,
    tool: str | None = None,
) -> None:
    """Emit the per-request metadata line.

    ``tool`` is always ``None`` today — server-side tool execution arrives
    with the m1 Phase D taint rule and the m2 ``web_fetch`` tool; the field
    exists so the schema does not change when it does.
    """
    logger.info(
        json.dumps(
            {
                "event": "chat_completion",
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
