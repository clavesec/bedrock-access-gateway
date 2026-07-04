import time
from typing import Annotated, AsyncIterable, Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.access_log import emit_access_log
from api.auth import api_key_auth
from api.identity import require_identity
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL
from api.taint import capture_conversation
from api.tools.executor import ServerToolContext

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth), Depends(require_identity), Depends(capture_conversation)],
    # responses={404: {"description": "Not found"}},
)


async def _stream_with_access_log(
    model: BedrockModel,
    chat_request: ChatRequest,
    emit: Callable[..., None],
    tool_ctx=None,
) -> AsyncIterable[bytes]:
    """Pass the stream through untouched; emit the metadata line at the end.

    ``chat_stream`` converts internal errors into an SSE error event on a
    wire-status-200 response, recording ``stream_error`` on the model — so
    outcome, not status, reflects streaming failures. A client disconnect
    surfaces as GeneratorExit and is recorded as ``aborted``.
    """
    outcome = "success"
    try:
        async for chunk in model.chat_stream(chat_request, tool_ctx):
            yield chunk
    except GeneratorExit:
        outcome = "aborted"
        raise
    except BaseException:
        outcome = "error"
        raise
    finally:
        if model.stream_error:
            outcome = "error"
        emit(stream=True, status=200, outcome=outcome, usage=model.stream_usage)


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def chat_completions(
    request: Request,
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            ],
        ),
    ],
):
    started = time.monotonic()
    identity = getattr(request.state, "tpai_identity", None)

    # Identity/scope material for server-side tools (m2 web_fetch loop).
    # None whenever identity enforcement is off — planning then declines and
    # the request takes the exact pre-m2 path.
    tool_ctx = None
    binding = getattr(request.state, "tpai_mint_binding", None)
    if identity and binding:
        tool_ctx = ServerToolContext(
            identity=identity,
            binding=binding,
            subject_id=getattr(request.state, "tpai_mint_subject_id", None),
            chat_id=getattr(request.state, "tpai_chat_id", None),
        )

    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Bound before emit's closure reads it: the except path emits even when
    # model construction itself failed.
    model = None

    def emit(*, stream: bool, status: int, outcome: str, usage=None) -> None:
        emit_access_log(
            event="chat_completion",
            identity=identity,
            model=chat_request.model,
            stream=stream,
            status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            outcome=outcome,
            tool=getattr(model, "server_tool_used", None),
        )

    try:
        model = BedrockModel()
        # Exception will be raised if model not supported.
        model.validate(chat_request)
        if chat_request.stream:
            return StreamingResponse(
                content=_stream_with_access_log(model, chat_request, emit, tool_ctx),
                media_type="text/event-stream",
            )
        response = await model.chat(chat_request, tool_ctx)
    except Exception as exc:
        emit(
            stream=bool(chat_request.stream),
            status=exc.status_code if isinstance(exc, HTTPException) else 500,
            outcome="error",
        )
        raise

    emit(stream=False, status=200, outcome="success", usage=response.usage)
    return response
