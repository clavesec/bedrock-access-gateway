import time
from typing import Annotated, AsyncIterable

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.access_log import emit_chat_access_log
from api.auth import api_key_auth
from api.identity import require_identity
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth), Depends(require_identity)],
    # responses={404: {"description": "Not found"}},
)


async def _stream_with_access_log(
    model: BedrockModel,
    chat_request: ChatRequest,
    identity: str | None,
    started: float,
) -> AsyncIterable[bytes]:
    """Pass the stream through untouched; emit the metadata line at the end.

    ``chat_stream`` converts internal errors into an SSE error event on a
    wire-status-200 response, recording ``stream_error`` on the model — the
    outcome field reflects that even though the HTTP status cannot.
    """
    try:
        async for chunk in model.chat_stream(chat_request):
            yield chunk
    finally:
        usage = getattr(model, "stream_usage", None)
        emit_chat_access_log(
            identity=identity,
            model=chat_request.model,
            stream=True,
            status=200,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            outcome="error" if getattr(model, "stream_error", False) else "success",
        )


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

    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    model = BedrockModel()
    try:
        # Exception will be raised if model not supported.
        model.validate(chat_request)
        if chat_request.stream:
            return StreamingResponse(
                content=_stream_with_access_log(model, chat_request, identity, started),
                media_type="text/event-stream",
            )
        response = await model.chat(chat_request)
    except Exception as exc:
        emit_chat_access_log(
            identity=identity,
            model=chat_request.model,
            stream=chat_request.stream,
            status=exc.status_code if isinstance(exc, HTTPException) else 500,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=None,
            completion_tokens=None,
            outcome="error",
        )
        raise

    usage = response.usage
    emit_chat_access_log(
        identity=identity,
        model=chat_request.model,
        stream=False,
        status=200,
        latency_ms=int((time.monotonic() - started) * 1000),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        outcome="success",
    )
    return response
