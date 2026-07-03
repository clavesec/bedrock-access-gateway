import time
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from api.access_log import emit_access_log
from api.auth import api_key_auth
from api.identity import require_identity
from api.models.bedrock import get_embeddings_model
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL

router = APIRouter(
    prefix="/embeddings",
    dependencies=[Depends(api_key_auth), Depends(require_identity)],
)


@router.post("", response_model=EmbeddingsResponse)
async def embeddings(
    request: Request,
    embeddings_request: Annotated[
        EmbeddingsRequest,
        Body(
            examples=[
                {
                    "model": "cohere.embed-multilingual-v3",
                    "input": ["Your text string goes here"],
                }
            ],
        ),
    ],
):
    started = time.monotonic()
    identity = getattr(request.state, "tpai_identity", None)

    if embeddings_request.model.lower().startswith("text-embedding-"):
        embeddings_request.model = DEFAULT_EMBEDDING_MODEL

    def emit(*, status: int, outcome: str, usage=None) -> None:
        emit_access_log(
            event="embeddings",
            identity=identity,
            model=embeddings_request.model,
            stream=None,
            status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=None,
            outcome=outcome,
        )

    try:
        # Exception will be raised if model not supported.
        model = get_embeddings_model(embeddings_request.model)
        response = model.embed(embeddings_request)
    except Exception as exc:
        emit(
            status=exc.status_code if isinstance(exc, HTTPException) else 500,
            outcome="error",
        )
        raise

    emit(status=200, outcome="success", usage=response.usage)
    return response
