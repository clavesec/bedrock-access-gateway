import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterable

from api.schema import (
    # Chat
    ChatRequest,
    ChatResponse,
    ChatStreamResponse,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
    Error,
)

logger = logging.getLogger(__name__)


class BaseChatModel(ABC):
    """Represent a basic chat model

    Currently, only Bedrock model is supported, but may be used for SageMaker models if needed.
    """

    # Streaming metadata contract, read by the access log once the stream
    # ends: chat_stream implementations must set stream_usage when the
    # provider reports token usage, and stream_error when they swallow a
    # failure into an in-band SSE error event.
    stream_usage = None
    stream_error = False
    # Name of the server-side tool the request attempted (m2 web_fetch loop);
    # the access log's E3 metadata line carries the tool name only.
    server_tool_used = None

    def list_models(self) -> list[str]:
        """Return a list of supported models"""
        return []

    def validate(self, chat_request: ChatRequest):
        """Validate chat completion requests."""
        pass

    @abstractmethod
    async def chat(self, chat_request: ChatRequest, tool_ctx=None) -> ChatResponse:
        """Handle a basic chat completion requests. ``tool_ctx`` carries the
        request's identity/scope material for server-side tools (m2)."""
        pass

    @abstractmethod
    async def chat_stream(self, chat_request: ChatRequest, tool_ctx=None) -> AsyncIterable[bytes]:
        """Handle a basic chat completion requests with stream response."""
        pass

    @staticmethod
    def generate_message_id() -> str:
        return "chatcmpl-" + str(uuid.uuid4())[:8]

    @staticmethod
    def stream_response_to_bytes(response: ChatStreamResponse | Error | None = None) -> bytes:
        if isinstance(response, Error):
            logger.error("Stream error: %s", response.error.message if response.error else "Unknown error")
            data = response.model_dump_json()
        elif isinstance(response, ChatStreamResponse):
            # to populate other fields when using exclude_unset=True
            response.system_fingerprint = "fp"
            response.object = "chat.completion.chunk"
            response.created = int(time.time())
            data = response.model_dump_json(exclude_unset=True)
        else:
            data = "[DONE]"

        return f"data: {data}\n\n".encode("utf-8")


class BaseEmbeddingsModel(ABC):
    """Represents a basic embeddings model.

    Currently, only Bedrock-provided models are supported, but it may be used for SageMaker models if needed.
    """

    @abstractmethod
    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Handle a basic embeddings request."""
        pass
