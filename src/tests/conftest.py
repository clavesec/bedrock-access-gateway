"""Shared test fixtures for the gateway test suite.

Environment must be configured before ``api.*`` modules import: both
``api.auth`` and ``api.identity`` read their configuration at import time.
"""

import os

# Exercise the same env-var paths production uses (ECS container secrets
# inject API_KEY and TPAI_IDENTITY_HMAC_KEY as plain env vars).
os.environ.setdefault("API_KEY", "test-gateway-api-key")
os.environ.setdefault("TPAI_IDENTITY_HMAC_KEY", "test-identity-hmac-key")
os.environ.setdefault("AWS_REGION", "us-east-1")
# Hermetic tests: never inherit the developer's AWS profile/credentials —
# api.models.bedrock creates boto3 clients at import time, which resolves
# the profile chain even though no API call is ever made.
os.environ.pop("AWS_PROFILE", None)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.schema import ChatResponse, ChatResponseMessage, Choice, Usage


AUTH = {"Authorization": "Bearer test-gateway-api-key"}

CANNED_RESPONSE = ChatResponse(
    id="chatcmpl-test",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    choices=[
        Choice(
            index=0,
            message=ChatResponseMessage(role="assistant", content="Hello."),
            finish_reason="stop",
        )
    ],
    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
)


class StubBedrockModel:
    """Stands in for BedrockModel so tests never touch AWS."""

    def validate(self, chat_request):
        return None

    async def chat(self, chat_request):
        return CANNED_RESPONSE

    def chat_stream(self, chat_request):
        yield b"data: [DONE]\n\n"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("api.routers.chat.BedrockModel", StubBedrockModel)
    return TestClient(app)


CHAT_BODY = {
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [{"role": "user", "content": "Hello!"}],
}
