import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC
from typing import AsyncIterable, Iterable, Literal
from urllib.parse import urlsplit

import boto3
import numpy as np
import requests
from botocore.config import Config
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

# Early import logging to catch any issues
logger = logging.getLogger(__name__)
logger.info("🟢 BEDROCK.PY: Starting imports - EARLY INSTRUMENTATION CHECK")

from api import setting
from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    Embedding,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Error,
    ErrorMessage,
    Function,
    ImageContent,
    ResponseFunction,
    TextContent,
    ToolCall,
    ToolContent,
    ToolMessage,
    Usage,
    UserMessage,
)
from api.setting import (
    AWS_REGION,
    DEBUG,
    DEFAULT_MODEL,
    ENABLE_APPLICATION_INFERENCE_PROFILES,
    ENABLE_CROSS_REGION_INFERENCE,
    ENABLE_TIKTOKEN_DECODING,
)
from api.tools import executor, web_fetch

logger = logging.getLogger(__name__)

config = Config(
            connect_timeout=60,      # Connection timeout: 60 seconds
            read_timeout=900,        # Read timeout: 15 minutes (suitable for long streaming responses)
            retries={
                'max_attempts': 8,   # Maximum retry attempts
                'mode': 'adaptive'   # Adaptive retry mode
            },
            max_pool_connections=50  # Maximum connection pool size
        )

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=AWS_REGION,
    config=config,
)


def get_inference_region_prefix():
    if AWS_REGION.startswith("ap-"):
        return "apac"
    return AWS_REGION[:2]


# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
cr_inference_prefix = get_inference_region_prefix()

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    "amazon.titan-embed-text-v2:0": "Titan Embeddings G2 - Text",
    # Disable Titan embedding.
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

# Lazy tiktoken initialization with comprehensive logging
ENCODER = None

logger.info("🚀 BEDROCK.PY MODULE LOADING - Starting module import")
logger.info(f"🔒 ENABLE_TIKTOKEN_DECODING = {ENABLE_TIKTOKEN_DECODING}")
logger.info("📅 CODE VERSION: LAZY_TIKTOKEN_V2_WITH_INSTRUMENTATION_2025_07_21")
logger.info("✅ BEDROCK.PY MODULE LOADED - tiktoken import deferred successfully")

# If you see this in logs, the new code is deployed correctly
logger.info("🎯 VERIFICATION: This message proves lazy tiktoken code is active")

def _get_tiktoken_encoder():
    """Lazy initialization of tiktoken encoder - only imports when actually needed"""
    global ENCODER
    logger.info(f"🔍 _get_tiktoken_encoder() called - ENCODER={ENCODER}, ENABLE_TIKTOKEN_DECODING={ENABLE_TIKTOKEN_DECODING}")
    
    if ENCODER is None and ENABLE_TIKTOKEN_DECODING:
        logger.info("📥 Attempting lazy tiktoken import...")
        try:
            import tiktoken  # Import only when needed, not at module level
            logger.info("📦 tiktoken imported successfully, getting encoding...")
            ENCODER = tiktoken.get_encoding("cl100k_base")
            logger.info("✅ tiktoken encoder initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize tiktoken encoder: {e}")
            ENCODER = False  # Use False to indicate failed initialization
    elif ENCODER is None:
        logger.info("🚫 tiktoken decoding disabled, skipping initialization")
    else:
        logger.info(f"♻️ Using cached tiktoken encoder: {type(ENCODER)}")
    
    result = ENCODER if ENCODER is not False else None
    logger.info(f"🔄 _get_tiktoken_encoder() returning: {type(result)}")
    return result


def list_bedrock_models() -> dict:
    """Automatically getting a list of supported models.

    Returns a model list combines:
        - ON_DEMAND models.
        - Cross-Region Inference Profiles (if enabled via Env)
        - Application Inference Profiles (if enabled via Env)
    """
    model_list = {}
    try:
        profile_list = []
        app_profile_dict = {}
        
        if ENABLE_CROSS_REGION_INFERENCE:
            # List system defined inference profile IDs
            response = bedrock_client.list_inference_profiles(maxResults=1000, typeEquals="SYSTEM_DEFINED")
            profile_list = [p["inferenceProfileId"] for p in response["inferenceProfileSummaries"]]

        if ENABLE_APPLICATION_INFERENCE_PROFILES:
            # List application defined inference profile IDs and create mapping
            response = bedrock_client.list_inference_profiles(maxResults=1000, typeEquals="APPLICATION")
            
            for profile in response["inferenceProfileSummaries"]:
                try:
                    profile_arn = profile.get("inferenceProfileArn")
                    if not profile_arn:
                        continue
                    
                    # Process all models in the profile
                    models = profile.get("models", [])
                    for model in models:
                        model_arn = model.get("modelArn", "")
                        if model_arn:
                            model_id = model_arn.split('/')[-1] if '/' in model_arn else model_arn
                            if model_id:
                                app_profile_dict[model_id] = profile_arn
                except Exception as e:
                    logger.warning(f"Error processing application profile: {e}")
                    continue

        # List foundation models, only cares about text outputs here.
        response = bedrock_client.list_foundation_models(byOutputModality="TEXT")

        for model in response["modelSummaries"]:
            model_id = model.get("modelId", "N/A")
            stream_supported = model.get("responseStreamingSupported", True)
            status = model["modelLifecycle"].get("status", "ACTIVE")

            # currently, use this to filter out rerank models and legacy models
            if not stream_supported or status not in ["ACTIVE", "LEGACY"]:
                continue

            inference_types = model.get("inferenceTypesSupported", [])
            input_modalities = model["inputModalities"]
            # Add on-demand model list
            if "ON_DEMAND" in inference_types:
                model_list[model_id] = {"modalities": input_modalities}

            # Add cross-region inference model list.
            profile_id = cr_inference_prefix + "." + model_id
            if profile_id in profile_list:
                model_list[profile_id] = {"modalities": input_modalities}

            # Add application inference profiles
            if model_id in app_profile_dict:
                model_list[app_profile_dict[model_id]] = {"modalities": input_modalities}

    except Exception as e:
        logger.error(f"Unable to list models: {str(e)}")

    if not model_list:
        # In case stack not updated.
        model_list[DEFAULT_MODEL] = {"modalities": ["TEXT", "IMAGE"]}

    return model_list


# Initialize the model list lazily
bedrock_model_list = {}


class BedrockModel(BaseChatModel):
    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        global bedrock_model_list
        try:
            bedrock_model_list = list_bedrock_models()
        except Exception as e:
            logger.error(f"Failed to list bedrock models: {e}")
            # Fallback to default model if AWS calls fail
            if not bedrock_model_list:
                bedrock_model_list = {DEFAULT_MODEL: {"modalities": ["TEXT", "IMAGE"]}}
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # Ensure model list is initialized
        global bedrock_model_list
        if not bedrock_model_list:
            try:
                bedrock_model_list = list_bedrock_models()
            except Exception as e:
                logger.warning(f"Failed to list bedrock models during validation: {e}")
                # Fallback to default model
                bedrock_model_list = {DEFAULT_MODEL: {"modalities": ["TEXT", "IMAGE"]}}
        
        # check if model is supported
        if chat_request.model not in bedrock_model_list.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"
            logger.error("Unsupported model: %s", chat_request.model)

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False, args_override: dict | None = None):
        """Common logic for invoke bedrock models.

        Deliberately no request/response body logging here — not even behind
        DEBUG (external-content decision E3): message content must never
        reach CloudWatch. Metadata-only logging lives in api.access_log.

        ``args_override`` carries the already-built Converse args across the
        server-tool loop's continuation rounds (the loop appends toolResult
        messages between calls); without it the args are parsed fresh.
        """
        # convert OpenAI chat request to Bedrock SDK request
        args = args_override if args_override is not None else self._parse_request(chat_request)

        try:
            if stream:
                # Run the blocking boto3 call in a thread pool
                response = await run_in_threadpool(
                    bedrock_runtime.converse_stream, **args
                )
            else:
                # Run the blocking boto3 call in a thread pool
                response = await run_in_threadpool(bedrock_runtime.converse, **args)
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Bedrock validation error for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except bedrock_runtime.exceptions.ThrottlingException as e:
            logger.warning("Bedrock throttling for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error("Bedrock invocation failed for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=500, detail=str(e))
        return response

    async def chat(self, chat_request: ChatRequest, tool_ctx=None) -> ChatResponse:
        """Default implementation for Chat API.

        When the m2 server-tool plan is active (flag on, identity present,
        no client tools, fetchable human-turn URLs), the request runs through
        the internal Converse tool loop instead of the single-shot path.
        """

        message_id = self.generate_message_id()
        plan = await self._plan_server_tools(chat_request, tool_ctx)
        if plan is not None:
            return await self._chat_tool_loop(chat_request, plan, tool_ctx, message_id)
        response = await self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]

        chat_response = self._create_response(
            model=chat_request.model,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return chat_response

    async def _plan_server_tools(self, chat_request: ChatRequest, tool_ctx):
        """Plan server-tool injection for this request, or None.

        The cheap I/O-free gate runs first so the default/dark path never
        pays the threadpool hop (planning reads taint state in DynamoDB).
        """
        if tool_ctx is None or not executor.server_tools_enabled(chat_request):
            return None
        return await run_in_threadpool(executor.plan_server_tools, chat_request, tool_ctx)

    async def _execute_tool_round(self, plan, tool_ctx, tool_uses: list[dict], fetch_allowed: bool):
        """Run one round of server-tool calls through the executor choke
        point; returns ``[(toolUseId, ToolOutcome), ...]`` in request order.

        Every branch — including the iteration-cap deny and unexpected
        executor failures (routed through ``executor.unexpected_failure`` so
        even a crashed call leaves an audit record) — produces a toolResult,
        so the continuation conversation always answers every toolUse
        (Converse requires it). Calls within a round are independent and run
        concurrently; the per-call control order lives inside the executor.
        """

        async def run_one(tool_use: dict):
            name = tool_use["name"]
            # The raw model-emitted name is attacker-influenceable (via
            # fetched content); only a registered name may reach the E3
            # access-log line.
            self.server_tool_used = executor.loggable_tool_name(name)
            try:
                if not fetch_allowed:
                    outcome = await run_in_threadpool(
                        executor.deny_iteration_cap, plan, tool_ctx, name, tool_use["input"]
                    )
                else:
                    outcome = await run_in_threadpool(
                        executor.run_server_tool, plan, tool_ctx, name, tool_use["input"]
                    )
            except Exception as exc:
                # E3: exception class only — no URL, no content.
                logger.error("server tool execution failed (%s)", type(exc).__name__)
                outcome = await run_in_threadpool(
                    executor.unexpected_failure, plan, tool_ctx, name, tool_use["input"]
                )
            return (tool_use["toolUseId"], outcome)

        return list(await asyncio.gather(*(run_one(tool_use) for tool_use in tool_uses)))

    def _append_tool_round(self, messages: list, assistant_content: list, results, model: str) -> None:
        """Append the assistant turn and its toolResults to the loop
        conversation, with an optional cachePoint on the growing prefix (R9).

        cachePoints are pruned to Claude's four-checkpoint ceiling (oldest
        first) — a run of WEB_FETCH_MAX_ITERATIONS+1 rounds would otherwise
        accumulate five markers and fail Converse validation.
        """
        messages.append({"role": "assistant", "content": assistant_content})
        content = [
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": outcome.result_text}],
                    "status": "success" if outcome.ok else "error",
                }
            }
            for tool_use_id, outcome in results
        ]
        if setting.WEB_FETCH_PROMPT_CACHE and "anthropic" in model.lower():
            content.append({"cachePoint": {"type": "default"}})
        messages.append({"role": "user", "content": content})
        self._prune_cache_points(messages)

    @staticmethod
    def _prune_cache_points(messages: list, max_points: int = 4) -> None:
        """Drop the oldest cachePoint blocks beyond Claude's per-request
        checkpoint limit, keeping the most recent prefixes cacheable."""
        positions = [
            (message_index, block_index)
            for message_index, message in enumerate(messages)
            if isinstance(message.get("content"), list)
            for block_index, block in enumerate(message["content"])
            if isinstance(block, dict) and "cachePoint" in block
        ]
        excess = positions[: max(0, len(positions) - max_points)]
        for message_index, block_index in reversed(excess):
            del messages[message_index]["content"][block_index]

    async def _chat_tool_loop(self, chat_request: ChatRequest, plan, tool_ctx, message_id: str) -> ChatResponse:
        """Internal Converse loop for server-side tools (non-stream).

        Invariants (m2 plan / superseded web-fetch plan):
        - only reached when the client declared no tools, so nothing here can
          disturb client tool passthrough;
        - executes only while every toolUse in the turn is a planned server
          tool; anything else fails closed inside the executor;
        - at most ``WEB_FETCH_MAX_ITERATIONS`` fetch rounds, then one denied
          round, then a forced text-only final — the loop is bounded at
          ``WEB_FETCH_MAX_ITERATIONS + 2`` Converse calls.
        """
        args = self._parse_request(chat_request)
        args["toolConfig"] = plan.tool_config
        messages = args["messages"]
        total_input = 0
        total_output = 0
        fetch_rounds = 0
        denied_final = False
        while True:
            response = await self._invoke_bedrock(chat_request, args_override=args)
            total_input += response["usage"]["inputTokens"]
            total_output += response["usage"]["outputTokens"]
            content = response["output"]["message"]["content"]
            stop_reason = response["stopReason"]
            if stop_reason != "tool_use":
                return self._create_response(
                    model=chat_request.model,
                    message_id=message_id,
                    content=content,
                    finish_reason=stop_reason,
                    input_tokens=total_input,
                    output_tokens=total_output,
                )
            tool_uses = [part["toolUse"] for part in content if "toolUse" in part]
            if denied_final:
                # The model kept calling tools after the denied round: audit
                # the attempts (the WORM trail records attempts, not just
                # executions), then force a text-only final. The client
                # declared no tools, so surfacing tool_calls here would
                # break it (loop invariant).
                await self._execute_tool_round(plan, tool_ctx, tool_uses, fetch_allowed=False)
                text = "\n".join(part["text"] for part in content if "text" in part)
                return self._create_response(
                    model=chat_request.model,
                    message_id=message_id,
                    content=[{"text": text}],
                    finish_reason="end_turn",
                    input_tokens=total_input,
                    output_tokens=total_output,
                )
            fetch_allowed = fetch_rounds < setting.WEB_FETCH_MAX_ITERATIONS
            results = await self._execute_tool_round(plan, tool_ctx, tool_uses, fetch_allowed)
            if fetch_allowed:
                fetch_rounds += 1
            else:
                denied_final = True
            self._append_tool_round(messages, content, results, chat_request.model)

    async def _async_iterate(self, stream):
        """Helper method to convert sync iterator to async iterator"""
        for chunk in stream:
            await run_in_threadpool(lambda: chunk)
            yield chunk

    async def chat_stream(self, chat_request: ChatRequest, tool_ctx=None) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API.

        Records ``stream_usage`` (token counts from the Bedrock metadata
        chunk) and ``stream_error`` on the instance so the router can emit
        the metadata-only access-log line once the stream finishes.

        With an active server-tool plan the stream is produced by the tool
        loop instead: web_fetch toolUse deltas are suppressed, text deltas
        forwarded, and a fresh converse_stream started per continuation.
        """
        self.stream_usage = None
        self.stream_error = False
        try:
            plan = await self._plan_server_tools(chat_request, tool_ctx)
        except Exception as e:
            # Planning failures must surface as the documented in-band SSE
            # error on the wire-200 stream, not a severed connection. E3:
            # the log line carries the exception class only.
            self.stream_error = True
            logger.error("Stream planning error for model %s (%s)", chat_request.model, type(e).__name__)
            yield self.stream_response_to_bytes(Error(error=ErrorMessage(message=str(e))))
            return
        if plan is not None:
            async for chunk_bytes in self._chat_stream_tool_loop(chat_request, plan, tool_ctx):
                yield chunk_bytes
            return
        try:
            response = await self._invoke_bedrock(chat_request, stream=True)
            message_id = self.generate_message_id()
            stream = response.get("stream")
            async for chunk in self._async_iterate(stream):
                args = {"model_id": chat_request.model, "message_id": message_id, "chunk": chunk}
                stream_response = self._create_response_stream(**args)
                if not stream_response:
                    continue
                if stream_response.usage:
                    self.stream_usage = stream_response.usage
                if stream_response.choices:
                    yield self.stream_response_to_bytes(stream_response)
                elif chat_request.stream_options and chat_request.stream_options.include_usage:
                    # An empty choices for Usage as per OpenAI doc below:
                    # if you set stream_options: {"include_usage": true}.
                    # an additional chunk will be streamed before the data: [DONE] message.
                    # The usage field on this chunk shows the token usage statistics for the entire request,
                    # and the choices field will always be an empty array.
                    # All other chunks will also include a usage field, but with a null value.
                    yield self.stream_response_to_bytes(stream_response)

            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()
        except Exception as e:
            self.stream_error = True
            logger.error("Stream error for model %s: %s", chat_request.model, str(e))
            error_event = Error(error=ErrorMessage(message=str(e)))
            yield self.stream_response_to_bytes(error_event)

    def _stream_chunk(
        self, message_id: str, model: str, message: ChatResponseMessage, finish_reason: str | None = None
    ) -> bytes:
        """One OpenAI-format SSE chunk for the tool-loop stream path."""
        return self.stream_response_to_bytes(
            ChatStreamResponse(
                id=message_id,
                model=model,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
            )
        )

    async def _chat_stream_tool_loop(self, chat_request: ChatRequest, plan, tool_ctx) -> AsyncIterable[bytes]:
        """Internal Converse loop for server-side tools (stream).

        Per the m2 loop invariants: web_fetch toolUse deltas are buffered and
        suppressed — the client never sees tool_calls it did not declare —
        text/reasoning deltas are forwarded as they arrive, each continuation
        starts a fresh converse_stream, and usage is summed across rounds
        into a single final usage chunk. An optional "Fetching <host>…"
        status line is surfaced when WEB_FETCH_STREAM_STATUS is on.
        """
        message_id = self.generate_message_id()
        model = chat_request.model
        try:
            args = self._parse_request(chat_request)
            args["toolConfig"] = plan.tool_config
            messages = args["messages"]
            total_input = 0
            total_output = 0
            fetch_rounds = 0
            denied_final = False
            first_round = True
            while True:
                response = await self._invoke_bedrock(chat_request, stream=True, args_override=args)
                stream = response.get("stream")
                blocks: dict[int, dict] = {}
                stop_reason = None
                async for chunk in self._async_iterate(stream):
                    if "messageStart" in chunk:
                        if first_round:
                            yield self._stream_chunk(
                                message_id, model, ChatResponseMessage(role="assistant", content="")
                            )
                    elif "contentBlockStart" in chunk:
                        start = chunk["contentBlockStart"]["start"]
                        index = chunk["contentBlockStart"]["contentBlockIndex"]
                        if "toolUse" in start:
                            # Buffered for execution; never forwarded.
                            blocks[index] = {
                                "toolUseId": start["toolUse"]["toolUseId"],
                                "name": start["toolUse"]["name"],
                                "input_json": "",
                            }
                    elif "contentBlockDelta" in chunk:
                        delta = chunk["contentBlockDelta"]["delta"]
                        index = chunk["contentBlockDelta"]["contentBlockIndex"]
                        if "text" in delta:
                            # List-append, joined once at round end — string
                            # += here would be quadratic over long outputs.
                            entry = blocks.setdefault(index, {"text_parts": []})
                            entry.setdefault("text_parts", []).append(delta["text"])
                            yield self._stream_chunk(
                                message_id, model, ChatResponseMessage(content=delta["text"])
                            )
                        elif "reasoningContent" in delta:
                            # Defensive: reasoning requests are excluded from
                            # planning; forward text deltas if one appears.
                            if "text" in delta["reasoningContent"]:
                                yield self._stream_chunk(
                                    message_id,
                                    model,
                                    ChatResponseMessage(
                                        reasoning_content=delta["reasoningContent"]["text"]
                                    ),
                                )
                        elif "toolUse" in delta:
                            entry = blocks.get(index)
                            if entry is not None and "toolUseId" in entry:
                                entry["input_json"] += delta["toolUse"]["input"]
                    elif "messageStop" in chunk:
                        stop_reason = chunk["messageStop"]["stopReason"]
                    elif "metadata" in chunk:
                        usage = chunk["metadata"].get("usage")
                        if usage:
                            total_input += usage["inputTokens"]
                            total_output += usage["outputTokens"]
                first_round = False
                if stop_reason != "tool_use":
                    yield self._stream_chunk(
                        message_id, model, ChatResponseMessage(), finish_reason=stop_reason
                    )
                    break
                assistant_content = []
                tool_uses = []
                for index in sorted(blocks):
                    entry = blocks[index]
                    if "toolUseId" in entry:
                        try:
                            tool_input = json.loads(entry["input_json"]) if entry["input_json"] else {}
                        except ValueError:
                            tool_input = {}
                        if not isinstance(tool_input, dict):
                            tool_input = {}
                        tool_use = {
                            "toolUseId": entry["toolUseId"],
                            "name": entry["name"],
                            "input": tool_input,
                        }
                        assistant_content.append({"toolUse": tool_use})
                        tool_uses.append(tool_use)
                    elif entry.get("text_parts"):
                        assistant_content.append({"text": "".join(entry["text_parts"])})
                if denied_final:
                    # The model kept calling tools after the denied round:
                    # audit the attempts, then end the stream cleanly (tool
                    # deltas were suppressed).
                    await self._execute_tool_round(plan, tool_ctx, tool_uses, fetch_allowed=False)
                    yield self._stream_chunk(
                        message_id, model, ChatResponseMessage(), finish_reason="end_turn"
                    )
                    break
                fetch_allowed = fetch_rounds < setting.WEB_FETCH_MAX_ITERATIONS
                if fetch_allowed and setting.WEB_FETCH_STREAM_STATUS:
                    for tool_use in tool_uses:
                        url = web_fetch.resolve_url(plan.urls, tool_use["input"])
                        if url:
                            host = urlsplit(url).hostname or ""
                            yield self._stream_chunk(
                                message_id,
                                model,
                                ChatResponseMessage(content=f"\n\n🔎 Fetching {host}…\n\n"),
                            )
                results = await self._execute_tool_round(plan, tool_ctx, tool_uses, fetch_allowed)
                if fetch_allowed:
                    fetch_rounds += 1
                else:
                    denied_final = True
                self._append_tool_round(messages, assistant_content, results, model)

            self.stream_usage = Usage(
                prompt_tokens=total_input,
                completion_tokens=total_output,
                total_tokens=total_input + total_output,
            )
            if chat_request.stream_options and chat_request.stream_options.include_usage:
                yield self.stream_response_to_bytes(
                    ChatStreamResponse(
                        id=message_id, model=model, choices=[], usage=self.stream_usage
                    )
                )
            yield self.stream_response_to_bytes()
        except Exception as e:
            self.stream_error = True
            # E3: class name only — on this path exception text can embed
            # request material (the URL-carrying toolSpec, fenced content).
            logger.error("Stream error for model %s (%s)", model, type(e).__name__)
            error_event = Error(error=ErrorMessage(message=str(e)))
            yield self.stream_response_to_bytes(error_event)

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts.
        Note that not all models support system prompts.

        example output: [{"text" : system_prompt}]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """

        system_prompts = []
        for message in chat_request.messages:
            if message.role != "system":
                # ignore system messages here
                continue
            assert isinstance(message.content, str)
            system_prompts.append({"text": message.content})

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                # Check if message has content that's not empty
                has_content = False
                if isinstance(message.content, str):
                    has_content = message.content.strip() != ""
                elif isinstance(message.content, list):
                    has_content = len(message.content) > 0
                elif message.content is not None:
                    has_content = True
                
                if has_content:
                    # Text message
                    messages.append(
                        {
                            "role": message.role,
                            "content": self._parse_content_parts(
                                message, chat_request.model
                            ),
                        }
                    )
                if message.tool_calls:
                    # Tool use message
                    for tool_call in message.tool_calls:
                        tool_input = json.loads(tool_call.function.arguments)
                        messages.append(
                            {
                                "role": message.role,
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": tool_call.id,
                                            "name": tool_call.function.name,
                                            "input": tool_input,
                                        }
                                    }
                                ],
                            }
                        )
            elif isinstance(message, ToolMessage):
                # Bedrock does not support tool role,
                # Add toolResult to content
                # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
                
                # Handle different content formats from OpenAI SDK
                tool_content = self._extract_tool_content(message.content)
                
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.tool_call_id,
                                    "content": [{"text": tool_content}],
                                }
                            }
                        ],
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return self._reframe_multi_payloard(messages)

    def _extract_tool_content(self, content) -> str:
        """Extract text content from various OpenAI SDK tool message formats.
        
        Handles:
        - String content (legacy format)
        - List of content objects (OpenAI SDK 1.91.0+)
        - Nested JSON structures within text content
        """
        try:
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                text_parts = []
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        # Handle dict with 'text' field
                        if "text" in item:
                            item_text = item["text"]
                            if isinstance(item_text, str):
                                # Try to parse as JSON if it looks like JSON
                                if item_text.strip().startswith('{') and item_text.strip().endswith('}'):
                                    try:
                                        parsed_json = json.loads(item_text)
                                        # Convert JSON object to readable text
                                        text_parts.append(json.dumps(parsed_json, indent=2))
                                    except json.JSONDecodeError:
                                        # Silently fallback to original text
                                        text_parts.append(item_text)
                                else:
                                    text_parts.append(item_text)
                            else:
                                text_parts.append(str(item_text))
                        else:
                            # Handle other dict formats - convert to JSON string
                            text_parts.append(json.dumps(item, indent=2))
                    elif hasattr(item, 'text'):
                        # Handle ToolContent objects
                        text_parts.append(item.text)
                    else:
                        # Convert any other type to string
                        text_parts.append(str(item))
                return "\n".join(text_parts)
            
            # Fallback for any other type
            return str(content)
        except Exception as e:
            logger.warning("Tool content extraction failed: %s", str(e))
            # Return a safe fallback
            return str(content) if content is not None else ""

    def _reframe_multi_payloard(self, messages: list) -> list:
        """Receive messages and reformat them to comply with the Claude format

        With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
        with Claude format requests, you cannot repeatedly receive messages from the same role.

        This method searches through the OpenAI format messages in order and reformats them to the Claude format.

        ```
        openai_format_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Who are you?"},
        ]

        bedrock_format_messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Hello"},
                    {"text": "Who are you?"}
                ]
            },
        ]
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message["role"]
            next_content = message["content"]

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append(
                        {"role": current_role, "content": current_content}
                    )
                # Switch to the new role
                current_role = next_role
                current_content = []

            # Add the message content to current_content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)

        # Add the last role's messages to the list
        if current_content:
            reformatted_messages.append(
                {"role": current_role, "content": current_content}
            )

        return reformatted_messages

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        # Base inference parameters.
        inference_config = {
            "maxTokens": chat_request.max_tokens,
        }
        if chat_request.temperature is not None:
            inference_config["temperature"] = chat_request.temperature
        if chat_request.top_p is not None:
            inference_config["topP"] = chat_request.top_p

        if chat_request.stop is not None:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            inference_config["stopSequences"] = stop

        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        if chat_request.reasoning_effort:
            # From OpenAI api, the max_token is not supported in reasoning mode
            # Use max_completion_tokens if provided.

            max_tokens = (
                chat_request.max_completion_tokens
                if chat_request.max_completion_tokens
                else chat_request.max_tokens
            )
            budget_tokens = self._calc_budget_tokens(
                max_tokens, chat_request.reasoning_effort
            )
            inference_config["maxTokens"] = max_tokens
            # unset topP - Not supported
            inference_config.pop("topP", None)

            args["additionalModelRequestFields"] = {
                "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
            }

        # Anthropic models reject requests with both temperature and top_p.
        # Drop top_p, keeping temperature as the primary sampling control.
        if "anthropic" in chat_request.model.lower():
            inference_config.pop("topP", None)

        # add tool config
        if chat_request.tools:
            tool_config = {"tools": [self._convert_tool_spec(t.function) for t in chat_request.tools]}

            if chat_request.tool_choice and not chat_request.model.startswith(
                "meta.llama3-1-"
            ):
                if isinstance(chat_request.tool_choice, str):
                    # auto (default) is mapped to {"auto" : {}}
                    # required is mapped to {"any" : {}}
                    if chat_request.tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    assert "function" in chat_request.tool_choice
                    tool_config["toolChoice"] = {"tool": {"name": chat_request.tool_choice["function"].get("name", "")}}
            args["toolConfig"] = tool_config
        # add Additional fields to enable extend thinking
        if chat_request.extra_body:
            # reasoning_config will not be used 
            args["additionalModelRequestFields"] = chat_request.extra_body
        return args

    def _create_response(
        self,
        model: str,
        message_id: str,
        content: list[dict] | None = None,
        finish_reason: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ChatResponse:
        message = ChatResponseMessage(
            role="assistant",
        )
        if finish_reason == "tool_use":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html#tool-use-examples
            tool_calls = []
            for part in content:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            id=tool["toolUseId"],
                            type="function",
                            function=ResponseFunction(
                                name=tool["name"],
                                arguments=json.dumps(tool["input"]),
                            ),
                        )
                    )
            message.tool_calls = tool_calls
            message.content = None
        else:
            message.content = ""
            for c in content:
                if "reasoningContent" in c:
                    message.reasoning_content = c["reasoningContent"][
                        "reasoningText"
                    ].get("text", "")
                elif "text" in c:
                    message.content = c["text"]
                else:
                    logger.warning(
                        "Unknown tag in message content " + ",".join(c.keys())
                    )

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
        self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        finish_reason = None
        message = None
        usage = None
        if "messageStart" in chunk:
            message = ChatResponseMessage(
                role=chunk["messageStart"]["role"],
                content="",
            )
        if "contentBlockStart" in chunk:
            # tool call start
            delta = chunk["contentBlockStart"]["start"]
            if "toolUse" in delta:
                # first index is content
                index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            type="function",
                            id=delta["toolUse"]["toolUseId"],
                            function=ResponseFunction(
                                name=delta["toolUse"]["name"],
                                arguments="",
                            ),
                        )
                    ]
                )
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                # stream content
                message = ChatResponseMessage(
                    content=delta["text"],
                )
            elif "reasoningContent" in delta:
                # ignore "signature" in the delta.
                if "text" in delta["reasoningContent"]:
                    message = ChatResponseMessage(
                        reasoning_content=delta["reasoningContent"]["text"],
                    )
            else:
                # tool use
                index = chunk["contentBlockDelta"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            function=ResponseFunction(
                                arguments=delta["toolUse"]["input"],
                            ),
                        )
                    ]
                )
        if "messageStop" in chunk:
            message = ChatResponseMessage()
            finish_reason = chunk["messageStop"]["stopReason"]

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=metadata["usage"]["inputTokens"],
                        completion_tokens=metadata["usage"]["outputTokens"],
                        total_tokens=metadata["usage"]["totalTokens"],
                    ),
                )
        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                usage=usage,
            )

        return None

    def _parse_image(self, image_url: str) -> tuple[bytes, str]:
        """Try to get the raw data from an image url.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
        returns a tuple of (Image Data, Content Type)
        """
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)
        # if already base64 encoded.
        # Only supports 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)

        # Send a request to the image URL
        response = requests.get(image_url)
        # Check if the request was successful
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type")
            if not content_type.startswith("image"):
                content_type = "image/jpeg"
            # Get the image content
            image_content = response.content
            return image_content, content_type
        else:
            raise HTTPException(
                status_code=500, detail="Unable to access the image url"
            )

    def _parse_content_parts(
        self,
        message: UserMessage | AssistantMessage,
        model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self.is_supported_modality(model_id, modality="IMAGE"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        global bedrock_model_list
        # Ensure model list is initialized
        if not bedrock_model_list:
            try:
                bedrock_model_list = list_bedrock_models()
            except Exception as e:
                logger.warning(f"Failed to list bedrock models for modality check: {e}")
                # Fallback to default model
                bedrock_model_list = {DEFAULT_MODEL: {"modalities": ["TEXT", "IMAGE"]}}
        
        model = bedrock_model_list.get(model_id, {})
        modalities = model.get("modalities", [])
        if modality in modalities:
            return True
        return False

    def _convert_tool_spec(self, func: Function) -> dict:
        return {
            "toolSpec": {
                "name": func.name,
                "description": func.description,
                "inputSchema": {
                    "json": func.parameters,
                },
            }
        }

    def _calc_budget_tokens(
        self, max_tokens: int, reasoning_effort: Literal["low", "medium", "high"]
    ) -> int:
        # Helper function to calculate budget_tokens based on the max_tokens.
        # Ratio for efforts:  Low - 30%, medium - 60%, High: Max token - 1
        # Note that The minimum budget_tokens is 1,024 tokens so far.
        # But it may be changed for different models in the future.
        if reasoning_effort == "low":
            return int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            return int(max_tokens * 0.6)
        else:
            return max_tokens - 1

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "finished": "stop",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "complete": "stop",
                "content_filtered": "content_filter",
            }
            return finish_reason_mapping.get(
                finish_reason.lower(), finish_reason.lower()
            )
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            # Model id only — request bodies are never logged (E3).
            logger.info("Invoke Bedrock Model: " + model_id)
        try:
            return bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except bedrock_runtime.exceptions.ThrottlingException as e:
            logger.error("Throttling Error: " + str(e))
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def _create_response(
        self,
        embeddings: list[float],
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        return response


class CohereEmbeddingsModel(BedrockEmbeddingsModel):
    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        texts = []
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            # For encoded input
            # The workaround is to use tiktoken to decode to get the original text.
            encoder = _get_tiktoken_encoder()
            if encoder is None:
                raise HTTPException(
                    status_code=400,
                    detail="Encoded input is not supported when tiktoken decoding is disabled. "
                           "Set ENABLE_TIKTOKEN_DECODING=true to support encoded inputs, "
                           "or provide text input directly."
                )
            
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    # Iterable[int]
                    encodings.append(inner)
                else:
                    # Iterable[Iterable[int]]
                    text = encoder.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(encoder.decode(encodings))

        # Maximum of 2048 characters
        args = {
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END",  # "NONE|START|END"
        }
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        return self._create_response(
            embeddings=response_body["embeddings"],
            model=embeddings_request.model,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):
    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        if isinstance(embeddings_request.input, str):
            input_text = embeddings_request.input
        elif (
            isinstance(embeddings_request.input, list)
            and len(embeddings_request.input) == 1
        ):
            input_text = embeddings_request.input[0]
        else:
            raise ValueError(
                "Amazon Titan Embeddings models support only single strings as input."
            )
        args = {
            "inputText": input_text,
            # Note: inputImage is not supported!
        }
        if embeddings_request.model == "amazon.titan-embed-image-v1":
            args["embeddingConfig"] = (
                embeddings_request.embedding_config
                if embeddings_request.embedding_config
                else {"outputEmbeddingLength": 1024}
            )
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        return self._create_response(
            embeddings=[response_body["embedding"]],
            model=embeddings_request.model,
            input_tokens=response_body["inputTextTokenCount"],
        )


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    match model_name:
        case "Cohere Embed Multilingual" | "Cohere Embed English":
            return CohereEmbeddingsModel()
        case "Titan Embeddings G2 - Text":
            return TitanEmbeddingsModel()
        case _:
            logger.error("Unsupported model id " + model_id)
            raise HTTPException(
                status_code=400,
                detail="Unsupported embedding model id " + model_id,
            )
