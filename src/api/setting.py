import os

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"
ENABLE_APPLICATION_INFERENCE_PROFILES = os.environ.get("ENABLE_APPLICATION_INFERENCE_PROFILES", "true").lower() != "false"
ENABLE_TIKTOKEN_DECODING = os.environ.get("ENABLE_TIKTOKEN_DECODING", "false").lower() != "false"

# --- Built-in web_fetch tool ---------------------------------------------------
# Master switch for the server-side web_fetch tool. Default OFF so the gateway
# behaves identically to before until egress + allowlist are in place.
ENABLE_WEB_FETCH_TOOL = os.environ.get("ENABLE_WEB_FETCH_TOOL", "false").lower() != "false"
# When false, web_fetch is only injected for Anthropic (Claude) models. Set true
# to offer it to every tool-capable Converse model.
WEB_FETCH_MODELS_ALL = os.environ.get("WEB_FETCH_MODELS_ALL", "false").lower() != "false"
# Max number of server-side fetch rounds before the gateway forces a final answer.
WEB_FETCH_MAX_ITERATIONS = int(os.environ.get("WEB_FETCH_MAX_ITERATIONS", "4"))
# Per-request read timeout (seconds) for a single fetch.
WEB_FETCH_TIMEOUT_S = int(os.environ.get("WEB_FETCH_TIMEOUT_S", "8"))
# Hard cap on bytes downloaded per fetch (default 2 MiB).
WEB_FETCH_MAX_BYTES = int(os.environ.get("WEB_FETCH_MAX_BYTES", str(2 * 1024 * 1024)))
# Hard cap on extracted characters returned to the model per fetch.
WEB_FETCH_MAX_CHARS = int(os.environ.get("WEB_FETCH_MAX_CHARS", "50000"))
# Comma-separated domain allowlist, enforced gateway-side before dispatch —
# defense-in-depth on top of the connector's URL-policy layer (which owns the
# real decision, R5). Empty = beta allow-all: any URL from the human turn that
# the connector's SSRF guard and policy accept.
WEB_FETCH_ALLOWED_DOMAINS = os.environ.get("WEB_FETCH_ALLOWED_DOMAINS", "")
# Surface a "🔎 Fetching <host>…" status line into the stream when fetching.
WEB_FETCH_STREAM_STATUS = os.environ.get("WEB_FETCH_STREAM_STATUS", "false").lower() != "false"
# Append a cachePoint block to the growing conversation on loop continuation
# rounds (Claude only) so repeated context is served from the prompt cache.
# Off until m2 Phase 2 (R9).
WEB_FETCH_PROMPT_CACHE = os.environ.get("WEB_FETCH_PROMPT_CACHE", "false").lower() != "false"
# Read timeout (seconds) for the gateway->connector call. Must exceed the
# connector's own origin-fetch timeout (WEB_FETCH_TIMEOUT_S, enforced
# connector-side) plus its quarantined-model extraction pass (R7).
WEB_FETCH_CONNECTOR_TIMEOUT_S = int(os.environ.get("WEB_FETCH_CONNECTOR_TIMEOUT_S", "30"))
# Base URL of the external-content connector's PrivateLink interface endpoint
# (https://vpce-….vpce-svc-….…), injected by the bedrock-gateway-stack CDK
# wiring since S09. Unset leaves the whole fetch path dark (fail closed).
TPAI_CONNECTOR_URL = os.environ.get("TPAI_CONNECTOR_URL", "")
