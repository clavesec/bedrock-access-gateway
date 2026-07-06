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
# --- Built-in gmail tools (m3 G3) ----------------------------------------------
# Master switch for the server-side gmail_search / gmail_get_message tools AND
# the /connectors/gmail passthrough surface OWUI's Settings → Connectors page
# uses. Default OFF: the gateway behaves identically to before until the
# connector's gmail adapter (S14) and metadata layer (S15) are enabled.
ENABLE_GMAIL_TOOLS = os.environ.get("ENABLE_GMAIL_TOOLS", "false").lower() != "false"
# When false, gmail tools are only injected for Anthropic (Claude) models —
# same convention as WEB_FETCH_MODELS_ALL.
GMAIL_MODELS_ALL = os.environ.get("GMAIL_MODELS_ALL", "false").lower() != "false"
# Max index records a single gmail_search returns to the model.
GMAIL_SEARCH_MAX_RESULTS = int(os.environ.get("GMAIL_SEARCH_MAX_RESULTS", "20"))
# Hard cap on typed-output characters returned to the model per gmail call.
GMAIL_MAX_CHARS = int(os.environ.get("GMAIL_MAX_CHARS", "50000"))
# Read timeout (seconds) for gateway->connector gmail calls. Must exceed the
# connector's Gmail-API timeout plus its quarantined-model pass (R3).
GMAIL_CONNECTOR_TIMEOUT_S = int(os.environ.get("GMAIL_CONNECTOR_TIMEOUT_S", "30"))
# Hard cap on bytes read from a connector gmail response (default 2 MiB).
GMAIL_CONNECTOR_MAX_BYTES = int(os.environ.get("GMAIL_CONNECTOR_MAX_BYTES", str(2 * 1024 * 1024)))
# Hard cap on the encrypted metadata-index object read from S3 (default 1 MiB
# — the index is ≤200 header records, so anything near this is corrupt).
GMAIL_INDEX_MAX_BYTES = int(os.environ.get("GMAIL_INDEX_MAX_BYTES", str(1024 * 1024)))
# Per-identity DEK cache TTL (seconds). S15 R-5: MUST stay ≤ minutes — the
# metadata-key 403 is the revocation signal, and a cached DEK is the longest
# a revoked identity's index stays readable to the gateway.
GMAIL_DEK_CACHE_TTL_S = int(os.environ.get("GMAIL_DEK_CACHE_TTL_S", "120"))
# The S15 gmail metadata bucket (tpai-gmail-metadata-<env>-<acct>), injected
# by the bedrock-gateway-stack CDK wiring together with the read grants.
# Unset leaves gmail_search dark (fail closed); the connector's metadata-key
# response is verified against this value before any S3 read.
TPAI_GMAIL_METADATA_BUCKET = os.environ.get("TPAI_GMAIL_METADATA_BUCKET", "")

# base64 of the egress-local CA certificate (PEM — a multi-cert rotation
# bundle is accepted — or a single DER cert) that signs the connector's
# boot-time TLS certs (TPAI#365). Set: the connector HTTP session trusts
# EXACTLY this CA and asserts the fixed connector SAN (web_fetch.py
# CONNECTOR_TLS_SERVER_NAME) instead of the URL hostname. Unset: the session
# keeps default public-CA verification (which the connector's egress-local
# cert can never satisfy — fail closed). Injected by the bedrock-gateway-stack
# CDK wiring; only the connector session reads it, never the default bundle.
TPAI_CONNECTOR_CA_B64 = os.environ.get("TPAI_CONNECTOR_CA_B64", "")
