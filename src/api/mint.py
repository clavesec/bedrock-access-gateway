"""Connector-JWT mint client (TPAI external-content program, m1 Phase E(i);
decision D8, api-proxy analogue R12).

Server-side tool calls (web_fetch in m2, gmail_* in m3) reach the
external-content connector with a short-TTL per-user JWT
(``aud=tpai-connector``, ``iss=tpai-auth``). The gateway never signs
tokens itself: a dedicated mint Lambda in the auth-server stack signs them
with a KMS asymmetric key after cross-checking that the caller is *live* —

- ``owui-session`` binding: an **active VPN session** exists for the
  subject (live login — the property that justified auth-server minting
  over gateway self-signing);
- ``api-key`` binding: the subject's ``tpai-api-keys`` record exists and
  is **not revoked** (live credential, R12 — API-key callers have no OWUI
  session).

Transport is IAM SigV4 over the dedicated Lambda interface endpoint the
network stack pins to exactly this function ARN (D8: no shared secret
anywhere in the mint path — IAM is the caller auth, KMS is the signer).

Identity-space note (E2): the mint request carries both pseudonym spaces —
``identity`` (the gateway's audit-space HMAC from ``api.identity``) and
``subject_id`` (the enrollment-space ``user_id`` that keys the session and
api-key tables). Since the SH1 D7 adjustment the OWUI identity is *derived
from* the subject (``HMAC(owui-user-id, subject)``), so the pairing is
computable by anyone holding the Product-account identity key — see the E2
scope note in ``api.identity``; what remains load-bearing here is hygiene,
not unlinkability: neither this module nor the Lambda may log the two
together, and the minted JWT carries only ``sub=identity`` so the connector
side never sees the enrollment pseudonym.

Operating contract (mirrors ``api.audit`` / ``api.taint``):

- **Fail closed.** ``get_connector_token`` raises ``MintError`` on any
  transport/Lambda failure, including an unconfigured function ARN. A tool
  call that cannot present a live-bound token must not reach the connector.
- **Clean deny.** A refusal by the mint Lambda (no active session, revoked
  or unknown api-key) raises ``MintRefusedError`` — callers surface it as
  a failed tool call, not a 5xx.
- **The token never touches logs.** This module logs outcomes and the
  audit-space identity only — never the JWT, never ``subject_id``
  (see the log-capture test in ``tests/test_mint.py``).
- **Synchronous.** boto3 is blocking; async callers (the m2 Converse tool
  choke point) must wrap calls in
  ``starlette.concurrency.run_in_threadpool`` exactly as
  ``docs/TaintAndBudgets.md`` specifies for taint/budget/audit.

Tokens are cached per identity for their TTL (minus a refresh margin) so
the mint Lambda is invoked once per identity per ~15-minute window, not
once per tool call. The 15-minute default TTL is the same D8 window the
session-keyed taint fallback is pinned to (``api.taint``).

This path is dark today: no server-side tool executes until m2 Phase 0,
so nothing calls ``get_connector_token``. Full contract:
``docs/ConnectorMint.md``.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# Injected by the bedrock-gateway-stack CDK wiring (deterministic ARN of the
# auth-server stack's tpai-connector-mint-<env> function). Unset means this
# deployment cannot mint — get_connector_token refuses rather than degrades.
MINT_FUNCTION_ARN = os.environ.get("TPAI_CONNECTOR_MINT_FUNCTION_ARN", "")
# The mint interface endpoint's own DNS name (https://vpce-….lambda.….vpce.
# amazonaws.com). The endpoint deliberately does NOT use private DNS — that
# would capture every lambda.<region>.amazonaws.com resolution in the VPC
# (including VPN operators' aws CLI, which the endpoint policy would then
# deny) — so this client must address the endpoint explicitly. Unset falls
# back to the default resolver (unroutable in the airgapped VPC → the invoke
# fails closed, matching the dark posture).
MINT_ENDPOINT_URL = os.environ.get("TPAI_CONNECTOR_MINT_ENDPOINT_URL", "")

REQUEST_SCHEMA = "tpai.connector-mint.request.v1"

# The two live-ness bindings (R12). Values are part of the wire contract
# with the mint Lambda — keep in sync with lambda/connector-mint/index.ts.
BINDING_OWUI_SESSION = "owui-session"
BINDING_API_KEY = "api-key"
VALID_BINDINGS = (BINDING_OWUI_SESSION, BINDING_API_KEY)

# Stop serving a cached token this many seconds before its exp so an
# in-flight connector call never presents a token that expires mid-request.
REFRESH_MARGIN_SECONDS = 30

# The mint invoke sits on the tool-call critical path (fail closed) and each
# in-flight call pins a threadpool thread — keep the worst case bounded:
# 2 attempts x (3s connect + 10s read) + backoff, against a Lambda whose own
# timeout is 10s. (Same shape as api.audit's _S3_CONFIG.)
_LAMBDA_CONFIG = Config(
    connect_timeout=3,
    read_timeout=10,
    retries={"max_attempts": 2, "mode": "standard"},
)

_lambda_lock = threading.Lock()
_lambda_client = None

_cache_lock = threading.Lock()
_token_cache: dict[str, "MintedToken"] = {}


class MintError(RuntimeError):
    """The connector token could not be minted (transport/Lambda/config
    failure). Callers must fail the tool call — never proceed tokenless."""


class MintRefusedError(RuntimeError):
    """The mint Lambda refused: the caller has no live session / credential.
    A clean deny, not an infrastructure failure."""

    def __init__(self, reason: str):
        super().__init__(f"connector token refused: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class MintedToken:
    token: str
    expires_at: int  # epoch seconds (the JWT's exp claim)
    # The subject the Lambda's live-ness cross-check ran against. A cache hit
    # requires the presented subject to match — a token verified for subject A
    # must never be served to a request asserting subject B under the same
    # identity. Post-SH1 the owui-session identity is derived 1:1 from the
    # subject, so divergence there is impossible; the live case is the
    # api-key binding, where the identity HMAC lowercases the asserted user
    # while the subject is case-preserved — two case variants of one api-key
    # user share an identity but are distinct subjects, and the cross-check
    # result must not leak between them.
    subject_id: str

    def fresh(self, now: float) -> bool:
        return now < self.expires_at - REFRESH_MARGIN_SECONDS


def _lambda():
    """Lazy singleton Lambda client (like api.audit._s3: importing the app
    never resolves AWS credentials for a dark code path). Addressed at the
    interface endpoint's own DNS name — see MINT_ENDPOINT_URL above."""
    global _lambda_client
    with _lambda_lock:
        if _lambda_client is None:
            _lambda_client = boto3.client(
                "lambda",
                region_name=os.environ.get("AWS_REGION"),
                config=_LAMBDA_CONFIG,
                **({"endpoint_url": MINT_ENDPOINT_URL} if MINT_ENDPOINT_URL else {}),
            )
        return _lambda_client


def invalidate(identity: str) -> None:
    """Drop the cached token for one identity (e.g. after the connector
    rejects it as stale before its cached expiry)."""
    with _cache_lock:
        _token_cache.pop(identity, None)


def _invoke_mint(identity: str, binding: str, subject_id: str) -> MintedToken:
    payload = {
        "schema": REQUEST_SCHEMA,
        "identity": identity,
        "binding": binding,
        "subject_id": subject_id,
    }
    try:
        response = _lambda().invoke(
            FunctionName=MINT_FUNCTION_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        if response.get("FunctionError"):
            # The error payload may echo request fields — never log it.
            raise MintError(
                f"mint Lambda returned FunctionError={response['FunctionError']}"
            )
        body = json.loads(response["Payload"].read())
    except MintError:
        raise
    except Exception as exc:
        logger.error("connector mint invoke failed (%s)", type(exc).__name__)
        raise MintError("failed to invoke the connector mint Lambda") from exc

    if not isinstance(body, dict):
        raise MintError("mint Lambda returned a non-object payload")
    if body.get("ok") is not True:
        reason = str(body.get("reason") or "unspecified")
        logger.info(
            "connector token refused identity=%s binding=%s reason=%s",
            identity,
            binding,
            reason,
        )
        raise MintRefusedError(reason)

    token = body.get("token")
    expires_at = body.get("expires_at")
    if not isinstance(token, str) or not token:
        raise MintError("mint Lambda response is missing the token")
    if not isinstance(expires_at, int) or expires_at <= time.time():
        raise MintError("mint Lambda response has a missing or expired expires_at")
    return MintedToken(token=token, expires_at=expires_at, subject_id=subject_id)


def get_connector_token(identity: str, binding: str, subject_id: str | None) -> MintedToken:
    """Return a live-bound connector JWT for this identity, minting one via
    the auth-server Lambda if no fresh cached token exists.

    Raises ``MintError`` (fail closed) on config/transport/Lambda failure,
    ``MintRefusedError`` (clean deny) when the caller has no active session
    or a revoked/unknown api-key, and ``ValueError`` for malformed inputs —
    all three mean the tool call must not reach the connector.
    """
    if not identity:
        raise ValueError("connector token requires a non-empty HMAC identity")
    if binding not in VALID_BINDINGS:
        raise ValueError(f"binding must be one of {VALID_BINDINGS}")
    if not subject_id:
        # OWUI deployments predating the User-Id header, or a stripped proxy:
        # without a subject there is nothing to cross-check a live session
        # against — refuse rather than mint an unbound token.
        raise MintRefusedError("missing-subject")
    if not MINT_FUNCTION_ARN:
        raise MintError(
            "TPAI_CONNECTOR_MINT_FUNCTION_ARN is not configured - refusing to "
            "execute a connector call without a live-bound token"
        )

    now = time.time()
    with _cache_lock:
        cached = _token_cache.get(identity)
        if cached is not None and cached.fresh(now) and cached.subject_id == subject_id:
            # Subject equality guards the owui-session binding: the cached
            # token's live-session check ran against cached.subject_id, so a
            # request asserting a different subject re-mints (and re-checks).
            return cached

    minted = _invoke_mint(identity, binding, subject_id)
    with _cache_lock:
        _token_cache[identity] = minted
    logger.info(
        "connector token minted identity=%s binding=%s expires_at=%d",
        identity,
        binding,
        minted.expires_at,
    )
    return minted
