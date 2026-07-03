"""Per-user identity at ingestion (TPAI external-content program, m1 Phase A).

The gateway derives a pseudonymous per-user identity from trusted headers
(D7 — trust root is OWUI's asserted header, made non-spoofable by network
scoping + API-key rotation, not by cryptography):

- OWUI traffic:      ``X-OpenWebUI-User-Id``     (the immutable OWUI user id;
  for billing-enrolled users this is the enrollment pseudonym — no PII).
  Identity keys on the user id, NOT the email (D7 adjustment, ratified by
  the flag-day owner during SH1 gate-day prep 2026-07-03, TPAI#346):
  billing-enrolled users — every real production user — have ``email=None``
  (the fork coalesces the header to ``""``), and emails are mutable (the
  provisioning name-reconciliation writes a ``@placeholder.tpai.internal``
  address), so an email-derived identity would either fail outright or
  silently re-key mid-history, breaking budget and 7-year-audit continuity.
  Requests carrying an email header but no user id are REJECTED (401): every
  supported OWUI (fork or upstream) sends both headers in the same block, so
  an email-only request is anomalous, and accepting it would open a second,
  unlinked identity space (fail closed instead).
- api-proxy traffic: ``X-TPAI-ApiKey-User``      (the ``tpai-api-keys`` per-user
  id, already pseudonymous; the api-proxy strips any inbound identity headers
  before asserting this one)

Identity = HMAC-SHA256 over a domain-separated input, keyed with a dedicated
Product-account secret (``TPAI_IDENTITY_HMAC_KEY``, decision E2). This key is
never billing's Services-account HMAC secret and never leaves the Product
account. E2 scope note (SH1): because the OWUI user id *is* the enrollment
pseudonym for billing-enrolled users, a Product-account actor holding this
key plus any enrollment-keyed table can join audit identities to enrollment
pseudonyms with this key alone — two-key unlinkability is unachievable for
email-less users (every input visible to this gateway is Product-visible,
and the OWUI user row already carries ``billing_customer_id``). What the
dedicated key still buys: audit identities are opaque without it, and
Services-side actors cannot map into the audit space at all. Recorded for
the S09 THREAT_MODEL rewrite.

The raw email is never used: the header is read only to detect conflicting
identity assertions and is never logged, stored, hashed, or attached to
request state — the log-capture tests in ``tests/test_identity.py`` assert
this on both the accept and reject paths.

Enforcement is active iff ``TPAI_IDENTITY_HMAC_KEY`` is configured. The E1
flag-day change-set injects the key via ECS container secrets together with
``TPAI_IDENTITY_ENFORCE=true``; a plain CloudFormation rollback removes both
and restores pre-flip behavior without re-tagging the container image (task
definitions reference ``:latest``, so a code-level kill switch would
otherwise force an image rollback too).

CAUTION for live task-definition surgery (the phase4-MFA-style incident
rollback): removing only the HMAC key while ``TPAI_IDENTITY_ENFORCE`` stays
``true`` intentionally CRASHES the container at startup (fail closed).
Disable both together, or roll back the whole template.
"""

import hashlib
import hmac
import logging
import os

from fastapi import HTTPException, Request, status

from api.mint import BINDING_API_KEY, BINDING_OWUI_SESSION

logger = logging.getLogger(__name__)

# Read only for conflict detection; never an identity input (module docstring).
OWUI_EMAIL_HEADER = "X-OpenWebUI-User-Email"
# The OWUI identity header: both the identity-HMAC input and the mint path's
# cross-check subject (api.mint, D8) — one immutable key, two derived views.
# For billing-enrolled users this is the enrollment pseudonym
# (user_id == email_hmac), i.e. the key space of the auth-server's session
# and api-key tables.
OWUI_USER_ID_HEADER = "X-OpenWebUI-User-Id"
API_PROXY_USER_HEADER = "X-TPAI-ApiKey-User"

IDENTITY_HMAC_KEY = os.environ.get("TPAI_IDENTITY_HMAC_KEY", "")
IDENTITY_ENFORCE = os.environ.get("TPAI_IDENTITY_ENFORCE", "false").lower() == "true"


def _require_key_when_enforced(enforce: bool, key: str) -> None:
    """Fail closed: a deployment that declares enforcement but lost the key
    (bad task revision, secret-injection failure, drifted CDK) must crash at
    startup rather than boot healthy with the identity control silently off."""
    if enforce and not key:
        raise RuntimeError(
            "TPAI_IDENTITY_ENFORCE is set but TPAI_IDENTITY_HMAC_KEY is missing - "
            "refusing to start with identity enforcement silently disabled"
        )


_require_key_when_enforced(IDENTITY_ENFORCE, IDENTITY_HMAC_KEY)

if IDENTITY_HMAC_KEY:
    logger.info("TPAI identity enforcement ENABLED (identity HMAC key configured)")
else:
    logger.warning(
        "TPAI identity enforcement DISABLED (TPAI_IDENTITY_HMAC_KEY not set) - "
        "identity-required routes will not demand user identity headers"
    )


def _identity_hmac(domain: str, value: str) -> str:
    """HMAC-SHA256 of a domain-separated identity input, hex-encoded."""
    message = f"{domain}:{value}".encode("utf-8")
    return hmac.new(IDENTITY_HMAC_KEY.encode("utf-8"), message, hashlib.sha256).hexdigest()


async def require_identity(request: Request) -> str | None:
    """FastAPI dependency: resolve the caller's pseudonymous identity.

    Rejects requests that carry no identity header (401) or conflicting
    identity headers (400 — exactly one trusted proxy must assert identity).
    The resolved HMAC identity is stored on ``request.state.tpai_identity``
    for downstream audit/budget consumers.
    """
    if not IDENTITY_HMAC_KEY:
        # Pre-flip (or rolled-back) deployment: no key, no enforcement.
        # Still set the attributes so downstream consumers can read them
        # unconditionally in every enforcement state.
        request.state.tpai_identity = None
        request.state.tpai_mint_binding = None
        request.state.tpai_mint_subject_id = None
        return None

    # Case-preserved: the user id is an exact key in the auth-server's session
    # table (the mint subject) and must round-trip exactly; for the identity
    # HMAC, exactness also guarantees stability (ids are system-generated and
    # never re-cased, unlike emails).
    user_id = (request.headers.get(OWUI_USER_ID_HEADER) or "").strip()
    # Never an identity input — read only so a caller asserting both an OWUI
    # identity (id or email) and an api-proxy identity is rejected.
    email = (request.headers.get(OWUI_EMAIL_HEADER) or "").strip()
    api_user = (request.headers.get(API_PROXY_USER_HEADER) or "").strip().lower()

    if (user_id or email) and api_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting user identity headers",
        )

    if user_id:
        # OWUI path: identity from the immutable user id. The email header,
        # when present alongside it, is deliberately unused. Email-only
        # requests fall through to the 401 below (module docstring).
        identity = _identity_hmac("owui-user-id", user_id)
        # Mint-path subject (D8): the enrollment-space user id, cross-checked
        # by the mint Lambda against active sessions.
        binding = BINDING_OWUI_SESSION
        subject_id = user_id
    elif api_user:
        identity = _identity_hmac("api-key-user", api_user)
        # api-proxy callers have no OWUI session: the mint Lambda cross-checks
        # the api-key's validity/revocation instead (R12) — the subject is the
        # per-user id the api-proxy asserted, case-preserved: it is an exact
        # DynamoDB key in tpai-api-keys, so the .lower() normalization applied
        # to the HMAC input above must NOT leak into it (legacy mixed-case
        # userIds would never match and the live credential would be refused).
        binding = BINDING_API_KEY
        subject_id = (request.headers.get(API_PROXY_USER_HEADER) or "").strip()
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity",
        )

    request.state.tpai_identity = identity
    request.state.tpai_mint_binding = binding
    request.state.tpai_mint_subject_id = subject_id
    return identity
