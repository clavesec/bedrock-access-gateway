"""Per-user identity at ingestion (TPAI external-content program, m1 Phase A).

The gateway derives a pseudonymous per-user identity from trusted headers
(D7 — trust root is OWUI's asserted header, made non-spoofable by network
scoping + API-key rotation, not by cryptography):

- OWUI traffic:      ``X-OpenWebUI-User-Email``  (raw email — hashed immediately)
- api-proxy traffic: ``X-TPAI-ApiKey-User``      (the ``tpai-api-keys`` per-user
  id, already pseudonymous; the api-proxy strips any inbound identity headers
  before asserting this one)

Identity = HMAC-SHA256 over a domain-separated input, keyed with a dedicated
Product-account secret (``TPAI_IDENTITY_HMAC_KEY``, decision E2). This key is
never billing's Services-account HMAC secret — the audit and billing pseudonym
spaces must stay unlinkable without both keys.

The raw email has exactly one line of existence in this process: the
``_identity_hmac(...)`` call inside ``require_identity``. It must never be
logged, stored, or attached to request state — the log-capture test in
``tests/test_identity.py`` asserts this.

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

logger = logging.getLogger(__name__)

OWUI_EMAIL_HEADER = "X-OpenWebUI-User-Email"
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
        # Still set the attribute so downstream consumers can read it
        # unconditionally in every enforcement state.
        request.state.tpai_identity = None
        return None

    email = (request.headers.get(OWUI_EMAIL_HEADER) or "").strip().lower()
    api_user = (request.headers.get(API_PROXY_USER_HEADER) or "").strip().lower()

    if email and api_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting user identity headers",
        )

    if email:
        identity = _identity_hmac("owui-email", email)
    elif api_user:
        identity = _identity_hmac("api-key-user", api_user)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity",
        )

    request.state.tpai_identity = identity
    return identity
