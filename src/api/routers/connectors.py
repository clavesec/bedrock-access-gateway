"""Connectors passthrough surface for OWUI's Settings → Connectors page
(m3 G3, R14).

OWUI has no network path to the external-content connector (and must never
get one — AT-AS-4 counts exactly one egress path, owned by this gateway).
The page therefore manages the user's Gmail connection through the gateway:

    OWUI backend (forwarded user headers)
      -> this router (api_key_auth + require_identity)
      -> mint (short-TTL per-user JWT, active-session cross-check, D8)
      -> connector /v1/gmail/{status,consent-session,confirm,disconnect}

These are connection-management calls, not retrievals: the connector owns
their audit records (disconnect and confirm are WORM-audited there; status
and consent-session are not retrieval surfaces). No taint or budget applies
— nothing here returns mailbox content.

The confirm call is the R-1 account-linking close: consent finalization
must arrive as an authenticated request from the *initiating* user's OWUI
session — the connector compares this JWT's subject against the consent
session's bound identity and purges the pending connection on mismatch.
The gateway relays the confirm nonce opaquely; it never treats the nonce
itself as authorization.

E3: no line here logs connector response bodies, consent URLs, or nonces.
"""

import logging
import re

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api import mint, setting
from api.auth import api_key_auth
from api.identity import require_identity
from api.tools import base, gmail, web_fetch

logger = logging.getLogger(__name__)

# Connector responses this surface may relay verbatim; anything else maps
# to a 502 so a misbehaving connector cannot speak arbitrary statuses to
# OWUI through the gateway.
_RELAY_STATUSES = (200, 201, 403)

# secrets.token_urlsafe output is URL-safe base64; bound it defensively.
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{16,256}$")

# Wire schema the connector's GmailConfirmRequest requires (aliased `schema`,
# no default — omitting it is a 422 the connector rejects before its handler,
# which the gateway would then map to a 502 and break the whole confirm flow).
_CONFIRM_REQUEST_SCHEMA = "tpai.connector.gmail-confirm.request.v1"


def require_gmail_connectors_enabled() -> None:
    """The whole surface 404s while the gmail tool family is dark, matching
    the connector's route-registration gate."""
    if not setting.ENABLE_GMAIL_TOOLS:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(
    prefix="/connectors/gmail",
    dependencies=[
        Depends(require_gmail_connectors_enabled),
        Depends(api_key_auth),
        Depends(require_identity),
    ],
)


class ConfirmRequest(BaseModel):
    nonce: str = Field(min_length=16, max_length=256)


def _mint_material(request: Request) -> tuple[str, str, str | None]:
    identity = getattr(request.state, "tpai_identity", None)
    binding = getattr(request.state, "tpai_mint_binding", None)
    subject_id = getattr(request.state, "tpai_mint_subject_id", None)
    if not identity or not binding:
        # require_identity enforces this when TPAI_IDENTITY_ENFORCE is on;
        # without an identity there is nothing to bind a connector JWT to.
        raise HTTPException(status_code=403, detail="identity-required")
    return identity, binding, subject_id


def _call_connector(
    request: Request,
    method: str,
    path: str,
    payload: dict | None,
    invalidate_dek_on_success: bool = False,
) -> JSONResponse:
    """Blocking connector round-trip (callers wrap in run_in_threadpool):
    mint, call, re-mint exactly once on a 401, relay the bounded JSON body
    for the small allowlisted status set.

    ``invalidate_dek_on_success`` drops the cached gmail_search DEK on a 2xx
    too — the disconnect path sets it so the plaintext DEK cannot outlive an
    explicit disconnect by up to the cache TTL (S15 R-5: a transient failure
    of the connector's index delete would otherwise leave the ciphertext
    readable to a cached DEK for that window)."""
    identity, binding, subject_id = _mint_material(request)

    try:
        token = mint.get_connector_token(identity, binding, subject_id)
        response = _connector_request(method, path, token.token, payload)
        if response.status_code == 401:
            response.close()
            mint.invalidate(identity)
            fresh = mint.get_connector_token(identity, binding, subject_id)
            response = _connector_request(method, path, fresh.token, payload)
    except mint.MintRefusedError:
        raise HTTPException(status_code=403, detail="no-live-session")
    except (mint.MintError, ValueError):
        raise HTTPException(status_code=502, detail="connector-unavailable")
    except base.ToolExecutionError:
        raise HTTPException(status_code=502, detail="connector-unavailable")

    with response:
        if response.status_code not in _RELAY_STATUSES:
            logger.error("connector gmail surface returned status %d", response.status_code)
            raise HTTPException(status_code=502, detail="connector-error")
        if response.status_code == 403:
            # Any deny doubles as the revocation signal — drop the cached
            # DEK so gmail_search honors it promptly too (S15 R-5).
            gmail.invalidate_dek(identity)
        elif invalidate_dek_on_success and response.status_code == 200:
            gmail.invalidate_dek(identity)
        try:
            body = base.read_capped_json(
                response,
                max_bytes=65536,
                timeout_s=setting.GMAIL_CONNECTOR_TIMEOUT_S,
                error=base.ToolExecutionError,
            )
        except base.ToolExecutionError:
            raise HTTPException(status_code=502, detail="connector-error")
    return JSONResponse(status_code=response.status_code, content=body)


def _connector_request(method: str, path: str, token: str, payload: dict | None):
    try:
        return web_fetch.connector_session().request(
            method,
            _connector_base() + path,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=(web_fetch.CONNECT_TIMEOUT_S, setting.GMAIL_CONNECTOR_TIMEOUT_S),
            stream=True,
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as exc:
        logger.error("connector gmail surface transport failure (%s)", type(exc).__name__)
        raise base.ToolExecutionError("connector call failed") from exc


def _connector_base() -> str:
    connector_url = setting.TPAI_CONNECTOR_URL
    if not connector_url:
        raise base.ToolExecutionError("TPAI_CONNECTOR_URL is not configured")
    return connector_url.rstrip("/")


@router.get("/status")
async def gmail_status(request: Request) -> JSONResponse:
    return await run_in_threadpool(_call_connector, request, "GET", "/v1/gmail/status", None)


@router.post("/consent-session")
async def gmail_consent_session(request: Request) -> JSONResponse:
    return await run_in_threadpool(
        _call_connector, request, "POST", "/v1/gmail/consent-session", None
    )


@router.post("/confirm")
async def gmail_confirm(request: Request, body: ConfirmRequest) -> JSONResponse:
    if not _NONCE_RE.fullmatch(body.nonce):
        raise HTTPException(status_code=400, detail="invalid-nonce")
    return await run_in_threadpool(
        _call_connector,
        request,
        "POST",
        "/v1/gmail/confirm",
        {"schema": _CONFIRM_REQUEST_SCHEMA, "nonce": body.nonce},
    )


@router.post("/disconnect")
async def gmail_disconnect(request: Request) -> JSONResponse:
    return await run_in_threadpool(
        _call_connector, request, "POST", "/v1/gmail/disconnect", None,
        True,  # invalidate_dek_on_success: disconnect drops the cached DEK
    )
