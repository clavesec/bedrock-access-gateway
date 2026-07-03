"""Shared DynamoDB client plumbing for the taint and budget stores.

One client, one timeout policy, one conditional-failure test — the taint
and budget modules must never drift apart on any of these (they gate the
same tool call), so the definitions live here instead of being mirrored.
"""

import os
import threading

import boto3
from botocore.config import Config

# Taint/budget reads and writes sit on the tool-call critical path (fail
# closed), and each in-flight call pins a threadpool thread, so the worst
# case must stay small — same rationale and numbers as api.audit's S3
# config: 2 attempts x (3s connect + 5s read) + backoff, vs ~50s with
# boto3's defaults.
DDB_CONFIG = Config(
    connect_timeout=3,
    read_timeout=5,
    retries={"max_attempts": 2, "mode": "standard"},
)

_lock = threading.Lock()
_client = None


def client():
    """Lazy singleton DynamoDB client (importing the app never resolves AWS
    credentials for a dark code path). Unlocked fast path: boto3 clients are
    thread-safe; the lock only guards initialization."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = boto3.client(
                "dynamodb",
                region_name=os.environ.get("AWS_REGION"),
                config=DDB_CONFIG,
            )
        return _client


def is_conditional_check_failure(exc: Exception) -> bool:
    """True when a conditional write was refused by its ConditionExpression.

    Detected by class name because the modeled exception class hangs off the
    client instance (``client.exceptions.ConditionalCheckFailedException``),
    not an importable module path. Defined once so a change to the detection
    can never split the taint and budget deny paths.
    """
    return type(exc).__name__ == "ConditionalCheckFailedException"
