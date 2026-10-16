"""Chat UI for the AgentCore-hosted deep agent (AWS Lambda + Function URL).

The browser can't call AgentCore directly (it needs AWS SigV4 auth), so this
small Lambda serves the page and forwards each message with its IAM role:

  browser --POST /api/chat {prompt, session_id}--> Lambda --invoke_agent_runtime--> AgentCore

Routes:
  GET  /           -> chat page (index.html)
  GET  /health     -> {"ok": true}
  POST /api/chat   -> forward to AgentCore. Requires header x-access-code.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import time
from pathlib import Path

INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{33,100}$")  # AgentCore needs >= 33 chars
MAX_CHARS = 4000
_client = None


def _json(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps(body),
    }


def _agentcore():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        _client = boto3.client(
            "bedrock-agentcore",
            config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 1}),
        )
    return _client


def _validate(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError('Body must be JSON like {"prompt": "...", "session_id": "..."}')
    sid = str(payload.get("session_id", ""))
    if not SESSION_RE.match(sid):
        raise ValueError("session_id must be 33-100 letters, digits, '-' or '_'")
    if payload.get("reset"):
        return {"session_id": sid, "body": {"reset": True}}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    if len(prompt) > MAX_CHARS:
        raise ValueError(f"prompt is too long (max {MAX_CHARS} characters)")
    return {"session_id": sid, "body": {"prompt": prompt.strip()}}


def call_agent(session_id: str, body: dict) -> dict:
    t0 = time.perf_counter()
    resp = _agentcore().invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
        qualifier=os.environ.get("AGENT_QUALIFIER", "DEFAULT"),
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(body).encode("utf-8"),
    )
    raw = resp["response"].read()
    data = json.loads(raw or b"{}")
    if isinstance(data, str):  # some SDK versions double-encode
        data = json.loads(data)
    data["agentcore_s"] = round(time.perf_counter() - t0, 1)
    return data


def lambda_handler(event, context):
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method", "GET").upper()
    path = event.get("rawPath") or "/"

    if method == "GET" and path in ("/", "/index.html"):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"},
            "body": INDEX_HTML,
        }
    if method == "GET" and path == "/health":
        return _json(200, {"ok": True})
    if path != "/api/chat":
        return _json(404, {"error": "Not found"})
    if method != "POST":
        return _json(405, {"error": "Use POST"})

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected or not hmac.compare_digest(headers.get("x-access-code", ""), expected):
        return _json(401, {"error": "Wrong or missing access code"})

    try:
        raw = event.get("body") or ""
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        req = _validate(json.loads(raw or "{}"))
    except (ValueError, UnicodeDecodeError) as exc:
        return _json(400, {"error": str(exc)})

    try:
        data = call_agent(req["session_id"], req["body"])
    except Exception as exc:  # noqa: BLE001
        print(f"AgentCore error: {exc!r}")
        return _json(502, {"error": f"AgentCore call failed: {type(exc).__name__}: {exc}"[:500]})
    if data.get("error"):
        return _json(502, {"error": str(data["error"])[:500]})
    return _json(200, data)
