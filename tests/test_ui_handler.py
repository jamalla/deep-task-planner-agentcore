"""Chat UI Lambda: routes, access code, validation, AgentCore call (faked)."""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import handler  # noqa: E402

SID = "dtp-" + "a" * 48


def ev(method="GET", path="/", body=None, code="secret"):
    return {"requestContext": {"http": {"method": method}}, "rawPath": path,
            "headers": {"x-access-code": code}, "body": json.dumps(body) if body is not None else None}


class FakeClient:
    def __init__(self):
        self.calls = []

    def invoke_agent_runtime(self, **kw):
        self.calls.append(kw)
        body = json.loads(kw["payload"])
        return {"response": io.BytesIO(json.dumps({"reply": "ok:" + body.get("prompt", "reset"), "turns": 1}).encode())}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    monkeypatch.setenv("AGENT_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/x")
    client = FakeClient()
    monkeypatch.setattr(handler, "_client", client)
    return client


def test_page_health_and_routes(fake):
    assert "AgentCore" in handler.lambda_handler(ev(), None)["body"]
    assert handler.lambda_handler(ev(path="/health"), None)["statusCode"] == 200
    assert handler.lambda_handler(ev(path="/nope"), None)["statusCode"] == 404
    assert handler.lambda_handler(ev(path="/api/chat"), None)["statusCode"] == 405


def test_access_code_and_validation(fake):
    ok = {"prompt": "hi", "session_id": SID}
    assert handler.lambda_handler(ev("POST", "/api/chat", ok, code="bad"), None)["statusCode"] == 401
    assert handler.lambda_handler(ev("POST", "/api/chat", {"prompt": "hi", "session_id": "short"}), None)["statusCode"] == 400
    assert handler.lambda_handler(ev("POST", "/api/chat", {"prompt": " ", "session_id": SID}), None)["statusCode"] == 400
    assert fake.calls == []


def test_forwards_prompt_with_session(fake):
    r = handler.lambda_handler(ev("POST", "/api/chat", {"prompt": "Plan it", "session_id": SID}), None)
    assert r["statusCode"] == 200 and json.loads(r["body"])["reply"] == "ok:Plan it"
    call = fake.calls[0]
    assert call["runtimeSessionId"] == SID and json.loads(call["payload"]) == {"prompt": "Plan it"}
    r = handler.lambda_handler(ev("POST", "/api/chat", {"reset": True, "session_id": SID}), None)
    assert json.loads(fake.calls[1]["payload"]) == {"reset": True}
