"""The AgentCore entrypoint keeps chat history per session (no real model)."""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("bedrock_agentcore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def main(monkeypatch):
    mod = importlib.import_module("main")
    mod._sessions.clear()
    seen = []

    def fake_run_chat(messages):
        seen.append(messages)
        return {"reply": f"plan v{len(seen)}", "todos": [], "tools_used": [], "timings": {}}

    monkeypatch.setattr(mod, "run_chat", fake_run_chat)
    mod.seen = seen
    return mod


def test_follow_up_reuses_session_history(main):
    main.handle({"prompt": "Plan an offsite"}, "s1")
    out = main.handle({"prompt": "Weekends only"}, "s1")
    assert [m["content"] for m in main.seen[1]] == ["Plan an offsite", "plan v1", "Weekends only"]
    assert out["turns"] == 2 and out["session_id"] == "s1"


def test_sessions_are_isolated_and_reset_works(main):
    main.handle({"prompt": "A"}, "s1")
    main.handle({"prompt": "B"}, "s2")
    assert [m["content"] for m in main.seen[1]] == ["B"]
    assert main.handle({"reset": True}, "s1")["turns"] == 0
    main.handle({"prompt": "C"}, "s1")
    assert [m["content"] for m in main.seen[2]] == ["C"]


def test_stateless_messages_and_empty_prompt(main):
    main.handle({"messages": [{"role": "user", "content": "X"}]}, "s3")
    assert main.seen[0] == [{"role": "user", "content": "X"}] and "s3" not in main._sessions
    assert "error" in main.handle({}, "s4")
