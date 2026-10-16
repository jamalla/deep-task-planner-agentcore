"""Amazon Bedrock AgentCore Runtime entrypoint for the Deep Task Planner.

Same deep agent as the Lambda version (app/agent.py), different host:
AgentCore gives every session its own microVM, so we can keep the chat
history in memory per session instead of making the browser resend it.

Payloads (test endpoint in the AgentCore console or invoke_agent_runtime):
  {"prompt": "Plan a 2-day team offsite for 25 people..."}   new turn in this session
  {"prompt": "Make it weekends only", "reset": false}        follow-up, same session id
  {"messages": [{"role": "user", "content": "..."}]}         full history (stateless mode)
  {"reset": true}                                            forget this session's chat
"""

from __future__ import annotations

import threading

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from app.agent import run_chat

app = BedrockAgentCoreApp()

MAX_TURNS = 20  # keep the prompt small: last 20 user+assistant messages
_sessions: dict[str, list[dict]] = {}
_lock = threading.Lock()


def handle(payload: dict, session_id: str | None) -> dict:
    """Pure function so it can be unit-tested without the AgentCore server."""
    payload = payload or {}
    key = session_id or "default"
    if payload.get("reset"):
        with _lock:
            _sessions.pop(key, None)
        if not payload.get("prompt"):
            return {"reply": "Session cleared.", "session_id": key, "turns": 0}

    if payload.get("messages"):
        history = list(payload["messages"])
    else:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return {"error": 'Send {"prompt": "your task"}'}
        with _lock:
            history = [*_sessions.get(key, []), {"role": "user", "content": prompt[:4000]}]

    out = run_chat(history[-MAX_TURNS:])
    if not payload.get("messages"):
        with _lock:
            _sessions[key] = [*history, {"role": "assistant", "content": out["reply"]}][-MAX_TURNS:]
    out["session_id"] = key
    out["turns"] = len(_sessions.get(key, [])) // 2
    return out


@app.entrypoint
def invoke(payload, context):
    return handle(payload, getattr(context, "session_id", None))


if __name__ == "__main__":
    app.run()  # serves POST /invocations and GET /ping on port 8080
