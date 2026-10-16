"""The deep agent: a Task Breakdown Coach built with LangChain `deepagents`.

How it works:
  main agent  --write_todos-->  plans its own work (TodoListMiddleware)
              --task()------->  'plan-reviewer' sub-agent critiques the draft
              --schedule_plan-> real dates from effort estimates (tools.py)

Speed (Final Lab): every LLM call is sequential and the slow part is the
tokens each call *writes*. So we (1) let code paste the schedule table into
the reply via a {{SCHEDULE}} placeholder, (2) drop two round trips (today's
date is in the prompt, todos are completed in code), and (3) use a small fast
model with a short answer for the reviewer. TimingCallback measures it all.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from app.tools import TOOLS

SCHEDULE_PLACEHOLDER = "{{SCHEDULE}}"

MAIN_PROMPT = """You are "Task Breakdown Coach", an expert project planner.
The user gives you a task or goal in a chat. Your job: turn it into a smart,
realistic breakdown of small action items they can start on immediately.
Today's date is given in a system message; use it as start_date.

Method (follow it every time you produce or revise a plan):
1. Understand the goal. If one or two facts are essential and missing
   (e.g. deadline, budget, hours available), make a sensible assumption and
   state it. Only ask questions instead of planning if the request is too vague
   to plan at all (e.g. "help me"). Ask at most 2 short questions.
2. Call write_todos ONCE with 3-4 short todos for your own work. Do not call
   it again; the app marks them completed for you.
3. Draft the breakdown:
   - 3-6 phases, in a logical order.
   - Each action item: starts with a verb, is specific and checkable,
     takes at most ~1 day (split anything bigger), and has: effort in hours,
     priority (Must / Should / Could), and depends_on (ids) where relevant.
   - Number items 1..N across all phases.
4. Delegate a quality check: call the task tool with subagent 'plan-reviewer'
   EXACTLY ONCE per user message. Keep the description compact: one line for
   the goal + assumptions, then one line per item "id|title|h|priority|deps".
   Apply its fixes yourself; do not send the plan back for re-review.
5. Call schedule_plan with all items, start_date = today, and hours_per_day and
   work_days matching the user's availability (e.g. "weekends only, 5 h" ->
   hours_per_day=5, work_days="Sat,Sun"). If unknown assume 2 h/day, "Mon-Fri".
   If the finish date misses the user's deadline, say so plainly and offer tweaks.
6. Reply in Markdown using exactly these sections:
   ## Goal
   ## Assumptions
   ## Plan  (one '### Phase' heading per phase, items as a numbered list:
             **#. Action** — effort h · Priority · depends on #x)
   ## Schedule
   {{SCHEDULE}}
   (write the line {{SCHEDULE}} literally: the app replaces it with the exact
   table from schedule_plan. Do NOT copy the table yourself. After it, add one
   line about the finish date vs the deadline.)
   ## Do this first (next 15 minutes)
   ## Want changes?  (one line offering 2-3 concrete tweaks)

For follow-up messages ("make it 1 month", "I only have weekends"),
revise the existing plan instead of starting over, and repeat steps 4-6.
Keep the tone encouraging and concise. Never invent facts about the user.
"""

REVIEWER_PROMPT = """You are a strict senior project manager reviewing a task
breakdown. Check it against this list and return ONLY up to 5 bullets of
concrete fixes, max 20 words each (or 'Looks good' if none):
- Is every item specific, verb-first and checkable (clear definition of done)?
- Is any item bigger than ~1 day? Split it.
- Missing steps (research, approvals, testing, launch, follow-up)?
- Wrong order or missing dependencies?
- Unrealistic effort estimates?
- Are the Must items really the minimum needed to reach the goal?
"""

_agent = None


def _chat_model(name: str, timeout: int = 120) -> ChatOpenAI:
    extra = {}
    if name.startswith(("gpt-5", "o")):
        # Reasoning models "think" before every call, and a deep agent makes
        # many calls. Effort: minimal < low < medium (speed vs. care).
        extra["reasoning_effort"] = os.environ.get("OPENAI_REASONING_EFFORT", "low")
    return ChatOpenAI(model=name, timeout=timeout, max_retries=2, **extra)


def build_agent(model=None, reviewer_model=None):
    """Create the deep agent. Models can be injected for tests."""
    injected = model is not None
    if model is None:
        model = _chat_model(os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    reviewer = {
        "name": "plan-reviewer",
        "description": "Reviews a draft task breakdown and returns concrete fixes "
        "(missing steps, items too big, wrong order, bad estimates).",
        "system_prompt": REVIEWER_PROMPT,
        # No tools: otherwise the sub-agent inherits schedule_plan and a fast model
        # may "helpfully" schedule inside the review (seen in the A/B test).
        "tools": [],
    }
    if reviewer_model is None and not injected:
        name = os.environ.get("OPENAI_REVIEWER_MODEL", "gpt-4.1-mini")
        if name != "same":  # "same" = reuse the main model
            reviewer_model = _chat_model(name, timeout=60)
    if reviewer_model is not None:
        reviewer["model"] = reviewer_model
    return create_deep_agent(
        model=model,
        tools=TOOLS,
        system_prompt=MAIN_PROMPT,
        subagents=[reviewer],
        # deepagents >= 0.7 no longer adds the planner by default: this middleware
        # gives the agent the write_todos tool and keeps `todos` in its state.
        middleware=[TodoListMiddleware()],
        name="task-breakdown-coach",
    )


def get_agent():
    """Build once per Lambda container and reuse across requests (warm starts)."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class TimingCallback(BaseCallbackHandler):
    """Records how long every LLM call and tool call takes (incl. the sub-agent)."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self._open: dict = {}
        self.steps: list[dict] = []

    def _start(self, run_id, **info):
        self._open[run_id] = (time.perf_counter(), info)

    def _end(self, run_id, **more):
        start, info = self._open.pop(run_id, (None, {}))
        if start is not None:
            now = time.perf_counter()
            self.steps.append({**info, **more, "at_s": round(start - self.t0, 1),
                               "s": round(now - start, 1)})

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        model = (metadata or {}).get("ls_model_name") or kw.get("invocation_params", {}).get("model", "?")
        self._start(run_id, kind="llm", model=model)

    def on_llm_end(self, response, *, run_id, **kw):
        out = 0
        for gen in (response.generations or [[]])[0]:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            out += usage.get("output_tokens", 0)
        self._end(run_id, out_tokens=out)

    def on_llm_error(self, error, *, run_id, **kw):
        self._end(run_id, error=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._start(run_id, kind="tool", name=(serialized or {}).get("name", "?"))

    def on_tool_end(self, output, *, run_id, **kw):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, **kw):
        self._end(run_id, error=type(error).__name__)

    def summary(self) -> dict:
        llm = [s for s in self.steps if s["kind"] == "llm"]
        return {
            "total_s": round(time.perf_counter() - self.t0, 1),
            "llm_calls": len(llm),
            "llm_s": round(sum(s["s"] for s in llm), 1),
            "out_tokens": sum(s.get("out_tokens", 0) for s in llm),
            "steps": sorted(self.steps, key=lambda s: s["at_s"]),
        }


def _today_message() -> dict:
    tz = os.environ.get("PLANNER_TIMEZONE", "UTC")
    now = datetime.now(ZoneInfo(tz))
    return {"role": "system", "content": f"Today is {now:%Y-%m-%d} ({now:%A}), timezone {tz}."}


def _fill_schedule(reply: str, messages) -> str:
    """Replace {{SCHEDULE}} with the last successful schedule_plan output."""
    if SCHEDULE_PLACEHOLDER not in reply:
        return reply
    tables = [str(m.content) for m in messages
              if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "schedule_plan"
              and not str(m.content).startswith("Error")]
    table = tables[-1] if tables else "_(no schedule was produced — ask me to retry)_"
    return reply.replace(SCHEDULE_PLACEHOLDER, table)


def run_chat(messages: list[dict], agent=None) -> dict:
    """Run one chat turn. `messages` is the full history [{role, content}, ...]."""
    agent = agent or get_agent()
    timer = TimingCallback()
    state = agent.invoke(
        {"messages": [_today_message(), *messages]},
        config={"recursion_limit": 80, "callbacks": [timer]},
    )
    last = state["messages"][-1]
    reply = _fill_schedule(str(last.text), state["messages"])  # .text joins text blocks
    # The plan is done when we reply, so mark the agent's todos completed in code
    # (saves a whole LLM round trip that only flipped statuses).
    todos = [
        {"content": t.get("content", ""), "status": "completed"}
        for t in state.get("todos", []) or []
    ]
    used = [
        call["name"]
        for msg in state["messages"]
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    timings = timer.summary()
    print(json.dumps({"event": "chat_timing", **{k: v for k, v in timings.items() if k != "steps"},
                      "steps": [f"{s.get('model') or s.get('name')}:{s['s']}s" for s in timings["steps"]]}))
    return {"reply": reply, "todos": todos, "tools_used": used, "timings": timings}
