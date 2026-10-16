"""Runs the real deep agent graph with a scripted fake model (no OpenAI calls).

The fake model: 1) writes todos, 2) delegates to plan-reviewer,
3) (as the reviewer) answers, 4) calls schedule_plan, 5) final reply that uses
the {{SCHEDULE}} placeholder, which run_chat fills with the real table.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agent import build_agent, run_chat
from app.tools import TOOLS


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_deep_agent_plans_delegates_and_schedules():
    script = iter([
        call("write_todos", {"todos": [{"content": "Draft plan", "status": "in_progress"},
                                       {"content": "Review + schedule", "status": "pending"}]}, 1),
        call("task", {"subagent_type": "plan-reviewer", "description": "Bake bread\n1|Buy flour|1|Must|-"}, 2),
        AIMessage(content="Looks good"),  # the reviewer sub-agent's answer
        call("schedule_plan", {"items": [{"id": 1, "title": "Buy flour", "hours": 1}],
                               "start_date": "2026-10-19"}, 3),
        AIMessage(content="## Goal\nBake bread\n## Schedule\n{{SCHEDULE}}\nOn time.\n## Do this first"),
    ])
    agent = build_agent(model=ScriptedModel(messages=script))
    out = run_chat([{"role": "user", "content": "Bake bread this week"}], agent=agent)

    assert out["reply"].startswith("## Goal")
    # code pasted the exact tool table, the model never had to copy it
    assert "{{SCHEDULE}}" not in out["reply"]
    assert "| 1 | Buy flour | 1 | 2026-10-19 | 2026-10-19 |" in out["reply"]
    assert {"write_todos", "task", "schedule_plan"} <= set(out["tools_used"])
    # todos are marked completed in code (no extra write_todos round trip)
    assert out["todos"] == [{"content": "Draft plan", "status": "completed"},
                            {"content": "Review + schedule", "status": "completed"}]
    t = out["timings"]
    assert t["llm_calls"] == 5 and t["total_s"] >= 0
    assert [s["name"] for s in t["steps"] if s["kind"] == "tool"] == ["write_todos", "task", "schedule_plan"]


def test_get_today_is_not_a_tool_anymore():
    assert [t.name for t in TOOLS] == ["schedule_plan"]


def test_reviewer_has_no_tools(monkeypatch):
    import app.agent as agent_mod
    captured = {}
    monkeypatch.setattr(agent_mod, "create_deep_agent", lambda **kw: captured.update(kw))
    agent_mod.build_agent(model=ScriptedModel(messages=iter([])))
    assert captured["subagents"][0]["tools"] == []


def test_today_is_injected_and_missing_schedule_is_explained():
    seen = []

    class Recorder(ScriptedModel):
        def _generate(self, messages, *a, **kw):
            seen.append(messages)
            return super()._generate(messages, *a, **kw)

    agent = build_agent(model=Recorder(messages=iter([AIMessage(content="## Schedule\n{{SCHEDULE}}")])))
    out = run_chat([{"role": "user", "content": "hi"}], agent=agent)
    assert any("Today is " in str(m.content) for m in seen[0])
    assert "no schedule was produced" in out["reply"]
