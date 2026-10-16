"""Deterministic tools the deep agent can call.

LLMs are bad at calendar arithmetic, so dates and scheduling are done in
plain Python. The agent decides *what* to do, and these tools work out *when*.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain.tools import tool
from pydantic import BaseModel, Field


@tool
def get_today(timezone_name: str = "UTC") -> str:
    """Return today's date and weekday in an IANA timezone (e.g. 'Asia/Singapore')."""
    try:
        now = datetime.now(ZoneInfo(timezone_name))
    except Exception:  # noqa: BLE001 - unknown timezone falls back to UTC
        now = datetime.now(ZoneInfo("UTC"))
        timezone_name = "UTC"
    return f"{now:%Y-%m-%d} ({now:%A}), timezone {timezone_name}"


class PlanItem(BaseModel):
    id: int = Field(description="Unique number of the action item, e.g. 1, 2, 3")
    title: str = Field(description="Short verb-first action, e.g. 'Draft menu of 10 items'")
    hours: float = Field(gt=0, le=40, description="Estimated effort in hours")
    depends_on: list[int] = Field(default_factory=list, description="ids that must finish first")


def _order(items: list[PlanItem]) -> list[PlanItem]:
    """Topological sort (dependencies first, then by id). Raises on cycles."""
    by_id = {i.id: i for i in items}
    unknown = {d for i in items for d in i.depends_on if d not in by_id}
    if unknown:
        raise ValueError(f"depends_on refers to unknown ids: {sorted(unknown)}")
    done: set[int] = set()
    ordered: list[PlanItem] = []
    remaining = sorted(items, key=lambda i: i.id)
    while remaining:
        ready = [i for i in remaining if set(i.depends_on) <= done]
        if not ready:
            raise ValueError("dependency cycle between ids " + str([i.id for i in remaining]))
        nxt = ready[0]
        ordered.append(nxt)
        done.add(nxt.id)
        remaining.remove(nxt)
    return ordered


DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
FULL_DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
ALIASES = {"all": "mon-sun", "every day": "mon-sun", "everyday": "mon-sun",
           "weekdays": "mon-fri", "weekends": "sat,sun", "weekend": "sat,sun"}


def parse_work_days(spec: str) -> set[int]:
    """'Mon-Fri' | 'Sat,Sun' | 'weekends' | 'all' | 'Mon,Wed,Fri' | 'Mon-Wed,Sat' -> {0..6}."""
    text = spec.strip().lower()
    for sep in ("&", " and ", "/", ";", "+"):  # "Sat & Sun", "Sat and Sun" -> "sat,sun"
        text = text.replace(sep, ",")
    text = ALIASES.get(text.strip(), text)
    days: set[int] = set()
    for part in filter(None, (p.strip() for p in text.split(","))):
        names = [n.strip() for n in part.split("-")]
        # Accept "sat", "satur", "saturday"; reject anything else (no silent guesses).
        if len(names) > 2 or not all(len(n) >= 3 and any(f.startswith(n) for f in FULL_DAY_NAMES) for n in names):
            raise ValueError(f"work_days '{spec}' not understood; use e.g. 'Mon-Fri', 'Sat,Sun', 'all'")
        names = [n[:3] for n in names]
        a, b = DAY_NAMES.index(names[0]), DAY_NAMES.index(names[-1])
        days.update(range(a, b + 1) if a <= b else [*range(a, 7), *range(0, b + 1)])
    if not days:
        raise ValueError("work_days is empty")
    return days


def build_schedule(
    items: list[PlanItem],
    hours_per_day: float = 2.0,
    start_date: str | None = None,
    work_days: str = "Mon-Fri",
) -> list[dict]:
    """Pack items, in dependency order, into work days of `hours_per_day` each."""
    if hours_per_day <= 0:
        raise ValueError("hours_per_day must be > 0")
    allowed = parse_work_days(work_days)
    day = date.fromisoformat(start_date) if start_date else date.today()

    def next_work_day(d: date) -> date:
        while d.weekday() not in allowed:
            d += timedelta(days=1)
        return d

    day = next_work_day(day)
    used = 0.0
    out = []
    for item in _order(items):
        left = item.hours
        start = None
        while left > 1e-9:
            if used >= hours_per_day - 1e-9:
                day = next_work_day(day + timedelta(days=1))
                used = 0.0
            chunk = min(left, hours_per_day - used)
            start = start or day
            used += chunk
            left -= chunk
        out.append(
            {"id": item.id, "title": item.title, "hours": item.hours,
             "start": start.isoformat(), "end": day.isoformat()}
        )
    return out


@tool
def schedule_plan(
    items: list[PlanItem],
    hours_per_day: float = 2.0,
    start_date: str | None = None,
    work_days: str = "Mon-Fri",
) -> str:
    """Turn action items with effort estimates into a dated schedule.

    Respects depends_on order and fits `hours_per_day` of work on each allowed
    day. work_days says WHICH days the user works: 'Mon-Fri' (default),
    'Sat,Sun' (weekends only), 'all', or a list like 'Mon,Wed,Fri'.
    start_date is YYYY-MM-DD (default today). Returns a Markdown table plus the
    finish date. Do not re-type the table: write {{SCHEDULE}} in your reply and
    the app inserts it.
    """
    try:
        rows = build_schedule(items, hours_per_day, start_date, work_days)
    except ValueError as exc:
        return f"Error: {exc}. Fix the items and call schedule_plan again."
    total = sum(r["hours"] for r in rows)
    lines = ["| # | Action | Hours | Start | End |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['title']} | {r['hours']:g} | {r['start']} | {r['end']} |" for r in rows]
    finish = max(r["end"] for r in rows) if rows else "-"
    lines.append(f"\nTotal effort: {total:g} h at {hours_per_day:g} h/day on {work_days} -> finishes {finish}")
    return "\n".join(lines)


# get_today stays available for reuse, but the agent no longer calls it: today's
# date is injected into the prompt (one fewer LLM round trip per turn).
TOOLS = [schedule_plan]
