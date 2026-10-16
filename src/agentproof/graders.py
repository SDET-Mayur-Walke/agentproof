"""Reusable graders. Deterministic ones first: they're cheap, repeatable,
and they gate. The LLM judge is last and non-gating by default.

Each factory returns a Grader. Graders read what they need from the
Transcript (tool calls, outcome, final text) and task.expected.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from .core import Grader, Task, Transcript


def _sync(fn: Callable[[Task, Transcript], tuple[float, str]]):
    async def run(task: Task, t: Transcript) -> tuple[float, str]:
        return fn(task, t)
    return run


def custom(name: str, fn: Callable[[Task, Transcript], tuple[float, str]], gating: bool = True) -> Grader:
    """Wrap any plain (task, transcript) -> (score, reason) function."""
    return Grader(name, _sync(fn), gating)


# ----- trajectory: what the agent did -----

def tool_called(tool: str, min_times: int = 1) -> Grader:
    def fn(task, t):
        n = len(t.calls(tool))
        return (1.0, f"{tool} called {n}x") if n >= min_times else (0.0, f"{tool} called {n}x, need >= {min_times}")
    return custom(f"called:{tool}", fn)


def tool_not_called(tool: str) -> Grader:
    """For tasks where acting would be wrong (no flight exists, over budget, missing info)."""
    def fn(task, t):
        n = len(t.calls(tool))
        return (1.0, f"{tool} not called") if n == 0 else (0.0, f"{tool} called {n}x but must not be")
    return custom(f"not_called:{tool}", fn)


def tool_order(before: str, after: str) -> Grader:
    """`before` must happen before the first `after` (e.g. search before book)."""
    def fn(task, t):
        names = [c.name for c in t.tool_calls]
        if after not in names:
            return 1.0, f"{after} never called"
        first_after = names.index(after)
        ok = before in names[:first_after]
        return (1.0, f"{before} before {after}") if ok else (0.0, f"{after} called before any {before}")
    return custom(f"order:{before}<{after}", fn)


def args_match(tool: str, expected: dict[str, Any] | Callable[[Task], dict[str, Any]]) -> Grader:
    """Every call to `tool` carries these args (string compare is
    case-insensitive). Partial credit = fraction of expected args right."""
    def fn(task, t):
        exp = expected(task) if callable(expected) else expected
        calls = t.calls(tool)
        if not calls:
            return 0.0, f"{tool} never called"
        def norm(v):
            return v.strip().lower() if isinstance(v, str) else v
        worst, why = 1.0, "all args match"
        for c in calls:
            hits = [k for k, v in exp.items() if norm(c.args.get(k)) == norm(v)]
            score = len(hits) / len(exp) if exp else 1.0
            if score < worst:
                miss = {k: c.args.get(k) for k in exp if k not in hits}
                worst, why = score, f"wrong args {miss}, expected {({k: exp[k] for k in miss})}"
        return worst, why
    return custom(f"args:{tool}", fn)


# ----- outcome: what changed in the world -----

def outcome_equals(key: str, expected: Any | Callable[[Task], Any]) -> Grader:
    """The strongest check when the task has one: compare final world state."""
    def fn(task, t):
        exp = expected(task) if callable(expected) else expected
        got = (t.outcome or {}).get(key)
        return (1.0, f"{key} == {exp!r}") if got == exp else (0.0, f"{key} is {got!r}, expected {exp!r}")
    return custom(f"outcome:{key}", fn)


# ----- grounding: the reply only states facts the agent actually saw -----

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def grounded_numbers(min_value: float = 10, ignore: Callable[[Task], set[str]] | None = None) -> Grader:
    """Every number >= min_value in the final reply must appear in a tool
    result or in the user's input. Catches invented prices, times, IDs.
    Small numbers (counts, list markers) are skipped via min_value."""
    def fn(task, t):
        def nums(text: str) -> set[str]:
            return {m.replace(",", "") for m in _NUM.findall(text or "")}
        seen = nums(task.input) | {n for f in task.followups for n in nums(f)}
        for c in t.tool_calls:
            seen |= nums(str(c.result)) | nums(str(c.args))
        extra = ignore(task) if ignore else set()
        claimed = {n for n in nums(t.final_output) if float(n) >= min_value}
        # derived values the agent could legitimately compute aren't checked here:
        # list them in `ignore` per task, or write a task-specific grader
        invented = sorted(claimed - seen - extra, key=float)
        return (1.0, f"{len(claimed)} numbers, all traced") if not invented else \
            (0.0, f"untraced numbers: {invented}")
    return custom("grounded_numbers", fn)


# ----- text checks (use sparingly: brittle to valid rephrasing) -----

def mentions(pattern: str, name: str | None = None, flags=re.I) -> Grader:
    rx = re.compile(pattern, flags)
    def fn(task, t):
        return (1.0, f"matched /{pattern}/") if rx.search(t.final_output or "") else (0.0, f"no match for /{pattern}/")
    return custom(name or f"mentions:{pattern}", fn)


# ----- model-based -----

JudgeFn = Callable[[str, Task, Transcript], Awaitable[tuple[float, str]]]


def llm_judge(name: str, rubric: str, judge: JudgeFn, gating: bool = False) -> Grader:
    """judge(rubric, task, transcript) -> (score, reason). Kept as a plain
    callable so the core has no LLM dependency (see agentproof.judges).
    Non-gating until calibrated: measure it with calibration.calibrate() first."""
    async def run(task, t):
        return await judge(rubric, task, t)
    return Grader(name, run, gating)
