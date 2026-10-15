"""Core model: Task, Transcript, Grader, Trial, and the harness that runs them.

Vocabulary follows Anthropic's "Demystifying evals for AI agents":
  Task        an input plus what "correct" means, checked by several graders
  Trial       one attempt at a task; agents are non-deterministic, so run N
  Transcript  what the agent DID (tool calls, args, results, final state),
              not only what it said
  Grader      scores one transcript 0..1; code, outcome, or LLM judge
  Outcome     the state of the world after the trial (a DB row, a booking)

Two decisions differ from a naive "mean score >= 0.5" harness:
  1. A trial passes only if every GATING grader scores 1.0. Partial credit
     is reported, but a booking agent that books the wrong flight half
     correctly has still booked the wrong flight.
  2. LLM judges are NON-gating by default. They're reported next to the
     code graders so you can see how often the judge liked a failed trial
     (the "looks right, is wrong" rate) before trusting it to gate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class Tier(str, Enum):
    """capability = a hill to climb (low pass rate is expected);
    regression = a floor to protect (should stay ~100%)."""
    CAPABILITY = "capability"
    REGRESSION = "regression"


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    result: Any = None


@dataclass
class Transcript:
    """The full record of one trial."""
    final_output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    outcome: dict | None = None          # world state after the trial, if checkable
    turns: list[dict] = field(default_factory=list)   # [{"role": ..., "content": ...}]
    usage: dict = field(default_factory=dict)         # tokens, cost, latency, ...
    error: str | None = None             # the agent crashed; graders still run

    def calls(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]


@dataclass
class GraderResult:
    grader: str
    score: float                 # 0.0 .. 1.0, partial credit allowed
    reason: str = ""
    gating: bool = True


@dataclass
class Grader:
    """name + an async check (task, transcript) -> (score, reason).
    gating=False: reported, but doesn't decide pass/fail."""
    name: str
    check: Callable[["Task", Transcript], Awaitable[tuple[float, str]]]
    gating: bool = True

    async def __call__(self, task: "Task", t: Transcript) -> GraderResult:
        try:
            score, reason = await self.check(task, t)
        except Exception as e:   # a broken grader is a FAIL with a reason, not a crash
            score, reason = 0.0, f"grader error: {type(e).__name__}: {e}"
        return GraderResult(self.name, max(0.0, min(1.0, float(score))), reason, self.gating)


@dataclass
class Task:
    id: str
    input: str
    graders: list[Grader]
    tier: Tier = Tier.CAPABILITY
    expected: dict = field(default_factory=dict)    # whatever graders need: expected outcome, facts, ...
    followups: list[str] = field(default_factory=list)  # scripted simulated-user turns
    tags: list[str] = field(default_factory=list)


@dataclass
class TrialResult:
    task_id: str
    index: int
    transcript: Transcript
    grades: list[GraderResult]
    seconds: float

    @property
    def passed(self) -> bool:
        gating = [g for g in self.grades if g.gating]
        return bool(gating) and all(g.score >= 1.0 for g in gating) and self.transcript.error is None


@dataclass
class TaskReport:
    task: Task
    trials: list[TrialResult]

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def pass_rate(self) -> float:
        return sum(t.passed for t in self.trials) / self.n if self.n else 0.0

    @property
    def pass_at_k(self) -> bool:
        """passed at least once in k trials: 'can it do this at all?'"""
        return any(t.passed for t in self.trials)

    @property
    def pass_hat_k(self) -> bool:
        """passed ALL k trials: what you need before an agent acts unsupervised."""
        return self.n > 0 and all(t.passed for t in self.trials)

    def grader_means(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for t in self.trials:
            for g in t.grades:
                out.setdefault(g.grader, []).append(g.score)
        return {k: sum(v) / len(v) for k, v in out.items()}


AgentFn = Callable[[Task], Awaitable[Transcript]]
"""Adapter contract: run the agent on a task (including task.followups, if
the agent supports multi-turn), return a Transcript. Everything framework-
specific lives in the adapter, so the harness works with any agent."""


async def run_trial(task: Task, agent: AgentFn, index: int) -> TrialResult:
    start = time.perf_counter()
    try:
        transcript = await agent(task)
    except Exception as e:      # the agent crashing is a result, not a harness failure
        transcript = Transcript(final_output="", error=f"{type(e).__name__}: {e}")
    grades = [await g(task, transcript) for g in task.graders]
    return TrialResult(task.id, index, transcript, grades, time.perf_counter() - start)


async def run_suite(tasks: list[Task], agent: AgentFn, trials: int = 3,
                    concurrency: int = 1) -> "SuiteReport":
    """Run every task `trials` times. concurrency > 1 runs trials in
    parallel (keep 1 for a single local model; raise it for hosted APIs)."""
    sem = asyncio.Semaphore(concurrency)

    async def guarded(task: Task, i: int) -> TrialResult:
        async with sem:
            return await run_trial(task, agent, i)

    results = await asyncio.gather(*(guarded(t, i) for t in tasks for i in range(trials)))
    by_task: dict[str, list[TrialResult]] = {t.id: [] for t in tasks}
    for r in results:
        by_task[r.task_id].append(r)
    return SuiteReport([TaskReport(t, sorted(by_task[t.id], key=lambda r: r.index)) for t in tasks])


@dataclass
class SuiteReport:
    tasks: list[TaskReport]

    @property
    def pass_rate(self) -> float:
        trials = [t for r in self.tasks for t in r.trials]
        return sum(t.passed for t in trials) / len(trials) if trials else 0.0

    def tier(self, tier: Tier) -> list[TaskReport]:
        return [r for r in self.tasks if r.task.tier == tier]

    def looks_right_is_wrong(self, judge: str, threshold: float = 0.7) -> tuple[int, int]:
        """(failed trials the judge scored >= threshold, failed trials judged).
        High = the judge can't tell good from bad on this agent; don't gate on it."""
        failed = [t for r in self.tasks for t in r.trials if not t.passed]
        judged = [g for t in failed for g in t.grades if g.grader == judge]
        return sum(g.score >= threshold for g in judged), len(judged)
