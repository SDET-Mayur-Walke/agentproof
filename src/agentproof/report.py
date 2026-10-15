"""Reports: console table, JSON (for diffing runs), Markdown (for PRs/READMEs),
and a CI gate."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .core import SuiteReport, Tier


def console(report: SuiteReport, judge: str | None = None) -> str:
    lines = []
    for tier in Tier:
        rows = report.tier(tier)
        if not rows:
            continue
        lines.append(f"\n{tier.value.upper()} — {len(rows)} task(s)")
        for r in rows:
            dots = "".join("●" if t.passed else "○" for t in r.trials)
            lines.append(f"  {dots:<8} {r.task.id:<32} pass {r.pass_rate:>4.0%}  "
                         f"pass^k {'yes' if r.pass_hat_k else 'no ':<3}")
            for t in r.trials:
                for g in t.grades:
                    if g.gating and g.score < 1.0:
                        lines.append(f"           trial {t.index}: {g.grader}: {g.reason}")
                if t.transcript.error:
                    lines.append(f"           trial {t.index}: agent error: {t.transcript.error}")
    lines.append(f"\nOverall trial pass rate: {report.pass_rate:.0%}")
    if judge:
        hit, n = report.looks_right_is_wrong(judge)
        if n:
            lines.append(f"Judge '{judge}' rated {hit}/{n} FAILED trials as good (looks right, is wrong)")
    return "\n".join(lines)


def to_json(report: SuiteReport, meta: dict | None = None) -> dict:
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta or {},
        "pass_rate": report.pass_rate,
        "tasks": [{
            "id": r.task.id, "tier": r.task.tier.value, "tags": r.task.tags,
            "pass_rate": r.pass_rate, "pass_hat_k": r.pass_hat_k,
            "grader_means": r.grader_means(),
            "trials": [{
                "index": t.index, "passed": t.passed, "seconds": round(t.seconds, 2),
                "grades": [asdict(g) for g in t.grades],
                "final_output": t.transcript.final_output,
                "tool_calls": [asdict(c) for c in t.transcript.tool_calls],
                "outcome": t.transcript.outcome, "error": t.transcript.error,
                "usage": t.transcript.usage,
            } for t in r.trials],
        } for r in report.tasks],
    }


def save_json(report: SuiteReport, path: str, meta: dict | None = None) -> None:
    with open(path, "w") as f:
        json.dump(to_json(report, meta), f, indent=2, default=str)


def markdown(report: SuiteReport) -> str:
    out = ["| Task | Tier | Trials | Pass rate | pass^k |", "|---|---|---|---|---|"]
    for r in report.tasks:
        dots = "".join("●" if t.passed else "○" for t in r.trials)
        out.append(f"| `{r.task.id}` | {r.task.tier.value} | {dots} | {r.pass_rate:.0%} | "
                   f"{'✅' if r.pass_hat_k else '❌'} |")
    out.append(f"\n**Overall trial pass rate: {report.pass_rate:.0%}**")
    return "\n".join(out)


def gate(report: SuiteReport, regression_min: float = 1.0, capability_min: float | None = None) -> list[str]:
    """Reasons to block a change. Regression tasks must hold their floor;
    capability tasks only gate if you set a minimum."""
    reasons = [f"regression task {r.task.id} at {r.pass_rate:.0%} < {regression_min:.0%}"
               for r in report.tier(Tier.REGRESSION) if r.pass_rate < regression_min]
    if capability_min is not None:
        cap = [t for r in report.tier(Tier.CAPABILITY) for t in r.trials]
        rate = sum(t.passed for t in cap) / len(cap) if cap else 1.0
        if rate < capability_min:
            reasons.append(f"capability pass rate {rate:.0%} < {capability_min:.0%}")
    return reasons
