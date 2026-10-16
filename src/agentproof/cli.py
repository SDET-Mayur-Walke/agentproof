"""Command line: gate a saved eval run, or compare runs. Works on the JSON
that report.save_json() writes, so it needs no agent and no LLM. In CI the
eval runs first and writes results; this decides pass/fail.

    python -m agentproof gate results/run.json [--baseline results/main.json]
    python -m agentproof compare results/a.json results/b.json

Exit code 1 when the gate blocks, so a CI job fails and the PR can't merge.
On GitHub Actions the Markdown report is also written to the job summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _load(path: str) -> dict:
    matches = sorted(glob.glob(path))
    if not matches:
        sys.exit(f"no results file matches {path}")
    with open(matches[-1]) as f:          # a glob picks the newest-named file
        data = json.load(f)
    data["_path"] = matches[-1]
    return data


def _dots(task: dict) -> str:
    return "".join("●" if t["passed"] else "○" for t in task["trials"])


def gate(run: dict, baseline: dict | None, regression_min: float, capability_min: float | None,
         max_drop: float) -> tuple[list[str], str]:
    reasons = []
    for t in run["tasks"]:
        if t["tier"] == "regression" and t["pass_rate"] < regression_min:
            fails = [g["reason"] for tr in t["trials"] if not tr["passed"]
                     for g in tr["grades"] if g["gating"] and g["score"] < 1.0][:2]
            reasons.append(f"regression task `{t['id']}` at {t['pass_rate']:.0%} "
                           f"(min {regression_min:.0%}): {'; '.join(fails) or 'agent error'}")
    cap = [tr for t in run["tasks"] if t["tier"] == "capability" for tr in t["trials"]]
    cap_rate = sum(tr["passed"] for tr in cap) / len(cap) if cap else None
    if capability_min is not None and cap_rate is not None and cap_rate < capability_min:
        reasons.append(f"capability pass rate {cap_rate:.0%} < {capability_min:.0%}")
    if baseline:
        drop = baseline["pass_rate"] - run["pass_rate"]
        if drop > max_drop:
            reasons.append(f"overall pass rate fell {drop:.0%} vs baseline "
                           f"({baseline['pass_rate']:.0%} -> {run['pass_rate']:.0%}; max drop {max_drop:.0%})")

    meta = run.get("meta", {})
    lines = [f"### agentproof eval gate: {'✅ PASS' if not reasons else '❌ BLOCKED'}",
             f"`{os.path.basename(run['_path'])}` · model `{meta.get('model', '?')}` · "
             f"variant `{meta.get('variant', '?')}` · {meta.get('trials', '?')} trials per task", "",
             "| Task | Tier | Trials | Pass rate |", "|---|---|---|---|"]
    base_rates = {t["id"]: t["pass_rate"] for t in (baseline or {}).get("tasks", [])}
    for t in run["tasks"]:
        delta = f" ({t['pass_rate'] - base_rates[t['id']]:+.0%})" if t["id"] in base_rates else ""
        lines.append(f"| `{t['id']}` | {t['tier']} | {_dots(t)} | {t['pass_rate']:.0%}{delta} |")
    lines.append(f"\n**Overall: {run['pass_rate']:.0%}**"
                 + (f" (baseline {baseline['pass_rate']:.0%})" if baseline else ""))
    if reasons:
        lines += ["", "**Why it's blocked:**"] + [f"- {r}" for r in reasons]
    return reasons, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentproof")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="pass/fail a saved eval run")
    g.add_argument("results", help="results JSON (a glob picks the newest match)")
    g.add_argument("--baseline", help="results JSON to compare against, e.g. from main")
    g.add_argument("--regression-min", type=float, default=1.0)
    g.add_argument("--capability-min", type=float)
    g.add_argument("--max-drop", type=float, default=0.10)
    c = sub.add_parser("compare", help="side-by-side pass rates of saved runs")
    c.add_argument("results", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        runs = [_load(p) for p in args.results]
        names = [os.path.basename(r["_path"]) for r in runs]
        print("| Task | " + " | ".join(names) + " |\n|---|" + "---|" * len(runs))
        for i, t in enumerate(runs[0]["tasks"]):
            print(f"| `{t['id']}` | " + " | ".join(f"{_dots(r['tasks'][i])} {r['tasks'][i]['pass_rate']:.0%}"
                                                  for r in runs) + " |")
        print("| **overall** | " + " | ".join(f"**{r['pass_rate']:.0%}**" for r in runs) + " |")
        return 0

    run = _load(args.results)
    base = _load(args.baseline) if args.baseline else None
    reasons, md = gate(run, base, args.regression_min, args.capability_min, args.max_drop)
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(md + "\n")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
