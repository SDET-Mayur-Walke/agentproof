"""Grade τ²-bench airline runs with agentproof.

    python examples/tau2_airline/eval.py ~/Projects/tau2-bench/data/simulations/<run>/results.json \
        [--db ~/Projects/tau2-bench/data/tau2/domains/airline/db.json] [--out results/tau2_<run>.json]

Prints, per task: pass@k and pass^k, every grader's verdict, and the cases
where τ²-bench and the policy graders disagree:
  right outcome, wrong process   τ² reward 1, but a policy grader failed
  wrong outcome, explained       τ² reward 0, and a policy grader says why
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentproof import run_suite                 # noqa: E402
from agentproof import report as rpt             # noqa: E402
from adapter import infrastructure_errors, load  # noqa: E402
from policy_graders import all_graders           # noqa: E402

PROCESS = ["confirmed_before_write", "cancel_eligible", "cancel_reason_asked", "compensation_allowed",
           "no_tool_loop", "no_reply_loop",
           "membership_grounded"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--db", default=os.path.expanduser("~/Projects/tau2-bench/data/tau2/domains/airline/db.json"))
    ap.add_argument("--out")
    a = ap.parse_args()

    db = json.load(open(a.db)) if os.path.exists(a.db) else json.load(open(a.results)).get("db_subset")
    tasks, agent, k = load(a.results, all_graders(db))
    if not tasks:
        sys.exit("no saved trials in " + a.results)
    report = asyncio.run(run_suite(tasks, agent, trials=k))

    print(f"\nτ²-bench airline · {len(tasks)} tasks × {k} trials · {os.path.basename(os.path.dirname(a.results))}\n")
    print(rpt.console(report))
    infra = infrastructure_errors(a.results)
    if infra:
        print(f"\nleft out: {len(infra)} trial(s) τ²-bench couldn't run (infrastructure_error), tasks {sorted(set(infra))}")

    tau2_only = sum(g.score >= 1 for r in report.tasks for t in r.trials for g in t.grades if g.grader == "tau2_reward")
    n = sum(len(r.trials) for r in report.tasks)
    print(f"\nτ²-bench reward alone: {tau2_only}/{n} trials pass · with policy graders: "
          f"{sum(t.passed for r in report.tasks for t in r.trials)}/{n}")

    print("\nWhy each trial failed:")
    right_wrong = explained = 0
    for r in report.tasks:
        for t in r.trials:
            g = {x.grader: x for x in t.grades}
            fails = [x for x in PROCESS if g[x].score < 1]
            tau2_ok = g["tau2_reward"].score >= 1
            if tau2_ok and fails:
                right_wrong += 1
            if not tau2_ok and (fails or t.transcript.error):
                explained += 1
            if t.passed:
                continue
            tag = "RIGHT OUTCOME, WRONG PROCESS" if tau2_ok else "FAIL"
            print(f"  task {r.task.id} trial {t.index}  [{tag}]  {g['tau2_reward'].reason}")
            if t.transcript.error:
                print(f"      ✗ {t.transcript.error}")
            for x in fails:
                print(f"      ✗ {x}: {g[x].reason}")
            if g["user_sim_leak"].score < 1:
                print(f"      ⚠ test defect: {g['user_sim_leak'].reason}")
            if t.transcript.usage.get("max_prompt_tokens"):
                print(f"      · largest prompt {t.transcript.usage['max_prompt_tokens']} tokens, "
                      f"{t.transcript.usage.get('seconds') or 0:.0f}s")
    fails_total = n - tau2_only
    print(f"\nright outcome, wrong process: {right_wrong} · "
          f"τ² failures a policy grader explains: {explained}/{fails_total}")
    leaks = sum(g.score < 1 for r in report.tasks for t in r.trials for g in t.grades if g.grader == "user_sim_leak")
    print(f"simulated user leaked its script in {leaks}/{n} trials")

    if a.out:
        rpt.save_json(report, a.out, meta={"source": a.results, "benchmark": "tau2-bench airline", "k": k})
        print("saved", a.out)


if __name__ == "__main__":
    main()
