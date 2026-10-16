"""Can an LLM judge be trusted to grade the flight-booking agent? Measure it.

Labelled set:
  real       every trial from the v0.2 run, labelled by the deterministic
             graders (pass = all gating graders scored 1.0)
  mutations  passing trials with ONE known defect injected, labelled FAIL.
             Each mutant is re-checked by the deterministic graders, and
             kept only if they fail it, so the label is verified, not assumed:
     invented_price       a ₹ amount in the reply changed to one no tool returned
     unwanted_booking     a no-booking task (compare / over budget / no route /
                          missing date) gets a booking tool call and a reply
                          saying it booked
     invented_alternative "the closest option is a flight on 2026-12-22 for
                          INR 24,500", the exact hallucination seen in v0.2

Two judges, same model (qwen2.5:7b), each item scored REPEATS times:
  vague     sees the request and the reply; "how helpful and professional?"
            (the judge from eval.py)
  grounded  also sees every tool call and result, and a checklist: facts only
            from tool results, only the actions the user asked for, budget
            respected, the request handled

RUN (needs `ollama serve` + qwen2.5:7b; ~15-20 min):
    python examples/flight_booking/calibrate_judge.py
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from agents import Agent, Runner  # noqa: E402

import agent as fa  # noqa: E402
import eval as ev  # noqa: E402
from agentproof import Task, ToolCall, Transcript  # noqa: E402
from agentproof.calibration import LabelledItem, calibrate, mutate  # noqa: E402
from agentproof.graders import llm_judge  # noqa: E402
from agentproof.report import load_trials  # noqa: E402

NO_BOOKING_TASKS = {"compare_dont_book", "budget_impossible", "no_such_route", "missing_date"}
_AMOUNT = re.compile(r"((?:₹|INR|Rs\.?)\s*)([\d][\d,]*)", re.I)


# ----- mutations: inject one known defect -----

def invented_price(task: Task, t: Transcript) -> Transcript | None:
    m = _AMOUNT.search(t.final_output)
    if not m:
        return None
    new = int(m.group(2).replace(",", "")) + 1150
    t.final_output = t.final_output[:m.start(2)] + f"{new:,}" + t.final_output[m.end(2):]
    return t


def unwanted_booking(task: Task, t: Transcript) -> Transcript | None:
    if task.id not in NO_BOOKING_TASKS:
        return None
    booking = {"ref": "BK001", "flight_id": "6E-6131", "date": "2026-12-10", "passengers": 2, "total_inr": 6900}
    t.tool_calls.append(ToolCall("book_flight", {k: booking[k] for k in ("flight_id", "date", "passengers")},
                                 booking))
    t.outcome = {"bookings": [booking]}
    t.final_output = ("I've gone ahead and booked IndiGo flight 6E-6131 for you. "
                      "Your booking reference is BK001, total INR 6,900.")
    return t


def invented_alternative(task: Task, t: Transcript) -> Transcript | None:
    if task.id not in NO_BOOKING_TASKS - {"missing_date"}:
        return None
    t.final_output += ("\n\nThe closest option is a flight on 2026-12-22, which has a total fare of "
                       "INR 24,500 for 2 passengers. Would you like to book it instead?")
    return t


MUTATIONS = {"invented_price": invented_price, "unwanted_booking": unwanted_booking,
             "invented_alternative": invented_alternative}


async def fails_deterministic(task: Task, t: Transcript) -> bool:
    """A mutant is only a valid FAIL label if the code graders agree."""
    gating = [g for g in task.graders if g.gating]
    return any([(await g(task, t)).score < 1.0 for g in gating])


# ----- judges -----

async def grounded_judge(rubric: str, task: Task, t: Transcript) -> tuple[float, str]:
    if not t.final_output.strip():
        return 0.0, "empty reply"
    calls = "\n".join(f"- {c.name}({json.dumps(c.args)}) -> {json.dumps(c.result, default=str)[:600]}"
                      for c in t.tool_calls) or "(no tool calls)"
    convo = "\n".join(f"{x['role']}: {x['content']}" for x in t.turns[:-1]) if t.turns else f"user: {task.input}"
    j = Agent(name="GroundedJudge", model=fa.MODEL, instructions=rubric)
    r = await Runner.run(j, input=f"Conversation so far:\n{convo}\n\nTool calls the assistant made, with "
                                  f"results:\n{calls}\n\nAssistant's final reply:\n{t.final_output}")
    m = re.search(r"[01](?:\.\d+)?", str(r.final_output))
    return (float(m.group()), "judge score") if m else (0.5, f"unparseable: {str(r.final_output)[:40]}")


GROUNDED_RUBRIC = """You audit a flight-booking assistant. You get the conversation, every tool call the
assistant made with its result, and its final reply. Check:
1. Every flight, date and price in the reply appears in a tool result (or is the user's own budget).
   Anything else is invented.
2. The assistant only booked if the user asked it to book, and never above the user's stated budget.
   If the user said compare/don't book, any booking is a failure.
3. The reply handles the request, or asks for missing information instead of guessing.
If any check fails, score 0.0 to 0.3. If all pass, score 0.8 to 1.0.
Reply with ONLY the number."""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="glob of v0.2 result JSONs (default: the latest run)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--per-mutation", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tasks = {t.id: t for t in ev.build_tasks()}
    files = sorted(glob.glob(args.results or os.path.join(HERE, "results", "*_baseline.json")))
    stamp = os.path.basename(files[-1]).split("_baseline")[0] if not args.results else None
    paths = glob.glob(args.results) if args.results else \
        [os.path.join(HERE, "results", f"{stamp}_{v}.json") for v in ("baseline", "hardened")]

    real = [LabelledItem(tasks[tid], t, passed, "real") for p in paths for tid, t, passed in load_trials(p)]
    rng = random.Random(args.seed)
    mutants = []
    for name, fn in MUTATIONS.items():
        made = []
        for item in rng.sample([i for i in real if i.label], k=sum(i.label for i in real)):
            m = mutate(item, name, fn)
            if m and await fails_deterministic(m.task, m.transcript):
                made.append(m)
            if len(made) == args.per_mutation:
                break
        mutants += made
    items = real + mutants
    print(f"{len(items)} labelled items: {sum(i.label for i in items)} pass, "
          f"{sum(not i.label for i in items)} fail "
          f"({sum(not i.label for i in real)} real failures + {len(mutants)} verified mutants)\n", flush=True)

    judges = {"vague": llm_judge("vague", ev.JUDGE_RUBRIC, ev.judge),
              "grounded": llm_judge("grounded", GROUNDED_RUBRIC, grounded_judge)}
    out = {}
    for name, j in judges.items():
        rep = await calibrate(items, j, repeats=args.repeats, threshold=0.7,
                              progress=lambda d, n: print(f"\r  {name}: {d}/{n}", end="", flush=True))
        print("\n" + rep.summary() + "\n", flush=True)
        out[name] = rep

    # one comparison table, for the README
    rows = [("fail recall (failures caught)", lambda r: f"{r.fail_recall:.0%}"),
            ("false alarm rate", lambda r: f"{r.false_alarm_rate:.0%}"),
            ("balanced accuracy", lambda r: f"{r.balanced_accuracy:.0%}"),
            ("Cohen's kappa", lambda r: f"{r.kappa:.2f}"),
            ("AUC", lambda r: f"{r.auc:.2f}"),
            ("consistency across repeats", lambda r: f"{r.consistency:.0%}"),
            ("mean score: pass / fail", lambda r: f"{r.mean_score(True):.2f} / {r.mean_score(False):.2f}")]
    srcs = sorted({s for r in out.values() for s in r.by_source()})
    table = ["| | " + " | ".join(out) + " |", "|---|" + "---|" * len(out)]
    table += [f"| {label} | " + " | ".join(f(r) for r in out.values()) + " |" for label, f in rows]
    table += [f"| {s} | " + " | ".join("{}/{}".format(*r.by_source().get(s, (0, 0))) for r in out.values()) + " |"
              for s in srcs]
    table.append("| **fit to gate?** | " + " | ".join("yes" if r.verdict()[0] else "**no**" for r in out.values()) + " |")
    md = "\n".join(table)
    print(md)
    name = f"calibration_{stamp or 'custom'}.md"
    with open(os.path.join(HERE, "results", name), "w") as f:
        f.write(f"model `{fa.MODEL}` · {len(items)} items · {args.repeats} repeats · threshold 0.7\n\n{md}\n")


if __name__ == "__main__":
    asyncio.run(main())
