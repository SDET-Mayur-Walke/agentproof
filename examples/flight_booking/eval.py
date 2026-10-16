"""Evaluate the flight-booking agent with agentproof: 8 tasks, each aimed at
one way a booking agent goes wrong in production, graded on what it DID.

Gating graders (code, deterministic):
  outcome      the booking table is exactly right: the right flight, party
               and date, or NO booking when booking would be wrong
  trajectory   searched before booking; booked only a flight search returned
  arguments    the search carried the user's facts (date with year, party
               size, destination)
  grounding    every ₹ amount in the reply traces to a fare, a total, the
               user's budget, or a difference from it
  behaviour    task-specific words: asks for the missing date, says there
               are no flights instead of inventing one
Non-gating:
  judge        an LLM rates the reply 0-1 for "helpful and professional".
               Reported next to the code graders, to measure how often it
               likes an answer that actually failed.

RUN (needs `ollama serve` + qwen2.5:7b; ~20-30 min for 2 variants x 5 trials):
    pip install -e ".[openai-agents]"
    python examples/flight_booking/eval.py --trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from agents import Agent, Runner  # noqa: E402

import agent as fa  # noqa: E402
from agentproof import Task, Tier, Transcript, run_suite  # noqa: E402
from agentproof import graders as G  # noqa: E402
from agentproof.report import compare, console, save_json  # noqa: E402

YES = ["Yes, please book it."]
NO_FLIGHT_WORDS = ["no flight", "not find", "couldn't find", "could not find", "no available",
                   "unavailable", "not available", "none"]

# expected["booking"]: None -> there must be NO booking; else a spec the
# single booking must match (any of allowed_flights is acceptable).
SPECS = [
    dict(id="book_cheapest", tags=["happy path"], tier=Tier.REGRESSION,
         q="Book the cheapest flight from Pune to Goa on 10 December 2026 for 3 people.",
         search={"date": "2026-12-10", "passengers": 3, "destination": "goa"},
         booking={"allowed_flights": ["6E-6131"], "passengers": 3, "date": "2026-12-10"}, followups=YES),
    dict(id="compare_dont_book", tags=["unauthorised action"],
         q="Compare flights from Pune to Goa on 10 Dec 2026 for 2 people. Don't book anything yet.",
         search={"date": "2026-12-10", "passengers": 2, "destination": "goa"},
         booking=None, followups=["Thanks, that's all for now."]),
    dict(id="missing_date", tags=["guessing missing info"],
         q="Book a flight from Pune to Delhi for 2 people.",
         booking=None, must_say=["date", "when", "which day"]),
    dict(id="date_without_year", tags=["wrong date or year"],
         q="Book the 7am flight from Pune to Goa on 12th December for 1 person.",
         search={"date": "2026-12-12", "passengers": 1, "destination": "goa"},
         booking={"allowed_flights": ["6E-6133"], "passengers": 1, "date": "2026-12-12"}, followups=YES),
    dict(id="party_size_implicit", tags=["wrong passenger count"],
         q="Book the cheapest flight from Pune to Goa on 10 Dec 2026 for me and my two kids.",
         search={"date": "2026-12-10", "passengers": 3, "destination": "goa"},
         booking={"allowed_flights": ["6E-6131"], "passengers": 3, "date": "2026-12-10"}, followups=YES),
    dict(id="within_budget", tags=["ignoring constraints"],
         q="Book a flight from Pune to Goa on 10 Dec 2026 for 2 people. Our total budget is ₹8,000.",
         search={"date": "2026-12-10", "passengers": 2, "destination": "goa"}, budget=8000,
         booking={"allowed_flights": ["6E-6131", "IX-1250"], "passengers": 2, "date": "2026-12-10"},
         followups=YES),
    dict(id="budget_impossible", tags=["overspending"], tier=Tier.REGRESSION,
         q="Book a flight from Pune to Goa on 10 Dec 2026 for 2 people. Our total budget is ₹5,000.",
         search={"date": "2026-12-10", "passengers": 2, "destination": "goa"}, budget=5000,
         booking=None, followups=["Go ahead and book whatever fits our budget."]),
    dict(id="no_such_route", tags=["hallucinated result"],
         q="Book a flight from Pune to Leh on 15 December 2026 for 2 people.",
         booking=None, followups=YES, must_say=NO_FLIGHT_WORDS),
]


# ----- graders (domain-specific checks, built on agentproof.graders.custom) -----

def outcome(task: Task, t: Transcript) -> tuple[float, str]:
    exp, books = task.expected["booking"], (t.outcome or {}).get("bookings", [])
    if exp is None:
        return (1.0, "correctly made no booking") if not books else \
            (0.0, f"UNWANTED booking: {[(b['flight_id'], b['passengers']) for b in books]}")
    if len(books) != 1:
        return 0.0, f"expected exactly 1 booking, found {len(books)}"
    b = books[0]
    wrong = [k for k, ok in {"flight": b["flight_id"] in exp["allowed_flights"],
                             "passengers": b["passengers"] == exp["passengers"],
                             "date": b["date"] == exp["date"]}.items() if not ok]
    return (1.0, f"booked {b['flight_id']} x{b['passengers']} on {b['date']}") if not wrong else \
        (0.0, f"wrong {wrong}: booked {b['flight_id']} x{b['passengers']} on {b['date']}")


def booked_searched_ids(task: Task, t: Transcript) -> tuple[float, str]:
    searched = {f["flight_id"] for c in t.calls("search_flights") for f in (c.result or [])}
    unknown = [c.args["flight_id"] for c in t.calls("book_flight") if c.args["flight_id"] not in searched]
    return (0.0, f"booked flight ids search never returned: {unknown}") if unknown else \
        (1.0, "every booked id came from a search")


def search_args(task: Task, t: Transcript) -> tuple[float, str]:
    spec = task.expected.get("search")
    if not spec:
        return 1.0, "n/a"
    searches = t.calls("search_flights")
    if not searches:
        return 0.0, "never searched"
    if any(c.args["date"] == spec["date"] and c.args["passengers"] == spec["passengers"]
           and fa.norm_city(c.args["destination"]) == spec["destination"] for c in searches):
        return 1.0, f"searched {spec['date']} x{spec['passengers']}"
    return 0.0, f"expected ({spec['date']}, {spec['passengers']}), searched " \
                f"{[(c.args['date'], c.args['passengers']) for c in searches]}"


_AMT = re.compile(r"(?:₹|\binr\b|\brs\.?)\s*([\d][\d,]*(?:\.\d+)?)|([\d][\d,]*(?:\.\d+)?)\s*(?:\binr\b|rupees)", re.I)


def grounding(task: Task, t: Transcript) -> tuple[float, str]:
    """Currency amounts only (dates and times aren't money). Allowed: any
    fare x 1-6 travellers, the user's budget, and differences from it. The
    difference rule came from a false failure in the original POC: "the
    cheapest option exceeds your budget by ₹1,900" is derived, not invented."""
    if not t.final_output:
        return 0.0, "no answer"
    allowed = {f * k for f in fa.ALL_FARES for k in range(1, 7)}
    if budget := task.expected.get("budget"):
        allowed |= {budget} | {abs(x - budget) for x in list(allowed)}
    stated = [float((a or b).replace(",", "")) for a, b in _AMT.findall(t.final_output)]
    bad = [v for v in stated if v >= 100 and not any(abs(v - x) < 1 for x in allowed)]
    return (0.0, f"untraceable amounts {bad[:4]}") if bad else (1.0, f"{len(stated)} amounts, all traceable")


def says(task: Task, t: Transcript) -> tuple[float, str]:
    words = task.expected.get("must_say")
    if not words:
        return 1.0, "n/a"
    text = " ".join(x["content"] for x in t.turns if x["role"] == "assistant").lower() or t.final_output.lower()
    hit = [w for w in words if w in text]
    return (1.0, f"said {hit[0]!r}") if hit else (0.0, f"never said any of {words[:3]}...")


async def judge(rubric: str, task: Task, t: Transcript) -> tuple[float, str]:
    if not t.final_output.strip():
        return 0.0, "empty reply"
    j = Agent(name="Judge", model=fa.MODEL, instructions=rubric)
    r = await Runner.run(j, input=f"Customer: {task.input}\n\nAssistant reply:\n{t.final_output}")
    m = re.search(r"[01](?:\.\d+)?", str(r.final_output))
    return (float(m.group()), "judge score") if m else (0.5, f"unparseable: {str(r.final_output)[:40]}")


JUDGE_RUBRIC = ("You rate a travel assistant's reply to a customer. Score 0.0-1.0 for how helpful, "
                "correct-sounding and professional it is. Reply with ONLY the number.")


def build_tasks() -> list[Task]:
    graders = [G.custom("outcome", outcome),
               G.tool_order("search_flights", "book_flight"),
               G.custom("booked_searched_ids", booked_searched_ids),
               G.custom("arguments", search_args),
               G.custom("grounding", grounding),
               G.custom("behaviour", says),
               G.llm_judge("judge", JUDGE_RUBRIC, judge)]          # non-gating
    return [Task(id=s["id"], input=s["q"], graders=graders, tier=s.get("tier", Tier.CAPABILITY),
                 tags=s["tags"], followups=s.get("followups", []),
                 expected={k: s.get(k) for k in ("booking", "search", "budget", "must_say")})
            for s in SPECS]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--variants", default="baseline,hardened")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    tasks = build_tasks()
    if args.tasks:
        tasks = [t for t in tasks if t.id in args.tasks.split(",")]
    print(f"model {fa.MODEL} · {len(tasks)} tasks x {args.trials} trials", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    reports = {}
    for v in args.variants.split(","):
        print(f"\n=== {v} ===", flush=True)
        rep = await run_suite(tasks, fa.adapter(v), trials=args.trials)
        reports[v] = rep
        print(console(rep, judge="judge"), flush=True)
        save_json(rep, os.path.join(args.out_dir, f"{stamp}_{v}.json"),
                  meta={"model": fa.MODEL, "variant": v, "trials": args.trials})

    table = compare(reports, judge="judge")
    print("\n" + table)
    with open(os.path.join(args.out_dir, f"{stamp}_compare.md"), "w") as f:
        f.write(f"model `{fa.MODEL}` · {args.trials} trials per task\n\n{table}\n")


if __name__ == "__main__":
    asyncio.run(main())
