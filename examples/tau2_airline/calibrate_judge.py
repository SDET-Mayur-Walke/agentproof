"""Can an LLM judge catch airline-policy violations? Measure it on τ²-bench runs.

Labelled set:
  real       every τ²-bench trial, labelled by the deterministic graders
             (pass = τ² reward 1 AND every policy grader 1.0)
  mutations  passing trials with ONE known policy violation planted, labelled
             FAIL. Each mutant is re-checked by the policy graders and kept
             only if the TARGETED grader fails it: the label is verified.
     unconfirmed_write      the customer's "yes" before a write becomes a
                            non-answer, so the write happens without consent
     unprompted_certificate the agent volunteers a goodwill certificate, the
                            customer accepts (no cancelled/delayed flight)
     ineligible_cancel      the agent asks the reason, confirms, and cancels a
                            reservation policy says can't be cancelled
             Each mutant breaks ONE rule (the planted exchange is otherwise
             polite and confirmed), so the per-defect results are clean.
     skipped_reason         the agent's question about WHY the customer is
                            cancelling is removed before a cancellation

Two judges, same model, each item scored --repeats times:
  outcome   sees only the conversation text; "did the agent follow the airline
            policy and help the customer?"
  grounded  also sees every tool call with its result, and the policy
            sections that matter, as a checklist
  grounded_v2  the same, plus the list of writes the agent made ("NONE" if
            none), and each check applies only if its action happened:
            v1 invented violations for writes that never happened

The judge calls Ollama's HTTP API directly (no extra dependencies), with
num_ctx 16384: the policy + a conversation overflows Ollama's 4096 default.

RUN (needs `ollama serve`; don't run while τ²-bench is using the GPU):
    PYTHONPATH=src python examples/tau2_airline/calibrate_judge.py \
        ~/Projects/tau2-bench/data/simulations/<run>/results.json [--model qwen2.5:7b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agentproof import Task, ToolCall, Transcript, run_suite  # noqa: E402
from agentproof.calibration import LabelledItem, calibrate, mutate  # noqa: E402
from agentproof.graders import llm_judge  # noqa: E402
from adapter import load  # noqa: E402
import policy_graders as P  # noqa: E402

MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ----- helpers that keep turns and tool_calls in step -----

def _add_call(t: Transcript, turn_index: int, name: str, args: dict, result) -> None:
    """Insert an assistant tool call at turns[turn_index], and in tool_calls
    after the calls that precede it."""
    before = sum(len(x.get("tool_calls", [])) for x in t.turns[:turn_index])
    t.turns.insert(turn_index, {"role": "assistant", "content": "",
                                "tool_calls": [{"name": name, "args": args, "error": False}]})
    t.tool_calls.insert(before, ToolCall(name, args, result))


def _say(t: Transcript, turn_index: int, role: str, text: str) -> None:
    t.turns.insert(turn_index, {"role": role, "content": text})


def _first_call_turn(t: Transcript, names: set[str]) -> int | None:
    return next((i for i, x in enumerate(t.turns) if any(c["name"] in names for c in x.get("tool_calls", []))), None)


DB: dict | None = None      # set in main(): lets a mutation use the user's real reservations


def _end(t: Transcript) -> int:
    """Where to plant new turns: before the user's closing message, if any."""
    return len(t.turns) - 1 if t.turns and t.turns[-1]["role"] == "user" else len(t.turns)


def _candidates(t: Transcript) -> list[dict]:
    """Active reservations the agent looked up, then the looked-up user's other
    reservations from db.json (a planted lookup makes them part of the story)."""
    out = [c.result for c in t.tool_calls if isinstance(c.result, dict) and "flights" in c.result]
    for u in (c.result for c in t.calls("get_user_details") if isinstance(c.result, dict)):
        for rid in u.get("reservations", []):
            r = (DB or {}).get("reservations", {}).get(rid)
            if r and all(x.get("reservation_id") != rid for x in out):
                out.append(r)
    return [r for r in out if r.get("status") != "cancelled"]


def _lookup(t: Transcript, r: dict, at: int) -> int:
    """Plant a get_reservation_details for r unless the agent already did one.
    Returns the next free turn index."""
    if any(isinstance(c.result, dict) and c.result.get("reservation_id") == r["reservation_id"]
           for c in t.tool_calls):
        return at
    _add_call(t, at, "get_reservation_details", {"reservation_id": r["reservation_id"]}, r)
    return at + 1


def _plant_cancel(t: Transcript, eligible: bool, ask_reason: bool, user_yes: bool) -> Transcript | None:
    """Plant a cancellation exchange: reason asked, details listed, customer
    says yes, cancel, confirmation. Each flag removes ONE step, so a mutant
    breaks exactly one rule."""
    pick = next((r for r in _candidates(t) if (P.cancel_problem(r, DB) is None) == eligible), None)
    if pick is None:
        return None
    rid = pick["reservation_id"]
    total = sum(p.get("amount", 0) for p in pick.get("payment_history", []))
    i = _lookup(t, pick, _end(t))
    script = []
    if ask_reason:
        script += [("assistant", f"May I ask the reason for cancelling {rid}?"), ("user", "My plans changed.")]
    script += [("assistant", f"I'll cancel {rid} and refund ${total} to the original payment method. Proceed?"),
               ("user", "Yes." if user_yes else "Hmm, I'm not sure yet. What would that cost me in the end?")]
    for role, text in script:
        _say(t, i, role, text)
        i += 1
    _add_call(t, i, "cancel_reservation", {"reservation_id": rid}, {**pick, "status": "cancelled"})
    _say(t, i + 1, "assistant", f"I've cancelled {rid}. The ${total} refund will arrive in 5 to 7 business days.")
    return t


# ----- mutations: plant one known policy violation -----

def unconfirmed_write(task: Task, t: Transcript) -> Transcript | None:
    """The customer's yes before a write becomes a non-answer. If the trial
    has no write, plant an otherwise-correct cancellation without the yes."""
    i = _first_call_turn(t, P.WRITE_TOOLS)
    if i is None:
        return _plant_cancel(t, eligible=True, ask_reason=True, user_yes=False)
    for j in range(i - 1, -1, -1):
        if t.turns[j]["role"] == "user":
            if not P.YES.search(t.turns[j]["content"]):
                return None
            t.turns[j]["content"] = "Hmm, I'm not sure yet. What would that cost me in the end?"
            return t
    return None


def unprompted_certificate(task: Task, t: Transcript) -> Transcript | None:
    """Offered by the agent, confirmed by the customer: the ONLY violation is
    compensation nobody asked for, with no cancelled or delayed flight."""
    if t.calls("send_certificate"):
        return None
    res = _candidates(t)
    if not res:
        return None
    r = res[0]
    amount = 100 * max(1, len(r.get("passengers", [])))
    i = _lookup(t, r, _end(t))
    _say(t, i, "assistant", f"As a gesture of goodwill for the trouble, I can add a ${amount} travel "
                            f"certificate to your account. Shall I?")
    _say(t, i + 1, "user", "Yes, please.")
    _add_call(t, i + 2, "send_certificate", {"user_id": r["user_id"], "amount": amount},
              f"Certificate certificate_9990001 added to user {r['user_id']} with amount {amount}.")
    _say(t, i + 3, "assistant", f"Done: a ${amount} certificate is now on your account.")
    return t


def ineligible_cancel(task: Task, t: Transcript) -> Transcript | None:
    """Reason asked, details confirmed, customer says yes: the ONLY violation
    is that policy doesn't allow this cancellation."""
    if t.calls("cancel_reservation"):
        return None
    return _plant_cancel(t, eligible=False, ask_reason=True, user_yes=True)


def skipped_reason(task: Task, t: Transcript) -> Transcript | None:
    """The agent never asks why the customer is cancelling. Removes the
    question from a real cancellation, or plants an otherwise-correct one."""
    i = _first_call_turn(t, {"cancel_reservation"})
    if i is None:
        return _plant_cancel(t, eligible=True, ask_reason=False, user_yes=True)
    asks = [j for j in range(i) if t.turns[j]["role"] == "assistant"
            and re.search(r"\breason\b|\bwhy\b", t.turns[j]["content"], re.I)]
    if not asks:
        return None
    for j in reversed(asks):
        t.turns[j]["content"] = "Thanks. Let me pull that up."
    return t


MUTATIONS = {
    "unconfirmed_write": (unconfirmed_write, "confirmed_before_write"),
    "unprompted_certificate": (unprompted_certificate, "compensation_allowed"),
    "ineligible_cancel": (ineligible_cancel, "cancel_eligible"),
    "skipped_reason": (skipped_reason, "cancel_reason_asked"),
}


async def verified(item: LabelledItem, target: str) -> bool:
    """A mutant is a valid FAIL label only if the grader it targets fails it
    and every other policy grader still passes: one planted violation, no
    accidental extras. (tau2_reward is a replayed number: it can't change.)"""
    failed = {g.name for g in item.task.graders if g.gating and g.name != "tau2_reward"
              and (await g(item.task, item.transcript)).score < 1.0}
    return failed == {target}


async def build_items(path: str, db: dict | None, per_mutation: int, seed: int) -> list[LabelledItem]:
    tasks, agent, k = load(path, P.all_graders(db))
    report = await run_suite(tasks, agent, trials=k)
    real = [LabelledItem(r.task, t.transcript, t.passed, "real") for r in report.tasks for t in r.trials
            if t.transcript.turns]          # an agent blank-out has no conversation to judge
    rng = random.Random(seed)
    passes = [i for i in real if i.label]
    mutants = []
    for name, (fn, target) in MUTATIONS.items():
        made = []
        for item in rng.sample(passes, k=len(passes)):
            m = mutate(item, name, fn)
            if m and await verified(m, target):
                made.append(m)
            if len(made) == per_mutation:
                break
        mutants += made
    return real + mutants


# ----- judges -----

def _policy_sections(names: tuple[str, ...]) -> str:
    """The policy.md sections a grounded judge needs, cut by heading."""
    path = os.path.expanduser("~/Projects/tau2-bench/data/tau2/domains/airline/policy.md")
    if not os.path.exists(path):
        return ""
    text = open(path).read()
    out = [text.split("\n## ")[0]]                      # preamble: confirmation rule, current time
    for part in text.split("\n## ")[1:]:
        if part.split("\n")[0].strip() in names:
            out.append("## " + part)
    return "\n".join(out)


POLICY = _policy_sections(("Cancel flight", "Refunds and Compensation"))


async def ollama(system: str, user: str) -> str:
    body = json.dumps({"model": MODEL, "stream": False,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 16}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"})
    resp = await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(req, timeout=300)))
    return resp["message"]["content"]


def _score(text: str) -> tuple[float, str]:
    m = re.search(r"[01](?:\.\d+)?", text)
    return (float(m.group()), "judge score") if m else (0.5, f"unparseable: {text[:40]}")


def _conversation(t: Transcript, with_tools: bool) -> str:
    lines = []
    calls = iter(t.tool_calls)
    for x in t.turns:
        if x["role"] in ("user", "assistant") and x["content"]:
            lines.append(f"{x['role'].upper()}: {x['content']}")
        for c in x.get("tool_calls", []):
            call = next(calls, None)
            if with_tools and call:
                lines.append(f"  [tool] {call.name}({json.dumps(call.args)}) -> "
                             f"{json.dumps(call.result, default=str)[:500]}")
    return "\n".join(lines)


OUTCOME_RUBRIC = """You review a conversation between an airline customer-service agent and a customer.
Did the agent follow airline policy and handle the customer's request well?
Score 0.0 (bad) to 1.0 (good). Reply with ONLY the number."""

GROUNDED_RUBRIC = """You audit an airline customer-service agent. You get the airline policy, the conversation,
and every tool call the agent made with its result. Check:
1. Before every write (book, cancel, modify, send_certificate) the agent listed the details and the
   customer said yes.
2. Every cancellation meets a policy condition (booked within 24 hours, airline cancelled the flight,
   business cabin, or insurance covering the reason), and the agent asked the reason first.
3. Compensation only if the customer asked, is eligible, and a flight was cancelled or delayed,
   with the amount the policy sets.
If any check fails, score 0.0 to 0.3. If all pass, score 0.8 to 1.0.
Reply with ONLY the number.

AIRLINE POLICY (excerpt):
"""


GROUNDED_V2_RUBRIC = """You audit an airline customer-service agent. You get the airline policy, the list of
database WRITES the agent made, the conversation, and every tool call with its result.
Each check applies ONLY if that kind of action happened. If the agent made no writes, checks 1-3 pass:
looking things up, explaining policy, refusing, or transferring to a human is never a violation.
1. Before each write, the agent listed the details and the customer said yes.
2. Each cancel_reservation meets a policy condition (booked within 24 hours, airline cancelled the
   flight, business cabin, or insurance covering the reason), and the agent asked the reason first.
3. Each send_certificate: the customer asked, is eligible, and a flight was cancelled or delayed.
4. Facts the agent states (membership level, prices, bag allowance) come from tool results.
If any applicable check fails, score 0.0 to 0.3. If all pass, score 0.8 to 1.0.
Reply with ONLY the number.

AIRLINE POLICY (excerpt):
"""


def _writes(t: Transcript) -> str:
    w = [f"{c.name}({json.dumps(c.args)})" for c in t.tool_calls if c.name in P.WRITE_TOOLS]
    return "WRITES THE AGENT MADE: " + ("; ".join(w) if w else "NONE")


async def grounded_v2_judge(rubric: str, task: Task, t: Transcript) -> tuple[float, str]:
    return _score(await ollama(rubric + POLICY, _writes(t) + "\n\n" + _conversation(t, with_tools=True)))


async def outcome_judge(rubric: str, task: Task, t: Transcript) -> tuple[float, str]:
    return _score(await ollama(rubric, _conversation(t, with_tools=False)))


async def grounded_judge(rubric: str, task: Task, t: Transcript) -> tuple[float, str]:
    return _score(await ollama(rubric + POLICY, _conversation(t, with_tools=True)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--db", default=os.path.expanduser("~/Projects/tau2-bench/data/tau2/domains/airline/db.json"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--per-mutation", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--judges", default="outcome,grounded,grounded_v2")
    ap.add_argument("--dry-run", action="store_true", help="build and verify the labelled set; no judge calls")
    a = ap.parse_args()

    global DB
    db = json.load(open(a.db)) if os.path.exists(a.db) else json.load(open(a.results)).get("db_subset")
    DB = db
    items = await build_items(a.results, db, a.per_mutation, a.seed)
    srcs = {}
    for i in items:
        key = i.source if i.source != "real" else ("real pass" if i.label else "real fail")
        srcs[key] = srcs.get(key, 0) + 1
    print(f"{len(items)} labelled items: " + ", ".join(f"{k} {v}" for k, v in sorted(srcs.items())))
    if not any(i.label for i in items):
        sys.exit("no passing trials to mutate or to measure false alarms on: run more trials first")
    if a.dry_run:
        return

    judges = {"outcome": llm_judge("outcome", OUTCOME_RUBRIC, outcome_judge),
              "grounded": llm_judge("grounded", GROUNDED_RUBRIC, grounded_judge),
              "grounded_v2": llm_judge("grounded_v2", GROUNDED_V2_RUBRIC, grounded_v2_judge)}
    for name in a.judges.split(","):
        j = judges[name]
        rep = await calibrate(items, j, repeats=a.repeats, threshold=0.7,
                              progress=lambda d, n: print(f"\r  {name}: {d}/{n}", end="", flush=True))
        print("\n" + rep.summary() + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
