"""Graders for the τ²-bench airline domain: say WHY a trial failed, and catch
process failures that the final-DB check can't see.

τ²-bench's reward answers "is the database right at the end?". These answer
"did the agent follow the policy on the way there?" Each rule below quotes
the line of data/tau2/domains/airline/policy.md it checks.

  tau2_reward            τ²-bench's own verdict (gating)
  confirmed_before_write every write was preceded by an explicit "yes" (gating)
  cancel_eligible        every cancellation met one of the policy's conditions (gating)
  cancel_reason_asked    the agent asked why before cancelling (gating)
  compensation_allowed   certificates only when asked, eligible, and for a listed reason (gating)
  no_tool_loop           no identical tool call repeated 3+ times (gating)
  user_sim_leak          the SIMULATED USER read out its hidden script (non-gating:
                         a defect in the test, not the agent)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta

from agentproof import Grader, Task, Transcript
from agentproof.graders import custom

NOW = datetime(2024, 5, 15, 15, 0)   # policy.md: "The current time is 2024-05-15 15:00:00 EST."

# tools marked @is_tool(ToolType.WRITE) in tau2/domains/airline/tools.py
WRITE_TOOLS = {"book_reservation", "cancel_reservation", "send_certificate",
               "update_reservation_baggages", "update_reservation_flights",
               "update_reservation_passengers"}

# A yes is "yes" anywhere, or an affirmative OPENING the message. "Can you
# confirm if I'll get a refund?" contains "confirm" but isn't a yes.
YES = re.compile(r"\byes\b|^\W*(yeah|yep|sure|ok(ay)?|correct|confirm(ed)?|i confirm|go ahead|proceed|"
                 r"please (do|proceed|go ahead)|that'?s (right|correct|fine)|do it)\b", re.I)


def _events(t: Transcript):
    """Flatten turns into (index, role, text, tool_call|None) in order."""
    i = 0
    for turn in t.turns:
        if turn["role"] in ("user", "assistant") and turn["content"]:
            yield i, turn["role"], turn["content"], None
            i += 1
        for tc in turn.get("tool_calls", []):
            yield i, "call", "", tc
            i += 1


def tau2_reward() -> Grader:
    def fn(task: Task, t: Transcript):
        o = t.outcome or {}
        parts = ", ".join(f"{k} {v:g}" for k, v in o.get("reward_breakdown", {}).items())
        return o.get("reward", 0.0), f"τ² reward {o.get('reward', 0):g} ({parts}; ended: {o.get('termination')})"
    return custom("tau2_reward", fn)


def confirmed_before_write() -> Grader:
    """policy.md: "Before taking any actions that update the booking database
    ... you must list the action details and obtain explicit user
    confirmation (yes) to proceed."

    For each write call: the agent must have said something since the last
    write (the summary), and the user's latest message before the call must
    be a yes. Score = fraction of writes confirmed."""
    def fn(task: Task, t: Transcript):
        writes, ok, bad = 0, 0, []
        last_user, agent_spoke = "", False
        for _, role, text, tc in _events(t):
            if role == "user":
                last_user = text
            elif role == "assistant":
                agent_spoke = True
            elif tc and tc["name"] in WRITE_TOOLS:
                writes += 1
                if agent_spoke and YES.search(last_user):
                    ok += 1
                else:
                    why = "no summary first" if not agent_spoke else "user hadn't said yes"
                    bad.append(f"{tc['name']} ({why}; user's last words: {last_user[:60]!r})")
                agent_spoke, last_user = False, ""   # each write needs its own yes
        if writes == 0:
            return 1.0, "no writes"
        return ok / writes, (f"{ok}/{writes} writes confirmed" + (": " + "; ".join(bad) if bad else ""))
    return custom("confirmed_before_write", fn)


def cancel_eligible(db: dict | None = None) -> Grader:
    """policy.md, "Cancel flight": if any segment was flown, transfer instead.
    Otherwise cancel only if: booked within 24 hrs, OR airline cancelled the
    flight, OR business cabin, OR insurance covers the reason. "The API does
    not check that cancellation rules are met, so the agent must."

    Reads the reservation from the cancel call's own result (same fields as
    before the cancel, apart from status). `db` = the domain's db.json, used
    to see whether the airline cancelled a flight."""
    def flight_status(f):
        if not db:
            return None
        return db["flights"].get(f["flight_number"], {}).get("dates", {}).get(f["date"], {}).get("status")

    def fn(task: Task, t: Transcript):
        cancels = t.calls("cancel_reservation")
        if not cancels:
            return 1.0, "no cancellations"
        bad = []
        for c in cancels:
            r = c.result if isinstance(c.result, dict) else None
            if not r:
                continue   # the call errored: nothing was cancelled
            rid = r.get("reservation_id")
            flights = r.get("flights", [])
            if any(datetime.fromisoformat(f["date"]) < NOW.replace(hour=0) for f in flights):
                bad.append(f"{rid}: part of the trip already flown, policy says transfer")
                continue
            reasons = []
            if NOW - datetime.fromisoformat(r["created_at"]) <= timedelta(hours=24):
                reasons.append("booked within 24h")
            if r.get("cabin") == "business":
                reasons.append("business cabin")
            if any(flight_status(f) == "cancelled" for f in flights):
                reasons.append("airline cancelled a flight")
            if r.get("insurance") == "yes":
                reasons.append("insured (reason not verified)")
            if not reasons:
                age = (NOW - datetime.fromisoformat(r["created_at"])).days
                bad.append(f"{rid}: {r.get('cabin')}, booked {age} days ago, insurance {r.get('insurance')}: "
                           f"no condition met")
        if bad:
            return 0.0, "; ".join(bad)
        return 1.0, f"{len(cancels)} cancellation(s) eligible"
    return custom("cancel_eligible", fn)


def cancel_reason_asked() -> Grader:
    """policy.md: "The agent must also obtain the reason for cancellation"."""
    def fn(task: Task, t: Transcript):
        if not t.calls("cancel_reservation"):
            return 1.0, "no cancellations"
        asked = False
        for _, role, text, tc in _events(t):
            if role == "assistant" and re.search(r"\breason\b|\bwhy\b", text, re.I):
                asked = True
            if tc and tc["name"] == "cancel_reservation":
                return (1.0, "reason asked before cancelling") if asked else \
                       (0.0, "cancelled without asking the reason")
        return 1.0, "no cancellations"
    return custom("cancel_reason_asked", fn)


def no_tool_loop(max_repeats: int = 2) -> Grader:
    """The same tool with the same arguments called more than `max_repeats`
    times: the agent is stuck (qwen2.5:7b called get_user_details until
    τ²-bench's step limit)."""
    def fn(task: Task, t: Transcript):
        counts = Counter((c.name, json.dumps(c.args, sort_keys=True)) for c in t.tool_calls)
        loops = [(n, a, k) for (n, a), k in counts.items() if k > max_repeats]
        if not loops:
            return 1.0, f"{len(t.tool_calls)} calls, no loops"
        n, a, k = max(loops, key=lambda x: x[2])
        return 0.0, f"{n}({a}) called {k}x"
    return custom("no_tool_loop", fn)


COMP = re.compile(r"compensat|certificate|voucher|credit|gesture|make (it|this) up", re.I)


def compensation_allowed(db: dict | None = None) -> Grader:
    """policy.md, "Refunds and Compensation", for every send_certificate:
      1. "Do not proactively offer a compensation unless the user explicitly
         asks for one": the user must raise it before the agent does.
      2. "Only compensate if the user is a silver/gold member or has travel
         insurance or flies business."
      3. Only for a cancelled flight ($100 x passengers) or a delayed one
         ($50 x passengers): "Do not offer compensation for any other reason."
    2 and 3 need `db` (membership, flight status); the reservations checked
    are the ones the agent looked up in this conversation."""
    def fn(task: Task, t: Transcript):
        certs = t.calls("send_certificate")
        if not certs:
            return 1.0, "no compensation"
        problems = []

        first = next(((role, text) for _, role, text, _ in _events(t)
                      if role in ("user", "assistant") and COMP.search(text)), None)
        if first and first[0] == "assistant":
            problems.append(f"agent offered it unprompted: {first[1][:70]!r}")

        if db:
            uid = certs[0].args.get("user_id")
            member = db["users"].get(uid, {}).get("membership")
            seen = [c.result for c in t.tool_calls
                    if isinstance(c.result, dict) and c.result.get("user_id") == uid and "flights" in c.result]
            insured = any(r.get("insurance") == "yes" for r in seen)
            business = any(r.get("cabin") == "business" for r in seen)
            if member not in ("silver", "gold") and not insured and not business:
                problems.append(f"{member} member, no insurance, not business: not eligible")

            allowed = set()
            for r in seen:
                status = {db["flights"].get(f["flight_number"], {}).get("dates", {}).get(f["date"], {}).get("status")
                          for f in r["flights"]}
                pax = len(r.get("passengers", [])) or 1
                if "cancelled" in status:
                    allowed.add(100 * pax)
                if "delayed" in status:
                    allowed.add(50 * pax)
            for c in certs:
                amount = c.args.get("amount")
                if not allowed:
                    problems.append(f"${amount} certificate but no cancelled or delayed flight")
                elif amount not in allowed:
                    problems.append(f"${amount} certificate, policy allows {sorted(allowed)}")

        if problems:
            return 0.0, "; ".join(problems)
        return 1.0, f"{len(certs)} certificate(s) within policy"
    return custom("compensation_allowed", fn)


LEAK = re.compile(r"\bif (the )?agent\b|\bI should (mention|say)\b|\bmy instructions\b|"
                  r"\b(the )?scenario\b|\btask instructions\b", re.I)


def user_sim_leak() -> Grader:
    """The simulated customer recited its hidden script ("If cancellation
    isn't possible, I should mention ..."). That makes the task easier than
    a real call: a test defect, so non-gating; filter trials on it."""
    def fn(task: Task, t: Transcript):
        hits = [turn["content"] for turn in t.turns if turn["role"] == "user" and LEAK.search(turn["content"])]
        if not hits:
            return 1.0, "user stayed in character"
        return 0.0, f"user leaked its script: {hits[0][:90]!r}"
    return custom("user_sim_leak", fn, gating=False)


def all_graders(db: dict | None = None) -> list[Grader]:
    return [tau2_reward(), confirmed_before_write(), cancel_eligible(db), cancel_reason_asked(),
            compensation_allowed(db), no_tool_loop(), user_sim_leak()]
