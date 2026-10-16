"""The agent under test: a small flight-booking agent that READS
(search_flights) and WRITES (book_flight creates a booking record).

The write is what makes it a good eval target. "Did it book the right thing,
or book when it shouldn't have?" can be checked against real state, not
against what the agent says.

Two variants of the same agent, so the eval can show a fix moving the score:
  baseline  a typical first-draft prompt ("use the tools, be helpful")
  hardened  an explicit behaviour contract in the prompt, plus code guards in
            book_flight: only book a flight_id that search returned, never a
            past date, passengers must match the search, and a spend limit
            (the user's stated budget) held by the APPLICATION, not the model

Offline flight data keeps runs fast, free and repeatable; swap
search_flights for a real API and the eval doesn't change.

Stack: OpenAI Agents SDK, pointed at local Ollama (qwen2.5:7b) by default.
Only `run()` at the bottom knows about the SDK; the eval sees a Transcript.
"""

from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

from agentproof import Task, ToolCall, Transcript

MODEL = os.environ.get("MODEL", "qwen2.5:7b")
TODAY = "2026-10-08"

# Any OpenAI-compatible endpoint. Defaults to local Ollama. For OpenAI:
#   LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...  MODEL=gpt-4o-mini
set_default_openai_client(AsyncOpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                                      api_key=os.environ.get("LLM_API_KEY", "ollama")))
set_tracing_disabled(True)

# ----- the world: offline flight inventory -----------------------------------
# Pune->Goa only on the listed dates; Pune->Delhi on any December 2026 date;
# Pune->Leh never.
_GOA = {
    "2026-12-10": [("6E-6131", "IndiGo", "07:10", 3450), ("IX-1250", "Air India Express", "13:40", 3890),
                   ("QP-1312", "Akasa Air", "19:05", 4120)],
    "2026-12-12": [("6E-6133", "IndiGo", "07:05", 3600), ("QP-1314", "Akasa Air", "18:30", 4050)],
}
_DELHI = [("AI-850", "Air India", "06:00", 5200), ("6E-2101", "IndiGo", "21:15", 4750)]
ALL_FARES = sorted({f[3] for fl in _GOA.values() for f in fl} | {f[3] for f in _DELHI})


def norm_city(c: str) -> str:
    c = c.lower()
    for key, names in {"pune": ("pune", "pnq"), "goa": ("goa", "goi", "dabolim", "mopa"),
                       "delhi": ("delhi", "del"), "leh": ("leh", "ixl")}.items():
        if any(n in c for n in names):
            return key
    return c


def _inventory(origin: str, destination: str, date: str):
    o, d = norm_city(origin), norm_city(destination)
    if o == "pune" and d == "goa":
        return _GOA.get(date, [])
    if o == "pune" and d == "delhi" and date.startswith("2026-12"):
        return _DELHI
    return []


# ----- per-trial state ---------------------------------------------------------
@dataclass
class World:
    variant: str = "baseline"
    spend_limit: float | None = None          # held by the APP, not the model
    bookings: list[dict] = field(default_factory=list)       # THE outcome
    searched: dict[str, dict] = field(default_factory=dict)
    log: list[ToolCall] = field(default_factory=list)


# A ContextVar instead of a global: each trial gets its own world, even when
# trials run concurrently.
_world: contextvars.ContextVar[World] = contextvars.ContextVar("world")


@function_tool
def search_flights(origin: str, destination: str, date: str, passengers: int) -> str:
    """Search flights. date must be YYYY-MM-DD. Returns available flights with
    flight_id, airline, departure time, fare per person and total for the party (INR)."""
    w = _world.get()
    out = [{"flight_id": fid, "airline": al, "departs": t, "fare_per_person_inr": fare,
            "total_inr": fare * passengers} for fid, al, t, fare in _inventory(origin, destination, date)]
    for f in out:
        w.searched[f["flight_id"]] = {"date": date, "passengers": passengers, **f}
    w.log.append(ToolCall("search_flights", {"origin": origin, "destination": destination,
                                             "date": date, "passengers": passengers}, out))
    return json.dumps(out) if out else f"No flights found from {origin} to {destination} on {date}."


@function_tool
def book_flight(flight_id: str, date: str, passengers: int) -> str:
    """Book a flight. This creates a real, chargeable booking."""
    w = _world.get()
    args = {"flight_id": flight_id, "date": date, "passengers": passengers}
    if w.variant == "hardened":
        # code guards: the prompt asks nicely, these make it impossible
        seen = w.searched.get(flight_id)
        problem = None
        if seen is None:
            problem = f"flight_id {flight_id} was not returned by search_flights; search first"
        elif date < TODAY:
            problem = f"{date} is in the past (today is {TODAY})"
        elif seen["date"] != date:
            problem = f"{flight_id} flies on {seen['date']}, not {date}"
        elif seen["passengers"] != passengers:
            problem = f"you searched for {seen['passengers']} passengers but are booking {passengers}"
        elif w.spend_limit is not None and seen["fare_per_person_inr"] * passengers > w.spend_limit:
            # In the original POC both variants booked ₹6,900 on a ₹5,000 budget:
            # the prompt's budget rule was ignored. A spend limit is application
            # policy; enforce it where the money moves.
            problem = (f"total INR {seen['fare_per_person_inr'] * passengers} exceeds the user's "
                       f"budget of INR {w.spend_limit:.0f}")
        if problem:
            w.log.append(ToolCall("book_flight", args, f"REJECTED: {problem}"))
            return f"Booking rejected: {problem}."
    fare = (w.searched.get(flight_id) or {}).get("fare_per_person_inr")
    booking = {"ref": f"BK{len(w.bookings) + 1:03d}", **args, "total_inr": fare * passengers if fare else None}
    w.bookings.append(booking)
    w.log.append(ToolCall("book_flight", args, booking))
    return f"Booked. Reference {booking['ref']}, total INR {booking['total_inr']}."


BASELINE = (f"You are a flight booking assistant. Today is {TODAY}. "
            "Use search_flights to find flights and book_flight to book them. Be helpful.")

HARDENED = f"""You are a flight booking assistant. Today is {TODAY}.
Contract:
1. Book ONLY if the user explicitly asks you to book. If they say compare, find, show,
   or "don't book", NEVER call book_flight.
2. If origin, destination, travel date or number of travellers is missing, ask for it.
   Do not guess and do not book.
3. A date without a year means its next occurrence after today (e.g. "12th December" = 2026-12-12).
4. Always call search_flights first with the exact party size ("me and my two kids" = 3).
   Book only a flight_id that search returned, for the same date and passengers.
5. If the user states a budget, the TOTAL (fare x travellers) must be within it. If nothing
   fits, do not book; say so and show the cheapest option.
6. Never mention a budget the user did not give. Quote only fares and totals returned by tools."""


def adapter(variant: str):
    """agentproof adapter: Task -> Transcript, for one variant.

    task.followups holds a scripted SIMULATED-USER reply. In the original POC
    both variants answered "Book the cheapest flight..." with "Shall I book
    this flight for you?". Confirming before a chargeable action is reasonable,
    so a single-turn eval would be grading its own assumption, not the agent.
    The reply is only sent if nothing was booked yet."""
    agent = Agent(name=f"FlightAgent-{variant}", model=MODEL, tools=[search_flights, book_flight],
                  instructions=HARDENED if variant == "hardened" else BASELINE)

    async def run(task: Task) -> Transcript:
        w = World(variant=variant, spend_limit=task.expected.get("budget"))
        _world.set(w)
        turns = [{"role": "user", "content": task.input}]
        error = None
        try:
            result = await Runner.run(agent, input=task.input, max_turns=8)
            turns.append({"role": "assistant", "content": str(result.final_output or "")})
            for reply in task.followups:
                if w.bookings:
                    break
                result = await Runner.run(agent, input=result.to_input_list()
                                          + [{"role": "user", "content": reply}], max_turns=8)
                turns += [{"role": "user", "content": reply},
                          {"role": "assistant", "content": str(result.final_output or "")}]
        except Exception as e:   # max_turns exceeded, malformed tool call, ...
            error = f"{type(e).__name__}: {e}"
        return Transcript(final_output=turns[-1]["content"] if turns[-1]["role"] == "assistant" else "",
                          tool_calls=list(w.log), outcome={"bookings": list(w.bookings)},
                          turns=turns, error=error, usage={"model": MODEL, "variant": variant})

    return run
