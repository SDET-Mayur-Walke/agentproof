"""Offline tests for the harness and graders: a scripted fake agent, no LLM.
Run:  python tests/test_core.py      (or: pytest tests/)"""

import asyncio
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentproof import Task, Tier, ToolCall, Transcript, run_suite  # noqa: E402
from agentproof import graders as G  # noqa: E402
from agentproof.report import console, gate, to_json  # noqa: E402

FLIGHTS = [{"id": "AI-101", "price": 5400}]


def booking_transcript(book: bool = True, price_in_reply: int = 5400, order_ok: bool = True) -> Transcript:
    search = ToolCall("search_flights", {"origin": "Pune", "dest": "Delhi"}, FLIGHTS)
    booking = ToolCall("book_flight", {"flight_id": "AI-101", "passengers": 2}, {"ok": True})
    calls = ([search, booking] if order_ok else [booking, search]) if book else [search]
    return Transcript(final_output=f"Booked AI-101 for ₹{price_in_reply}." if book else "No booking made.",
                      tool_calls=calls, outcome={"bookings": ["AI-101"] if book else []})


def task(**kw) -> Task:
    base = dict(id="book", input="Book Pune to Delhi for 2 people",
                graders=[G.tool_order("search_flights", "book_flight"),
                         G.args_match("book_flight", {"flight_id": "AI-101", "passengers": 2}),
                         G.outcome_equals("bookings", ["AI-101"]),
                         G.grounded_numbers()])
    base.update(kw)
    return Task(**base)


def run(tasks, agent, trials=3):
    return asyncio.run(run_suite(tasks, agent, trials=trials))


def test_all_gating_graders_pass():
    async def agent(t):
        return booking_transcript()
    r = run([task()], agent)
    assert r.tasks[0].pass_rate == 1.0 and r.tasks[0].pass_hat_k


def test_invented_number_fails_grounding():
    async def agent(t):
        return booking_transcript(price_in_reply=4999)
    r = run([task()], agent, trials=1)
    g = {x.grader: x for x in r.tasks[0].trials[0].grades}
    assert g["grounded_numbers"].score == 0.0 and "4999" in g["grounded_numbers"].reason
    assert not r.tasks[0].trials[0].passed


def test_order_and_partial_args():
    async def agent(t):
        tr = booking_transcript(order_ok=False)
        tr.tool_calls[0].args["passengers"] = 3
        return tr
    r = run([task()], agent, trials=1)
    g = {x.grader: x.score for x in r.tasks[0].trials[0].grades}
    assert g["order:search_flights<book_flight"] == 0.0
    assert g["args:book_flight"] == 0.5          # partial credit: 1 of 2 args right


def test_flaky_agent_pass_rate_and_pass_hat_k():
    flips = itertools.cycle([True, False, True])
    async def agent(t):
        return booking_transcript(book=next(flips))
    r = run([task()], agent).tasks[0]
    assert abs(r.pass_rate - 2 / 3) < 1e-9 and r.pass_at_k and not r.pass_hat_k


def test_non_gating_judge_is_reported_not_deciding():
    async def lenient_judge(rubric, task, t):
        return 0.9, "looks great"
    judged = task(graders=task().graders + [G.llm_judge("judge", "helpful?", lenient_judge)])
    async def wrong_agent(t):
        return booking_transcript(book=False)           # fails the outcome check
    r = run([judged], wrong_agent, trials=2)
    assert r.pass_rate == 0.0
    assert r.looks_right_is_wrong("judge") == (2, 2)   # judge liked both failed trials


def test_agent_crash_is_a_failed_trial_not_a_crash():
    async def agent(t):
        raise TimeoutError("model timed out")
    r = run([task()], agent, trials=2)
    assert r.pass_rate == 0.0 and "TimeoutError" in r.tasks[0].trials[0].transcript.error


def test_broken_grader_scores_zero_with_reason():
    def boom(task, t):
        raise KeyError("expected_field")
    async def agent(t):
        return booking_transcript()
    r = run([task(graders=[G.custom("boom", boom)])], agent, trials=1)
    g = r.tasks[0].trials[0].grades[0]
    assert g.score == 0.0 and "grader error" in g.reason


def test_gate_blocks_regression_drop_only():
    flips = itertools.cycle([True, False])
    async def agent(t):
        return booking_transcript(book=next(flips)) if t.id == "reg" else booking_transcript(book=False)
    tasks = [task(id="reg", tier=Tier.REGRESSION), task(id="cap", tier=Tier.CAPABILITY)]
    r = run(tasks, agent, trials=2)
    reasons = gate(r)
    assert len(reasons) == 1 and "reg" in reasons[0]
    assert to_json(r)["tasks"][0]["id"] == "reg" and "pass" in console(r)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
