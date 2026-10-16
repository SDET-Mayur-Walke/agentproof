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


def test_compare_table_and_judge_split():
    async def judge_fn(rubric, task, t):
        return (0.9, "") if t.outcome["bookings"] else (0.6, "")
    graders = task().graders + [G.llm_judge("judge", "?", judge_fn)]
    async def good(t):
        return booking_transcript()
    async def bad(t):
        return booking_transcript(book=False)
    reports = {"a": run([task(graders=graders)], good, trials=2), "b": run([task(graders=graders)], bad, trials=2)}
    assert reports["a"].judge_split("judge") == (0.9, None)
    assert reports["b"].judge_split("judge") == (None, 0.6)
    from agentproof.report import compare
    table = compare(reports, judge="judge")
    assert "●● 100%" in table and "○○ 0%" in table and "0.90 / –" in table


def test_calibration_metrics_on_a_scripted_judge():
    from agentproof.calibration import LabelledItem, calibrate, mutate
    t0 = task()
    good = [LabelledItem(t0, booking_transcript(), True) for _ in range(4)]

    def invent_price(task, t):
        t.final_output = t.final_output.replace("5400", "4999")
        return t
    bad = [mutate(g, "invented_price", invent_price) for g in good[:2]]
    bad.append(mutate(good[0], "no_op", lambda task, t: None))      # mutation doesn't apply
    bad = [b for b in bad if b]
    assert len(bad) == 2 and all(not b.label and "4999" in b.transcript.final_output for b in bad)
    assert "5400" in good[0].transcript.final_output                # original untouched

    # judge: likes everything except the FIRST invented-price item
    seen = {"n": 0}
    async def judge_fn(rubric, task, t):
        if "4999" in t.final_output and seen["n"] < 2:
            seen["n"] += 1
            return 0.1, ""
        return 0.9, ""
    from agentproof.graders import llm_judge
    rep = asyncio.run(calibrate(good + bad, llm_judge("j", "?", judge_fn), repeats=2))
    assert rep.n_pass == 4 and rep.n_fail == 2
    assert rep.fail_recall == 0.5 and rep.false_alarm_rate == 0.0
    assert rep.balanced_accuracy == 0.75
    assert rep.consistency == 1.0
    assert rep.by_source()["invented_price"] == (1, 2)
    ok, why = rep.verdict()
    assert not ok and any("recall" in w for w in why)
    assert 0.5 < rep.auc < 1.0


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
