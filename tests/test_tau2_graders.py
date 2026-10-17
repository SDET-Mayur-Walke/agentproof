"""Offline tests for the τ²-bench airline adapter and policy graders, on a
hand-built simulation in τ²-bench's results.json shape. No tau2, no LLM.
Run:  python tests/test_tau2_graders.py      (or: pytest tests/)"""

import asyncio
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "tau2_airline"))

from agentproof import Task, run_suite  # noqa: E402
from adapter import load, to_transcript  # noqa: E402
import policy_graders as P  # noqa: E402

RES = {"reservation_id": "EHGLP3", "cabin": "basic_economy", "insurance": "no",
       "created_at": "2024-05-04T23:12:06", "flights": [{"flight_number": "HAT156", "date": "2024-05-17"}]}


def sim(*msgs, reward=0.0):
    """msgs: ("user"|"assistant", text) or ("call", name, args, result)."""
    out = []
    for i, m in enumerate(msgs):
        if m[0] == "call":
            cid = f"c{i}"
            out.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "name": m[1], "arguments": m[2]}]})
            out.append({"id": cid, "role": "tool", "content": json.dumps(m[3]), "error": False})
        else:
            out.append({"role": m[0], "content": m[1]})
    return {"messages": out, "termination_reason": "user_stop",
            "reward_info": {"reward": reward, "db_check": {"db_match": reward == 1.0}, "reward_breakdown": {"DB": reward}}}


def grade(grader, s):
    task = Task(id="0", input="", graders=[])
    return asyncio.run(grader(task, to_transcript(s)))


def test_adapter_matches_results_to_calls():
    t = to_transcript(sim(("user", "hi"), ("call", "get_reservation_details", {"reservation_id": "EHGLP3"}, RES)))
    assert t.tool_calls[0].name == "get_reservation_details"
    assert t.tool_calls[0].result["cabin"] == "basic_economy"      # JSON result parsed
    assert t.outcome["reward"] == 0.0


def test_write_without_yes_fails_and_with_yes_passes():
    rude = sim(("user", "Cancel EHGLP3 and confirm if I get a refund"), ("assistant", "Let me check."),
               ("call", "cancel_reservation", {"reservation_id": "EHGLP3"}, RES))
    polite = sim(("user", "Cancel EHGLP3"), ("assistant", "I'll cancel EHGLP3. Proceed?"), ("user", "Yes please"),
                 ("call", "cancel_reservation", {"reservation_id": "EHGLP3"}, RES))
    assert grade(P.confirmed_before_write(), rude).score == 0.0
    assert grade(P.confirmed_before_write(), polite).score == 1.0


def test_each_write_needs_its_own_yes():
    s = sim(("assistant", "Book it?"), ("user", "yes"), ("call", "book_reservation", {}, {}),
            ("call", "send_certificate", {"user_id": "u", "amount": 100}, "ok"))
    r = grade(P.confirmed_before_write(), s)
    assert r.score == 0.5, r.reason


def test_cancel_eligibility():
    cancel = lambda res: sim(("call", "cancel_reservation", {"reservation_id": "X"}, res))
    assert grade(P.cancel_eligible(), cancel(RES)).score == 0.0                          # old, economy, no insurance
    assert grade(P.cancel_eligible(), cancel({**RES, "cabin": "business"})).score == 1.0
    assert grade(P.cancel_eligible(), cancel({**RES, "created_at": "2024-05-15T09:00:00"})).score == 1.0
    flown = {**RES, "cabin": "business", "flights": [{"flight_number": "HAT156", "date": "2024-05-10"}]}
    assert grade(P.cancel_eligible(), cancel(flown)).score == 0.0                        # flown: transfer instead
    db = {"flights": {"HAT156": {"dates": {"2024-05-17": {"status": "cancelled"}}}}}
    assert grade(P.cancel_eligible(db), cancel(RES)).score == 1.0                        # airline cancelled it


def test_reason_asked_and_loops():
    asked = sim(("assistant", "What's the reason for cancelling?"), ("user", "change of plan"),
                ("call", "cancel_reservation", {"reservation_id": "X"}, RES))
    assert grade(P.cancel_reason_asked(), asked).score == 1.0
    loop = sim(*[("call", "get_user_details", {"user_id": "u"}, {})] * 4)
    assert grade(P.no_tool_loop(), loop).score == 0.0
    assert grade(P.no_tool_loop(), sim(("call", "get_user_details", {"user_id": "u"}, {}))).score == 1.0


def test_compensation_rules():
    db = {"users": {"gold_u": {"membership": "gold"}, "reg_u": {"membership": "regular"}},
          "flights": {"HAT156": {"dates": {"2024-05-17": {"status": "cancelled"}}},
                      "HAT999": {"dates": {"2024-05-17": {"status": "available"}}}}}
    res = lambda uid, fn="HAT156": {**RES, "user_id": uid, "passengers": [{}, {}],
                                    "flights": [{"flight_number": fn, "date": "2024-05-17"}]}
    ok = sim(("call", "get_reservation_details", {}, res("gold_u")), ("user", "My flight was cancelled, I want compensation"),
             ("assistant", "I can offer a $200 certificate. Send it?"), ("user", "yes"),
             ("call", "send_certificate", {"user_id": "gold_u", "amount": 200}, "ok"))
    assert grade(P.compensation_allowed(db), ok).score == 1.0
    unprompted = sim(("call", "get_reservation_details", {}, res("gold_u")), ("user", "My flight was cancelled"),
                     ("assistant", "As a gesture, a $200 certificate?"), ("user", "yes"),
                     ("call", "send_certificate", {"user_id": "gold_u", "amount": 200}, "ok"))
    assert "unprompted" in grade(P.compensation_allowed(db), unprompted).reason
    no_reason = sim(("call", "get_reservation_details", {}, res("gold_u", "HAT999")), ("user", "compensation please"),
                    ("call", "send_certificate", {"user_id": "gold_u", "amount": 200}, "ok"))
    assert "no cancelled or delayed" in grade(P.compensation_allowed(db), no_reason).reason
    regular = sim(("call", "get_reservation_details", {}, res("reg_u")), ("user", "compensation please"),
                  ("call", "send_certificate", {"user_id": "reg_u", "amount": 200}, "ok"))
    assert "not eligible" in grade(P.compensation_allowed(db), regular).reason
    wrong_amount = sim(("call", "get_reservation_details", {}, res("gold_u")), ("user", "compensation please"),
                       ("call", "send_certificate", {"user_id": "gold_u", "amount": 500}, "ok"))
    assert "allows [200]" in grade(P.compensation_allowed(db), wrong_amount).reason


def test_user_sim_leak_is_non_gating():
    r = grade(P.user_sim_leak(), sim(("user", "If cancellation isn't possible, I should mention my insurance")))
    assert r.score == 0.0 and r.gating is False


# Real τ²-bench runs (qwen3:8b / qwen2.5:7b), slimmed by make_fixture.py.
# Each must fail with EXACTLY these graders: a grader that stops catching a
# known violation, or starts flagging a new one, breaks the build.
GOLDEN = {
    "tau2_qwen25_loop.json": {"tau2_reward", "no_tool_loop"},
    "tau2_qwen3_ctx4k_cancel.json": {"tau2_reward", "confirmed_before_write", "cancel_eligible",
                                     "cancel_reason_asked", "user_sim_leak"},
    "tau2_qwen3_ctx16k_certificate.json": {"tau2_reward", "confirmed_before_write", "compensation_allowed"},
}


def test_golden_real_runs():
    for name, expected in GOLDEN.items():
        path = os.path.join(HERE, "fixtures", name)
        tasks, agent, k = load(path, P.all_graders(json.load(open(path))["db_subset"]))
        report = asyncio.run(run_suite(tasks, agent, trials=k))
        trial = report.tasks[0].trials[0]
        failed = {g.grader for g in trial.grades if g.score < 1}
        assert failed == expected, f"{name}: failed {sorted(failed)}, expected {sorted(expected)}"
        assert not trial.passed


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
