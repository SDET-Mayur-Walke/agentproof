"""Slim a τ²-bench results.json down to the fields the adapter reads, for use
as a test fixture (raw LLM responses, audio fields and timestamps dropped).

    python examples/tau2_airline/make_fixture.py <results.json> <out.json> [--sim N] [--db db.json]

The data is τ²-bench's synthetic airline data (Sierra Research, MIT).
"""

from __future__ import annotations

import argparse
import json

SIM_KEYS = ["task_id", "trial", "duration", "termination_reason", "reward_info"]
MSG_KEYS = ["role", "content", "tool_calls", "id", "error", "usage"]


def db_subset(db: dict, sims: list[dict]) -> dict:
    """The users and flights these transcripts touch, so DB-backed graders
    (membership, flight status) work without the 7 MB db.json."""
    text = json.dumps([s["messages"] for s in sims])
    return {"users": {k: {"membership": v["membership"]} for k, v in db["users"].items() if k in text},
            "flights": {k: v for k, v in db["flights"].items() if k in text}}


def slim(data: dict, sims: list[int] | None, db: dict | None = None) -> dict:
    keep = [data["simulations"][i] for i in sims] if sims else data["simulations"]
    task_ids = {str(s["task_id"]) for s in keep}
    return {
        "source": "tau2-bench airline (Sierra Research, MIT): github.com/sierra-research/tau2-bench",
        "info": {k: data["info"].get(k) for k in ("git_commit", "num_trials", "max_steps")} |
                {"agent_llm": data["info"]["agent_info"]["llm"], "agent_llm_args": data["info"]["agent_info"]["llm_args"]},
        "tasks": [t for t in data["tasks"] if str(t["id"]) in task_ids],
        "simulations": [{**{k: s.get(k) for k in SIM_KEYS},
                         "messages": [{k: m[k] for k in MSG_KEYS if m.get(k) not in (None, "")}
                                      for m in s["messages"]]}
                        for s in keep],
        **({"db_subset": db_subset(db, keep)} if db else {}),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--sim", type=int, nargs="*")
    ap.add_argument("--db", help="the domain's db.json, to embed the slice the graders need")
    a = ap.parse_args()
    json.dump(slim(json.load(open(a.src)), a.sim, json.load(open(a.db)) if a.db else None), open(a.out, "w"), indent=1, ensure_ascii=False)
    print(a.out)
