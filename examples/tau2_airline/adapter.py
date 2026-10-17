"""Load τ²-bench results into agentproof, so its graders and reports run on them.

τ²-bench (Sierra Research, MIT licence, https://github.com/sierra-research/tau2-bench)
runs the agent against an LLM-simulated customer and grades the final database.
It already ran the agent, so the "agent" here is a replay: each call returns
the next saved trial for that task, turned into an agentproof Transcript.

    tasks, agent, k = load("path/to/results.json", graders)
    report = await run_suite(tasks, agent, trials=k)

Nothing here imports tau2: results.json is plain JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict

from agentproof import Grader, Task, Transcript
from agentproof.core import ToolCall


def to_transcript(sim: dict) -> Transcript:
    """One τ²-bench simulation -> Transcript.

    tool_calls: every agent tool call, with the tool message's result matched
    by call id. turns: every message, tool calls included, in order (the
    graders need the order to see what the user said before a write).
    outcome: τ²-bench's own verdict (reward, DB check, termination)."""
    results = {m["id"]: m for m in sim["messages"] if m["role"] == "tool"}
    calls, turns, prompt_tokens = [], [], []
    for m in sim["messages"]:
        turn = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "tool":
            turn["error"] = m.get("error", False)
        for tc in m.get("tool_calls") or []:
            res = results.get(tc["id"], {})
            content = res.get("content")
            try:
                content = json.loads(content)
            except (TypeError, ValueError):
                pass
            call = ToolCall(tc["name"], tc.get("arguments") or {}, content)
            calls.append(call)
            turn.setdefault("tool_calls", []).append(
                {"name": call.name, "args": call.args, "error": res.get("error", False)})
        if m["role"] == "assistant" and m.get("usage"):
            prompt_tokens.append(m["usage"].get("prompt_tokens", 0))
        turns.append(turn)

    ri = sim.get("reward_info") or {}
    error = None
    if sim.get("termination_reason") == "infrastructure_error":
        error = "agent produced an empty reply on every attempt (reasoning used the whole reply budget)"
    final = next((t["content"] for t in reversed(turns)
                  if t["role"] == "assistant" and t["content"]), "")
    return Transcript(
        final_output=final,
        tool_calls=calls,
        outcome={"reward": ri.get("reward", 0.0),
                 "db_match": (ri.get("db_check") or {}).get("db_match"),
                 "reward_breakdown": ri.get("reward_breakdown") or {},
                 "termination": sim.get("termination_reason")},
        turns=turns,
        usage={"seconds": sim.get("duration"),
               "max_prompt_tokens": max(prompt_tokens, default=0)},
        error=error,
    )


def _harness_failed(sim: dict) -> bool:
    """τ²-bench gave up on the trial (infrastructure_error) for a reason that
    isn't the agent's fault. An empty AGENT message ("AssistantMessage must
    have either content or tool_calls": a thinking model used its whole reply
    budget reasoning) is the agent failing, so it stays in, as a failed trial.
    An empty simulated-USER message, or any other crash, is left out."""
    if sim.get("termination_reason") != "infrastructure_error":
        return False
    return "AssistantMessage" not in str((sim.get("info") or {}).get("error", ""))


def infrastructure_errors(path: str) -> list[str]:
    """Task ids of trials left out because the harness, not the agent, failed."""
    return [str(s["task_id"]) for s in json.load(open(path))["simulations"] if _harness_failed(s)]


def load(path: str, graders: list[Grader]) -> tuple[list[Task], callable, int]:
    """Read a results.json -> (tasks, replay agent, trials per task).

    Only tasks with saved trials are returned. k is the smallest number of
    trials any task has, so pass^k compares like with like on a partial run.
    Trials the harness failed (see _harness_failed) are left out; count them
    with infrastructure_errors()."""
    data = json.load(open(path))
    latest: dict[tuple, dict] = {}       # a resumed run re-runs failed trials: keep the last
    for sim in data["simulations"]:
        latest[(str(sim["task_id"]), sim.get("trial", 0))] = sim
    by_task: dict[str, list[dict]] = defaultdict(list)
    for sim in latest.values():
        if _harness_failed(sim):
            continue
        by_task[str(sim["task_id"])].append(sim)
    for sims in by_task.values():
        sims.sort(key=lambda s: s.get("trial", 0))

    tasks = []
    for t in data["tasks"]:
        tid = str(t["id"])
        if tid not in by_task:
            continue
        instr = t["user_scenario"]["instructions"]
        tasks.append(Task(
            id=tid,
            input=instr["reason_for_call"],
            graders=graders,
            expected={"tau2_task": t},
            tags=["tau2", "airline"],
        ))
    k = min((len(by_task[t.id]) for t in tasks), default=0)
    queues = {t.id: iter(by_task[t.id][:k]) for t in tasks}

    async def replay(task: Task) -> Transcript:
        return to_transcript(next(queues[task.id]))

    return tasks, replay, k
