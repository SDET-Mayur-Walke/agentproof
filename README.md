# agentproof

**Evaluate AI agents by what they *do*, not only what they say.**

Most LLM evals score the final answer. An agent that books flights, issues
refunds or writes to a database needs more: did it call the right tools, in
the right order, with the user's facts? Did the world end up in the right
state? Does every number in its reply trace to something it actually saw?
And does it do that *every* time, not just once?

> Status: **v0.1, work in progress.** The core harness and graders are done
> and tested; the examples, judge calibration, CI gate and tracing are next
> (see the roadmap).

## The idea in one table

| Concept | Meaning |
|---|---|
| **Task** | A user request plus what "correct" means, checked by several graders |
| **Trial** | One run. Every task runs N times, because agents are non-deterministic |
| **Transcript** | Tool calls, arguments, results, final state: what the agent *did* |
| **Graders** | Outcome, trajectory, arguments, grounding (code, deterministic) + LLM judge |
| **Pass rule** | A trial passes only if **every gating grader scores 1.0** |
| **pass^k** | Passed *all* k trials: the bar before an agent acts unsupervised |

LLM judges are **reported but non-gating** until calibrated. The report
shows how often the judge rated a *failed* trial as good ("looks right, is
wrong"), which is how you find out whether it can be trusted.

## Quick look

```python
from agentproof import Task, Tier, run_suite
from agentproof import graders as G

task = Task(
    id="book_cheapest_for_two",
    input="Book the cheapest Pune to Delhi flight on 12 Nov 2026 for 2 people",
    tier=Tier.REGRESSION,
    graders=[
        G.tool_order("search_flights", "book_flight"),       # searched first
        G.args_match("book_flight", {"passengers": 2}),       # used the user's facts
        G.outcome_equals("bookings", ["AI-101"]),             # right end state
        G.grounded_numbers(),                                 # no invented prices
    ],
)
report = await run_suite([task], my_agent_adapter, trials=5)
```

An adapter is one async function `Task -> Transcript`, so the harness works
with any framework.

## Run the tests

```bash
python tests/test_core.py      # no dependencies, no LLM
```

## Roadmap

- [x] Core: tasks, trials, transcripts, gating vs non-gating graders, pass@k / pass^k
- [x] Graders: tool order, args, outcome, grounded numbers, text, LLM judge
- [x] Reports: console dot grid, JSON, Markdown, regression gate
- [ ] Example: flight-booking agent (baseline vs hardened), runs free on Ollama
- [ ] Judge calibration: agreement of an LLM judge with labelled trials
- [ ] Adapters: OpenAI Agents SDK, LangGraph, CrewAI, Google ADK
- [ ] CLI + GitHub Action: block a PR when a regression task drops
- [ ] Langfuse tracing: every trial linked to its trace
- [ ] Safety pack: prompt-injection and tool-misuse tasks

## Background

The vocabulary follows Anthropic's
[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
