# agentproof on τ²-bench (airline)

[τ²-bench](https://github.com/sierra-research/tau2-bench) by Sierra Research
(MIT licence) is an industry benchmark for customer-service agents: the agent
follows an airline policy, talks to an LLM-simulated customer, and calls tools
that read and write a mock airline database. Its reward checks the **final
database**.

This example loads τ²-bench's saved runs into agentproof and adds graders that
check the **process**, rule by rule from the airline `policy.md`:

| Grader | Policy rule | Gating |
|---|---|---|
| `tau2_reward` | τ²-bench's own verdict (DB + communication) | yes |
| `confirmed_before_write` | "obtain explicit user confirmation (yes)" before any database write | yes |
| `cancel_eligible` | cancel only if booked < 24 h ago, airline cancelled, business cabin, or insured; never if part of the trip was flown | yes |
| `cancel_reason_asked` | "obtain the reason for cancellation" | yes |
| `compensation_allowed` | certificates only if the user asked, is silver/gold, insured or business, and a flight was cancelled ($100 × passengers) or delayed ($50 × passengers) | yes |
| `no_tool_loop` | same tool + same arguments 3+ times = stuck | yes |
| `no_reply_loop` | same agent reply 3+ times = the conversation is going in circles | yes |
| `membership_grounded` | any "gold/silver/regular member" the agent says matches the account it looked up (membership sets bag allowance and compensation) | yes |
| `user_sim_leak` | the *simulated customer* read out its hidden script: a defect in the test, not the agent | no |

So a failed trial says **why** it failed, and a trial where the database came
out right by luck ("right outcome, wrong process") gets caught too.

## Run it

```bash
# 1. τ²-bench, with a free local model (qwen3:8b on Ollama)
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
uv venv && uv pip install -e .
A='{"temperature": 0.0, "api_base": "http://localhost:11434", "num_ctx": 16384, "max_tokens": 4096}'
U='{"temperature": 0.0, "api_base": "http://localhost:11434", "num_ctx": 16384, "max_tokens": 1024, "think": false}'
tau2 run --domain airline --agent-llm ollama_chat/qwen3:8b --user-llm ollama_chat/qwen3:8b \
  --agent-llm-args "$A" --user-llm-args "$U" --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --num-trials 2 --max-steps 60 --max-retries 0 --save-to airline_qwen3

# 2. grade it with agentproof (no extra dependencies)
cd ../agentproof
PYTHONPATH=src python examples/tau2_airline/eval.py \
  ../tau2-bench/data/simulations/airline_qwen3/results.json
```

## Tests in CI ($0, no model)

Running τ²-bench on GitHub's free CPU runner would take hours per trial, so CI
**replays** real saved runs instead. `tests/fixtures/tau2_*.json` are three real
qwen runs (a 93-call loop, an ineligible cancellation, an unprompted $200
certificate), slimmed by `make_fixture.py` with the slice of `db.json` they
need. `tests/test_tau2_graders.py` asserts each one fails with exactly the
expected graders, so a grader that stops catching a known violation breaks the
build. The tests run in under a second.

## Results

qwen3:8b on airline tasks 0–9 × 2 trials: **6/20 trials pass, pass^2 = 3/10
tasks, and the policy graders name the broken rule in 14 of 14 failures**
where τ²-bench reports "reward 0". Per task, and the judge calibration:
[FINDINGS.md](FINDINGS.md).

## Gotcha: set `num_ctx` (and two more)

Ollama's default context window is **4,096 tokens**. The airline policy plus 14
tool definitions is about **4,700 tokens** before the conversation starts, and
Ollama silently drops the start of an over-long prompt, so the agent loses part
of the policy. Our first runs logged prompts of 3,954 and 4,040 tokens, pinned at
the limit, on the exact step where the agent wrongly cancelled a booking.
Pass `"num_ctx": 16384` in the LLM args. `eval.py` prints each trial's largest
prompt so you can spot this. Also cap the agent's reply (`"max_tokens"`; a
thinking model can reason until the 600 s timeout) and turn thinking off for the
simulated customer (`"think": false`; otherwise it can send empty messages).
Details in [FINDINGS.md](FINDINGS.md), section 3.

## Credit

Tasks, policy, database, user simulator and reward: τ²-bench, Sierra Research,
[github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
(MIT). The data is synthetic. The graders and adapter here are agentproof's.
