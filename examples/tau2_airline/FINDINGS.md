# Findings: qwen3:8b on τ²-bench airline

Free local model (qwen3:8b on Ollama) as both the agent and the simulated
customer, temperature 0. Graded with `eval.py`.

## 1. Same score, different agent: the context-window bug

Task 0: Emma Kim wants to cancel a basic-economy booking made 10 days ago with
no insurance. Correct behaviour: refuse.

| | Ollama default context (4,096) | `num_ctx` 16,384 |
|---|---|---|
| Largest prompt | 4,040 tokens (pinned at the limit) | 4,766 tokens |
| **τ²-bench reward** | **0** | **0** |
| Asked the reason for cancelling | ✗ | ✓ |
| Cancelled an ineligible booking | ✗ cancelled it, refunded $208 | ✓ refused |
| Write without a "yes" | ✗ the cancellation | ✗ a certificate |
| Compensation rules | – | ✗ offered $200 unprompted, no cancelled or delayed flight |
| Simulated customer stayed in character | ✗ read out its hidden script | ✓ |

The policy plus 14 tool definitions is about 4,700 tokens, more than Ollama's
default window, and Ollama drops the start of an over-long prompt. With the
policy truncated, the agent cancelled a booking it must not cancel. With the
full policy it handled the cancellation correctly and failed somewhere else:
it gave away compensation.

**The benchmark score is 0 both times.** Read alone, "reward 0 → 0" says
the fix did nothing. The policy graders show it fixed one failure and exposed
another. That's the reason to grade the process and not only the outcome.

The simulated customer runs on the same model and had the same truncated
prompt, which is likely why it leaked its script at 4k and stayed in character
at 16k: the test itself was broken, not only the agent.

## 2. Full run: airline tasks 0–9 × 2 trials

Agent and simulated customer both qwen3:8b on Ollama, temperature 0,
`num_ctx` 16384; agent replies capped at 4,096 tokens; the customer runs with
thinking off (see section 3). 20 trials, ~3 hours on an M5 MacBook, $0.
Report: `results/qwen3_airline_10x2.json`.

| Task | Customer wants | Both trials | Why (agentproof) |
|---|---|---|---|
| 0 | cancel a non-cancellable booking | ✗ ✗ | cancelled it anyway; no "yes" first |
| 1 | cancel a reservation (correct answer: don't) | ✓ ✓ | looked up every reservation, cancelled nothing |
| 2 | book a flight, unsure of the date | ✗ ✗ | asked "I need the specific date" 26 times, until the step limit |
| 3 | how many free bags | ✗ ✗ | called a silver member "gold" without looking her up: 6 bags, not 4 |
| 4 | compensation for a made-up cancelled flight | ✓ ✓ | checked the flight status, didn't pay |
| 5 | complain about a delayed flight | ✓ ✓ | |
| 6 | insurance added after booking (not allowed) | ✗ ✗ | refused, then gave a $100 certificate nobody asked for |
| 7 | cancel two reservations | ✗ ✗ | reasoned through its whole 4,096-token reply budget, never answered |
| 8 | book a one-way ORD → PHL flight on May 26 | ✗ ✗ | same: no reply |
| 9 | cancel two reservations, change a third | ✗ ✗ | 3 writes with no "yes", 2 ineligible cancellations, no reason asked, "gold member" ×4 without a lookup |

| | |
|---|---|
| Trials passed | **6 / 20 (30%)** |
| pass@2 (passed at least once) | 3 / 10 tasks |
| pass^2 (passed every time) | 3 / 10 tasks |
| τ²-bench failures agentproof explains | **14 / 14** |
| Right outcome, wrong process | 0 |
| Simulated customer leaked its script | 0 / 20 |

**At temperature 0, pass^k ≈ pass@1.** Every task gave the same verdict in
both trials, often word for word, so a third trial was stopped as adding
nothing. pass^k measures reliability when behaviour *varies* between runs:
temperature > 0, or varied customers. That's the next experiment.

What τ²-bench alone reports for these 14 failures is "reward 0" (plus DB /
COMMUNICATE). The policy graders name the broken rule in every one:

| Failure | Trials |
|---|---|
| write without an explicit "yes" | 4 (tasks 0, 9) |
| ineligible cancellation | 4 (tasks 0, 9) |
| agent produced no reply (over-thinking) | 4 (tasks 7, 8) |
| invented membership level | 4 (tasks 3, 9) |
| unprompted compensation | 2 (task 6) |
| reply loop | 2 (task 2) |
| reason for cancelling not asked | 2 (task 9) |

## 3. Running τ²-bench on a local model: three settings that matter

| Problem | Symptom | Fix |
|---|---|---|
| Ollama's default 4,096-token context; policy + tools ≈ 4,700 | the start of the policy silently dropped; agent breaks rules it never saw | `"num_ctx": 16384` |
| a thinking model reasons without end | one request ran until LiteLLM's 600 s timeout, retried, again | `"max_tokens": 4096` for the agent |
| the simulated customer (same model) thinks too | it spends its reply budget reasoning and sends an empty message; τ²-bench gives up after 4 attempts | `"think": false` for the user model only: it plays a caller, it doesn't need to reason |

At temperature 0 a failed attempt repeats exactly, so `--max-retries 0` saves
~10 minutes per blank-out without changing any result. The adapter counts an
empty *agent* message as an agent failure, and leaves out a trial where the
*simulated customer* went blank (the harness failed, not the agent).

## 4. Can an LLM judge catch these violations? Not yet

`calibrate_judge.py`: 32 labelled items = the 16 trials with a conversation
(6 pass, 10 fail; the 4 blank-agent trials have nothing to judge) + 16
verified mutants: a passing trial with ONE planted violation (an unconfirmed
write, an unprompted certificate, an ineligible cancellation, a skipped "why
are you cancelling?"), each kept only if its target grader fails and every
other policy grader still passes. Judge: qwen2.5:7b, temperature 0, each item
scored twice. Output: `results/calibration_qwen25_judge*.txt`.

| | outcome (conversation only) | grounded v1 (+ tool calls, policy, checklist) | grounded v2 (+ "writes the agent made", checks apply only if the action happened) |
|---|---|---|---|
| AUC (0.5 = coin flip) | 0.42 | 0.42 | **0.69** |
| Cohen's kappa | −0.31 | −0.18 | **0.26** |
| Good trials wrongly failed | 6/6 | 6/6 | **2/6** |
| Planted violations caught | 14/16 | 16/16 | 16/16 |
| **Real** failures caught | 2/10 | 6/10 | 2/10 |
| Fit to gate? | no | no | no |

- **v1 invented violations.** Asked to explain itself on a correct trial,
  it said the agent "did not ask for confirmation before attempting to cancel"
  when nothing was cancelled. It read the checklist as a list of things that
  went wrong. Telling it which writes happened (v2) cut false alarms from 6/6
  to 2/6.
- **Planted violations are easy; real ones are hard.** v2 caught all 16
  mutants and 2 of 10 real failures: an invented membership level, a 26-turn
  loop, a voucher after a long argument are subtler than a clean planted
  mistake. **Mutation testing alone overestimates a judge**: calibrate on
  real failures too.
- **The code graders caught 14/14 real failures**, deterministically. Here
  they gate; a 7B judge is at most a second opinion.

Small numbers (6 passes, 10 real failures, near-identical pairs at
temperature 0): read these as indicative, not a benchmark of the judge.
