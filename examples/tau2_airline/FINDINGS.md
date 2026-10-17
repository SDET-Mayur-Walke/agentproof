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

## 2. Full run: 10 tasks × 3 trials

*(Filled in when the run finishes.)*
