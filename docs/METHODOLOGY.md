# Methodology and references

agentproof combines established ideas from agent evaluation research and
classic software testing. This page says where each idea comes from, what
agentproof takes from it, and what it adds.

![How agentproof fits together](linkedin/1_architecture.png)

## The building blocks

| agentproof feature | Built on | What we take from it |
|---|---|---|
| **Tasks, trials, transcripts, graders, outcomes** (the core vocabulary) | Anthropic, [*Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Grade the **outcome** (the state of the world after the run), not only the reply; combine code-based and model-based graders; run several trials per task; separate **capability** suites (a hill to climb) from **regression** suites (should stay near 100%) |
| **pass^k** ("passed all k trials") | Yao et al., [*τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*](https://arxiv.org/abs/2406.12045), 2024 | pass^k = the probability that **all** k independent trials succeed: a reliability measure for agents that act on a user's behalf. τ-bench showed strong agents scoring far lower on pass^k than on a single run |
| **pass@k** ("passed at least once") | Chen et al., [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) (Codex / HumanEval), 2021 | The "can it do this at all?" counterpart to pass^k |
| **Industry benchmark: airline tasks, policy, LLM-simulated customer, DB reward** | Sierra Research, [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT licence, synthetic data) | agentproof replays τ²-bench runs and adds process graders, one per rule in its airline `policy.md`, so a failed trial says *why* and a lucky pass is caught ([example](../examples/tau2_airline/README.md)) |
| **Database-state outcome checks**, **simulated user follow-ups** | τ-bench (above) | Compare the final database state with the expected one; an LLM-simulated user drives multi-turn tasks. agentproof uses a simpler *scripted* follow-up ("Yes, please book it") |
| **LLM-as-a-judge**, reported but not trusted by default | Zheng et al., [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685), 2023 | Strong LLM judges can reach about 80% agreement with humans, but have known biases (position, verbosity, self-enhancement). So a judge must be **checked against labels** before it decides anything |
| **Judge calibration metrics** | Cohen, *A coefficient of agreement for nominal scales*, Educational and Psychological Measurement 20(1), 1960; standard classifier metrics (recall, false-alarm rate, ROC AUC) | Cohen's kappa measures agreement beyond chance; a judge is graded like any binary classifier |
| **Mutation testing for judges** | DeMillo, Lipton & Sayward, *Hints on Test Data Selection: Help for the Practicing Programmer*, IEEE Computer 11(4), 1978 | Mutation testing judges a **test suite** by injecting small known faults ("mutants") and checking whether the tests catch them |
| **Grounding check** (every number traced to a tool result) | The faithfulness idea in Es et al., [*RAGAS: Automated Evaluation of Retrieval Augmented Generation*](https://arxiv.org/abs/2309.15217), 2023 | An answer's claims should be supported by the evidence the system retrieved. agentproof applies it deterministically to prices and amounts, against the agent's own tool results |
| **CI regression gate** | Common MLOps practice (e.g. OpenAI's agent-evals guide: smoke tests per commit, full suites nightly) | Block a change when quality drops below a floor |

## What agentproof adds

These are combinations and design choices made for this project, not taken
from one source:

1. **Mutation testing applied to an LLM judge.** Mutation testing classically
   evaluates a *test suite*. agentproof treats the LLM judge as the test
   suite: it injects known defects (an invented price, an unwanted booking,
   a made-up flight) into transcripts that passed, and measures how many the
   judge catches. Each mutant is **re-checked by the deterministic graders**
   and kept only if they fail it, so every "fail" label is verified rather
   than assumed.
2. **A strict, gating pass rule.** A trial passes only if *every* gating
   grader scores 1.0; partial credit is reported but never averages a wrong
   action into a pass. LLM judges are **non-gating** until a calibration
   report says they're fit to gate.
3. **A "fit to gate?" verdict** for judges, with explicit thresholds (fail
   recall ≥ 90%, kappa ≥ 0.6, consistency ≥ 90% across repeats), plus a
   per-defect breakdown showing *which* kinds of mistakes a judge misses.
4. **"Looks right, is wrong" accounting:** how often the judge rated a
   failed trial as good, and the judge's mean score on passed vs failed
   trials.
5. **A $0 CI gate that runs the real agent:** a free local model (Ollama)
   installed on the GitHub Actions runner, no API keys.

## Results, and how much to read into them

The numbers in the README come from real runs on a free 7B model
(qwen2.5:7b): 80 eval trials, 104 labelled calibration items (75 pass,
5 real failures, 24 verified mutants), and two CI runs. They show the
**method** working on one agent; they aren't a general benchmark. Counts per
defect type are small (8 each), and results move a few points between runs
(the same eval measured 85%/100% in an earlier run and 90%/98% here). Re-run
the calibration on your own judge, agent and defects before trusting a judge
with a gate.

## Further reading

- OpenAI, [Agent evals guide](https://developers.openai.com/api/docs/guides/agent-evals)
- Google Cloud, [Evaluate Gen AI agents](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-agents) (trajectory match metrics)
- AWS, [AgentCore evaluation types](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations-types.html) (online, on-demand and batch evaluation)
- Debenedetti et al., [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract.html), NeurIPS 2024 (prompt-injection evaluation for agents; the planned safety pack)
- OWASP, [Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026)
