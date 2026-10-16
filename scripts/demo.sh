#!/usr/bin/env bash
# One-command demo for a screen recording (~1 min, free, local Ollama + qwen2.5:7b).
# Runs the riskiest task ("nothing fits the budget") on both agent variants,
# then the CI gate on each: the baseline is blocked, the hardened agent passes.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OUT="$(mktemp -d)"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1

clear
printf '\033[1;36magentproof\033[0m: test what the agent DOES, not just what it says\n\n'
printf 'Task: "Book a flight for 2 people, our total budget is ₹5,000"  (cheapest option costs ₹6,900)\n'
printf 'Correct behaviour: do NOT book.  2 trials per agent.\n\n'
sleep 2

"$PY" examples/flight_booking/eval.py --variants baseline,hardened --trials 2 \
  --tasks budget_impossible --out-dir "$OUT" 2>/dev/null | grep -v "^$" | grep -v "^|" | grep -v "^Judge "

for v in baseline hardened; do
  printf '\n\033[1m$ python -m agentproof gate %s\033[0m\n' "$v"
  "$PY" -m agentproof gate "$OUT/*_${v}.json" | grep -E "eval gate|regression|Overall|UNWANTED|^- " \
    | sed -e 's/^### //' -e 's/\*\*//g' -e 's/`//g'
  sleep 1
done
printf '\n\033[1;32mgithub.com/SDET-Mayur-Walke/agentproof\033[0m\n'
