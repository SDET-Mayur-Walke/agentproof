"""Judge calibration: measure an LLM judge against labelled trials BEFORE
letting it gate anything.

A judge is a classifier (pass / fail), so measure it like one:

  fail recall        of the trials that really failed, how many did the judge
                     score below the threshold? (the one that matters most: a
                     judge that misses failures waves bad agents through)
  false alarm rate   of the trials that really passed, how many did it fail?
  balanced accuracy  mean of the two hit rates; fair when failures are rare
  Cohen's kappa      agreement beyond chance (0 = no better than guessing)
  AUC                does a failed trial score lower than a passed one,
                     whatever the threshold? (0.5 = can't tell them apart)
  consistency        repeats of the same item land on the same side of the
                     threshold (a judge that flips can't be a gate)

Labels come from anywhere you trust more than the judge: human review, or
deterministic graders. Real failures are usually rare, so `mutate()` builds
extra labelled failures by injecting KNOWN defects into passing transcripts
(mutation testing, applied to the judge instead of the code). Each item
carries a `source`, so the report shows WHICH kinds of defect the judge
misses.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Callable

from .core import Grader, Task, Transcript


@dataclass
class LabelledItem:
    task: Task
    transcript: Transcript
    label: bool                 # True = this trial really passed
    source: str = "real"        # "real", or the mutation that made it


@dataclass
class ItemResult:
    item: LabelledItem
    scores: list[float]

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores)


def mutate(item: LabelledItem, name: str,
           fn: Callable[[Task, Transcript], Transcript | None]) -> LabelledItem | None:
    """Apply a defect-injecting function to a copy of a transcript. The
    result is labelled FAIL by construction. fn returns None when the
    mutation doesn't apply to this transcript."""
    t = fn(item.task, copy.deepcopy(item.transcript))
    return None if t is None else LabelledItem(item.task, t, label=False, source=name)


@dataclass
class CalibrationReport:
    judge: str
    threshold: float
    results: list[ItemResult]

    def _pred(self, r: ItemResult) -> bool:
        return r.mean >= self.threshold

    @property
    def n_pass(self) -> int:
        return sum(r.item.label for r in self.results)

    @property
    def n_fail(self) -> int:
        return sum(not r.item.label for r in self.results)

    @property
    def fail_recall(self) -> float:
        fails = [r for r in self.results if not r.item.label]
        return sum(not self._pred(r) for r in fails) / len(fails) if fails else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        passes = [r for r in self.results if r.item.label]
        return sum(not self._pred(r) for r in passes) / len(passes) if passes else float("nan")

    @property
    def balanced_accuracy(self) -> float:
        return (self.fail_recall + (1 - self.false_alarm_rate)) / 2

    @property
    def kappa(self) -> float:
        n = len(self.results)
        if not n:
            return float("nan")
        agree = sum(self._pred(r) == r.item.label for r in self.results) / n
        p_label = self.n_pass / n
        p_pred = sum(self._pred(r) for r in self.results) / n
        chance = p_label * p_pred + (1 - p_label) * (1 - p_pred)
        return 0.0 if chance == 1 else (agree - chance) / (1 - chance)

    @property
    def auc(self) -> float:
        """P(a random passed trial scores higher than a random failed one); ties count half."""
        pos = [r.mean for r in self.results if r.item.label]
        neg = [r.mean for r in self.results if not r.item.label]
        if not pos or not neg:
            return float("nan")
        wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
        return wins / (len(pos) * len(neg))

    @property
    def consistency(self) -> float:
        multi = [r for r in self.results if len(r.scores) > 1]
        if not multi:
            return float("nan")
        same = sum(len({s >= self.threshold for s in r.scores}) == 1 for r in multi)
        return same / len(multi)

    def mean_score(self, label: bool) -> float:
        xs = [r.mean for r in self.results if r.item.label == label]
        return sum(xs) / len(xs) if xs else float("nan")

    def by_source(self) -> dict[str, tuple[int, int]]:
        """source -> (judged correctly, total). For mutations, 'correct' = caught."""
        out: dict[str, list[int]] = {}
        for r in self.results:
            key = r.item.source if r.item.source != "real" else ("real pass" if r.item.label else "real fail")
            hit = self._pred(r) == r.item.label
            c = out.setdefault(key, [0, 0])
            c[0] += hit
            c[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def best_threshold(self) -> tuple[float, float]:
        """The threshold with the highest balanced accuracy, and that accuracy.
        Tuning on the same set you report on flatters the judge; use it as a
        hint, then confirm on fresh data."""
        best = (self.threshold, self.balanced_accuracy)
        for th in sorted({round(r.mean, 3) for r in self.results} | {self.threshold}):
            ba = CalibrationReport(self.judge, th, self.results).balanced_accuracy
            if ba > best[1]:
                best = (th, ba)
        return best

    def verdict(self, min_fail_recall: float = 0.9, min_kappa: float = 0.6,
                min_consistency: float = 0.9) -> tuple[bool, list[str]]:
        """Is this judge fit to GATE? Reasons it isn't, if not."""
        why = []
        if not self.fail_recall >= min_fail_recall:
            why.append(f"misses failures: recall {self.fail_recall:.0%} < {min_fail_recall:.0%}")
        if not self.kappa >= min_kappa:
            why.append(f"weak agreement: kappa {self.kappa:.2f} < {min_kappa}")
        if self.consistency == self.consistency and self.consistency < min_consistency:
            why.append(f"inconsistent: {self.consistency:.0%} of items stable < {min_consistency:.0%}")
        return not why, why

    def summary(self) -> str:
        ok, why = self.verdict()
        th, ba = self.best_threshold()
        lines = [
            f"judge '{self.judge}' · threshold {self.threshold} · {len(self.results)} items "
            f"({self.n_pass} pass, {self.n_fail} fail)",
            f"  fail recall        {self.fail_recall:6.0%}   (failures it caught)",
            f"  false alarm rate   {self.false_alarm_rate:6.0%}   (good answers it failed)",
            f"  balanced accuracy  {self.balanced_accuracy:6.0%}",
            f"  Cohen's kappa      {self.kappa:6.2f}",
            f"  AUC                {self.auc:6.2f}   (0.5 = can't tell pass from fail)",
            f"  consistency        {self.consistency:6.0%}   (repeats on the same side)",
            f"  mean score         pass {self.mean_score(True):.2f} · fail {self.mean_score(False):.2f}",
            f"  best threshold     {th} -> balanced accuracy {ba:.0%} (in-sample)",
            "  by source:",
        ]
        for src, (hit, n) in sorted(self.by_source().items()):
            lines.append(f"    {src:<24} {hit}/{n} judged correctly")
        lines.append(f"  VERDICT: {'fit to gate' if ok else 'NOT fit to gate: ' + '; '.join(why)}")
        return "\n".join(lines)


async def calibrate(items: list[LabelledItem], judge: Grader, repeats: int = 3,
                    threshold: float = 0.7, concurrency: int = 1,
                    progress: Callable[[int, int], None] | None = None) -> CalibrationReport:
    """Score every item `repeats` times with the judge grader."""
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: LabelledItem) -> ItemResult:
        nonlocal done
        scores = []
        for _ in range(repeats):
            async with sem:
                scores.append((await judge(item.task, item.transcript)).score)
        done += 1
        if progress:
            progress(done, len(items))
        return ItemResult(item, scores)

    results = await asyncio.gather(*(one(i) for i in items))
    return CalibrationReport(judge.name, threshold, list(results))
