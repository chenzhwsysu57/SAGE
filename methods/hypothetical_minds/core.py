"""Paper §3.2/D.3: predictive hypothesis values, not payoff-based policy search.

Independent implementation of Cross et al., arXiv:2407.07086v2. Values are
Rescorla-Wagner scores, NOT normalized probabilities or calibrated confidence.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class HMConfig:
    alpha: float = 0.3
    reward: float = 1.0
    threshold: float = 0.7
    top_k: int = 5
    output_retries: int = 3

    def __post_init__(self):
        for key in ("alpha", "reward", "threshold"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if not 0 < self.alpha <= 1 or self.reward <= 0 or not 0 <= self.threshold < self.reward:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= threshold < reward")
        if type(self.top_k) is not int or not 0 <= self.top_k <= 20:
            raise ValueError("top_k must be an integer in [0,20]")
        if type(self.output_retries) is not int or not 0 <= self.output_retries <= 3:
            raise ValueError("output_retries must be an integer in [0,3]")


@dataclass
class Hypothesis:
    id: int
    text: str
    value: float = 0.0
    correct: int = 0
    incorrect: int = 0


class HypothesisPool:
    def __init__(self, config: HMConfig):
        self.config = config
        self.hypotheses: dict[int, Hypothesis] = {}
        self.evaluations: list[dict] = []
        self._consumed: set[tuple[int, str]] = set()

    def add(self, text: str) -> Hypothesis:
        if type(text) is not str or not text.strip() or len(text) > 6000:
            raise ValueError("Hypothesis must be nonempty text of at most 6000 characters")
        h = Hypothesis(len(self.hypotheses) + 1, text.strip())
        self.hypotheses[h.id] = h
        return h

    def ranked(self) -> list[Hypothesis]:
        # Stable oldest-first ties, matching insertion-order sorting in reference.
        return sorted(self.hypotheses.values(), key=lambda h: (-h.value, h.id))

    def validated(self) -> Hypothesis | None:
        return next((h for h in self.ranked() if h.value > self.config.threshold), None)

    def selected(self) -> Hypothesis:
        if not self.hypotheses:
            raise ValueError("No hypothesis generated yet")
        return self.validated() or self.hypotheses[max(self.hypotheses)]

    def candidates(self) -> list[Hypothesis]:
        good = self.validated()
        if good is not None:
            return [good]
        newest = self.selected()
        old = [h for h in self.ranked() if h.id != newest.id]
        return [newest, *old[:self.config.top_k]]

    def refinement_context(self) -> list[dict]:
        return [asdict(h) for h in self.ranked()[:self.config.top_k] if h.value > 0]

    def evaluate(self, hypothesis_id: int, event_id: str, prediction: str, actual: str) -> dict:
        key = (hypothesis_id, event_id)
        if key in self._consumed:
            raise ValueError("Prediction already consumed for this event")
        if not event_id or not prediction or not actual:
            raise ValueError("A concrete prediction and observed label are required")
        h = self.hypotheses[hypothesis_id]
        before = h.value
        correct = prediction == actual
        reward = self.config.reward if correct else -self.config.reward
        h.value += self.config.alpha * (reward - h.value)
        h.correct += int(correct)
        h.incorrect += int(not correct)
        self._consumed.add(key)
        row = {"hypothesis_id": h.id, "event_id": event_id, "prediction": prediction,
               "actual": actual, "correct": correct, "intrinsic_reward": reward,
               "value_before": before, "value_after": h.value,
               "validated_before": before > self.config.threshold,
               "validated_after": h.value > self.config.threshold}
        self.evaluations.append(row)
        return row

    def snapshot(self) -> dict:
        return {"config": asdict(self.config),
                "hypotheses": [asdict(h) for h in self.hypotheses.values()],
                "evaluations": list(self.evaluations)}