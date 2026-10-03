"""Candidate-pool scoring from idea.md; no model or numerical dependencies."""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class LogProbs:
    """Raw sequence log probabilities, including one terminal EOS token."""

    l11: float
    l01: float
    l10: float
    l00: float

    def __post_init__(self) -> None:
        for name in ("l11", "l01", "l10", "l00"):
            value = _finite(getattr(self, name), name)
            if value > 0:
                raise ValueError(f"{name} is a log probability and must be <= 0")


def components(logp: LogProbs) -> dict[str, float]:
    """Return historical support and the history × current-feedback interaction."""
    values = {
        "base": logp.l11,
        "d0": logp.l10 - logp.l00,
        "d1": logp.l11 - logp.l01,
    }
    values["gamma"] = values["d1"] - values["d0"]
    return {name: _finite(value, name) for name, value in values.items()}


def score(logp: LogProbs, alpha: float = 0.0, beta: float = 0.0) -> float:
    """S = l11 + alpha D0 + beta Gamma, without length normalization."""
    alpha, beta = _finite(alpha, "alpha"), _finite(beta, "beta")
    terms = components(logp)
    return _finite(terms["base"] + alpha * terms["d0"] + beta * terms["gamma"], "score")


def method_scores(
    logp: LogProbs, weight: float = 1.0, alpha: float = 0.0, beta: float = 1.0
) -> dict[str, float]:
    """Common controls on the same candidate, model and four conditions."""
    weight = _finite(weight, "weight")
    return {
        "base": score(logp),
        "history_d0": score(logp, alpha=weight),
        "cd": score(logp, alpha=weight, beta=weight),
        "b": score(logp, beta=weight),
        "joint": score(logp, alpha=alpha, beta=beta),
    }


def softmax(scores: Sequence[float]) -> list[float]:
    """Stable candidate distribution. Nonfinite scores are errors, not masks."""
    values = [_finite(value, "candidate score") for value in scores]
    if not values:
        raise ValueError("at least one candidate is required")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


class MissingGroupError(ValueError):
    """The pool cannot realize q; callers must report coverage or resample."""

    def __init__(self, missing_groups: Sequence[str]) -> None:
        self.missing_groups = tuple(sorted(missing_groups))
        super().__init__(
            "positive-q groups have no candidates: " + ", ".join(self.missing_groups)
            + "; q was not renormalized"
        )


def calibrate_groups(
    scores: Sequence[float], groups: Sequence[str], q: Mapping[str, float]
) -> list[float]:
    """P_A(y) = q(g(y)) softmax(S restricted to g(y)).

    Every candidate group must be explicitly present in q, including groups with
    zero mass. A missing positive-mass group raises MissingGroupError. This
    guarantees group marginals for the distribution, not for argmax selection.
    """
    if len(scores) != len(groups) or not scores:
        raise ValueError("scores and groups must have equal, nonzero length")
    values = [_finite(value, "candidate score") for value in scores]
    if not q or any(not isinstance(group, str) or not group for group in q):
        raise ValueError("q must map nonempty group names to probabilities")
    masses = {group: _finite(mass, f"q[{group}]") for group, mass in q.items()}
    if any(mass < 0 or mass > 1 for mass in masses.values()):
        raise ValueError("q probabilities must lie in [0, 1]")
    if not math.isclose(math.fsum(masses.values()), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("q must sum to 1; implicit renormalization is forbidden")
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if not isinstance(group, str) or group not in masses:
            raise ValueError(f"candidate group {group!r} has no explicit q value")
        members[group].append(index)
    missing = [group for group, mass in masses.items() if mass > 0 and group not in members]
    if missing:
        raise MissingGroupError(missing)
    probabilities = [0.0] * len(values)
    for group, indices in members.items():
        conditional = softmax([values[index] for index in indices])
        for index, probability in zip(indices, conditional):
            probabilities[index] = masses[group] * probability
    return probabilities
