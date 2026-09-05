from __future__ import annotations

from collections import deque

from domain_types import Candidate, ReactionState


class TemporalFilter:
    def __init__(self, window_size: int, majority_ratio: float, min_duration: float, cooldown_seconds: float) -> None:
        self.history: deque[tuple[float, Candidate]] = deque(maxlen=window_size)
        self.majority_ratio = majority_ratio
        self.min_duration = min_duration
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_until = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.cooldown_until = 0.0

    def update(self, candidate: Candidate, now: float) -> Candidate | None:
        self.history.append((now, candidate))
        if candidate.state is ReactionState.NEUTRAL or now < self.cooldown_until:
            return None
        matching = [(at, item) for at, item in self.history if item.state == candidate.state and item.label == candidate.label]
        required = max(3, int(len(self.history) * self.majority_ratio + 0.999))
        if len(matching) < required:
            return None
        if now - matching[0][0] < self.min_duration:
            return None
        confidence = sum(item.confidence for _, item in matching) / len(matching)
        self.cooldown_until = now + self.cooldown_seconds
        self.history.clear()
        return Candidate(candidate.state, candidate.label, confidence)
