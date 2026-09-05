from __future__ import annotations

from domain_types import Candidate, HandFeatures, ReactionState


class GestureClassifier:
    def classify(self, hand: HandFeatures) -> Candidate:
        if hand.gesture == "thumbs_up":
            return Candidate(ReactionState.POSITIVE, "thumbs up", hand.confidence)
        if hand.gesture == "index_up":
            return Candidate(ReactionState.POSITIVE, "index finger up", hand.confidence)
        if hand.gesture == "thumbs_down":
            return Candidate(ReactionState.NEGATIVE, "thumbs down", hand.confidence)
        return Candidate()
