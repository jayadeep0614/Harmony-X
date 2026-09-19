"""
HARMONY-X
Facial Expression / Emotion Analyzer
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class EmotionResult:
    """Estimated facial expression state."""

    emotion: str
    confidence: float
    signals: Dict[str, float]


class EmotionAnalyzer:
    """
    Interprets MediaPipe face blendshape signals.

    These are expression-based estimates, not definitive
    measurements of a person's internal emotional state.
    """

    def __init__(self):
        self.name = "HARMONY-X Emotion Analyzer"

    def analyze(
        self,
        expressions: Dict[str, float]
    ) -> EmotionResult:

        if not expressions:
            return EmotionResult(
                emotion="UNKNOWN",
                confidence=0.0,
                signals={},
            )

        def score(name: str) -> float:
            return float(expressions.get(name, 0.0))

        # --------------------------------------------------
        # EXPRESSION FEATURES
        # --------------------------------------------------

        smile = (
            score("mouthSmileLeft")
            + score("mouthSmileRight")
        ) / 2.0

        frown = (
            score("mouthFrownLeft")
            + score("mouthFrownRight")
        ) / 2.0

        brow_down = (
            score("browDownLeft")
            + score("browDownRight")
        ) / 2.0

        brow_up = (
            score("browOuterUpLeft")
            + score("browOuterUpRight")
        ) / 2.0

        jaw_open = score("jawOpen")

        eye_wide = (
            score("eyeWideLeft")
            + score("eyeWideRight")
        ) / 2.0

        eye_squint = (
            score("eyeSquintLeft")
            + score("eyeSquintRight")
        ) / 2.0

        # --------------------------------------------------
        # EXPRESSION SCORES
        # --------------------------------------------------

        happy_score = (
            0.70 * smile
            + 0.30 * eye_squint
        )

        sad_score = (
            0.65 * frown
            + 0.35 * brow_up
        )

        angry_score = (
            0.75 * brow_down
            + 0.25 * eye_squint
        )

        surprised_score = (
            0.40 * brow_up
            + 0.35 * jaw_open
            + 0.25 * eye_wide
        )

        # --------------------------------------------------
        # NEUTRAL
        #
        # Neutral is now based on LOW expression activity
        # rather than 1 - strongest emotion.
        # --------------------------------------------------

        expression_activity = max(
            smile,
            frown,
            brow_down,
            brow_up,
            jaw_open,
            eye_wide,
            eye_squint,
        )

        neutral_score = max(
            0.0,
            1.0 - expression_activity
        )

        scores = {
            "HAPPY": happy_score,
            "SAD": sad_score,
            "ANGRY": angry_score,
            "SURPRISED": surprised_score,
            "NEUTRAL": neutral_score,
        }

        # --------------------------------------------------
        # SELECT STRONGEST STATE
        # --------------------------------------------------

        emotion = max(
            scores,
            key=scores.get
        )

        confidence = scores[emotion]

        confidence = min(
            max(confidence, 0.0),
            1.0
        )

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            signals={
                "smile": smile,
                "frown": frown,
                "brow_down": brow_down,
                "brow_up": brow_up,
                "jaw_open": jaw_open,
                "eye_wide": eye_wide,
                "eye_squint": eye_squint,
                "expression_activity":
                    expression_activity,
            },
        )


if __name__ == "__main__":

    analyzer = EmotionAnalyzer()

    test_expression = {
        "mouthSmileLeft": 0.85,
        "mouthSmileRight": 0.82,
        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.32,
        "browDownLeft": 0.05,
        "browDownRight": 0.04,
        "browOuterUpLeft": 0.10,
        "browOuterUpRight": 0.08,
        "jawOpen": 0.05,
        "eyeWideLeft": 0.10,
        "eyeWideRight": 0.12,
    }

    result = analyzer.analyze(
        test_expression
    )

    print()
    print("HARMONY-X Emotion Analyzer")
    print("--------------------------")

    print(
        f"Emotion: {result.emotion}"
    )

    print(
        f"Confidence: {result.confidence:.2f}"
    )

    print()
    print("Signals:")

    for name, value in result.signals.items():

        print(
            f"{name}: {value:.2f}"
        )