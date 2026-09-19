"""
HARMONY-X
Multimodal Reasoning Engine
"""

from dataclasses import dataclass

from src.fusion.multimodal_fusion import FusionResult


@dataclass
class ReasoningResult:
    interpretation: str
    action: str
    confidence: float
    priority: str


class ReasoningEngine:
    """
    Multimodal decision-making layer for HARMONY-X.

    Converts fused perception signals into
    interpretable interaction decisions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70
    ):
        self.name = "HARMONY-X Reasoning Engine"

        self.confidence_threshold = (
            confidence_threshold
        )

        self.initialized = False

    def initialize(self):

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

        print(
            f"Confidence threshold: "
            f"{self.confidence_threshold}"
        )

    def reason(
        self,
        fusion: FusionResult
    ) -> ReasoningResult:

        if not self.initialized:
            raise RuntimeError(
                "ReasoningEngine has not been initialized."
            )

        # ==================================================
        # Extract signals
        # ==================================================

        face = fusion.face_detected
        hands = fusion.hands_detected
        pose = fusion.pose_detected

        speech = fusion.speech_detected

        volume = fusion.volume

        emotion = fusion.emotion

        confidence = fusion.overall_confidence

        # ==================================================
        # LOW CONFIDENCE
        # ==================================================

        if confidence < self.confidence_threshold:

            return ReasoningResult(
                interpretation=(
                    "Insufficient confidence "
                    "in current observation"
                ),
                action="wait",
                confidence=confidence,
                priority="LOW",
            )

        # ==================================================
        # HIGH PRIORITY INTERACTION
        # ==================================================

        if (
            face
            and speech
            and volume > 0.25
        ):

            return ReasoningResult(
                interpretation=(
                    "Person is present and "
                    "actively speaking"
                ),
                action="respond",
                confidence=confidence,
                priority="HIGH",
            )

        # ==================================================
        # SPEECH INTERACTION
        # ==================================================

        if face and speech:

            return ReasoningResult(
                interpretation=(
                    "Person is present "
                    "and speaking"
                ),
                action="listen",
                confidence=confidence,
                priority="HIGH",
            )

        # ==================================================
        # EMOTION-AWARE OBSERVATION
        # ==================================================

        if face:

            if emotion == "HAPPY":

                return ReasoningResult(
                    interpretation=(
                        "Person appears "
                        "positively expressive"
                    ),
                    action="engage",
                    confidence=confidence,
                    priority="MEDIUM",
                )

            if emotion == "SAD":

                return ReasoningResult(
                    interpretation=(
                        "Person shows "
                        "sad-like facial expression"
                    ),
                    action="support",
                    confidence=confidence,
                    priority="HIGH",
                )

            if emotion == "ANGRY":

                return ReasoningResult(
                    interpretation=(
                        "Person shows "
                        "angry-like facial expression"
                    ),
                    action="caution",
                    confidence=confidence,
                    priority="HIGH",
                )

            if emotion == "SURPRISED":

                return ReasoningResult(
                    interpretation=(
                        "Person shows "
                        "surprise-like expression"
                    ),
                    action="observe",
                    confidence=confidence,
                    priority="MEDIUM",
                )

        # ==================================================
        # HAND INTERACTION
        # ==================================================

        if face and hands:

            return ReasoningResult(
                interpretation=(
                    "Person detected with "
                    "hand activity"
                ),
                action="observe",
                confidence=confidence,
                priority="MEDIUM",
            )

        # ==================================================
        # BODY / POSE INTERACTION
        # ==================================================

        if face and pose:

            return ReasoningResult(
                interpretation=(
                    "Person detected with "
                    "body posture information"
                ),
                action="observe",
                confidence=confidence,
                priority="MEDIUM",
            )

        # ==================================================
        # PERSON DETECTED
        # ==================================================

        if face:

            return ReasoningResult(
                interpretation=(
                    "Person detected"
                ),
                action="observe",
                confidence=confidence,
                priority="LOW",
            )

        # ==================================================
        # NO SIGNIFICANT INTERACTION
        # ==================================================

        return ReasoningResult(
            interpretation=(
                "No significant interaction "
                "detected"
            ),
            action="idle",
            confidence=confidence,
            priority="LOW",
        )

    def status(self):

        return {
            "name": self.name,
            "initialized": self.initialized,
            "confidence_threshold":
                self.confidence_threshold,
        }


if __name__ == "__main__":

    print(
        "HARMONY-X Reasoning Engine"
    )

    engine = ReasoningEngine(
        confidence_threshold=0.70
    )

    engine.initialize()

    print(
        engine.status()
    )