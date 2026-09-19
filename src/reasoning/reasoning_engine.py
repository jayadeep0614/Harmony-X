"""
HARMONY-X
Confidence & Uncertainty Aware Reasoning Engine
with Machine Learning Integration
"""

from dataclasses import dataclass

from src.fusion.multimodal_fusion import FusionResult
from src.reasoning.context_memory import ContextMemory
from src.reasoning.ml_predictor import MLPredictor


@dataclass
class ReasoningResult:
    interpretation: str
    action: str
    confidence: float
    priority: str
    context_signal: str
    confidence_level: str
    uncertainty: str
    uncertainty_reason: str

    ml_prediction: str = "UNKNOWN"
    ml_confidence: float = 0.0


class ReasoningEngine:

    def __init__(
        self,
        confidence_threshold: float = 0.70
    ):

        self.name = (
            "HARMONY-X Confidence-Aware "
            "Temporal Reasoning Engine"
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.ml_predictor = MLPredictor()

        self.initialized = False

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def initialize(self):

        print(
            f"{self.name}: initializing..."
        )

        self.ml_predictor.initialize()

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

        print(
            f"Confidence threshold: "
            f"{self.confidence_threshold}"
        )

        print(
            "ML Predictor: ready"
        )

    # --------------------------------------------------
    # CONFIDENCE CLASSIFICATION
    # --------------------------------------------------

    def classify_confidence(
        self,
        fusion: FusionResult
    ):

        confidence = (
            fusion.overall_confidence
        )

        if confidence >= 0.80:

            level = "HIGH"
            uncertainty = "LOW"

        elif confidence >= 0.50:

            level = "MEDIUM"
            uncertainty = "MODERATE"

        else:

            level = "LOW"
            uncertainty = "HIGH"

        reasons = []

        if fusion.visual_confidence < 0.50:

            reasons.append(
                "weak visual evidence"
            )

        if fusion.audio_confidence < 0.50:

            reasons.append(
                "weak audio evidence"
            )

        if fusion.emotion_confidence < 0.50:

            reasons.append(
                "uncertain expression estimate"
            )

        if not fusion.face_detected:

            reasons.append(
                "face not detected"
            )

        if not fusion.speech_detected:

            reasons.append(
                "speech not detected"
            )

        if not reasons:

            if level == "HIGH":

                reason = (
                    "strong multimodal evidence"
                )

            elif level == "MEDIUM":

                reason = (
                    "partial multimodal evidence"
                )

            else:

                reason = (
                    "weak multimodal evidence"
                )

        else:

            reason = ", ".join(
                reasons
            )

        return (
            level,
            uncertainty,
            reason,
        )

    # --------------------------------------------------
    # MAIN REASONING
    # --------------------------------------------------

    def reason(
        self,
        fusion: FusionResult,
        memory: ContextMemory
    ) -> ReasoningResult:

        if not self.initialized:

            raise RuntimeError(
                "ReasoningEngine has not "
                "been initialized."
            )

        # --------------------------------------------------
        # CURRENT MULTIMODAL STATE
        # --------------------------------------------------

        face = fusion.face_detected

        hands = fusion.hands_detected

        pose = fusion.pose_detected

        speech = fusion.speech_detected

        emotion = fusion.emotion

        confidence = (
            fusion.overall_confidence
        )

        # --------------------------------------------------
        # TEMPORAL MEMORY
        # --------------------------------------------------

        speech_frequency = (
            memory.speech_frequency()
        )

        dominant_emotion = (
            memory.dominant_emotion()
        )

        history_size = (
            memory.observation_count()
        )

        memory_summary = (
            memory.summary()
        )

        # --------------------------------------------------
        # ML PREDICTION
        # --------------------------------------------------

        ml_result = (
            self.ml_predictor.predict(
                fusion,
                memory_summary,
            )
        )

        ml_prediction = (
            ml_result["prediction"]
        )

        ml_confidence = (
            ml_result["confidence"]
        )

        # --------------------------------------------------
        # CONFIDENCE / UNCERTAINTY
        # --------------------------------------------------

        (
            confidence_level,
            uncertainty,
            uncertainty_reason,
        ) = self.classify_confidence(
            fusion
        )

        # --------------------------------------------------
        # LOW CONFIDENCE
        # --------------------------------------------------

        if confidence_level == "LOW":

            if (
                history_size >= 5
                and speech_frequency >= 0.30
            ):

                return ReasoningResult(
                    interpretation=(
                        "Current observation is uncertain, "
                        "but recent interaction is present"
                    ),

                    action="maintain_context",

                    confidence=confidence,

                    priority="MEDIUM",

                    context_signal=(
                        "Recent speech activity"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            return ReasoningResult(
                interpretation=(
                    "Insufficient confidence "
                    "in current observation"
                ),

                action="wait",

                confidence=confidence,

                priority="LOW",

                context_signal=(
                    "Weak current evidence"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # SPEECH + FACE
        # --------------------------------------------------

        if (
            face
            and speech
            and speech_frequency >= 0.30
        ):

            return ReasoningResult(
                interpretation=(
                    "Person is present and "
                    "sustained speech interaction "
                    "is detected"
                ),

                action="respond",

                confidence=confidence,

                priority="HIGH",

                context_signal=(
                    "Sustained speech"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        if face and speech:

            return ReasoningResult(
                interpretation=(
                    "Person is present "
                    "and speaking"
                ),

                action="listen",

                confidence=confidence,

                priority="HIGH",

                context_signal=(
                    "Current speech"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # RECENT INTERACTION
        # --------------------------------------------------

        if (
            face
            and not speech
            and speech_frequency >= 0.30
        ):

            return ReasoningResult(
                interpretation=(
                    "Person was recently interacting "
                    "but is currently silent"
                ),

                action="maintain_context",

                confidence=confidence,

                priority="MEDIUM",

                context_signal=(
                    "Recent speech activity"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # ML-AWARE EXPRESSION REASONING
        # --------------------------------------------------

        if face:

            # ML prediction is used as an
            # additional evidence source.

            if (
                ml_prediction == "HAPPY"
                and ml_confidence >= 0.60
            ):

                return ReasoningResult(
                    interpretation=(
                        "Person shows a "
                        "positive expression pattern"
                    ),

                    action="engage",

                    confidence=confidence,

                    priority="MEDIUM",

                    context_signal=(
                        "ML detected HAPPY expression pattern"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            if (
                emotion == dominant_emotion
                and emotion != "NEUTRAL"
            ):

                return ReasoningResult(
                    interpretation=(
                        "Current expression matches "
                        "the recent dominant "
                        "emotional pattern"
                    ),

                    action="engage",

                    confidence=confidence,

                    priority="MEDIUM",

                    context_signal=(
                        f"Persistent expression: "
                        f"{dominant_emotion}"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            if emotion == "HAPPY":

                return ReasoningResult(
                    interpretation=(
                        "Person shows a "
                        "positive expression"
                    ),

                    action="engage",

                    confidence=confidence,

                    priority="MEDIUM",

                    context_signal=(
                        "Positive expression"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            if emotion == "SAD":

                return ReasoningResult(
                    interpretation=(
                        "Person shows a "
                        "sad-like expression"
                    ),

                    action="support",

                    confidence=confidence,

                    priority="HIGH",

                    context_signal=(
                        "Sad-like expression"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            if emotion == "ANGRY":

                return ReasoningResult(
                    interpretation=(
                        "Person shows an "
                        "angry-like expression"
                    ),

                    action="caution",

                    confidence=confidence,

                    priority="HIGH",

                    context_signal=(
                        "Angry-like expression"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

            if emotion == "SURPRISED":

                return ReasoningResult(
                    interpretation=(
                        "Person shows a "
                        "surprise-like expression"
                    ),

                    action="observe",

                    confidence=confidence,

                    priority="MEDIUM",

                    context_signal=(
                        "Surprise-like expression"
                    ),

                    confidence_level=(
                        confidence_level
                    ),

                    uncertainty=uncertainty,

                    uncertainty_reason=(
                        uncertainty_reason
                    ),

                    ml_prediction=(
                        ml_prediction
                    ),

                    ml_confidence=(
                        ml_confidence
                    ),
                )

        # --------------------------------------------------
        # HAND ACTIVITY
        # --------------------------------------------------

        if face and hands:

            return ReasoningResult(
                interpretation=(
                    "Person detected "
                    "with hand activity"
                ),

                action="observe",

                confidence=confidence,

                priority="MEDIUM",

                context_signal=(
                    "Current hand activity"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # BODY POSTURE
        # --------------------------------------------------

        if face and pose:

            return ReasoningResult(
                interpretation=(
                    "Person detected with "
                    "body posture information"
                ),

                action="observe",

                confidence=confidence,

                priority="MEDIUM",

                context_signal=(
                    "Current body posture"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # FACE ONLY
        # --------------------------------------------------

        if face:

            return ReasoningResult(
                interpretation=(
                    "Person detected"
                ),

                action="observe",

                confidence=confidence,

                priority="LOW",

                context_signal=(
                    "Person presence"
                ),

                confidence_level=(
                    confidence_level
                ),

                uncertainty=uncertainty,

                uncertainty_reason=(
                    uncertainty_reason
                ),

                ml_prediction=(
                    ml_prediction
                ),

                ml_confidence=(
                    ml_confidence
                ),
            )

        # --------------------------------------------------
        # IDLE
        # --------------------------------------------------

        return ReasoningResult(
            interpretation=(
                "No significant interaction detected"
            ),

            action="idle",

            confidence=confidence,

            priority="LOW",

            context_signal=(
                "No interaction"
            ),

            confidence_level=(
                confidence_level
            ),

            uncertainty=uncertainty,

            uncertainty_reason=(
                uncertainty_reason
            ),

            ml_prediction=(
                ml_prediction
            ),

            ml_confidence=(
                ml_confidence
            ),
        )

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    def status(self):

        return {
            "name": self.name,

            "initialized":
                self.initialized,

            "confidence_threshold":
                self.confidence_threshold,

            "ml_predictor":
                self.ml_predictor.status(),
        }


# ------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------

if __name__ == "__main__":

    engine = ReasoningEngine()

    engine.initialize()

    print()

    print(
        "Reasoning Engine Status:"
    )

    print(
        engine.status()
    )