"""
HARMONY-X
Multimodal Fusion Module
"""

from dataclasses import dataclass

from src.perception.vision import VisionResult
from src.audio.audio_processor import AudioResult


@dataclass
class FusionResult:
    """Combined representation from multiple modalities."""

    # Vision
    face_detected: bool
    pose_detected: bool
    hands_detected: bool

    # Emotion
    emotion: str
    emotion_confidence: float

    # Audio
    speech_detected: bool
    volume: float

    # Confidence
    visual_confidence: float
    audio_confidence: float
    overall_confidence: float


class MultimodalFusion:
    """Combines vision, emotion and audio information."""

    def __init__(self):

        self.name = "HARMONY-X Multimodal Fusion"

        self.initialized = False

    def initialize(self):
        """Initialize the fusion system."""

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

    def fuse(
        self,
        vision: VisionResult,
        audio: AudioResult,
    ) -> FusionResult:
        """Combine vision and audio results."""

        if not self.initialized:

            raise RuntimeError(
                "MultimodalFusion has not been initialized."
            )

        # --------------------------------------------------
        # VISION CONFIDENCE
        # --------------------------------------------------

        visual_confidence = (
            vision.confidence
        )

        # --------------------------------------------------
        # AUDIO CONFIDENCE
        # --------------------------------------------------

        audio_confidence = (
            audio.confidence
        )

        # --------------------------------------------------
        # EMOTION CONFIDENCE
        # --------------------------------------------------

        emotion_confidence = (
            vision.emotion_confidence
        )

        # --------------------------------------------------
        # MULTIMODAL CONFIDENCE
        # --------------------------------------------------

        overall_confidence = (
            visual_confidence
            + audio_confidence
            + emotion_confidence
        ) / 3.0

        # --------------------------------------------------
        # RETURN FUSED RESULT
        # --------------------------------------------------

        return FusionResult(

            # Vision
            face_detected=(
                vision.face_detected
            ),

            pose_detected=(
                vision.pose_detected
            ),

            hands_detected=(
                vision.hands_detected
            ),

            # Emotion
            emotion=(
                vision.emotion
            ),

            emotion_confidence=(
                emotion_confidence
            ),

            # Audio
            speech_detected=(
                audio.speech_detected
            ),

            volume=(
                audio.volume
            ),

            # Confidence
            visual_confidence=(
                visual_confidence
            ),

            audio_confidence=(
                audio_confidence
            ),

            overall_confidence=(
                overall_confidence
            ),
        )

    def status(self):
        """Return current fusion status."""

        return {

            "name": self.name,

            "initialized": (
                self.initialized
            ),
        }


if __name__ == "__main__":

    # --------------------------------------------------
    # TEST VISION RESULT
    # --------------------------------------------------

    vision_result = VisionResult(

        face_detected=True,

        pose_detected=True,

        hands_detected=True,

        confidence=1.0,

        expressions={
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.82,
        },

        emotion="HAPPY",

        emotion_confidence=0.61,
    )

    # --------------------------------------------------
    # TEST AUDIO RESULT
    # --------------------------------------------------

    audio_result = AudioResult(

        speech_detected=True,

        volume=0.65,

        confidence=0.90,
    )

    # --------------------------------------------------
    # CREATE FUSION SYSTEM
    # --------------------------------------------------

    fusion = MultimodalFusion()

    fusion.initialize()

    # --------------------------------------------------
    # FUSE
    # --------------------------------------------------

    result = fusion.fuse(

        vision_result,

        audio_result,
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("\nFusion Status:")

    print(
        fusion.status()
    )

    print("\nFusion Result:")

    print(result)

    print("\nInterpretation:")

    print(
        f"Emotion: {result.emotion}"
    )

    print(
        f"Emotion confidence: "
        f"{result.emotion_confidence * 100:.1f}%"
    )

    print(
        f"Overall multimodal confidence: "
        f"{result.overall_confidence * 100:.1f}%"
    )