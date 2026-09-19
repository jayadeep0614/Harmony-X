"""
HARMONY-X
ML Predictor Integration Test
"""

from src.reasoning.ml_predictor import MLPredictor
from src.fusion.multimodal_fusion import FusionResult


def main():

    print("=" * 70)
    print("HARMONY-X ML PREDICTOR TEST")
    print("=" * 70)

    predictor = MLPredictor()

    predictor.initialize()

    # --------------------------------------------------
    # SIMULATED MULTIMODAL STATE
    # --------------------------------------------------

    fusion = FusionResult(
        face_detected=True,
        pose_detected=True,
        hands_detected=True,

        emotion="HAPPY",
        emotion_confidence=0.68,

        speech_detected=False,
        volume=0.05,

        visual_confidence=1.0,
        audio_confidence=0.05,
        overall_confidence=0.5767,
    )

    memory_summary = {
        "memory_size": 10,
        "speech_frequency": 0.20,
        "face_presence_frequency": 1.0,
        "average_volume": 0.05,
        "dominant_emotion": "HAPPY",
    }

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    result = predictor.predict(
        fusion,
        memory_summary,
    )

    print()
    print("ML Prediction:")
    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print()
    print("Class probabilities:")

    for label, probability in (
        result["probabilities"].items()
    ):

        print(
            f"{label}: "
            f"{probability:.4f}"
        )

    print()
    print("=" * 70)
    print("ML PREDICTOR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()