"""
HARMONY-X
Machine Learning Predictor

Loads the trained Random Forest model and
produces a HAPPY / NEUTRAL prediction from
the current multimodal state.
"""

from pathlib import Path

import joblib
import pandas as pd


class MLPredictor:

    def __init__(
        self,
        model_path="models/harmony_x_emotion_model.joblib",
    ):

        self.name = "HARMONY-X ML Predictor"

        project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.model_path = (
            project_root / model_path
        )

        self.model = None
        self.initialized = False

        self.features = [
            "face_detected",
            "hands_detected",
            "pose_detected",
            "visual_confidence",
            "emotion_confidence",
            "speech_detected",
            "volume",
            "audio_confidence",
            "overall_confidence",
            "speech_frequency",
            "face_presence_frequency",
            "average_volume",
            "memory_size",
        ]

    def initialize(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"ML model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

        print(
            f"Model: {self.model_path}"
        )

    def predict(
        self,
        fusion_result,
        memory_summary,
    ):

        if not self.initialized:

            raise RuntimeError(
                "MLPredictor has not been initialized."
            )

        row = {
            "face_detected":
                int(
                    fusion_result.face_detected
                ),

            "hands_detected":
                int(
                    fusion_result.hands_detected
                ),

            "pose_detected":
                int(
                    fusion_result.pose_detected
                ),

            "visual_confidence":
                fusion_result.visual_confidence,

            "emotion_confidence":
                fusion_result.emotion_confidence,

            "speech_detected":
                int(
                    fusion_result.speech_detected
                ),

            "volume":
                fusion_result.volume,

            "audio_confidence":
                fusion_result.audio_confidence,

            "overall_confidence":
                fusion_result.overall_confidence,

            "speech_frequency":
                memory_summary[
                    "speech_frequency"
                ],

            "face_presence_frequency":
                memory_summary[
                    "face_presence_frequency"
                ],

            "average_volume":
                memory_summary[
                    "average_volume"
                ],

            "memory_size":
                memory_summary[
                    "memory_size"
                ],
        }

        input_data = pd.DataFrame(
            [row],
            columns=self.features,
        )

        prediction = self.model.predict(
            input_data
        )[0]

        probabilities = {}

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probability_values = (
                self.model.predict_proba(
                    input_data
                )[0]
            )

            for label, probability in zip(
                self.model.classes_,
                probability_values,
            ):

                probabilities[
                    label
                ] = float(
                    probability
                )

        prediction_confidence = (
            probabilities.get(
                prediction,
                0.0
            )
        )

        return {
            "prediction": prediction,
            "confidence":
                prediction_confidence,
            "probabilities":
                probabilities,
        }

    def status(self):

        return {
            "name": self.name,
            "initialized":
                self.initialized,
            "model_path":
                str(self.model_path),
            "model_loaded":
                self.model is not None,
        }


if __name__ == "__main__":

    predictor = MLPredictor()

    predictor.initialize()

    print()

    print(
        "ML Predictor Status:"
    )

    print(
        predictor.status()
    )
