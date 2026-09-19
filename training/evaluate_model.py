"""
HARMONY-X
ML Model Evaluation

Evaluates the trained HAPPY vs NEUTRAL
Random Forest baseline.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "harmony_x_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "harmony_x_emotion_model.joblib"
)

FEATURES = [
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


def main():

    print("=" * 70)
    print("HARMONY-X MODEL EVALUATION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    df = df[
        df["session_label"].isin(
            ["HAPPY", "NEUTRAL"]
        )
    ].copy()

    X = df[FEATURES]

    y = df["session_label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print()
    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print(report)

    print("Confusion Matrix")
    print("-" * 70)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "HAPPY",
            "NEUTRAL",
        ],
    )

    print(matrix)

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()