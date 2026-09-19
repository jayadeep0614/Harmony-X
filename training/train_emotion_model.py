"""
HARMONY-X
First ML Baseline

Binary classification:
HAPPY vs NEUTRAL

This is an initial baseline experiment.
The dataset contains controlled-condition labels.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
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
    print("HARMONY-X FIRST ML BASELINE")
    print("=" * 70)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    print()
    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Total records: {len(df)}"
    )

    # --------------------------------------------------
    # CONTROLLED CLASSES
    # --------------------------------------------------

    df = df[
        df["session_label"].isin(
            ["HAPPY", "NEUTRAL"]
        )
    ].copy()

    print()
    print("Controlled dataset:")

    print(
        df["session_label"].value_counts()
    )

    # --------------------------------------------------
    # FEATURES / TARGET
    # --------------------------------------------------

    X = df[FEATURES]

    y = df["session_label"]

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    print()
    print("Training records:")
    print(len(X_train))

    print()
    print("Testing records:")
    print(len(X_test))

    print()
    print("Training distribution:")

    print(
        y_train.value_counts()
    )

    print()
    print("Testing distribution:")

    print(
        y_test.value_counts()
    )

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print()
    print("Training Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print()
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[
                "HAPPY",
                "NEUTRAL",
            ],
        )
    )

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    importance = (
        pd.Series(
            model.feature_importances_,
            index=FEATURES,
        )
        .sort_values(
            ascending=False
        )
    )

    print()
    print("Feature Importance:")

    print(
        importance
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print("Model saved to:")

    print(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("ML BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()