"""
HARMONY-X
Dataset Statistics Report
"""

from pathlib import Path

import pandas as pd


class DatasetReport:

    def __init__(
        self,
        dataset_path="data/processed/harmony_x_dataset.csv"
    ):

        project_root = Path(__file__).resolve().parents[2]

        self.dataset_path = project_root / dataset_path

    def generate(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        print()
        print("=" * 70)
        print("HARMONY-X DATASET STATISTICS REPORT")
        print("=" * 70)

        print()
        print(f"Total observations: {len(df)}")
        print(f"Total features: {len(df.columns)}")

        print()
        print("EMOTION DISTRIBUTION")
        print("-" * 70)

        emotion_counts = df["emotion"].value_counts()

        for emotion, count in emotion_counts.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{emotion:<12} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

        print()
        print("ACTION DISTRIBUTION")
        print("-" * 70)

        action_counts = df["action"].value_counts()

        for action, count in action_counts.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{action:<20} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

        print()
        print("SPEECH DISTRIBUTION")
        print("-" * 70)

        speech_counts = df["speech_detected"].value_counts()

        for value, count in speech_counts.items():

            label = (
                "Speech detected"
                if value == 1
                else "No speech"
            )

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{label:<20} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

        print()
        print("CONFIDENCE DISTRIBUTION")
        print("-" * 70)

        confidence_counts = (
            df["confidence_level"]
            .value_counts()
        )

        for level, count in confidence_counts.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{level:<12} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

        print()
        print("NUMERIC FEATURE SUMMARY")
        print("-" * 70)

        numeric_columns = [
            "visual_confidence",
            "emotion_confidence",
            "volume",
            "audio_confidence",
            "overall_confidence",
        ]

        print(
            df[numeric_columns]
            .describe()
            .round(4)
        )

        print()
        print("DATASET BALANCE CHECK")
        print("-" * 70)

        smallest_emotion = emotion_counts.min()
        largest_emotion = emotion_counts.max()

        balance_ratio = (
            smallest_emotion /
            largest_emotion
        )

        print(
            f"Largest emotion class: "
            f"{largest_emotion}"
        )

        print(
            f"Smallest emotion class: "
            f"{smallest_emotion}"
        )

        print(
            f"Balance ratio: "
            f"{balance_ratio:.4f}"
        )

        if balance_ratio < 0.20:

            print(
                "⚠️ Dataset is highly imbalanced."
            )

        elif balance_ratio < 0.50:

            print(
                "⚠️ Dataset has moderate imbalance."
            )

        else:

            print(
                "✅ Dataset has reasonable balance."
            )

        print()
        print("=" * 70)
        print("REPORT COMPLETE")
        print("=" * 70)


if __name__ == "__main__":

    report = DatasetReport()

    report.generate()