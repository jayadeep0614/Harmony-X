"""
HARMONY-X
Dataset Analysis & Quality Check

Analyzes the collected multimodal dataset.
"""

from pathlib import Path

import pandas as pd


class DatasetAnalyzer:

    def __init__(
        self,
        dataset_path="data/processed/harmony_x_dataset.csv"
    ):

        project_root = Path(__file__).resolve().parents[2]

        self.dataset_path = (
            project_root / dataset_path
        )

    def load_dataset(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        return pd.read_csv(self.dataset_path)

    def analyze(self):

        df = self.load_dataset()

        print()
        print("=" * 70)
        print("HARMONY-X DATASET ANALYSIS")
        print("=" * 70)

        print()
        print(f"Dataset path: {self.dataset_path}")

        print()
        print(f"Total records: {len(df)}")

        print()
        print("Columns:")
        print("-" * 70)

        for column in df.columns:
            print(column)

        print()
        print("Dataset Shape:")
        print("-" * 70)
        print(df.shape)

        print()
        print("Missing Values:")
        print("-" * 70)
        print(df.isnull().sum())

        print()
        print("Duplicate Rows:")
        print("-" * 70)
        print(df.duplicated().sum())

        print()
        print("Emotion Distribution:")
        print("-" * 70)
        print(df["emotion"].value_counts())

        print()
        print("Action Distribution:")
        print("-" * 70)
        print(df["action"].value_counts())

        print()
        print("Confidence Level Distribution:")
        print("-" * 70)
        print(df["confidence_level"].value_counts())

        print()
        print("Uncertainty Distribution:")
        print("-" * 70)
        print(df["uncertainty"].value_counts())

        print()
        print("Speech Detection:")
        print("-" * 70)
        print(df["speech_detected"].value_counts())

        print()
        print("Average Numeric Features:")
        print("-" * 70)

        numeric_columns = [
            "visual_confidence",
            "emotion_confidence",
            "volume",
            "audio_confidence",
            "overall_confidence",
            "speech_frequency",
            "face_presence_frequency",
            "average_volume",
        ]

        print(
            df[numeric_columns].mean().round(4)
        )

        print()
        print("Confidence Statistics:")
        print("-" * 70)

        print(
            df["overall_confidence"].describe().round(4)
        )

        print()
        print("Dataset Quality Check:")
        print("-" * 70)

        if len(df) == 0:
            print("❌ Dataset is empty.")

        else:
            print("✅ Dataset contains records.")

        if df.isnull().sum().sum() == 0:
            print("✅ No missing values.")

        else:
            print("⚠️ Missing values detected.")

        if df.duplicated().sum() == 0:
            print("✅ No duplicate rows.")

        else:
            print("⚠️ Duplicate rows detected.")

        print()
        print("=" * 70)
        print("DATASET ANALYSIS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":

    analyzer = DatasetAnalyzer()

    analyzer.analyze()