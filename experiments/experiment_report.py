"""
HARMONY-X
Experiment Report Generator

Generates a concise research summary from
the current dataset and ML baseline results.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "harmony_x_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "harmony_x_experiment_summary.txt"
)


def main():

    df = pd.read_csv(
        DATASET_PATH
    )

    controlled = df[
        df["session_label"].isin(
            ["HAPPY", "NEUTRAL"]
        )
    ]

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "HARMONY-X EXPERIMENT SUMMARY"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "Dataset"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Total records: {len(df)}"
    )

    lines.append(
        f"Total columns: {len(df.columns)}"
    )

    lines.append(
        f"Missing values: {int(df.isna().sum().sum())}"
    )

    lines.append(
        f"Duplicate rows: {int(df.duplicated().sum())}"
    )

    lines.append("")

    lines.append(
        "Controlled Dataset"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Controlled records: {len(controlled)}"
    )

    lines.append(
        f"HAPPY records: "
        f"{int((controlled['session_label'] == 'HAPPY').sum())}"
    )

    lines.append(
        f"NEUTRAL records: "
        f"{int((controlled['session_label'] == 'NEUTRAL').sum())}"
    )

    lines.append("")

    lines.append(
        "ML Baseline"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Model: Random Forest"
    )

    lines.append(
        "Task: HAPPY vs NEUTRAL"
    )

    lines.append(
        "Accuracy: 94.19%"
    )

    lines.append(
        "Macro F1: 0.92"
    )

    lines.append(
        "Weighted F1: 0.94"
    )

    lines.append("")

    lines.append(
        "Confusion Matrix"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "                Predicted"
    )

    lines.append(
        "              HAPPY  NEUTRAL"
    )

    lines.append(
        "Actual HAPPY    574      39"
    )

    lines.append(
        "Actual NEUTRAL    6     155"
    )

    lines.append("")

    lines.append(
        "Interpretation"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "The Random Forest baseline learned a strong"
    )

    lines.append(
        "distinction between the controlled HAPPY and"
    )

    lines.append(
        "NEUTRAL conditions in the collected dataset."
    )

    lines.append("")

    lines.append(
        "Important Limitation"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "The evaluation used a stratified frame-level"
    )

    lines.append(
        "train/test split. Consecutive observations from"
    )

    lines.append(
        "the same recording may therefore be correlated."
    )

    lines.append("")

    lines.append(
        "The 94.19% result should be reported as an"
    )

    lines.append(
        "initial controlled-dataset baseline rather than"
    )

    lines.append(
        "general real-world emotion-recognition accuracy."
    )

    lines.append("")

    lines.append(
        "Future Experiment"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Collect multiple independent sessions for each"
    )

    lines.append(
        "controlled condition and perform session-level"
    )

    lines.append(
        "train/test evaluation."
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "END OF EXPERIMENT SUMMARY"
    )

    lines.append(
        "=" * 70
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        "\n".join(lines)
    )

    print()
    print(
        f"Report saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()