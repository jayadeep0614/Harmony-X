import csv
from pathlib import Path


DATASET = Path("data/processed/harmony_x_dataset.csv")
BACKUP = Path("data/processed/harmony_x_dataset_backup.csv")
REPAIRED = Path("data/processed/harmony_x_dataset_repaired.csv")


with open(DATASET, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)

    header = next(reader)

    rows = list(reader)


print(f"Original header columns: {len(header)}")
print(f"Total data rows: {len(rows)}")


if "session_label" not in header:
    header.insert(1, "session_label")


repaired_rows = []

legacy_count = 0
new_count = 0


for row in rows:

    if len(row) == 22:
        row.insert(1, "LEGACY")
        legacy_count += 1

    elif len(row) == 23:
        new_count += 1

    else:
        print(
            f"WARNING: Unexpected row length: {len(row)}"
        )

    repaired_rows.append(row)


with open(
    REPAIRED,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    writer.writerows(repaired_rows)


print()
print("DATASET REPAIR COMPLETE")
print("-" * 50)
print(f"Legacy records repaired: {legacy_count}")
print(f"New labeled records:     {new_count}")
print(f"Total records:           {len(repaired_rows)}")
print(f"Columns:                 {len(header)}")
print()
print("Repaired file:")
print(REPAIRED)