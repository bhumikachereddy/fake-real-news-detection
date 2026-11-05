
"""
Create Prediction Drift Reference Baseline
Lab 12 - Experiment 12.4
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "cleaned_news.csv"
OUTPUT_FILE = ROOT / "artifacts" / "drift_baseline.json"

fake_count = 0
real_count = 0
total_words = 0
total_rows = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=["clean_text", "label"],
    chunksize=100000
):
    chunk["label"] = pd.to_numeric(
        chunk["label"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["label", "clean_text"])
    chunk = chunk[chunk["label"].isin([0, 1])]

    fake_count += int((chunk["label"] == 0).sum())
    real_count += int((chunk["label"] == 1).sum())

    total_words += int(
        chunk["clean_text"].astype(str).str.split().str.len().sum()
    )
    total_rows += len(chunk)

if total_rows == 0:
    raise ValueError("No valid training records found.")

baseline = {
    "total_records": total_rows,
    "fake_news": fake_count,
    "real_news": real_count,
    "fake_percentage": round(fake_count / total_rows * 100, 2),
    "real_percentage": round(real_count / total_rows * 100, 2),
    "average_words": round(total_words / total_rows, 2)
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(baseline, file, indent=4)

print("Drift baseline created successfully.")
print(json.dumps(baseline, indent=4))

