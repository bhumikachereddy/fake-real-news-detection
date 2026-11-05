
"""
Prediction Drift Detection
Lab 12 - Experiment 12.4
"""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent

BASELINE_FILE = ROOT / "artifacts" / "drift_baseline.json"
LOG_FILE = ROOT / "artifacts" / "prediction_logs.csv"

MINIMUM_PREDICTIONS = 20
WINDOW_SIZE = 100
DRIFT_THRESHOLD = 15.0


def detect_prediction_drift():

    if not BASELINE_FILE.exists():
        return {"status": "baseline_missing"}

    if not LOG_FILE.exists():
        return {"status": "no_prediction_logs"}

    with open(BASELINE_FILE, encoding="utf-8") as file:
        baseline = json.load(file)

    recent = []

    with open(LOG_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("status") == "success":
                recent.append(row)

    recent = recent[-WINDOW_SIZE:]

    if len(recent) < MINIMUM_PREDICTIONS:
        return {
            "status": "insufficient_data",
            "minimum_required": MINIMUM_PREDICTIONS,
            "available_predictions": len(recent)
        }

    fake_count = sum(
        1 for row in recent if row["prediction"] == "0"
    )
    real_count = sum(
        1 for row in recent if row["prediction"] == "1"
    )

    total = len(recent)

    current_fake_percentage = fake_count / total * 100
    current_real_percentage = real_count / total * 100

    fake_difference = abs(
        current_fake_percentage - baseline["fake_percentage"]
    )

    real_difference = abs(
        current_real_percentage - baseline["real_percentage"]
    )

    drift_detected = max(
        fake_difference, real_difference
    ) >= DRIFT_THRESHOLD

    return {
        "status": "drift_detected" if drift_detected else "stable",
        "baseline_fake_percentage": baseline["fake_percentage"],
        "current_fake_percentage": round(
            current_fake_percentage, 2
        ),
        "baseline_real_percentage": baseline["real_percentage"],
        "current_real_percentage": round(
            current_real_percentage, 2
        ),
        "fake_percentage_difference": round(fake_difference, 2),
        "real_percentage_difference": round(real_difference, 2),
        "predictions_analyzed": total,
        "threshold_percentage_points": DRIFT_THRESHOLD
    }
