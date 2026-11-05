
"""
Prediction Monitoring Module
Lab 12 - Experiment 12.1

Tracks:
1. Model predictions
2. Fake and Real news prediction counts
3. Input text statistics
4. Prediction latency
5. Successful and failed predictions

Raw news content is not stored.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import threading

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Create artifacts directory if missing
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Prediction log file
LOG_FILE = ARTIFACT_DIR / "prediction_logs.csv"

# Thread safety for concurrent API requests
file_lock = threading.Lock()

# CSV columns
FIELDNAMES = [
    "timestamp",
    "prediction",
    "result",
    "input_characters",
    "input_words",
    "latency_ms",
    "status"
]


def log_prediction(
    text,
    prediction,
    result,
    latency_ms,
    status="success"
):
    """
    Record one model prediction.

    Raw news text is not saved.
    """

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "result": result,
        "input_characters": len(text),
        "input_words": len(text.split()),
        "latency_ms": round(latency_ms, 3),
        "status": status
    }

    with file_lock:

        file_exists = LOG_FILE.exists()

        with open(
            LOG_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            if not file_exists or LOG_FILE.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(row)

    return row


def get_prediction_summary():
    """
    Generate prediction distribution statistics.
    """

    if not LOG_FILE.exists():
        return {
            "total_predictions": 0,
            "fake_news": 0,
            "real_news": 0,
            "fake_percentage": 0,
            "real_percentage": 0
        }

    fake_count = 0
    real_count = 0
    total = 0

    with file_lock:

        with open(
            LOG_FILE,
            mode="r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["status"] != "success":
                    continue

                total += 1

                if row["prediction"] == "0":
                    fake_count += 1

                elif row["prediction"] == "1":
                    real_count += 1

    return {
        "total_predictions": total,
        "fake_news": fake_count,
        "real_news": real_count,
        "fake_percentage": round(
            fake_count / total * 100, 2
        ) if total else 0,
        "real_percentage": round(
            real_count / total * 100, 2
        ) if total else 0
    }
