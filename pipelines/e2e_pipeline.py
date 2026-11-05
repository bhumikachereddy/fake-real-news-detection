
import json
import shutil
import subprocess
import sys
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline_config.json"

STEPS = []
METRICS = {}
DATASET_INFO = {}

PIPELINE_STATUS = "failed"
ERROR_MESSAGE = None
ROLLBACK_PERFORMED = False


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def validate_datasets(config):
    train_path = ROOT / config["dataset_path"]
    test_path = ROOT / config["test_dataset_path"]

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_columns = {"clean_text", "label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError("Training dataset is missing required columns.")

    if not required_columns.issubset(test_df.columns):
        raise ValueError("Test dataset is missing required columns.")

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test dataset is empty.")

    if train_df["clean_text"].fillna("").str.strip().eq("").all():
        raise ValueError("Training dataset has no usable text.")

    if test_df["clean_text"].fillna("").str.strip().eq("").all():
        raise ValueError("Test dataset has no usable text.")

    train_labels = set(
        pd.to_numeric(train_df["label"], errors="raise").astype(int).unique()
    )

    test_labels = set(
        pd.to_numeric(test_df["label"], errors="raise").astype(int).unique()
    )

    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError("Unexpected labels. Expected 0 and 1.")

    return {
    "training_records": int(len(train_df)),
    "test_records": int(len(test_df)),
    "training_labels": [int(x) for x in sorted(train_labels)],
    "test_labels": [int(x) for x in sorted(test_labels)]
    }

def run_command(step_name, command):
    print("\n" + "=" * 60)
    print("RUNNING:", step_name)
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    STEPS.append({
        "step": step_name,
        "command": command,
        "return_code": result.returncode,
        "status": "passed" if result.returncode == 0 else "failed"
    })

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed.")


def validate_model_artifacts(config):
    model_path = ROOT / config["model_path"]
    vectorizer_path = ROOT / config["vectorizer_path"]

    for path in [model_path, vectorizer_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Invalid model artifact: {path}")

    STEPS.append({
        "step": "model_artifact_validation",
        "status": "passed"
    })


def evaluate_quality(config):
    report_path = ROOT / config["evaluation_report_path"]

    if not report_path.exists():
        raise FileNotFoundError("Model evaluation report not found.")

    with open(report_path, "r", encoding="utf-8-sig") as file:
        report = json.load(file)

    accuracy = float(report["accuracy"])
    f1_score = float(report["f1_score"])

    METRICS.update({
        "test_records": report.get("test_records"),
        "accuracy": accuracy,
        "precision": report.get("precision"),
        "recall": report.get("recall"),
        "f1_score": f1_score,
        "confusion_matrix": report.get("confusion_matrix")
    })

    if accuracy < config["min_accuracy"]:
        raise ValueError("Accuracy quality gate failed.")

    if f1_score < config["min_f1_score"]:
        raise ValueError("F1-score quality gate failed.")

    STEPS.append({
        "step": "model_quality_gate",
        "status": "passed",
        "actual_accuracy": accuracy,
        "actual_f1_score": f1_score
    })


def save_pipeline_report(config, started_at, duration):
    report = {
        "pipeline": "Fake and Real News Detection - End-to-End MLOps",
        "status": PIPELINE_STATUS,
        "started_at": started_at,
        "completed_at": timestamp(),
        "duration_seconds": round(duration, 2),
        "dataset": DATASET_INFO,
        "model_metrics": METRICS,
        "rollback_performed": ROLLBACK_PERFORMED,
        "steps": STEPS,
        "error": ERROR_MESSAGE
    }

    report_path = ROOT / config["pipeline_report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nFINAL PIPELINE REPORT")
    print(json.dumps(report, indent=4))
    print("\nReport saved at:", report_path)


def main():
    global PIPELINE_STATUS
    global ERROR_MESSAGE
    global ROLLBACK_PERFORMED
    global DATASET_INFO

    start_time = time.time()
    started_at = timestamp()
    config = load_config()

    backup_targets = [
        ROOT / config["model_path"],
        ROOT / config["vectorizer_path"],
        ROOT / config["evaluation_report_path"]
    ]

    try:
        with tempfile.TemporaryDirectory(
            prefix="lab13_model_backup_"
        ) as temporary_directory:

            backup_directory = Path(temporary_directory)
            backups = {}

            for original_path in backup_targets:
                if not original_path.exists():
                    raise FileNotFoundError(
                        f"Required backup file missing: {original_path}"
                    )

                backup_path = backup_directory / original_path.name
                shutil.copy2(original_path, backup_path)
                backups[original_path] = backup_path

            try:
                print("\nSTAGE 1: DATA VALIDATION")
                DATASET_INFO = validate_datasets(config)

                STEPS.append({
                    "step": "dataset_validation",
                    "status": "passed",
                    "details": DATASET_INFO
                })

                run_command(
                    "model_training",
                    [sys.executable, "src/train.py"]
                )

                run_command(
                    "model_evaluation",
                    [
                        sys.executable,
                        "-m",
                        "monitoring.evaluate_model"
                    ]
                )

                evaluate_quality(config)
                validate_model_artifacts(config)

                if config.get("run_api_tests", True):
                    run_command(
                        "api_testing",
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "tests/test_api.py",
                            "-v"
                        ]
                    )

                PIPELINE_STATUS = "success"

            except Exception:
                print("\nPIPELINE FAILED. RESTORING PREVIOUS FILES.")

                for original_path, backup_path in backups.items():
                    shutil.copy2(backup_path, original_path)

                ROLLBACK_PERFORMED = True
                raise

    except Exception as error:
        PIPELINE_STATUS = "failed"
        ERROR_MESSAGE = str(error)

        STEPS.append({
            "step": "pipeline_execution",
            "status": "failed",
            "error": ERROR_MESSAGE
        })

    duration = time.time() - start_time

    save_pipeline_report(config, started_at, duration)

    if PIPELINE_STATUS != "success":
        sys.exit(1)

    print("\nEND-TO-END MLOPS PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
