
import json
import tempfile
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
DATASET_PATH = BASE_DIR / "data" / "processed" / "cleaned_news.csv"

REPORT_PATH = BASE_DIR / "artifacts" / "failure_injection_report.json"

REQUIRED_COLUMNS = ["clean_text", "label"]


def load_artifact(path):
    """Load a model or vectorizer safely."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    return joblib.load(path)


def validate_text(text):
    """Validate input news text."""
    if not isinstance(text, str):
        raise TypeError("News input must be a string")

    if not text.strip():
        raise ValueError("News input cannot be empty")

    return text.strip()


def validate_dataset(path):
    """Check dataset availability and required columns."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError("Dataset file is missing")

    df = pd.read_csv(path, nrows=5)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    return True


def predict_news(text):
    """Load the actual model and predict news."""
    text = validate_text(text)

    vectorizer = load_artifact(VECTORIZER_PATH)
    model = load_artifact(MODEL_PATH)

    features = vectorizer.transform([text])
    prediction = model.predict(features)

    return int(prediction[0])


def run_scenario(name, operation, expected_exception=None):
    """Execute a failure scenario and record its result."""
    try:
        result = operation()

        return {
            "scenario": name,
            "status": "PASSED" if expected_exception is None else "FAILED",
            "failure_detected": False,
            "message": "Operation completed successfully",
            "result": str(result)
        }

    except Exception as error:
        detected = (
            expected_exception is not None
            and isinstance(error, expected_exception)
        )

        return {
            "scenario": name,
            "status": "PASSED" if detected else "FAILED",
            "failure_detected": True,
            "message": str(error),
            "error_type": type(error).__name__
        }


def run_failure_injection():

    results = []

    # Scenario 1: Missing model artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_model = Path(temp_dir) / "missing_model.pkl"

        results.append(
            run_scenario(
                "Missing model artifact",
                lambda: load_artifact(missing_model),
                FileNotFoundError
            )
        )

    # Scenario 2: Corrupted model artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        corrupt_model = Path(temp_dir) / "corrupt_model.pkl"
        corrupt_model.write_bytes(b"INVALID PICKLE CONTENT")

        results.append(
            run_scenario(
                "Corrupted model artifact",
                lambda: load_artifact(corrupt_model),
                Exception
            )
        )

    # Scenario 3: Invalid input data type
    results.append(
        run_scenario(
            "Invalid input data type",
            lambda: validate_text(12345),
            TypeError
        )
    )

    # Scenario 4: Empty news input
    results.append(
        run_scenario(
            "Empty news input",
            lambda: validate_text("   "),
            ValueError
        )
    )

    # Scenario 5: Missing dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_dataset = Path(temp_dir) / "missing.csv"

        results.append(
            run_scenario(
                "Missing dataset",
                lambda: validate_dataset(missing_dataset),
                FileNotFoundError
            )
        )

    # Scenario 6: Dataset with missing columns
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_dataset = Path(temp_dir) / "invalid.csv"
        pd.DataFrame({
            "title": ["Test news"],
            "text": ["Sample content"]
        }).to_csv(invalid_dataset, index=False)

        results.append(
            run_scenario(
                "Dataset schema validation",
                lambda: validate_dataset(invalid_dataset),
                ValueError
            )
        )

    # Scenario 7: Actual model recovery and prediction
    sample_text = (
        "The government announced a new development initiative."
    )

    results.append(
        run_scenario(
            "Model recovery and prediction",
            lambda: predict_news(sample_text)
        )
    )

    passed = sum(
        result["status"] == "PASSED" for result in results
    )

    failed = len(results) - passed

    report = {
        "lab": "Lab 9 - Failure Injection and Reliability Testing",
        "timestamp": datetime.now().isoformat(),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": "PASSED" if failed == 0 else "FAILED",
        "scenarios": results
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== LAB 9 FAILURE INJECTION REPORT ==========")

    for result in results:
        print(
            f"{result['scenario']}: {result['status']}"
        )

    print("\nTotal scenarios:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)
    print("Overall status:", report["overall_status"])
    print("\nReport saved at:", REPORT_PATH)

    return report


if __name__ == "__main__":
    run_failure_injection()
