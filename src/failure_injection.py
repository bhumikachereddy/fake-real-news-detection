
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_PATH = ARTIFACT_DIR / "failure_injection_report.json"


def load_artifact(path):
    """Load a model or vectorizer and detect missing/corrupted files."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    try:
        return joblib.load(path)
    except Exception as error:
        raise ValueError(f"Artifact is corrupted: {error}")


def validate_text(text):
    """Validate news input."""

    if not isinstance(text, str):
        raise TypeError("News input must be a string")

    if not text.strip():
        raise ValueError("News input cannot be empty")

    return True


def validate_dataset(path):
    """Validate dataset existence and required columns."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as error:
        raise ValueError(f"Unable to read dataset: {error}")

    required_columns = {"clean_text", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset missing required columns: {missing_columns}"
        )

    return True


def predict_news(text):
    """Load trained artifacts and predict Fake or Real news."""

    validate_text(text)

    model = load_artifact(MODEL_PATH)
    vectorizer = load_artifact(VECTORIZER_PATH)

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)

    return int(prediction[0])


def run_scenario(name, function, expected_failure=False):
    """Execute one failure-injection scenario."""

    result = {
        "scenario": name,
        "status": "FAILED",
        "expected_failure": expected_failure,
        "message": ""
    }

    try:
        output = function()

        if expected_failure:
            result["message"] = (
                "Expected failure did not occur"
            )
        else:
            result["status"] = "PASSED"
            result["message"] = f"Executed successfully: {output}"

    except Exception as error:

        if expected_failure:
            result["status"] = "PASSED"
            result["message"] = (
                f"Expected failure detected: {error}"
            )
        else:
            result["message"] = (
                f"Unexpected failure: {error}"
            )

    return result


def run_failure_injection():

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Scenario 1: Missing model artifact
    results.append(
        run_scenario(
            "Missing Model Artifact",
            lambda: load_artifact(
                BASE_DIR / "models" / "missing_model.pkl"
            ),
            expected_failure=True
        )
    )

    # Scenario 2: Corrupted model artifact
    def corrupted_model_test():

        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            delete=False
        ) as temp_file:

            temp_file.write(b"This is corrupted model data")
            temp_path = temp_file.name

        try:
            return load_artifact(temp_path)
        finally:
            os.remove(temp_path)

    results.append(
        run_scenario(
            "Corrupted Model Artifact",
            corrupted_model_test,
            expected_failure=True
        )
    )

    # Scenario 3: Invalid input datatype
    results.append(
        run_scenario(
            "Invalid Input Data Type",
            lambda: validate_text(12345),
            expected_failure=True
        )
    )

    # Scenario 4: Empty news input
    results.append(
        run_scenario(
            "Empty News Input",
            lambda: validate_text("   "),
            expected_failure=True
        )
    )

    # Scenario 5: Missing dataset
    results.append(
        run_scenario(
            "Missing Dataset",
            lambda: validate_dataset(
                BASE_DIR / "data" / "missing_dataset.csv"
            ),
            expected_failure=True
        )
    )

    # Scenario 6: Invalid dataset schema
    def invalid_schema_test():

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
            newline=""
        ) as temp_file:

            pd.DataFrame({
                "wrong_column": ["sample news"],
                "category": [0]
            }).to_csv(temp_file.name, index=False)

            temp_path = temp_file.name

        try:
            return validate_dataset(temp_path)
        finally:
            os.remove(temp_path)

    results.append(
        run_scenario(
            "Invalid Dataset Schema",
            invalid_schema_test,
            expected_failure=True
        )
    )

    # Scenario 7: Actual model recovery and prediction
    sample_news = (
        "The government announced a new development "
        "programme for the country."
    )

    results.append(
        run_scenario(
            "Model Recovery and Prediction",
            lambda: predict_news(sample_news),
            expected_failure=False
        )
    )

    # Calculate report summary
    passed = sum(
        1 for result in results
        if result["status"] == "PASSED"
    )

    failed = len(results) - passed

    report = {
        "project": "Fake and Real News Detection",
        "lab": "Lab 9 - Failure Injection and Reliability Testing",
        "execution_time": datetime.now().isoformat(),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": (
            "PASSED" if failed == 0 else "FAILED"
        ),
        "results": results
    }

    # Save JSON report
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Display results
    print("\n===== LAB 9 FAILURE INJECTION REPORT =====")

    for result in results:
        print(
            f"{result['scenario']}: "
            f"{result['status']} - {result['message']}"
        )

    print("\nTotal scenarios:", report["total_scenarios"])
    print("Passed:", report["passed"])
    print("Failed:", report["failed"])
    print("Overall status:", report["overall_status"])

    print("\nReport saved to:", REPORT_PATH)

    if failed > 0:
        raise RuntimeError(
            "Failure injection testing did not pass all scenarios"
        )

    return report


if __name__ == "__main__":
    run_failure_injection()
