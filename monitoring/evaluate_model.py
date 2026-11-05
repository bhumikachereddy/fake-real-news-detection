
"""
Model Evaluation and Degradation Analysis
Lab 12 - Experiment 12.5
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = ROOT / "data" / "processed" / "test_data.csv"
MODEL_FILE = ROOT / "models" / "final_svm_model.pkl"
VECTORIZER_FILE = ROOT / "models" / "tfidf_vectorizer.pkl"
REPORT_FILE = ROOT / "artifacts" / "model_evaluation_report.json"


def evaluate():

    data = pd.read_csv(TEST_FILE)

    required = {"clean_text", "label"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"Test dataset must contain: {required}"
        )

    data = data.dropna(subset=["clean_text", "label"])
    data["label"] = pd.to_numeric(
        data["label"], errors="coerce"
    )
    data = data.dropna(subset=["label"])
    data = data[data["label"].isin([0, 1])]

    if data.empty:
        raise ValueError("No valid test records.")

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    X = vectorizer.transform(
        data["clean_text"].astype(str)
    )
    y_true = data["label"].astype(int)

    y_pred = model.predict(X)

    report = {
        "evaluation": "Model Evaluation",
        "test_records": int(len(data)),
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "precision": round(
            precision_score(
                y_true, y_pred, zero_division=0
            ), 4
        ),
        "recall": round(
            recall_score(
                y_true, y_pred, zero_division=0
            ), 4
        ),
        "f1_score": round(
            f1_score(
                y_true, y_pred, zero_division=0
            ), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Fake News", "Real News"],
            output_dict=True,
            zero_division=0
        )
    }

    REPORT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print("\nEvaluation report saved:", REPORT_FILE)


if __name__ == "__main__":
    evaluate()
