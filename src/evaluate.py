"""
Evaluation module for Fake and Real News Detection.

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
"""

from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the trained model on the unseen test set.
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Generate predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"],
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1 Score : {f1:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    return metrics, cm, y_pred


def save_metrics(metrics, cm):
    """
    Save evaluation results as JSON.
    """

    metrics_path = RESULTS_DIR / "metrics.json"

    results = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist()
    }

    with open(metrics_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nMetrics saved to: {metrics_path}")

    return metrics_path


if __name__ == "__main__":
    from preprocess import load_dataset, prepare_features
    from train import train_pipeline

    df = load_dataset()
    X, y = prepare_features(df)

    results = train_pipeline(X, y)

    metrics, cm, y_pred = evaluate_model(
        results["model"],
        results["X_test_tfidf"],
        results["y_test"]
    )

    save_metrics(metrics, cm)

    print("\nEvaluation completed successfully.")