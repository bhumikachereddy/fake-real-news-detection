"""
Train Fake/Real News Detection model with MLflow tracking.

Run from project root:

python src/train_mlflow.py
"""

from pathlib import Path
import sys
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from preprocess import load_dataset, prepare_features
from train import train_pipeline
from evaluate import evaluate_model, save_metrics


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

EXPERIMENT_NAME = "Fake_Real_News_Detection"

mlflow.set_experiment(EXPERIMENT_NAME)


def main():

    print("=" * 70)
    print("Fake/Real News - MLflow Training")
    print("=" * 70)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\n[1] Loading dataset...")

    df = load_dataset()

    X, y = prepare_features(df)

    # -----------------------------------------------------
    # Start MLflow run
    # -----------------------------------------------------

    with mlflow.start_run() as run:

        print(f"\nMLflow Run ID: {run.info.run_id}")

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        print("\n[2] Training model...")

        results = train_pipeline(
            X,
            y,
            save_artifacts=True
        )

        model = results["model"]
        tfidf = results["tfidf"]
        tuning = results["tuning"]

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------

        print("\n[3] Evaluating model...")

        metrics, cm, predictions = evaluate_model(
            model,
            results["X_test_tfidf"],
            results["y_test"]
        )

        # -------------------------------------------------
        # Log parameters
        # -------------------------------------------------

        print("\n[4] Logging parameters...")

        mlflow.log_params({
            "model": "LinearSVC",
            "best_C": tuning.best_params_["C"],
            "tfidf_max_features": 50000,
            "ngram_range": "(1, 2)",
            "min_df": 2,
            "max_df": 0.95,
            "test_size": 0.20,
            "random_state": 42,
            "cv_folds": 5
        })

        # -------------------------------------------------
        # Log metrics
        # -------------------------------------------------

        print("\n[5] Logging metrics...")

        mlflow.log_metrics({
            "cv_f1_score": float(tuning.best_score_),
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1_score": metrics["f1_score"]
        })

        # -------------------------------------------------
        # Save evaluation metrics
        # -------------------------------------------------

        metrics_path = save_metrics(
            metrics,
            cm
        )

        # -------------------------------------------------
        # Log model using MLflow
        # -------------------------------------------------

        print("\n[6] Logging model...")

        mlflow.sklearn.log_model(
            model,
            name="fake_real_news_svm_model"
        )

        # -------------------------------------------------
        # Log TF-IDF vectorizer
        # -------------------------------------------------

        vectorizer_path = (
            PROJECT_ROOT /
            "models" /
            "tfidf_vectorizer.pkl"
        )

        mlflow.log_artifact(
            str(vectorizer_path),
            artifact_path="vectorizer"
        )

        # -------------------------------------------------
        # Log metrics file
        # -------------------------------------------------

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation"
        )

        # -------------------------------------------------
        # Final output
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print("MLFLOW TRAINING COMPLETED")
        print("=" * 70)

        print(f"\nRun ID: {run.info.run_id}")

        print("\nMetrics:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

        print(
            f"\nCross-validation F1: "
            f"{tuning.best_score_:.6f}"
        )

        print(
            f"Best C: "
            f"{tuning.best_params_['C']}"
        )


if __name__ == "__main__":
    main()