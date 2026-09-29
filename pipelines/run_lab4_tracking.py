"""
Lab 4: MLflow Experiment Tracking

Run from project root:

python pipelines/run_lab4_tracking.py
"""

import sys
from pathlib import Path

import mlflow


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
    print("Starting Lab 4: MLflow Experiment Tracking")
    print("=" * 70)

    # ---------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------

    with mlflow.start_run() as run:

        print(f"\nMLflow Run ID: {run.info.run_id}")

        # -----------------------------------------------------
        # STEP 1: Load dataset
        # -----------------------------------------------------

        print("\n[STEP 1] Loading dataset...")

        df = load_dataset()

        X, y = prepare_features(df)

        # -----------------------------------------------------
        # STEP 2: Train model
        # -----------------------------------------------------

        print("\n[STEP 2] Training model...")

        results = train_pipeline(
            X,
            y,
            save_artifacts=True
        )

        model = results["model"]
        tfidf = results["tfidf"]
        tuning = results["tuning"]

        # -----------------------------------------------------
        # STEP 3: Evaluate model
        # -----------------------------------------------------

        print("\n[STEP 3] Evaluating model...")

        metrics, confusion_matrix, predictions = evaluate_model(
            model,
            results["X_test_tfidf"],
            results["y_test"]
        )

        # -----------------------------------------------------
        # STEP 4: Log parameters
        # -----------------------------------------------------

        print("\n[STEP 4] Logging parameters...")

        mlflow.log_param(
            "model_type",
            "LinearSVC"
        )

        mlflow.log_param(
            "best_C",
            tuning.best_params_["C"]
        )

        mlflow.log_param(
            "tfidf_max_features",
            50000
        )

        mlflow.log_param(
            "tfidf_ngram_range",
            "(1, 2)"
        )

        mlflow.log_param(
            "tfidf_min_df",
            2
        )

        mlflow.log_param(
            "tfidf_max_df",
            0.95
        )

        mlflow.log_param(
            "tfidf_stop_words",
            "english"
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "cv_folds",
            5
        )

        # -----------------------------------------------------
        # STEP 5: Log metrics
        # -----------------------------------------------------

        print("\n[STEP 5] Logging metrics...")

        mlflow.log_metric(
            "cv_f1_score",
            float(tuning.best_score_)
        )

        mlflow.log_metric(
            "test_accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "test_precision",
            metrics["precision"]
        )

        mlflow.log_metric(
            "test_recall",
            metrics["recall"]
        )

        mlflow.log_metric(
            "test_f1_score",
            metrics["f1_score"]
        )

        # -----------------------------------------------------
        # STEP 6: Save metrics
        # -----------------------------------------------------

        metrics_path = save_metrics(
            metrics,
            confusion_matrix
        )

        # -----------------------------------------------------
        # STEP 7: Log model artifacts
        # -----------------------------------------------------

        print("\n[STEP 6] Logging model artifacts...")

        model_path = PROJECT_ROOT / "models" / "final_svm_model.pkl"

        vectorizer_path = (
            PROJECT_ROOT /
            "models" /
            "tfidf_vectorizer.pkl"
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model"
        )

        mlflow.log_artifact(
            str(vectorizer_path),
            artifact_path="vectorizer"
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation"
        )

        # -----------------------------------------------------
        # STEP 8: Print final results
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("LAB 4 PIPELINE FULLY EXECUTED")
        print("=" * 70)

        print("\nFinal Metrics:")

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

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        print("\nTracked artifacts:")
        print("- SVM model")
        print("- TF-IDF vectorizer")
        print("- Evaluation metrics")

        print("\nOpen MLflow UI using:")
        print("mlflow ui")


if __name__ == "__main__":
    main()