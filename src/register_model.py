"""
Experiment 6: Model Registry and Version Management

Fake & Real News Detection

This script:
1. Loads the dataset
2. Trains the TF-IDF + Linear SVM model
3. Evaluates the model
4. Logs the model to MLflow
5. Registers the model
6. Logs preprocessing configuration
7. Logs evaluation metrics
8. Logs the TF-IDF vectorizer
9. Creates a registered model version

Run from project root:

python src/register_model.py
"""

from pathlib import Path
import sys

import mlflow
import mlflow.sklearn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from preprocess import (
    load_dataset,
    prepare_features
)

from train import train_pipeline

from evaluate import (
    evaluate_model,
    save_metrics
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

EXPERIMENT_NAME = (
    "Fake_Real_News_Detection"
)

REGISTERED_MODEL_NAME = (
    "Fake_Real_News_SVM"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 6: MODEL REGISTRY")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Load dataset
    # --------------------------------------------------------

    print("\n[STEP 1] Loading dataset...")

    df = load_dataset()

    X, y = prepare_features(df)

    # --------------------------------------------------------
    # STEP 2: Start MLflow
    # --------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # STEP 3: Train model
        # ----------------------------------------------------

        print(
            "\n[STEP 2] Training model..."
        )

        results = train_pipeline(
            X,
            y,
            save_artifacts=True
        )

        model = results["model"]

        tfidf = results["tfidf"]

        tuning = results["tuning"]

        # ----------------------------------------------------
        # STEP 4: Evaluate model
        # ----------------------------------------------------

        print(
            "\n[STEP 3] Evaluating model..."
        )

        metrics, cm, predictions = (
            evaluate_model(
                model,
                results["X_test_tfidf"],
                results["y_test"]
            )
        )

        # ----------------------------------------------------
        # STEP 5: Log parameters
        # ----------------------------------------------------

        print(
            "\n[STEP 4] Logging parameters..."
        )

        mlflow.log_params({

            "model_type":
                "LinearSVC",

            "best_C":
                tuning.best_params_["C"],

            "tfidf_max_features":
                50000,

            "tfidf_ngram_range":
                "(1, 2)",

            "tfidf_min_df":
                2,

            "tfidf_max_df":
                0.95,

            "tfidf_stop_words":
                "english",

            "test_size":
                0.20,

            "random_state":
                42,

            "cv_folds":
                5
        })

        # ----------------------------------------------------
        # STEP 6: Log metrics
        # ----------------------------------------------------

        print(
            "\n[STEP 5] Logging metrics..."
        )

        mlflow.log_metrics({

            "cv_f1_score":
                float(tuning.best_score_),

            "test_accuracy":
                float(metrics["accuracy"]),

            "test_precision":
                float(metrics["precision"]),

            "test_recall":
                float(metrics["recall"]),

            "test_f1_score":
                float(metrics["f1_score"])
        })

        # ----------------------------------------------------
        # STEP 7: Save evaluation metrics
        # ----------------------------------------------------

        metrics_path = save_metrics(
            metrics,
            cm
        )

        # ----------------------------------------------------
        # STEP 8: Register model
        # ----------------------------------------------------

        print(
            "\n[STEP 6] Registering model..."
        )

        model_info = (
            mlflow.sklearn.log_model(

                sk_model=model,

                name="fake_real_news_model",

                registered_model_name=
                    REGISTERED_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # STEP 9: Log vectorizer
        # ----------------------------------------------------

        print(
            "\n[STEP 7] Logging TF-IDF vectorizer..."
        )

        vectorizer_path = (
            PROJECT_ROOT
            / "models"
            / "tfidf_vectorizer.pkl"
        )

        mlflow.log_artifact(
            str(vectorizer_path),
            artifact_path="vectorizer"
        )

        # ----------------------------------------------------
        # STEP 10: Log evaluation file
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation"
        )

        # ----------------------------------------------------
        # STEP 11: Add tags
        # ----------------------------------------------------

        print(
            "\n[STEP 8] Adding model metadata..."
        )

        mlflow.set_tags({

            "project":
                "Fake and Real News Detection",

            "model_type":
                "Linear SVM",

            "feature_extraction":
                "TF-IDF",

            "dataset":
                "cleaned_news.csv",

            "task":
                "Binary Text Classification",

            "framework":
                "scikit-learn"
        })

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("MODEL REGISTRY COMPLETED")
        print("=" * 70)

        print(
            f"\nRegistered Model:"
            f"\n{REGISTERED_MODEL_NAME}"
        )

        print(
            f"\nRun ID:"
            f"\n{run.info.run_id}"
        )

        print(
            f"\nModel URI:"
            f"\n{model_info.model_uri}"
        )

        print("\nMetrics:")

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.6f}"
            )

        print(
            f"\nBest C: "
            f"{tuning.best_params_['C']}"
        )

        print(
            "\nModel version has been "
            "created in MLflow Registry."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()