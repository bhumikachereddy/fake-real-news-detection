"""
Lab 3: Baseline Machine Learning Pipeline

Run from project root:

python pipelines/run_lab3_baseline.py
"""

import sys
from pathlib import Path


# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from preprocess import load_dataset, prepare_features
from train import train_pipeline
from evaluate import evaluate_model, save_metrics


def main():

    print("=" * 70)
    print("Starting Lab 3: Fake/Real News Baseline Pipeline")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: Load and preprocess dataset
    # ---------------------------------------------------------

    print("\n[STEP 1] Loading dataset...")

    df = load_dataset()

    X, y = prepare_features(df)

    # ---------------------------------------------------------
    # STEP 2: Train model
    # ---------------------------------------------------------

    print("\n[STEP 2] Training Linear SVM...")

    results = train_pipeline(
        X,
        y,
        save_artifacts=True
    )

    # ---------------------------------------------------------
    # STEP 3: Evaluate model
    # ---------------------------------------------------------

    print("\n[STEP 3] Evaluating model...")

    metrics, confusion_matrix, predictions = evaluate_model(
        results["model"],
        results["X_test_tfidf"],
        results["y_test"]
    )

    # ---------------------------------------------------------
    # STEP 4: Save metrics
    # ---------------------------------------------------------

    save_metrics(
        metrics,
        confusion_matrix
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LAB 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nFinal Test Metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nArtifacts:")
    print("models/final_svm_model.pkl")
    print("models/tfidf_vectorizer.pkl")
    print("results/metrics.json")


if __name__ == "__main__":
    main()