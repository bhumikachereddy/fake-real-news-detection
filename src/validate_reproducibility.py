"""
Validate reproducibility of Fake/Real News Detection.

The script trains the model twice using the same:
- Dataset
- Random seed
- Train/test split
- TF-IDF configuration
- SVM configuration

Then it compares the results.

Run from project root:

python src/validate_reproducibility.py
"""

from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from preprocess import load_dataset, prepare_features
from train import train_pipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42


def train_once(X, y):
    """
    Train the complete pipeline once.
    """

    results = train_pipeline(
        X,
        y,
        save_artifacts=False
    )

    model = results["model"]

    X_test_tfidf = results["X_test_tfidf"]
    y_test = results["y_test"]

    predictions = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    return results, metrics, predictions


def compare_results(
    metrics1,
    metrics2,
    predictions1,
    predictions2
):
    """
    Compare two training runs.
    """

    print("\n" + "=" * 70)
    print("REPRODUCIBILITY COMPARISON")
    print("=" * 70)

    all_same = True

    # -----------------------------------------------------
    # Compare metrics
    # -----------------------------------------------------

    print("\nMetric comparison:")

    for metric in metrics1:

        value1 = metrics1[metric]
        value2 = metrics2[metric]

        same = value1 == value2

        print(
            f"{metric:12s}: "
            f"Run 1 = {value1:.10f} | "
            f"Run 2 = {value2:.10f} | "
            f"Same = {same}"
        )

        if not same:
            all_same = False

    # -----------------------------------------------------
    # Compare predictions
    # -----------------------------------------------------

    predictions_same = (
        predictions1 == predictions2
    ).all()

    print(
        f"\nPredictions identical: "
        f"{predictions_same}"
    )

    if not predictions_same:
        all_same = False

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    if all_same:
        print("REPRODUCIBILITY VALIDATION PASSED")
        print("=" * 70)

        print(
            "\nThe two training runs produced "
            "identical results."
        )

    else:
        print("REPRODUCIBILITY VALIDATION FAILED")
        print("=" * 70)

        print(
            "\nThe two training runs produced "
            "different results."
        )

    return all_same


def main():

    print("=" * 70)
    print("Fake/Real News - Reproducibility Validation")
    print("=" * 70)

    print(
        f"\nRandom seed: {RANDOM_SEED}"
    )

    # -----------------------------------------------------
    # Load dataset ONCE
    # -----------------------------------------------------

    print("\n[1] Loading dataset...")

    df = load_dataset()

    X, y = prepare_features(df)

    # -----------------------------------------------------
    # First training run
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RUN 1")
    print("=" * 70)

    results1, metrics1, predictions1 = train_once(
        X,
        y
    )

    print("\nRun 1 results:")

    for metric, value in metrics1.items():
        print(f"{metric}: {value:.10f}")

    # -----------------------------------------------------
    # Second training run
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RUN 2")
    print("=" * 70)

    results2, metrics2, predictions2 = train_once(
        X,
        y
    )

    print("\nRun 2 results:")

    for metric, value in metrics2.items():
        print(f"{metric}: {value:.10f}")

    # -----------------------------------------------------
    # Compare
    # -----------------------------------------------------

    compare_results(
        metrics1,
        metrics2,
        predictions1,
        predictions2
    )


if __name__ == "__main__":
    main()