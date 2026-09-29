"""
Experiment 5: Idempotency and Reproducibility Validation

Fake & Real News Detection

This script runs the data validation process twice using
the same dataset and compares the results.

The validation is considered reproducible when:

1. Dataset hash is identical
2. Dataset shape is identical
3. Schema validation result is identical
4. Datatype validation result is identical
5. Missing-value result is identical
6. Empty-text result is identical
7. Label validation result is identical
8. Corrupted-record result is identical
9. Overall validation result is identical

Run from project root:

python src/validate_idempotency.py
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT VALIDATION FUNCTION
# ============================================================

from validate_data import run_validation


# ============================================================
# COMPARE RESULTS
# ============================================================

def compare_results(report1, report2):

    print("\n" + "=" * 70)
    print("REPRODUCIBILITY COMPARISON")
    print("=" * 70)

    all_same = True

    # --------------------------------------------------------
    # Dataset hash
    # --------------------------------------------------------

    hash1 = report1["dataset"]["sha256"]
    hash2 = report2["dataset"]["sha256"]

    hash_same = hash1 == hash2

    print("\nDataset hash:")
    print(f"Run 1: {hash1}")
    print(f"Run 2: {hash2}")
    print(f"Same : {hash_same}")

    if not hash_same:
        all_same = False

    # --------------------------------------------------------
    # Dataset shape
    # --------------------------------------------------------

    rows1 = report1["dataset"]["rows"]
    rows2 = report2["dataset"]["rows"]

    columns1 = report1["dataset"]["columns"]
    columns2 = report2["dataset"]["columns"]

    shape_same = (
        rows1 == rows2
        and columns1 == columns2
    )

    print("\nDataset shape:")
    print(
        f"Run 1: {rows1} rows x {columns1} columns"
    )
    print(
        f"Run 2: {rows2} rows x {columns2} columns"
    )
    print(f"Same : {shape_same}")

    if not shape_same:
        all_same = False

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    checks = [
        (
            "Schema validation",
            "schema_validation"
        ),
        (
            "Datatype validation",
            "datatype_validation"
        ),
        (
            "Missing value validation",
            "missing_value_validation"
        ),
        (
            "Empty text validation",
            "empty_text_validation"
        ),
        (
            "Label validation",
            "label_validation"
        ),
        (
            "Corrupted record validation",
            "corrupted_record_validation"
        ),
        (
            "Data consistency validation",
            "consistency_validation"
        ),
        (
            "Overall validation",
            "overall_validation"
        )
    ]

    print("\nValidation comparison:")

    for display_name, key in checks:

        result1 = report1[key]["passed"]
        result2 = report2[key]["passed"]

        same = result1 == result2

        print(
            f"{display_name:35s} "
            f"Run 1 = {result1} | "
            f"Run 2 = {result2} | "
            f"Same = {same}"
        )

        if not same:
            all_same = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if all_same:

        print("REPRODUCIBILITY VALIDATION PASSED")
        print("=" * 70)

        print(
            "\nThe validation process produced identical "
            "results across both executions."
        )

        print(
            "\nThe dataset and validation workflow are "
            "reproducible."
        )

    else:

        print("REPRODUCIBILITY VALIDATION FAILED")
        print("=" * 70)

        print(
            "\nThe two executions produced different results."
        )

    return all_same


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 5: IDEMPOTENCY & REPRODUCIBILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # RUN 1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RUN 1")
    print("=" * 70)

    report1 = run_validation()

    # --------------------------------------------------------
    # RUN 2
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RUN 2")
    print("=" * 70)

    report2 = run_validation()

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    result = compare_results(
        report1,
        report2
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if result:
        print(
            "EXPERIMENT 5 REPRODUCIBILITY CHECK COMPLETED"
        )
    else:
        print(
            "EXPERIMENT 5 REPRODUCIBILITY CHECK FAILED"
        )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()