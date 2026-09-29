"""
Experiment 5: Data Validation Pipeline

Fake & Real News Detection

This pipeline:
1. Loads the dataset
2. Validates schema
3. Validates datatypes
4. Checks missing values
5. Checks empty text
6. Validates labels
7. Checks duplicates
8. Checks corrupted records
9. Checks data consistency
10. Generates a dataset hash
11. Saves validation report

Run from project root:

python pipelines/run_data_validation.py
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT VALIDATION MODULE
# ============================================================

from validate_data import run_validation


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 5: DATA VALIDATION PIPELINE")
    print("=" * 70)

    print("\n[STEP 1] Starting data validation...")

    report = run_validation()

    print("\n[STEP 2] Checking final validation status...")

    validation_status = (
        report["overall_validation"]["passed"]
    )

    print("\n" + "=" * 70)

    if validation_status:

        print("DATA VALIDATION PIPELINE COMPLETED SUCCESSFULLY")

    else:

        print("DATA VALIDATION PIPELINE COMPLETED WITH ERRORS")

    print("=" * 70)

    print("\nValidation Summary:")

    print(
        f"Dataset rows: "
        f"{report['dataset']['rows']}"
    )

    print(
        f"Dataset columns: "
        f"{report['dataset']['columns']}"
    )

    print(
        f"Schema validation: "
        f"{report['schema_validation']['passed']}"
    )

    print(
        f"Datatype validation: "
        f"{report['datatype_validation']['passed']}"
    )

    print(
        f"Missing value validation: "
        f"{report['missing_value_validation']['passed']}"
    )

    print(
        f"Empty text validation: "
        f"{report['empty_text_validation']['passed']}"
    )

    print(
        f"Label validation: "
        f"{report['label_validation']['passed']}"
    )

    print(
        f"Corrupted record validation: "
        f"{report['corrupted_record_validation']['passed']}"
    )

    print(
        f"Data consistency validation: "
        f"{report['consistency_validation']['passed']}"
    )

    print(
        f"\nOverall validation: "
        f"{validation_status}"
    )

    print(
        "\nValidation report:"
        "\nresults/validation_report.json"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()