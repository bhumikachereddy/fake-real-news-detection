"""
Experiment 5: Data Validation and Reproducible ML Pipelines

Fake & Real News Detection

This module validates:
1. Dataset existence
2. Required columns
3. Datatypes
4. Missing values
5. Empty text records
6. Invalid labels
7. Duplicate records
8. Corrupted/invalid records
9. Dataset consistency
10. Dataset hash for reproducibility

Expected dataset:
data/processed/cleaned_news.csv

Expected columns:
- clean_text
- label

Label meaning:
0 = Fake News
1 = Real News
"""

from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_news.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

VALIDATION_REPORT_PATH = (
    RESULTS_DIR / "validation_report.json"
)


# Create results directory if it does not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED SCHEMA
# ============================================================

REQUIRED_COLUMNS = [
    "clean_text",
    "label"
]

EXPECTED_LABELS = {0, 1}


# ============================================================
# DATASET HASH
# ============================================================

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of the dataset file.

    The hash helps us identify whether the dataset
    has changed between different executions.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(data_path=DATA_PATH):
    """
    Load the Fake/Real News dataset.
    """

    data_path = Path(data_path)

    print("=" * 70)
    print("EXPERIMENT 5: DATA VALIDATION")
    print("=" * 70)

    print("\n[1] Checking dataset...")

    if not data_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{data_path}\n\n"
            "Please make sure cleaned_news.csv exists inside:\n"
            "data/processed/"
        )

    print(f"Dataset found: {data_path}")

    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df):
    """
    Validate that all required columns are present.
    """

    print("\n" + "=" * 70)
    print("[2] SCHEMA VALIDATION")
    print("=" * 70)

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in actual_columns
    ]

    extra_columns = [
        column
        for column in actual_columns
        if column not in REQUIRED_COLUMNS
    ]

    schema_valid = len(missing_columns) == 0

    print(f"Required columns: {REQUIRED_COLUMNS}")
    print(f"Actual columns  : {actual_columns}")

    if missing_columns:
        print(f"\nMissing columns: {missing_columns}")
    else:
        print("\nMissing columns: None")

    if extra_columns:
        print(f"Extra columns: {extra_columns}")
    else:
        print("Extra columns: None")

    if schema_valid:
        print("\nSchema validation: PASSED")
    else:
        print("\nSchema validation: FAILED")

    return {
        "passed": schema_valid,
        "required_columns": REQUIRED_COLUMNS,
        "actual_columns": actual_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns
    }


# ============================================================
# DATATYPE VALIDATION
# ============================================================

def validate_datatypes(df):
    """
    Validate datatypes of the required columns.

    clean_text should be string/object.
    label should contain integer-like values.
    """

    print("\n" + "=" * 70)
    print("[3] DATATYPE VALIDATION")
    print("=" * 70)

    datatype_errors = []

    if "clean_text" in df.columns:

        # Object/string is acceptable for text data
        if not (
            pd.api.types.is_object_dtype(df["clean_text"])
            or pd.api.types.is_string_dtype(df["clean_text"])
        ):
            datatype_errors.append(
                "clean_text must be a string/object datatype."
            )

    if "label" in df.columns:

        # Allow integer labels
        if not pd.api.types.is_integer_dtype(df["label"]):

            # Also allow values that can be safely converted to integers
            try:
                converted = pd.to_numeric(
                    df["label"],
                    errors="coerce"
                )

                invalid_conversion = converted.isna().sum()

                if invalid_conversion > 0:
                    datatype_errors.append(
                        "label contains values that cannot be converted "
                        "to numeric values."
                    )

            except Exception:
                datatype_errors.append(
                    "label datatype is invalid."
                )

    print("\nColumn datatypes:")

    for column in df.columns:
        print(f"{column:20s} : {df[column].dtype}")

    if datatype_errors:
        print("\nDatatype validation: FAILED")

        for error in datatype_errors:
            print(f"- {error}")

    else:
        print("\nDatatype validation: PASSED")

    return {
        "passed": len(datatype_errors) == 0,
        "datatypes": {
            column: str(df[column].dtype)
            for column in df.columns
        },
        "errors": datatype_errors
    }


# ============================================================
# MISSING VALUE VALIDATION
# ============================================================

def validate_missing_values(df):
    """
    Check for missing values in the required columns.
    """

    print("\n" + "=" * 70)
    print("[4] MISSING VALUE VALIDATION")
    print("=" * 70)

    missing_counts = {}

    for column in REQUIRED_COLUMNS:

        if column in df.columns:
            missing_counts[column] = int(
                df[column].isna().sum()
            )

    total_missing = sum(missing_counts.values())

    print("\nMissing values:")

    for column, count in missing_counts.items():
        print(f"{column:20s} : {count}")

    passed = total_missing == 0

    if passed:
        print("\nMissing value validation: PASSED")
    else:
        print("\nMissing value validation: FAILED")

    return {
        "passed": passed,
        "missing_counts": missing_counts,
        "total_missing": total_missing
    }


# ============================================================
# EMPTY TEXT VALIDATION
# ============================================================

def validate_empty_text(df):
    """
    Check whether clean_text contains empty or whitespace-only records.
    """

    print("\n" + "=" * 70)
    print("[5] EMPTY TEXT VALIDATION")
    print("=" * 70)

    if "clean_text" not in df.columns:

        return {
            "passed": False,
            "empty_text_records": None
        }

    text_series = df["clean_text"].fillna("").astype(str)

    empty_text_count = int(
        (text_series.str.strip() == "").sum()
    )

    print(
        f"Empty/whitespace-only text records: "
        f"{empty_text_count}"
    )

    passed = empty_text_count == 0

    if passed:
        print("\nEmpty text validation: PASSED")
    else:
        print("\nEmpty text validation: FAILED")

    return {
        "passed": passed,
        "empty_text_records": empty_text_count
    }


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(df):
    """
    Validate the target labels.

    Expected:
    0 = Fake News
    1 = Real News
    """

    print("\n" + "=" * 70)
    print("[6] LABEL VALIDATION")
    print("=" * 70)

    if "label" not in df.columns:

        return {
            "passed": False,
            "invalid_labels": None,
            "unique_labels": []
        }

    numeric_labels = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    invalid_numeric_count = int(
        numeric_labels.isna().sum()
    )

    unique_labels = sorted(
        numeric_labels.dropna().unique().tolist()
    )

    invalid_labels = [
        label
        for label in unique_labels
        if label not in EXPECTED_LABELS
    ]

    total_invalid = (
        invalid_numeric_count
        + len(invalid_labels)
    )

    print(f"Unique labels found: {unique_labels}")
    print("Expected labels:")
    print("0 = Fake News")
    print("1 = Real News")

    print(
        f"\nInvalid/non-numeric label records: "
        f"{total_invalid}"
    )

    passed = total_invalid == 0

    if passed:
        print("\nLabel validation: PASSED")
    else:
        print("\nLabel validation: FAILED")

        if invalid_labels:
            print(
                f"Invalid label values: {invalid_labels}"
            )

    return {
        "passed": passed,
        "unique_labels": unique_labels,
        "expected_labels": [0, 1],
        "invalid_labels": invalid_labels,
        "invalid_label_records": total_invalid
    }


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validate_duplicates(df):
    """
    Check for duplicate records.
    """

    print("\n" + "=" * 70)
    print("[7] DUPLICATE RECORD VALIDATION")
    print("=" * 70)

    duplicate_count = int(
        df.duplicated().sum()
    )

    print(f"Duplicate records: {duplicate_count}")

    passed = duplicate_count == 0

    if passed:
        print("\nDuplicate validation: PASSED")
    else:
        print("\nDuplicate validation: WARNING")
        print(
            "Duplicate records exist and should be reviewed."
        )

    return {
        "passed": passed,
        "duplicate_records": duplicate_count
    }


# ============================================================
# CORRUPTED RECORD VALIDATION
# ============================================================

def validate_corrupted_records(df):
    """
    Detect obviously corrupted or invalid records.

    Checks:
    - Text must contain at least one character.
    - Label must be 0 or 1.
    """

    print("\n" + "=" * 70)
    print("[8] CORRUPTED RECORD VALIDATION")
    print("=" * 70)

    corrupted_records = 0

    if "clean_text" not in df.columns:
        corrupted_records += len(df)

    else:
        text_invalid = (
            df["clean_text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        corrupted_records += int(
            text_invalid.sum()
        )

    if "label" in df.columns:

        numeric_labels = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

        label_invalid = (
            numeric_labels.isna()
            | ~numeric_labels.isin(EXPECTED_LABELS)
        )

        corrupted_records += int(
            label_invalid.sum()
        )

    else:
        corrupted_records += len(df)

    print(
        f"Potentially corrupted/invalid records: "
        f"{corrupted_records}"
    )

    passed = corrupted_records == 0

    if passed:
        print("\nCorrupted record validation: PASSED")
    else:
        print("\nCorrupted record validation: FAILED")

    return {
        "passed": passed,
        "corrupted_records": corrupted_records
    }


# ============================================================
# DATA CONSISTENCY VALIDATION
# ============================================================

def validate_consistency(df):
    """
    Validate overall dataset consistency.

    Checks:
    - Dataset is not empty.
    - Required columns exist.
    - Number of rows is greater than zero.
    """

    print("\n" + "=" * 70)
    print("[9] DATA CONSISTENCY VALIDATION")
    print("=" * 70)

    row_count = len(df)

    has_required_columns = all(
        column in df.columns
        for column in REQUIRED_COLUMNS
    )

    dataset_not_empty = row_count > 0

    passed = (
        dataset_not_empty
        and has_required_columns
    )

    print(f"Number of records: {row_count}")
    print(
        f"Required columns present: "
        f"{has_required_columns}"
    )

    if passed:
        print("\nData consistency validation: PASSED")
    else:
        print("\nData consistency validation: FAILED")

    return {
        "passed": passed,
        "row_count": row_count,
        "required_columns_present": has_required_columns
    }


# ============================================================
# DATASET HASH
# ============================================================

def generate_dataset_hash(data_path=DATA_PATH):
    """
    Generate SHA-256 hash of the dataset.

    If the same dataset is used repeatedly,
    the hash should remain unchanged.
    """

    print("\n" + "=" * 70)
    print("[10] DATASET HASH")
    print("=" * 70)

    dataset_hash = calculate_file_hash(data_path)

    print(f"SHA-256: {dataset_hash}")

    return dataset_hash


# ============================================================
# FINAL VALIDATION
# ============================================================

def run_validation(data_path=DATA_PATH):
    """
    Run all validation checks and generate a report.
    """

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset(data_path)

    # --------------------------------------------------------
    # Calculate dataset hash
    # --------------------------------------------------------

    dataset_hash = generate_dataset_hash(data_path)

    # --------------------------------------------------------
    # Run validation checks
    # --------------------------------------------------------

    schema_result = validate_schema(df)

    datatype_result = validate_datatypes(df)

    missing_result = validate_missing_values(df)

    empty_text_result = validate_empty_text(df)

    label_result = validate_labels(df)

    duplicate_result = validate_duplicates(df)

    corrupted_result = validate_corrupted_records(df)

    consistency_result = validate_consistency(df)

    # --------------------------------------------------------
    # Overall validation
    # --------------------------------------------------------

    critical_checks = [
        schema_result["passed"],
        datatype_result["passed"],
        missing_result["passed"],
        empty_text_result["passed"],
        label_result["passed"],
        corrupted_result["passed"],
        consistency_result["passed"]
    ]

    overall_passed = all(critical_checks)

    # --------------------------------------------------------
    # Create validation report
    # --------------------------------------------------------

    report = {
        "experiment": "Experiment 5 - Data Validation and Reproducible ML Pipelines",

        "dataset": {
            "path": str(data_path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "sha256": dataset_hash
        },

        "schema_validation": schema_result,

        "datatype_validation": datatype_result,

        "missing_value_validation": missing_result,

        "empty_text_validation": empty_text_result,

        "label_validation": label_result,

        "duplicate_validation": duplicate_result,

        "corrupted_record_validation": corrupted_result,

        "consistency_validation": consistency_result,

        "overall_validation": {
            "passed": overall_passed
        }
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        VALIDATION_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if overall_passed:

        print("\nDATA VALIDATION PASSED")

    else:

        print("\nDATA VALIDATION FAILED")

        print(
            "\nPlease review the validation messages above."
        )

    print(
        f"\nValidation report saved to:\n"
        f"{VALIDATION_REPORT_PATH}"
    )

    print(
        f"\nDataset SHA-256:\n"
        f"{dataset_hash}"
    )

    return report


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_validation()