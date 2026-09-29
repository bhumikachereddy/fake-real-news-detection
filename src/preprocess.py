"""
Preprocessing module for Fake and Real News Detection.

This module:
1. Loads the processed news dataset.
2. Validates required columns.
3. Removes missing/empty text records.
4. Returns text features and labels.

Expected dataset:
data/processed/cleaned_news.csv

Expected columns:
- clean_text
- label

Labels:
0 = Fake News
1 = Real News
"""

from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_news.csv"


def load_dataset(data_path=DATA_PATH):
    """
    Load and validate the cleaned news dataset.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Please make sure data/processed/cleaned_news.csv exists."
        )

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("Loading Fake/Real News Dataset")
    print("=" * 60)

    print(f"Dataset path: {data_path}")
    print(f"Original shape: {df.shape}")

    # Required columns
    required_columns = ["clean_text", "label"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only rows with valid text and labels
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    before_rows = len(df)

    df = df[
        (df["clean_text"].str.strip() != "") &
        (df["label"].notna())
    ].copy()

    removed_rows = before_rows - len(df)

    print(f"Removed invalid rows: {removed_rows}")
    print(f"Final shape: {df.shape}")

    # Convert labels to integer
    df["label"] = df["label"].astype(int)

    # Check labels
    valid_labels = {0, 1}

    if not set(df["label"].unique()).issubset(valid_labels):
        raise ValueError(
            "Labels must contain only 0 (Fake) and 1 (Real)."
        )

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nLabel meaning:")
    print("0 = Fake News")
    print("1 = Real News")

    return df


def prepare_features(df):
    """
    Prepare X and y for machine learning.
    """

    X = df["clean_text"]
    y = df["label"]

    print("\nFeature preparation completed.")
    print(f"Number of samples: {len(X)}")
    print(f"Number of labels: {len(y)}")

    return X, y


if __name__ == "__main__":
    df = load_dataset()
    X, y = prepare_features(df)

    print("\nPreprocessing completed successfully.")