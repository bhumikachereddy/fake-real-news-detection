"""
Training module for Fake and Real News Detection.

Pipeline:
Raw cleaned text
        ↓
Train/Test Split
        ↓
TF-IDF
        ↓
Linear SVM
        ↓
Trained Model + Vectorizer

The TF-IDF vectorizer is fitted ONLY on training data
to prevent test-data leakage.
"""

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split data into training and testing sets using stratification.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_tfidf():
    """
    Create the TF-IDF vectorizer.

    These settings match the existing notebook.
    """

    tfidf = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    return tfidf


def transform_text(X_train, X_test):
    """
    Fit TF-IDF on training data and transform both
    training and testing data.
    """

    tfidf = create_tfidf()

    # IMPORTANT:
    # Fit ONLY on training data.
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Transform test data using the already fitted vectorizer.
    X_test_tfidf = tfidf.transform(X_test)

    print("\n" + "=" * 60)
    print("TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Testing TF-IDF shape : {X_test_tfidf.shape}")
    print(
        f"Number of TF-IDF features: "
        f"{len(tfidf.get_feature_names_out())}"
    )

    return tfidf, X_train_tfidf, X_test_tfidf


def train_linear_svm(X_train_tfidf, y_train):
    """
    Tune Linear SVM using 5-fold cross-validation.

    C is optimized using F1 score.
    """

    print("\n" + "=" * 60)
    print("LINEAR SVM HYPERPARAMETER TUNING")
    print("=" * 60)

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    svm_tuning = GridSearchCV(
        estimator=LinearSVC(random_state=42),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    svm_tuning.fit(X_train_tfidf, y_train)

    print("\nBest parameters:")
    print(svm_tuning.best_params_)

    print(
        "Best cross-validation F1 score:",
        f"{svm_tuning.best_score_:.6f}"
    )

    best_model = svm_tuning.best_estimator_

    return best_model, svm_tuning


def save_model(model, tfidf):
    """
    Save trained SVM and TF-IDF vectorizer.
    """

    model_path = MODEL_DIR / "final_svm_model.pkl"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)

    print("\n" + "=" * 60)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 60)

    print(f"SVM model      : {model_path}")
    print(f"TF-IDF vectorizer: {vectorizer_path}")

    return model_path, vectorizer_path


def train_pipeline(X, y, save_artifacts=True):
    """
    Complete training pipeline.
    """

    X_train, X_test, y_train, y_test = split_data(X, y)

    tfidf, X_train_tfidf, X_test_tfidf = transform_text(
        X_train,
        X_test
    )

    model, tuning = train_linear_svm(
        X_train_tfidf,
        y_train
    )

    if save_artifacts:
        save_model(model, tfidf)

    return {
        "model": model,
        "tfidf": tfidf,
        "tuning": tuning,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf
    }


if __name__ == "__main__":
    from preprocess import load_dataset, prepare_features

    df = load_dataset()
    X, y = prepare_features(df)

    results = train_pipeline(X, y)

    print("\nTraining completed successfully.")