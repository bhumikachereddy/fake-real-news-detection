"""
Load trained Fake and Real News Detection model.
"""

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def load_model_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if not hasattr(model, "predict"):
        raise ValueError("Invalid model artifact")

    if not hasattr(vectorizer, "transform"):
        raise ValueError("Invalid vectorizer artifact")

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(
            vectorizer.get_feature_names_out()
        ):
            raise ValueError("Model and vectorizer feature mismatch")

    return model, vectorizer