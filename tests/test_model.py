
from pathlib import Path
import joblib
import pytest

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and vectorizer paths
MODEL_PATH = BASE_DIR / "models" / "final_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


@pytest.fixture(scope="module")
def model():
    """Load the trained LinearSVC model."""
    assert MODEL_PATH.exists(), "Trained model file is missing"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def vectorizer():
    """Load the trained TF-IDF vectorizer."""
    assert VECTORIZER_PATH.exists(), "TF-IDF vectorizer file is missing"
    return joblib.load(VECTORIZER_PATH)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_vectorizer_file_exists():
    assert VECTORIZER_PATH.exists()


def test_model_loading(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_vectorizer_loading(vectorizer):
    assert vectorizer is not None
    assert hasattr(vectorizer, "transform")


def test_vectorizer_output(vectorizer):
    sample_text = ["The government announced a new development initiative."]
    transformed = vectorizer.transform(sample_text)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(vectorizer.get_feature_names_out())


def test_model_prediction(model, vectorizer):
    sample_text = [
        "The government announced a new development initiative."
    ]

    features = vectorizer.transform(sample_text)
    prediction = model.predict(features)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_prediction_multiple_samples(model, vectorizer):
    sample_texts = [
        "The government announced a new development initiative.",
        "Breaking news: Scientists discovered a new planet."
    ]

    features = vectorizer.transform(sample_texts)
    predictions = model.predict(features)

    assert len(predictions) == 2
    assert all(prediction in [0, 1] for prediction in predictions)
