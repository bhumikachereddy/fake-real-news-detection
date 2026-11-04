
import pytest
import pandas as pd

from src.failure_injection import (
    load_artifact,
    validate_text,
    validate_dataset,
    predict_news,
    run_failure_injection
)


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        load_artifact("missing_model.pkl")


def test_invalid_input_type():
    with pytest.raises(TypeError):
        validate_text(12345)


def test_empty_input():
    with pytest.raises(ValueError):
        validate_text("   ")


def test_valid_input():
    assert validate_text("Sample news") == "Sample news"


def test_missing_dataset():
    with pytest.raises(FileNotFoundError):
        validate_dataset("missing_dataset.csv")


def test_invalid_dataset_schema(tmp_path):
    file_path = tmp_path / "invalid.csv"

    pd.DataFrame({
        "title": ["Sample"],
        "text": ["Sample content"]
    }).to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        validate_dataset(file_path)


def test_valid_dataset_schema(tmp_path):
    file_path = tmp_path / "valid.csv"

    pd.DataFrame({
        "clean_text": ["Sample news"],
        "label": [0]
    }).to_csv(file_path, index=False)

    assert validate_dataset(file_path) is True


def test_actual_model_prediction():
    prediction = predict_news(
        "The government announced a new development initiative."
    )

    assert prediction in [0, 1]


def test_failure_injection_report():
    report = run_failure_injection()

    assert report["total_scenarios"] == 7
    assert report["failed"] == 0
    assert report["overall_status"] == "PASSED"
