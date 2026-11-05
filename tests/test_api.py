
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["algorithm"] == "LinearSVC"
    assert response.json()["feature_extraction"] == "TF-IDF"


def test_valid_news_prediction():
    response = client.post(
        "/predict",
        json={
            "text": "The government announced a new development programme."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Fake News", "Real News"]
    assert "message" in data


def test_missing_news_input():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_invalid_input_datatype():
    response = client.post(
        "/predict",
        json={"text": 12345}
    )

    assert response.status_code == 422


def test_empty_news_input():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_whitespace_news_input():
    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    assert response.status_code == 422


def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "The government announced a new development programme.",
                "Scientists discovered a new medical treatment."
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]


def test_empty_batch_input():
    response = client.post(
        "/predict/batch",
        json={"texts": []}
    )

    assert response.status_code == 422
