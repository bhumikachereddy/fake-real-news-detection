"""
FastAPI application for Fake and Real News Detection.
"""

from fastapi import Request
from time import perf_counter
from monitoring.drift_monitor import detect_prediction_drift
from monitoring.system_monitor import get_system_metrics

from monitoring.api_performance_monitor import (
    record_request,
    get_performance_summary
)
from fastapi import FastAPI, HTTPException

from api.model_loader import load_model_artifacts
from api.preprocessing import preprocess_text
from api.schemas import (
    NewsRequest,
    BatchNewsRequest,
    PredictionResponse,
    BatchPredictionResponse
)


app = FastAPI(
    title="Fake and Real News Detection API",
    description=(
        "Machine learning API using TF-IDF and Linear SVM "
        "to classify news as Fake or Real."
    ),
    version="1.0.0"
)
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    start_time = perf_counter()

    try:
        response = await call_next(request)
        latency_ms = (perf_counter() - start_time) * 1000

        record_request(
            status_code=response.status_code,
            latency_ms=latency_ms
        )

        return response

    except Exception:
        latency_ms = (perf_counter() - start_time) * 1000

        record_request(
            status_code=500,
            latency_ms=latency_ms
        )

        raise


# Load model artifacts when the application starts
try:
    model, vectorizer = load_model_artifacts()
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as error:
    model = None
    vectorizer = None
    MODEL_LOADED = False
    MODEL_ERROR = str(error)



def make_prediction(text: str):

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {MODEL_ERROR}"
        )

    start_time = time.perf_counter()

    try:
        cleaned_text = preprocess_text(text)

        transformed_text = vectorizer.transform(
            [cleaned_text]
        )

        prediction = int(
            model.predict(transformed_text)[0]
        )

        result = (
            "Fake News" if prediction == 0 else "Real News"
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Record successful prediction
        log_prediction(
            text=text,
            prediction=prediction,
            result=result,
            latency_ms=latency_ms
        )

        return PredictionResponse(
            prediction=prediction,
            result=result,
            message=f"The news is classified as {result}."
        )

    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )

@app.get("/")
def home():

    return {
        "project": "Fake and Real News Detection",
        "message": "Welcome to the News Detection API",
        "status": "Running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED
    }


@app.get("/model-info")
def model_info():

    return {
        "algorithm": "LinearSVC",
        "feature_extraction": "TF-IDF",
        "labels": {
            "0": "Fake News",
            "1": "Real News"
        },
        "model_loaded": MODEL_LOADED
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: NewsRequest):

    return make_prediction(request.text)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_batch(request: BatchNewsRequest):

    if len(request.texts) == 0:
        raise HTTPException(
            status_code=422,
            detail="At least one news article is required"
        )

    predictions = [
        make_prediction(text)
        for text in request.texts
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions)
    )
import time

from monitoring.prediction_monitor import (
    log_prediction,
    get_prediction_summary
)
@app.get("/monitoring/predictions")
def prediction_monitoring():

    return {
        "monitoring": "Prediction Monitoring",
        "status": "active",
        "summary": get_prediction_summary()
    }
@app.get("/monitoring/performance")
def api_performance_monitoring():
    return {
        "monitoring": "API Performance Monitoring",
        "status": "active",
        "performance": get_performance_summary()

    }
@app.get("/monitoring/system")
def system_monitoring():
    return {
        "monitoring": "System Resource Monitoring",
        "metrics": get_system_metrics()
    }
@app.get("/monitoring/drift")
def prediction_drift_monitoring():
    return {
        "monitoring": "Prediction Drift Detection",
        "result": detect_prediction_drift()
    }