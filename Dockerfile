
FROM python:3.13-slim

WORKDIR /app

# Install API dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code and trained model
COPY api/ ./api/
COPY models/final_svm_model.pkl ./models/final_svm_model.pkl
COPY models/tfidf_vectorizer.pkl ./models/tfidf_vectorizer.pkl

# Expose FastAPI port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
