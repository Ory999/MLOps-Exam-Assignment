FROM python:3.11-slim

# Metadata
LABEL description="DST Sector Health Forecaster — MLOps Exam"
LABEL version="1.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config.yaml .
COPY run_pipeline.py .

# Create artifact directories (these will be mounted as volumes in production)
RUN mkdir -p artifacts/{raw,processed,models,metrics,reports,mlflow}

# Expose FastAPI port
EXPOSE 8000

# Default command: start the API
# Override with 'python run_pipeline.py' to run the pipeline instead
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
