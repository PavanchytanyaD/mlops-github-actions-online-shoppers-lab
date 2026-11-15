# Same environment as the GitHub Actions jobs: code, data, params and the current champion
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml .
COPY src/ src/
COPY data/ data/
COPY models/ models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Default: score the sample sessions with the champion model
CMD ["python", "src/predict.py", "data/sample_visitors.csv"]
