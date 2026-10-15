FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY trigger_pipeline.py .
CMD ["gunicorn","-b","0.0.0.0:8080","trigger_pipeline:app"]
