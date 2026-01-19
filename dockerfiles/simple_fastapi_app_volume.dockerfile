FROM python:3.11-slim

EXPOSE $PORT
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi pydantic uvicorn python-multipart

COPY cloud_run/simple_fastapi_app_volume.py simple_fastapi_app.py

CMD exec uvicorn simple_fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1
