# API image (code only: forecasts are downloaded from the HF dataset at runtime).
#   docker build -t bikeshare-api . && docker run -p 8000:8000 bikeshare-api
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Render sets PORT; exec so uvicorn is PID 1 and gets SIGTERM
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
