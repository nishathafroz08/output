# Dockerfile
FROM python:3.10-slim

# avoid belligerent prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1) system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

# 2) set workdir
WORKDIR /app

# 3) install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) copy your entire project (code, models, data, config.json, tests, etc.)
COPY . .

# 5) ensure spaCy can find its model if you use a custom SPACY_DATA_PATH
ENV SPACY_DATA_PATH=/app/app/models/spacy/en_core_web_sm/en_core_web_sm-3.8.0

# 6) expose the FastAPI port
EXPOSE 8000

# 7) default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]