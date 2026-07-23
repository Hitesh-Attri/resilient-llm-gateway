FROM python:3.12-slim

# Fail fast, no .pyc, unbuffered logs (so container logs stream in real time).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy source (changes here don't bust the dependency layer above).
COPY src/ ./src/

# Run as a non-root user.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# PYTHONPATH=/app/src makes `main:app` and the `core.*` imports resolve.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]