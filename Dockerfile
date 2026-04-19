# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .


# ── Stage 2: api ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── Stage 3: dashboard ────────────────────────────────────────────────────────
FROM python:3.11-slim AS dashboard

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY config/ ./config/

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
