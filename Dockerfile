# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: api ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

COPY --from=builder /install /usr/local
COPY api/ ./api/
COPY config/ ./config/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── Stage 3: dashboard ────────────────────────────────────────────────────────
FROM python:3.11-slim AS dashboard

WORKDIR /app

COPY --from=builder /install /usr/local
COPY dashboard/ ./dashboard/
COPY config/ ./config/

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
