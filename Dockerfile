# Backend image: FastAPI API server + Feishu long-connection worker.
# Both services share the exact same dependency set (see requirements.txt), so one
# image is built once and reused with a different CMD per docker-compose service.

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./requirements.txt
COPY LightRAG ./LightRAG
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .

RUN mkdir -p knowledge_base lightrag_workspaces data memory logs

EXPOSE 24680

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "24680"]
