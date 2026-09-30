# Titan Agent — Dockerfile (Phase 16)
# Builds the web dashboard / API server. Runtime deps only; tests live outside
# the image (run them on the host or a dev image).
#
#   docker build -t titan-agent .
#   docker run --rm -p 7860:7860 --env-file .env titan-agent
FROM python:3.12-slim

# Python + logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TITAN_PORT=7860

WORKDIR /app

# System tools for GitOps, TLS, and stdio MCP servers (npx / uvx).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Layer of stability: install deps before copying source so rebuilds are fast
# when the code changes but requirements.txt does not.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Build-time smoke check: the server module must import (FastAPI app + all
# singletons) without network. Fail the build early if wiring is broken.
RUN python -c "import titan_agent.server" && echo "smoke OK"

EXPOSE 7860

CMD ["uvicorn", "titan_agent.server:app", "--host", "0.0.0.0", "--port", "7860"]