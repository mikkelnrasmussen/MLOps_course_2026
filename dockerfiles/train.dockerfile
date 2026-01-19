FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first (maximizes layer caching)
COPY uv.lock ./uv.lock
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
COPY LICENSE ./LICENSE
RUN mkdir outputs
RUN mkdir reports
RUN mkdir reports/figures

# Enable safe use of cached wheels/sdists inside Docker layers
ENV UV_LINK_MODE=copy

# Install dependencies (no project) using a persistent BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy application code and runtime data
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# Install the project itself (also benefits from the same cache mount)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/mlops_course/train.py"]
