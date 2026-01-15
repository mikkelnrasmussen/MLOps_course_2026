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

# Enable safe use of cached wheels/sdists inside Docker layers
ENV UV_LINK_MODE=copy

# Install dependencies (no project) using a persistent BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy application code and runtime data
COPY src/ ./src/

# Install the project itself (also benefits from the same cache mount)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENTRYPOINT ["uv", "run", "python", "src/mlops_course/evaluate.py"]
