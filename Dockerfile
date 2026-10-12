
FROM python:3.12-slim

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Use the project's virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies without building the project itself
RUN uv sync --no-dev --no-default-groups --group production --no-install-project

# Copy application source code and artifacts
COPY src ./src
COPY artifacts ./artifacts

WORKDIR /app/src/final_project

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
