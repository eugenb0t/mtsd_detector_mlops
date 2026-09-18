FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -LsSf https://astral.sh/uv/0.6.14/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && chown -R app:app /app

COPY --chown=app:app config.yaml ./config.yaml

USER app

EXPOSE 8000

CMD ["uvicorn", "mtsd_detector.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
