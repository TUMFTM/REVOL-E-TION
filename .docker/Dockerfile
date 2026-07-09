ARG GUROBI_VERSION=12.0.0
ARG PYTHON_VERSION=3.12

FROM gurobi/python:${GUROBI_VERSION}_${PYTHON_VERSION} AS builder

WORKDIR /build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./revol-e-tion/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=secret,id=gitlab_token \
    UV_INDEX_GITLAB_LRZ_USERNAME=oauth2 \
    UV_INDEX_GITLAB_LRZ_PASSWORD="$(cat /run/secrets/gitlab_token)" \
    uv sync --no-install-project --no-editable --project /build/revol-e-tion

COPY . ./revol-e-tion/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=secret,id=gitlab_token \
    UV_INDEX_GITLAB_LRZ_USERNAME=oauth2 \
    UV_INDEX_GITLAB_LRZ_PASSWORD="$(cat /run/secrets/gitlab_token)" \
    uv sync --locked --no-editable --project /build/revol-e-tion


FROM gurobi/python:${GUROBI_VERSION}_${PYTHON_VERSION} AS runtime

WORKDIR /app

RUN useradd --uid 1001 --home-dir /app --no-create-home --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser /build/revol-e-tion/revoletion /app/revoletion

RUN mkdir /app/results && chown appuser /app/results

USER appuser

ENV VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}" \
    MPLCONFIGDIR="/tmp/matplotlib"
