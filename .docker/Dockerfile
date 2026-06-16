# Build from the revol-e-tion/ directory, e.g.:
#   docker build -f .docker/Dockerfile -t <registry>/<image>:<tag> .
#
# Gurobi WLS credentials are NOT baked in — pass at runtime via env vars:
#   WLSACCESSID, WLSSECRET, LICENSEID
# or set GRB_LICENSE_FILE to point to a mounted gurobi.lic file.

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

RUN useradd --uid 1001 --no-create-home --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser /build/revol-e-tion/revoletion /app/revoletion
# TODO: remove once deployments always receive an explicit scenario_path parameter
COPY --from=builder --chown=appuser /build/revol-e-tion/example /app/example

RUN mkdir /app/results && chown appuser /app/results

USER appuser

ENV VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}" \
    MPLCONFIGDIR="/tmp/matplotlib"

# Gurobi WLS credentials — set these at runtime, never here.
# WLSACCESSID, WLSSECRET, LICENSEID (Gurobi 10+)
# or GRB_LICENSE_FILE=/path/to/mounted/gurobi.lic

# Prefect reads PREFECT_API_URL and PREFECT_API_KEY from the environment.
# The worker injects these; no defaults are set here.
