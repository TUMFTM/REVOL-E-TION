# Build context must be the parent `parent/` directory, e.g.:
#   docker build -f revol-e-tion/.docker/Dockerfile -t <registry>/<image>:<tag> .
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

# Install dependencies first for better layer caching — this layer is only
# invalidated when pyproject.toml, uv.lock, or revoletion-core source change.
# revoletion-core must be fully present here because uv builds it as a wheel
# (it's a local path dependency), not just resolved from the lock file.
COPY revol-e-tion/pyproject.toml revol-e-tion/uv.lock ./revol-e-tion/
COPY revol-e-tion-core/ ./revol-e-tion-core/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable --project /build/revol-e-tion

COPY revol-e-tion/ ./revol-e-tion/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --project /build/revol-e-tion


FROM gurobi/python:${GUROBI_VERSION}_${PYTHON_VERSION} AS runtime

WORKDIR /app

RUN useradd --uid 1001 --no-create-home --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser /build/revol-e-tion/revoletion /app/revoletion

USER appuser

ENV VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}"

# Gurobi WLS credentials — set these at runtime, never here.
# WLSACCESSID, WLSSECRET, LICENSEID (Gurobi 10+)
# or GRB_LICENSE_FILE=/path/to/mounted/gurobi.lic

# Prefect reads PREFECT_API_URL and PREFECT_API_KEY from the environment.
# The worker injects these; no defaults are set here.
