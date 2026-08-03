ARG GUROBI_VERSION=12.0.0
ARG PYTHON_VERSION=3.12

FROM gurobi/python:${GUROBI_VERSION}_${PYTHON_VERSION} AS builder

# Non-empty: point the "internal" uv index at this local pypiserver mirror instead of GitLab,
# skipping credentials entirely. Empty (default): use pyproject.toml's remote GitLab index,
# authenticated via the gitlab_token secret.
ARG UV_INDEX_INTERNAL_URL=""

WORKDIR /build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./revol-e-tion/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=secret,id=gitlab_token,required=false \
    sh -c '\
      if [ -n "$UV_INDEX_INTERNAL_URL" ]; then \
        export UV_INDEX="internal=$UV_INDEX_INTERNAL_URL"; \
      else \
        export UV_INDEX_INTERNAL_USERNAME=oauth2; \
        export UV_INDEX_INTERNAL_PASSWORD="$(cat /run/secrets/gitlab_token)"; \
      fi; \
      uv sync --no-install-project --no-editable --project /build/revol-e-tion'

COPY . ./revol-e-tion/

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=secret,id=gitlab_token,required=false \
    sh -c '\
      if [ -n "$UV_INDEX_INTERNAL_URL" ]; then \
        export UV_INDEX="internal=$UV_INDEX_INTERNAL_URL"; \
        uv sync --no-editable --project /build/revol-e-tion; \
      else \
        export UV_INDEX_INTERNAL_USERNAME=oauth2; \
        export UV_INDEX_INTERNAL_PASSWORD="$(cat /run/secrets/gitlab_token)"; \
        uv sync --locked --no-editable --project /build/revol-e-tion; \
      fi'


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
