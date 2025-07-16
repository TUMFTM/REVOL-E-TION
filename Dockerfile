ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS builder

WORKDIR /app/revoletion

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget

# Download and install Gurobi.
ARG GUROBI_MAJOR_MINOR_VERSION=12.0
ARG GUROBI_PATCH_VERSION=0
RUN mkdir -p /opt/gurobi && \
    wget "https://packages.gurobi.com/${GUROBI_MAJOR_MINOR_VERSION}/gurobi${GUROBI_MAJOR_MINOR_VERSION}.${GUROBI_PATCH_VERSION}_linux64.tar.gz" && \
    tar xvf "gurobi${GUROBI_MAJOR_MINOR_VERSION}.${GUROBI_PATCH_VERSION}_linux64.tar.gz" --directory=/opt/gurobi

# Download and install CBC.
ARG CBC_VERSION=2.10.12
# By default use the static version, which makes the dependency management easier.
ARG CBC_BUILD=x86_64-ubuntu22-gcc1140-static
RUN mkdir -p /opt/cbc && \
    wget "https://github.com/coin-or/Cbc/releases/download/releases%2F${CBC_VERSION}/Cbc-releases.${CBC_VERSION}-${CBC_BUILD}.tar.gz" && \
    tar xvf "Cbc-releases.${CBC_VERSION}-${CBC_BUILD}.tar.gz" --directory=/opt/cbc

# Enable bytecode compilation.
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Only install the dependencies, to improve layer caching.
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=cache,target=/app/revoletion/.cache/uv \
    uv sync --no-install-project --no-editable

COPY . .
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/app/revoletion/.cache/uv \
    uv sync --locked --no-editable

FROM python:${PYTHON_VERSION} AS runtime

WORKDIR /app/revoletion

# Copy Gurobi from builder stage.
COPY --from=builder /opt/gurobi /opt/gurobi
# Add Gurobi to Path, so that REVOL-E-TION can execute it.
ENV PATH="/opt/gurobi/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/gurobi/lib:${LD_LIBRARY_PATH:-}"

# Copy CBC from builder stage.
COPY --from=builder /opt/cbc /opt/cbc
# Add CBC to Path, so that REVOL-E-TION can execute it.
ENV PATH="/opt/cbc/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/cbc/lib:${LD_LIBRARY_PATH:-}"

# Normally, revoletion would be executed as root in the container, and
# the result files it creates would therefore be owned by root.
# For non-root host users, this complicates hanlding the result files.
# To circumvent this an extra user to execute revoletion is created.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} -r revoletion && \
 useradd --no-log-init --no-create-home --home-dir /app/revoletion -r --uid ${UID}  -g revoletion revoletion


# Enable bytecode compilation.
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

COPY --from=builder --chown=revoletion:revoletion /app/revoletion/.venv /app/revoletion/.venv
RUN chown -R revoletion:revoletion /app/revoletion

ENV VIRTUAL_ENV="/app/revoletion/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER revoletion

ENTRYPOINT ["revoletion"]
CMD []
