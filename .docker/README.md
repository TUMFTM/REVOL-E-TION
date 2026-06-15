# Docker image for Prefect deployments

This image packages `revoletion` together with the Gurobi solver for use by a Prefect Docker worker.

## Build

Because `revoletion-core` is a local path dependency (`../revol-e-tion-core`), the build context must be the **parent directory**, not this repository root.

```bash
# Run from parent/
docker build \
  -f revol-e-tion/.docker/Dockerfile \
  -t <registry>/<image>:<tag> \
  .
```

## Gurobi WLS licensing

Credentials are **not** baked into the image. Pass them at runtime as environment variables:

| Variable       | Description                        |
|----------------|------------------------------------|
| `WLSACCESSID`  | WLS access ID from gurobi.com      |
| `WLSSECRET`    | WLS secret                         |
| `LICENSEID`    | Numeric license ID                 |

Alternatively, mount a `gurobi.lic` file and point `GRB_LICENSE_FILE` at it.

In a Prefect deployment these are set via Secret blocks referenced under `job_variables.env` in your deployment YAML.

## Prefect worker

The image has no `ENTRYPOINT`, which lets the Prefect Docker worker inject its own command. The worker also injects `PREFECT_API_URL` and `PREFECT_API_KEY` at container start — these do not need to be set in the image.

Start a Docker worker against your pool:

```bash
prefect worker start --pool <pool-name>
```

## Prefect deployment

```bash
PREFECT_API_URL=http://localhost:4200/api .venv/bin/prefect --no-prompt deploy --all
```

for this to work, the docker image for the deployment must already exist.
