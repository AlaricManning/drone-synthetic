# The conversion job image — the artifact AWS Batch runs.
# Build from the repo root:  docker build -f docker/Dockerfile -t dronesynth-convert .
#
# Credentials are never baked in: locally they arrive via a mounted
# ~/.aws (read-only) and AWS_PROFILE; on Batch, via the job role.

FROM python:3.12-slim

RUN useradd --create-home app

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .

USER app

# run-id and dataset version are appended at run time:
#   docker run ... dronesynth-convert --run-id run_0001 --version v001
ENTRYPOINT ["dronesynth", "convert", "--config", "configs/convert.s3.yaml"]
