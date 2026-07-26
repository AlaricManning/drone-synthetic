# The conversion job image — the artifact AWS Batch runs.
# Build from the repo root:  scripts/build_image.sh
#
# Credentials are never baked in: locally they arrive via a mounted
# ~/.aws (read-only) and AWS_PROFILE; on Batch, via the job role.

FROM python:3.12-slim

# Which commit built this image, stamped into every dataset it converts.
# The image has no .git, so the converter cannot work this out at run time --
# it has to be told here. build_image.sh fills these in from the working tree;
# a bare `docker build` leaves them "unknown", which the provenance record
# reports as an unknown commit rather than inventing one.
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ENV DRONESYNTH_GIT_COMMIT=$GIT_COMMIT
ENV DRONESYNTH_GIT_DIRTY=$GIT_DIRTY

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
