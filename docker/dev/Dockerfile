FROM python:3.13.0-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && \
    pip install -U pdm

FROM base as dev

RUN apt-get update && apt-get install -y git curl && \
    sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /bin
