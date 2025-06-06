## docker build -t pix2poly  . 

## For Development:

## docker run --rm --name pix2poly-container pix2poly

## 
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG BUILD_TARGET=cpu



FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8080

CMD ["bin/bash"]