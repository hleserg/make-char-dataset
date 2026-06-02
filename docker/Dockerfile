# syntax=docker/dockerfile:1.7
# Multi-stage image for make-char-dataset. Build context = repo root:
#
#   # light image (default) — pure pipeline core, no torch/diffusers/onnxruntime:
#   docker build -f docker/Dockerfile -t make-char-dataset .
#
#   # heavy image — adds the `gpu` dependency-group (torch/diffusers/onnxruntime)
#   # for the real ComfyUI/diffusers + WD14 backends:
#   docker build -f docker/Dockerfile --build-arg GPU=1 -t make-char-dataset:gpu .

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# GPU=0 keeps the image light (CI/default); GPU=1 also installs the heavy `gpu` group.
ARG GPU=0

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev
# Heavy backends are an opt-in build arg so the default/CI image stays light.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then uv sync --locked --no-editable --no-dev --group gpu; fi

FROM python:3.13-slim AS runtime

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

USER appuser
ENTRYPOINT ["make-char-dataset"]
