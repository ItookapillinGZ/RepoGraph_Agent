FROM python:3.13-slim

ARG PYTEST_VERSION=9.1.1
ARG RUFF_VERSION=0.16.5
ARG BANDIT_VERSION=1.8.6

RUN python -m pip install --no-cache-dir \
      "pytest==${PYTEST_VERSION}" "ruff==${RUFF_VERSION}" "bandit==${BANDIT_VERSION}" \
    && useradd --create-home --uid 10001 repograph

USER 10001:10001
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "--version"]
