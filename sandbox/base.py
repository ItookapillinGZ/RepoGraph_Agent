"""Stable backend protocol; callers never construct Docker commands."""

from typing import Protocol

from sandbox.models import SandboxExecutionRequest, SandboxExecutionResult


class SandboxBackend(Protocol):
    def run(self, request: SandboxExecutionRequest) -> SandboxExecutionResult: ...
