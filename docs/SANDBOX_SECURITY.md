# RepoGraph Docker sandbox security model

## Scope and threat model

Pytest collection imports repository modules, so a reviewed repository can run
code. Malicious or defective code may try to read host files or secrets, write
outside its workspace, modify the real source repository, open network
connections, spawn many processes, consume memory, or run forever.

Docker mode moves targeted tests, Explorer `run_repository_test` calls,
candidate test verification, and supported local evaluator commands into a
short-lived container. LLM orchestration, LangGraph, repository reading,
planning, candidate generation, review, and G4/G5 delivery remain on the host.
Static Ruff and Bandit parsing currently remains on the host; these tools parse
source but do not import repository modules.

## Boundary and protected resources

RepoGraph makes a bounded, secret-filtered disposable copy before starting a
Docker execution. Only that copy is mounted read-write at `/workspace`. The
source repository, home directory, project root, `.env` files, SSH material,
Docker socket, and arbitrary caller mounts are not exposed. Container writes
are discarded with the temporary copy and never authorize a G4 repository
mutation.

The container receives fixed argv with `shell=False`. Repository and model text
cannot choose Docker flags, mounts, image, user, network settings,
capabilities, or resource limits. The only allowed container environment names
are `PYTHONUNBUFFERED` and `PYTHONDONTWRITEBYTECODE`; the parent environment is
not inherited into the container. In particular, OpenAI, GitHub, AWS, SSH, and
database credentials are not forwarded.

## Deterministic Docker policy

Docker mode always applies:

- `--network none`
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--read-only`
- one read-write disposable workspace mount
- a bounded, `noexec,nosuid` `/tmp` tmpfs
- 512 MiB memory, 1 CPU, and 64 PIDs by default
- a host-controlled wall-clock timeout and bounded stdout/stderr
- a random `repograph-sbx-<uuid>` container name and `--rm`

Timeout cleanup attempts `docker stop`, `docker kill`, and `docker rm --force`
with fixed argv and bounded waits. A missing CLI, unavailable daemon, or missing
configured image returns `sandbox_unavailable`. RepoGraph never pulls, builds,
enables networking, or falls back to host execution during an ordinary Docker
run.

The limits are trusted operator configuration with hard maximums. Defaults can
be configured using `REPOGRAPH_SANDBOX_IMAGE`,
`REPOGRAPH_SANDBOX_MAX_TIMEOUT_SECONDS`, `REPOGRAPH_SANDBOX_MEMORY_MB`,
`REPOGRAPH_SANDBOX_CPUS`, `REPOGRAPH_SANDBOX_PIDS`, and
`REPOGRAPH_SANDBOX_MAX_OUTPUT_CHARS`. Repository content and the LLM cannot set
them.

## Image trust and provenance

The default image name is `repograph-sandbox:local`. Build it explicitly with:

```text
python -m sandbox build
```

Ordinary runs never build or pull it. The configured image reference and the
locally resolved immutable image ID are recorded in execution provenance. A
tag alone is mutable and is not presented as immutable provenance. The image,
Docker daemon, container runtime, kernel, and host configuration are trusted
infrastructure. Third-party prebuilt images must be selected and trusted by the
operator; repository input cannot select one.

The Dockerfile creates a non-root user and contains Python, pytest, Ruff, and
Bandit only. It does not contain RepoGraph `.env` data, API keys, Git
credentials, or automatic repository dependency installation. Projects that
need additional offline dependencies require a trusted prebuilt image.

## Modes and fail-closed behavior

`--sandbox host` is the backward-compatible default. It executes repository
code on the host and is **not a security sandbox**.

`--sandbox docker` requests resource-constrained container execution. If its
prerequisites are unavailable, the requested check stops with structured
`sandbox_unavailable` evidence. It never silently executes the same command on
the host. Network-dependent tests fail; RepoGraph does not retry them with a
network-enabled policy.

Use `python -m sandbox doctor` for bounded prerequisite status. It reports
only CLI, daemon, image, image ID, and smoke availability; it does not disclose
host paths. Real security acceptance is intentionally opt-in and is never part
of ordinary unit tests.

## Authorization invariants

Sandbox results are verification evidence, not human approval. The three human
approvals and all G4 transactional apply, G5A local Git delivery, and G5B remote
delivery semantics are unchanged. Passing container tests cannot write to the
real source tree or authorize delivery.

## Target-platform acceptance

H3A real-Docker acceptance passed on 2026-09-15 using:

- Windows with Docker Desktop 4.73.1 and the WSL2 Linux engine
- Docker Client and Server 29.4.3
- Linux kernel 6.6.114.1-microsoft-standard-WSL2 on amd64
- trusted base digest
  sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285
- RepoGraph sandbox image ID
  sha256:c66fc8697a59ed31747845eb432dd05ae3371d12a4ed23286708cc8d67f96d57

The run validated container-side network and secret isolation, host sentinel
and source-repository protection, a writable disposable workspace, read-only
root filesystem, UID/GID 10001 non-root execution, an absent Docker socket,
zero effective capabilities, no-new-privileges, PID/memory/CPU cgroup
limits, timeout recovery, bounded stdout/stderr, and zero leaked RepoGraph
containers. Explorer test execution, candidate verification, and the local
evaluator each entered the real Docker boundary. Five deterministically
selected frozen H2 candidates produced 5/5 Host/Docker semantic equivalence.
Persisted artifacts contained no credential, Docker token, test secret, or
.env content. No live model or OpenAI API call was made.

The full bounded evidence bundle is stored in
sandbox/acceptance/results/. The acceptance runner is
python -m sandbox.acceptance; it remains an explicit operator action.

## Known limitations

Docker materially reduces the impact of repository-controlled execution, but
it is not a perfect security boundary. Docker daemon, runtime, kernel, host
filesystem sharing, and local policy configuration remain in the trusted
computing base. The initial H3 implementation does not impose a separate
workspace disk quota; the pre-execution copy is capped, but post-start file
growth is limited only by Docker/host storage. The successful target-platform
acceptance is evidence for the tested configuration, not a guarantee for other
Docker, kernel, architecture, dependency, or host file-sharing combinations.
No claim of perfect arbitrary-code containment is made.
