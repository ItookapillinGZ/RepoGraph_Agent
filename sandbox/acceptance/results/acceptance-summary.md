# H3A real Docker sandbox acceptance

- Status: **H3 FULLY ACCEPTED**
- Timestamp: 2026-09-15T07:21:54.483240+00:00
- Image: repograph-sandbox:local
- Image ID: sha256:c66fc8697a59ed31747845eb432dd05ae3371d12a4ed23286708cc8d67f96d57
- OpenAI API calls: 0
- G4/G5 semantics modified: no

## Security contracts

- Fixed Docker argv: PASS
- Network, secret, rootfs, non-root, socket, capabilities: PASS
- Host sentinel and real source protection: PASS

## Resource and lifecycle contracts

- PID limit: PASS
- Memory limit: PASS
- CPU quota: PASS
- Timeout cleanup: PASS
- Output bounding: PASS
- RepoGraph container leaks: 0

## Real integrations

- Explorer run_repository_test: PASS
- Candidate verification: PASS
- Local evaluator: PASS

## Frozen H2 Host/Docker replay

- Rule: From the authoritative H2.3E recovery results, use config=no_exploration; sort unique task IDs alphabetically; select the first four, then include registry-alias-collision. Selection is frozen before Docker replay and is independent of stored or replayed pass/fail outcomes.
- Compared: 5
- Semantically equivalent: 5
- Mismatches: 0

## Integrity

- Acceptance artifact security scan: PASS
- Acceptance-related source unchanged during run: PASS
- Remaining TCB: Windows host, Docker Desktop, WSL2 kernel, Docker daemon,
  container runtime, and the trusted sandbox image.
