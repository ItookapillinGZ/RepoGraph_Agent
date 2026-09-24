# Security

## Trust model

RepoGraph treats repository content, model output, test output, and pull-request
content as untrusted data. The local operator, RepoGraph controller code,
configured sandbox image, and explicit approval decisions are trusted.

## Primary controls

- Models receive bounded tools and strict schemas, not unrestricted shell or
  filesystem access.
- Repository paths are normalized, kept within explicit roots, checked for
  symlinks and secrets, and limited by file/byte budgets.
- Candidate execution happens in a disposable copy. Docker mode is fail-closed
  and has no host fallback.
- Docker verification uses network-none, `cap-drop ALL`,
  `no-new-privileges`, read-only root, non-root execution, CPU/memory/PID
  limits, hard timeout, bounded output, and a secret-filtered environment.
- G4 apply, G5A Git delivery, and G5B GitHub delivery require separate human
  approvals. There is no automatic chaining.
- Trace metadata is redacted and artifact payloads are rejected if they contain
  known credentials or non-portable local paths.

## Secrets

Do not commit `.env`. Offline flows need no credential. Doctor reports only
`configured` or `missing`; it never prints a value, fragment, length, or hash.
GitHub credentials are used only by an explicitly approved G5B operation.
Repository-provided environment configuration cannot widen trusted sandbox
policy.

## Git and GitHub safety

Transactional apply verifies the saved bundle and repository state before
writeback. Local Git delivery and remote publication are separate operations.
RepoGraph does not auto-commit, auto-push, or auto-create a pull request.

## Validation evidence

H3 ran real Docker acceptance for network, secret, filesystem, resource, and
cleanup isolation. Five Host/Docker equivalence cases matched. H3.1 validated
hierarchical tracing, artifact lineage, secret-free persistence, and five of
five frozen deterministic replays.

## Known limitations

Docker shares a host kernel and is not equivalent to a VM. Host mode is not a
security sandbox. Network-none means unvendored dependencies cannot be fetched
during verification. The product has no multi-tenant isolation, hosted identity
layer, cloud worker boundary, or distributed authorization service.

Detailed invariant and acceptance evidence is in
[`docs/SANDBOX_SECURITY.md`](docs/SANDBOX_SECURITY.md). This portfolio project
does not publish a dedicated private disclosure channel. Report issues through
the repository issue mechanism when one is available, without posting secrets
or exploit material.
