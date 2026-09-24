# RepoGraph demo guide

The demo is offline, deterministic, preview-only, and defaults to the real
Docker sandbox. Run `python -m sandbox build` once, then:

```powershell
.\scripts\demo.ps1 -Fresh
```

The command prints the plan/candidate counts, verification and review results,
trace ID, candidate artifact ID, state root, and exact inspection/replay
commands. It never invokes G4, G5A, G5B, or an LLM.

## Two-minute demo

1. Run `.\scripts\doctor.ps1` and point out secret-safe PASS/WARN/BLOCKED output.
2. Run `.\scripts\demo.ps1 -Fresh`.
3. Explain the two-file fixture task and preview-only mutation boundary.
4. Show `Verification: verified`, `Review: good`, and the trace ID.
5. Launch `.\scripts\start-studio.ps1` and open the printed URL to show the
   run-oriented UI panels.

## Five-minute technical demo

After the two-minute flow:

1. Run `python -m observability --state-root <root> show <run-id>` to show the
   planning, candidate, and sandbox hierarchy.
2. Run `... artifacts <run-id>` and `... lineage <run-id>` to show plan ->
   candidate -> diff/verification/review relationships.
3. Run `... replay <run-id> --candidate <artifact-id>` and show
   `semantic_match: true`.
4. Explain that replay validates fixture identity, verification spec, backend,
   Docker image provenance, and mutation-free metadata before execution.
5. Contrast preview with the three independent approval gates.

## Ten-minute interview walkthrough

Tell the story in this order:

1. Problem: free-form coding agents mix reasoning, execution, and mutation.
2. Control plane: five bounded workflows exchange strict models.
3. Exploration and planning: read-only evidence grounds authorized paths.
4. Multi-file execution: complete file states are validated and materialized in
   a disposable copy.
5. Safety: fixed arguments, hardened Docker, fail-closed selection.
6. Human control: preview, G4 apply, G5A Git, and G5B GitHub stay separate.
7. Evaluation: frozen N=17 paired benchmark with honest uncertainty.
8. Observability: hierarchical spans and content-addressed lineage.
9. Replay: deterministic verification without reconstructing LLM reasoning.
10. Tradeoffs and limitations: local SQLite, Docker not VM isolation, fixed
    budgets, Windows-first validation, and no cloud/multi-tenant layer.

## Fresh and repeated runs

`-Fresh` clears only the selected directory whose final name is `demo`. Without
it, each run gets a unique run ID and coexists with older runs. The fixture is
copied to a temporary workspace and hash-identical before and after the run.

## Docker unavailable

Default Docker mode reports `BLOCKED` and does not fall back. For local
diagnosis only:

```powershell
.\scripts\demo.ps1 -Sandbox host
```

Host mode visibly warns that it is not a security sandbox. Do not use it for an
untrusted repository.
