# Official SWE-bench evaluation

RepoGraph H2.1 can pass deterministic prediction JSONL to the upstream
SWE-bench Docker harness for one instance or an explicitly selected small
subset. It never starts a full dataset run by default.

## Install optional dependencies

Use a dedicated evaluation environment when practical:

```powershell
python -m pip install -r requirements-eval.txt
python -m evaluation.cli doctor
```

`requirements-eval.txt` is intentionally separate from `requirements.txt`.
Normal RepoGraph commands and Studio do not require SWE-bench or `datasets`.
The doctor reports only package/CLI/daemon availability and does not install,
start, or reconfigure Docker.

## Docker prerequisites

Install Docker using the instructions for the host platform, start its daemon,
and verify that the current user can run `docker version`. Docker image pulls
and builds can require substantial disk, memory, and network capacity.

Official SWE-bench grading runs repository test environments in Docker through
the upstream harness. RepoGraph planning, candidate generation, and the local
trusted evaluator are not thereby turned into a general-purpose OS sandbox.

## Prepare a prediction

Export a persisted RepoGraph experiment:

```powershell
python -m evaluation.cli swebench-export `
  --workspace C:\Temp\repograph-eval `
  --experiment baseline `
  --output C:\Temp\predictions.jsonl `
  --model-name-or-path repograph-baseline
```

Each line is deterministic JSON with the official fields:

```json
{"instance_id":"owner__repo-123","model_name_or_path":"repograph-baseline","model_patch":"diff --git ..."}
```

The LLM does not generate this JSON envelope. `model_patch` comes from Git diff
against the task's resolved base commit.

## Grade one selected instance

```powershell
python -m evaluation.cli swebench-evaluate `
  --predictions C:\Temp\predictions.jsonl `
  --dataset princeton-nlp/SWE-bench_Lite `
  --split test `
  --instance-id owner__repo-123 `
  --run-id baseline `
  --timeout 1800 `
  --output C:\Temp\swebench-results
```

H2.1 invokes the installed interpreter's
`swebench.harness.run_evaluation` module with a fixed argument vector,
`shell=False`, `--max_workers 1`, and the selected instance. The outer timeout
terminates the evaluator process tree; the same bounded timeout is passed to
the official per-instance harness.

The command writes:

- the official upstream JSON report;
- `repograph-swebench-metadata/<run-id>.json` with dataset, split, instance,
  SWE-bench version, and prediction digest;
- `<run-id>.repograph-result.json` with the strict normalized outcome.

Only the official JSON artifact decides `resolved`. Console text and
RepoGraph's internal `GOOD`/`verified` states never do.

## Cache and run identity

The upstream harness can reuse results by `run_id + instance_id`. RepoGraph
therefore derives the actual run ID from:

```text
requested experiment/run label + instance ID + SHA-256(model_patch) prefix
```

The same patch has a stable identity. A changed patch receives a different
identity. If a sidecar metadata file exists but its dataset, split, instance,
run ID, or full prediction SHA-256 differs, H2.1 rejects the cached result as an
`evaluation_error`.

## Inspect the result

Normalized statuses are:

- `resolved`: the official report marks the selected completed instance
  resolved;
- `unresolved`: the official report marks it completed and unresolved;
- `timeout`: the external grading process exceeded `evaluator_timeout`;
- `evaluation_error`: dependency, Docker, process, cache, or official report
  integrity failed.

The normalized result records SWE-bench/Docker versions when actually
available, dataset, split, instance, full prediction SHA-256, derived run ID,
grading duration, official report path, completion, and resolution. An image
identifier remains `null` unless the upstream artifact exposes one; it is not
invented.

## Optional manual smoke acceptance

After `doctor` reports both SWE-bench and the Docker daemon available:

1. Select one known small SWE-bench Lite instance.
2. Export a non-empty prediction for only that instance.
3. Run `swebench-evaluate` with an explicit timeout and output directory.
4. Confirm the normalized result is `resolved` or `unresolved`, not an
   infrastructure status.
5. Confirm the selected instance appears in the upstream `completed_ids` and
   exactly one of `resolved_ids` or `unresolved_ids`.
6. Change one byte of `model_patch` and confirm the derived run ID changes.

The normal automated test suite mocks the upstream subprocess/report boundary;
it does not pull or build large Docker images.

## Upstream references

- [SWE-bench evaluation guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md)
- [Official evaluation entry point](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py)
- [Official report construction](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/reporting.py)
