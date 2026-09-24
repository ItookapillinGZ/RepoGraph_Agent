# H2.2 Offline Regrade of Live-LLM Predictions

This is a small exploratory benchmark. Confidence intervals are wide and should not be interpreted as production-level statistical evidence.

The original local grading is invalid. Portable dataset commands beginning with 'python' resolved to a system interpreter without pytest inside the isolated worker. Pytest's import failure returned exit code 1 and was misclassified as an assertion failure.
The live model predictions and append-only result history were not changed.
This regrade made zero LLM API calls.

## Quality, cost, and latency

| Configuration | N | First-pass | Final | Patch grade | Tokens/task | LLM calls/task | Median runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full RepoGraph | 10 | 80.0%–100.0% | 100.0% | 100.0% (10 patches) | 50060.20 | 9.10 | 117.60s |
| No Self-Correction | 10 | 100.0% | 100.0% | 100.0% (10 patches) | 43714.90 | 8.00 | 71.47s |
| No Agentic Exploration | 10 | 50.0%–70.0% | 70.0% | 100.0% (7 patches) | N/A | 8.57 | 66.33s |
| No Agentic Test | 10 | 50.0%–70.0% | 70.0% | 100.0% (7 patches) | N/A | 9.71 | 48.52s |

First-pass ranges are bounds, not confidence intervals. Corrected-task first candidates were not persisted and cannot be reconstructed.

## Paired final-result matrix

| Task | Full | No Correction | No Explore | No Agentic Test |
|---|---:|---:|---:|---:|
| `bugfix-arithmetic` | ✓ | ✓ | ✓ | API error |
| `cache-key-case` | ✓ | ✓ | ✓ | API error |
| `edge-clamp` | ✓ | ✓ | ✓ | API error |
| `edge-division` | ✓ | ✓ | ✓ | ✓ |
| `feature-currency` | ✓ | ✓ | ✓ | ✓ |
| `feature-even` | ✓ | ✓ | API error | ✓ |
| `format-slug` | ✓ | ✓ | API error | ✓ |
| `multi-file-validation` | ✓ | ✓ | API error | ✓ |
| `refactor-tags` | ✓ | ✓ | ✓ | ✓ |
| `regression-bool` | ✓ | ✓ | ✓ | ✓ |

## Bootstrap 95% confidence intervals

- Full RepoGraph: 100.0% [100.0%, 100.0%]
- No Self-Correction: 100.0% [100.0%, 100.0%]
- No Agentic Exploration: 70.0% [40.0%, 100.0%]
- No Agentic Test: 70.0% [40.0%, 100.0%]

## Paired deltas (Full minus ablation)

- No Self-Correction: final resolve 0.0% CI [0.0%, 0.0%]; LLM calls 1.10; tokens 6345.30; runtime 52.53s.
- No Agentic Exploration: final resolve 30.0% CI [0.0%, 60.0%]; LLM calls N/A; tokens N/A; runtime 48.84s.
- No Agentic Test: final resolve 30.0% CI [0.0%, 60.0%]; LLM calls N/A; tokens N/A; runtime 64.04s.

## Self-correction analysis

- Correction attempted: 2 tasks
- Confirmed rescues: None
- Possible rescues with unrecoverable first attempt: format-slug, multi-file-validation
- Rescue-rate bound: 0.0%–100.0%

The 30-point Full advantage over No Exploration and No Agentic Test is entirely attributable to three retained API connection failures in each ablation. It is not evidence that those capabilities improved solution quality.
