# H2.2 Local Live Benchmark Comparison

This is a small exploratory benchmark. Confidence intervals are wide and should not be interpreted as production-level statistical evidence.

## Quality, cost, and latency

| Configuration | N | First-pass | Final | Tokens/task | LLM calls/task | Median runtime |
|---|---:|---:|---:|---:|---:|---:|
| Full RepoGraph | 10 | 0.0% | 0.0% | 50060.20 | 9.10 | 117.60s |
| No Self-Correction | 10 | 0.0% | 0.0% | 43714.90 | 8.00 | 71.47s |
| No Agentic Exploration | 10 | 0.0% | 0.0% | N/A — incomplete provider usage | 8.57 | 66.33s |
| No Agentic Test | 10 | 0.0% | 0.0% | N/A — incomplete provider usage | 9.71 | 48.52s |

## Paired task matrix

| Task | Full | No Correction | No Explore | No Agentic Test |
|---|---:|---:|---:|---:|
| `bugfix-arithmetic` | ✗ | ✗ | ✗ | ✗ |
| `cache-key-case` | ✗ | ✗ | ✗ | ✗ |
| `edge-clamp` | ✗ | ✗ | ✗ | ✗ |
| `edge-division` | ✗ | ✗ | ✗ | ✗ |
| `feature-currency` | ✗ | ✗ | ✗ | ✗ |
| `feature-even` | ✗ | ✗ | ✗ | ✗ |
| `format-slug` | ✗ | ✗ | ✗ | ✗ |
| `multi-file-validation` | ✗ | ✗ | ✗ | ✗ |
| `refactor-tags` | ✗ | ✗ | ✗ | ✗ |
| `regression-bool` | ✗ | ✗ | ✗ | ✗ |

## Bootstrap 95% confidence intervals

- Full RepoGraph: 0.0% [0.0%, 0.0%]
- No Self-Correction: 0.0% [0.0%, 0.0%]
- No Agentic Exploration: 0.0% [0.0%, 0.0%]
- No Agentic Test: 0.0% [0.0%, 0.0%]

## Paired deltas (Full minus ablation)

- No Self-Correction: resolve 0.0% CI [0.0%, 0.0%]; LLM calls 1.10; tokens 6345.30; runtime 52.53s.
- No Agentic Exploration: resolve 0.0% CI [0.0%, 0.0%]; LLM calls N/A — incomplete provider usage; tokens N/A — incomplete provider usage; runtime 48.84s.
- No Agentic Test: resolve 0.0% CI [0.0%, 0.0%]; LLM calls N/A — incomplete provider usage; tokens N/A — incomplete provider usage; runtime 64.04s.

## Self-correction analysis

- Rescued: None
- Regressed after correction: None
