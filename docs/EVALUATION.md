# RepoGraph evaluation

## What H2 measures

H2 measures whether a frozen RepoGraph configuration can produce a candidate
that resolves each controlled repository task under a frozen evaluator. It
separates first-pass capability from final results after bounded correction,
and separates Agent capability failures from end-to-end and infrastructure
failures.

The task manifest, evaluator, candidate artifacts, provenance, telemetry, and
campaign configuration are preserved with the historical campaign. H4 does not
rerun or rewrite those results.

## Authoritative controlled benchmark

- Tasks: N=17 controlled local tasks
- Provider: Direct OpenAI
- Model: `gpt-5.6-luna`
- Design: paired full/ablation comparison with frozen evaluator

| Variant | First pass | Final |
|---|---:|---:|
| Full | 15/17 (88.24%) | 17/17 (100%) |
| No Self-Correction | 16/17 (94.12%) | 16/17 (94.12%) |
| No Exploration | 16/17 (94.12%) | 16/17 (94.12%) |
| No Agentic Test | 16/17 (94.12%) | 16/17 (94.12%) |

In Full, 8 tasks entered correction, 2 were rescued, and 0 regressed.

## Interpretation

This is a small controlled exploratory benchmark. Many tasks were easy for the
selected model, producing a ceiling effect. Each final ablation difference is
one task, and paired confidence intervals include zero. The results support
debugging and design inspection; they do not establish statistical superiority,
SOTA performance, 95% general software-engineering accuracy, or a production
success rate.

Bootstrap results quantify uncertainty but cannot compensate for a small or
narrow task set. The paired design is more informative than treating variants
as unrelated samples, while still requiring conservative interpretation.

## Denominators

- Capability denominator: tasks that reached the Agent/evaluator path.
- End-to-end denominator: all requested task runs, including product failures.
- Infrastructure denominator: failures caused by environment or campaign
  infrastructure rather than candidate capability.

Keeping these separate prevents infrastructure availability from being
misreported as model quality and prevents capability-only numbers from being
presented as end-to-end reliability.

## H3 and H3.1 validation

H3 validated the real Docker boundary: network isolation, secret isolation,
filesystem isolation, resource limits, cleanup, and 5/5 Host/Docker semantic
equivalence cases. H3.1 validated hierarchical traces, artifact lineage,
secret-free persisted artifacts, and 5/5 frozen deterministic replay matches.

The observability microbenchmark measured roughly 0.170 s disabled versus
2.068 s for 100 durable SQLite event writes. This is a synthetic per-event
durable-write microbenchmark, not end-to-end product latency.

## SWE-bench status

SWE-bench dependencies are optional and separated in `requirements-eval.txt`.
No official full SWE-bench Docker campaign has been run for the portfolio
release. Future operators can export predictions and run the official evaluator
explicitly; H4 does not make live model calls or silently start an expensive
campaign.
