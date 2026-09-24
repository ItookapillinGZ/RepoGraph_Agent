# H2.3D Direct OpenAI clean paired benchmark

This is a controlled exploratory benchmark (N=17), not statistical proof or a general software-engineering success rate.

| Configuration | Attempted | Valid | Capability | End-to-end / 17 | Infrastructure / 17 |
|---|---:|---:|---:|---:|---:|
| full | 17/17 | 17 | 100.0% | 100.0% | 0.0% |
| no_correction | 17/17 | 17 | 94.1% | 94.1% | 0.0% |
| no_exploration | 4/17 | 3 | 100.0% | 17.6% | 5.9% |
| no_agentic_test | 0/17 | 0 | n/a | 0.0% | 0.0% |

R = resolved valid prediction; U = unresolved valid prediction; I = infrastructure failure; N = no valid candidate from a non-infrastructure agent failure.

| Task | Full | No Correction | No Exploration | No Agentic Test |
|---|---:|---:|---:|---:|
| batch-rollback-state | R | R | R | N |
| dependency-load-order | R | R | R | N |
| domain-timezone-serialization | R | R | R | N |
| email-domain-policy | R | R | I | N |
| event-retry-deduplication | R | R | N | N |
| inventory-reservation-atomicity | R | R | N | N |
| invoice-rounding-policy | R | R | N | N |
| nested-secret-redaction | R | R | N | N |
| pagination-zero-cursor | R | R | N | N |
| permission-parent-inheritance | R | R | N | N |
| record-parser-contract | R | R | N | N |
| registry-alias-collision | R | U | N | N |
| retry-idempotency-key | R | R | N | N |
| session-refresh-persistence | R | R | N | N |
| settings-source-precedence | R | R | N | N |
| specific-route-precedence | R | R | N | N |
| subscription-listener-removal | R | R | N | N |

## Comparable pairs

- Full − no_correction: comparable 17; Full wins 1; ablation wins 0; resolved ties 16; unresolved ties 0; excluded infrastructure 0; capability delta 0.058823529411764705.
- Full − no_exploration: comparable 3; Full wins 0; ablation wins 0; resolved ties 3; unresolved ties 0; excluded infrastructure 1; capability delta 0.0.
- Full − no_agentic_test: comparable 0; Full wins 0; ablation wins 0; resolved ties 0; unresolved ties 0; excluded infrastructure 0; capability delta None.

Capability pairs require valid predictions on both sides. Historical RelayAPI/Sol observations are contextual only because provider, model, and campaign time all changed.
