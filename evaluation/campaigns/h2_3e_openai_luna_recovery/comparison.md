# H2.3D Direct OpenAI clean paired benchmark

This is a controlled exploratory benchmark (N=17), not statistical proof or a general software-engineering success rate.

| Configuration | Attempted | Valid | Capability | End-to-end / 17 | Infrastructure / 17 |
|---|---:|---:|---:|---:|---:|
| full | 17/17 | 17 | 100.0% | 100.0% | 0.0% |
| no_correction | 17/17 | 17 | 94.1% | 94.1% | 0.0% |
| no_exploration | 17/17 | 17 | 94.1% | 94.1% | 0.0% |
| no_agentic_test | 17/17 | 17 | 94.1% | 94.1% | 0.0% |

R = resolved valid prediction; U = unresolved valid prediction; I = infrastructure failure; N = no valid candidate from a non-infrastructure agent failure.

| Task | Full | No Correction | No Exploration | No Agentic Test |
|---|---:|---:|---:|---:|
| batch-rollback-state | R | R | R | R |
| dependency-load-order | R | R | R | R |
| domain-timezone-serialization | R | R | R | R |
| email-domain-policy | R | R | R | R |
| event-retry-deduplication | R | R | R | R |
| inventory-reservation-atomicity | R | R | R | R |
| invoice-rounding-policy | R | R | R | R |
| nested-secret-redaction | R | R | R | R |
| pagination-zero-cursor | R | R | R | R |
| permission-parent-inheritance | R | R | R | R |
| record-parser-contract | R | R | R | R |
| registry-alias-collision | R | U | U | U |
| retry-idempotency-key | R | R | R | R |
| session-refresh-persistence | R | R | R | R |
| settings-source-precedence | R | R | R | R |
| specific-route-precedence | R | R | R | R |
| subscription-listener-removal | R | R | R | R |

## Comparable pairs

- Full − no_correction: comparable 17; Full wins 1; ablation wins 0; resolved ties 16; unresolved ties 0; excluded infrastructure 0; capability delta 0.058823529411764705.
- Full − no_exploration: comparable 17; Full wins 1; ablation wins 0; resolved ties 16; unresolved ties 0; excluded infrastructure 0; capability delta 0.058823529411764705.
- Full − no_agentic_test: comparable 17; Full wins 1; ablation wins 0; resolved ties 16; unresolved ties 0; excluded infrastructure 0; capability delta 0.058823529411764705.

Capability pairs require valid predictions on both sides. Historical RelayAPI/Sol observations are contextual only because provider, model, and campaign time all changed.
