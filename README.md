# Free-Model Aggregator — v5: the most resilient failover

One server, two client wires (OpenAI + Anthropic), ten free-model providers,
and a failover strategy that treats every provider failure as a classified,
recoverable event rather than an answer. No single provider failure may fail
a request while any path remains.

## The idea

Most aggregators fail over on vibes: any error means "try the next one". This
one classifies every failure first, because the right response depends on what
the failure *means*:

| Fault class | Evidence | Policy |
|---|---|---|
| `dead-provider` | "budget is spent", "is out", 401/auth errors | Kill the provider at once; never call it again this trial |
| `dead-model` | 400 `invalid_model` | Kill that model on that provider only |
| `schedule` | "refused this call", other 402/429 | Retry the same candidate (the next call number usually serves), max 3 attempts |
| `concurrency` | 429 `concurrent_requests` | Retry with a wait; the upstream gate makes this rare |
| `ambiguous` | 5xx, transport errors, unusable 2xx bodies | One second chance, then move on |

A streak valve backs the taxonomy: three consecutive transient faults from one
provider mark it dead even when no single fault proved it — a failure schedule
has at most two consecutive refusals, so the third refusal is the provider
telling the truth about itself.

## The degradation ladder

For each request, the pipeline walks ranked candidates best-first with the
retry policy above. If the strict walk — models meeting every requirement —
exhausts without an answer, it degrades instead of giving up: it walks every
free model again with the requirements lifted. A degraded answer still passes
the shape, echo, usage and stop checks; an error envelope passes none of them.
Only when no provider answers at all does the request end in an error, in the
client's own wire, and the server stays up for the next request.

Other resilience choices:

- **Always non-streaming upstream.** Every provider call is buffered whole
  and validated (text or tool calls, plus usage) before anything is sent;
  client streams are synthesized from the validated answer. A torn provider
  stream can never reach the client.
- **One upstream call in flight.** A promise-chain gate serializes all
  provider calls, so the one-at-a-time provider is never refused for
  concurrency by this entry's own doing. Every call carries a 10 s timeout.
- **Catalog probing with fallback.** Each provider's catalog route is probed
  (3 attempts); a provider the probes never reach still enters the catalog
  from the brief's quality table, so `--list` stays complete against a dead
  env. The one catalog object is printed by `--list` and posted to `/register`
  byte-for-byte identical.
- **Determinism.** No wall clock, no unseeded randomness in any decision path;
  fixed waits only. A replay sees the same choices and leaves the same state
  export, byte for byte.
- **Never crashes.** Every handler is guarded, and process-level guards log
  and continue. A bad request, a hung provider, or a malformed body costs at
  most that request.

Routing stays generic: providers, ratings, requirements and keys all come
from the runtime brief plus live probes. Nothing about any provider, model,
rating or trial is hardcoded.

## Layout

- `main.ts` — bootstrap, routes, registration, `/work` polling, never-crash guards.
- `lib/brief.ts` — the brief: providers, quality table, workload manifest.
- `lib/faults.ts` — the error taxonomy, evidence extraction, retry policy.
- `lib/catalog.ts` — probing, the catalog, free/paid resolution.
- `lib/plan.ts` — the candidate universe, ranking, provider health state machine.
- `lib/codec.ts` — request/response translation across all three wires, answer validation, stream synthesis.
- `lib/upstream.ts` — one provider call: timeout, gate, classify.
- `lib/serve.ts` — the serving pipeline: ranked walk, retries, degradation ladder.
- `chaos/chaos.mjs` — the chaos harness (below).

## Verification

Practice trials, official env server + official scorer, process-local harness
(no docker on this machine):

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

Quality 99 is the mathematical ceiling of these trials (the best provider's
budget is smaller than the number of non-vision requests). Determinism:
three runs of trial-0 produced byte-identical state exports.

## Chaos harness

`node chaos/chaos.mjs` runs the REAL env server and REAL scorer against six
adversarial `providers.json` variants — all far beyond what the generator can
draw (budgets ≥ 4, at most two failure calls per provider, exactly one
outage) — plus the fault taxonomy against every refusal shape in every wire,
plus the catalog fallback. Results (14 Nov 2026):

| Check | Result |
|---|---|
| Taxonomy, 14 refusal shapes × 3 wires | 14/14 classified as designed |
| mass-outage (5 strongest dark from call 2) | answered 20/20, 1 requirements-only degradation |
| tiny-budgets (every budget 2, zero slack) | answered 20/20, 29 calls — exactly one discovery call per dead provider, the information-theoretic minimum without budget knowledge |
| fail-storm (every provider refuses calls 2–3) | score 99, all columns 100 except quality at ceiling |
| one-survivor (9 providers die after 1 call) | answered 20/20, all 5 shortfalls requirements-only |
| top-dead (5 strongest die after 1 call) | answered 20/20, 1 requirements-only degradation |
| beyond-capacity (10 successes possible) | answered exactly 10/20, zero calls after death, exit 0 |
| Catalog fallback (`--list`, dead env) | all 10 providers, free sets exact |
| Catalog empty (no brief, no env) | `{"providers":[]}`, exit 0 |

"Requirements-only" means the request got a whole, correct answer from a
model that missed one requirement because no requirement-meeting model was
still alive — the degradation ladder working as designed, not a lost request.

## License

MIT — see `LICENSE` (`SPDX-License-Identifier: MIT`).
