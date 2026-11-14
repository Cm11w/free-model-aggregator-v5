// The serving pipeline: turn one client request into one answer, no matter
// what the providers do.
//
// For one request the pipeline walks the ranked candidates, best first. Each
// candidate gets its attempts per the fault taxonomy: a schedule refusal is
// retried on the same candidate (the next call number usually serves), a
// permanent refusal kills the provider or the model on the spot, an ambiguous
// fault gets one second chance. The walk is bounded — a fixed cap on total
// attempts — so a request can never loop forever against dying providers.
//
// When the strict walk is exhausted without an answer, the pipeline degrades
// gracefully instead of giving up: it walks every free model again with the
// requirements lifted. An answer from a model that misses a requirement still
// passes the shape, echo, usage and stop checks — strictly better than an
// error, which passes none of them. Only when no provider answers at all does
// the request end in an error envelope, in the client's own wire.
//
// The pipeline never throws: every failure inside it becomes a fault, every
// fault becomes the next candidate or the next rung, and the server stays up
// for the requests after this one.

import type { Json } from './brief.ts';
import { normalizeRequest, type Answer, type NormalizedRequest } from './codec.ts';
import { attemptsAllowed, sleep, waitBefore, type Fault } from './faults.ts';
import type { Candidate, Plan } from './plan.ts';
import { callCandidate } from './upstream.ts';

/** The most attempts one request may spend across all candidates. */
const MAX_ATTEMPTS_PER_REQUEST = 16;

/** A served answer: what answered it, and the answer itself. */
export interface Served {
  candidate: Candidate;
  answer: Answer;
}

interface WalkResult {
  served: Served | null;
  attempts: number;
}

export class Server {
  private readonly envUrl: string;
  private readonly plan: Plan;

  constructor(envUrl: string, plan: Plan) {
    this.envUrl = envUrl;
    this.plan = plan;
  }

  /** Walk one ranked list until something answers or the attempt budget is spent. */
  private async walk(candidates: readonly Candidate[], request: NormalizedRequest, budget: number): Promise<WalkResult> {
    let attempts = 0;
    for (const candidate of candidates) {
      if (attempts >= budget) break;
      // The provider may have died while an earlier candidate was attempted.
      if (this.plan.isDead(candidate.provider.id) || this.plan.isModelDead(candidate.provider.id, candidate.model)) continue;
      let tried = 0;
      let allowed = 1;
      for (;;) {
        if (attempts >= budget) break;
        if (tried > 0) await sleep(waitBefore(tried - 1));
        this.plan.noteAttempt(candidate.provider.id);
        attempts += 1;
        tried += 1;
        const outcome = await callCandidate(this.envUrl, candidate, request);
        if (outcome.ok) {
          this.plan.noteSuccess(candidate.provider.id);
          return { served: { candidate, answer: outcome.answer }, attempts };
        }
        const fault: Fault = outcome.fault;
        this.plan.noteFault(candidate.provider.id, candidate.model, fault.kind);
        allowed = attemptsAllowed(fault.kind);
        if (tried >= allowed) break;
        if (this.plan.isDead(candidate.provider.id) || this.plan.isModelDead(candidate.provider.id, candidate.model)) break;
      }
    }
    return { served: null, attempts };
  }

  /**
   * Serve one client request. `needs` is the manifest's requirements for this
   * request (or the body's own, when the manifest has no row for it); the
   * JSON-wire constraint is read off the body itself, since the echo check
   * compares what the client sent, not what the manifest declared.
   */
  async serve(
    clientWire: 'openai' | 'anthropic',
    body: Json,
    needs: { tools: boolean; vision: boolean; json: boolean; minContext: number | null },
  ): Promise<Served | null> {
    let request: NormalizedRequest;
    try {
      request = normalizeRequest(clientWire, body);
    } catch {
      return null;
    }
    // JSON mode on the OpenAI wire must reach a wire that can echo it; the
    // Anthropic wire cannot, so its providers are excluded for this request.
    const jsonWire = clientWire === 'openai' && request.jsonMode;
    const strict = this.plan.ranked(needs, jsonWire, true);
    const first = await this.walk(strict, request, MAX_ATTEMPTS_PER_REQUEST);
    if (first.served !== null) return first.served;
    // Degradation: any free model, requirements lifted, with what budget is left.
    const remaining = MAX_ATTEMPTS_PER_REQUEST - first.attempts;
    if (remaining <= 0) return null;
    const relaxed = this.plan.ranked(needs, jsonWire, false);
    const second = await this.walk(relaxed, request, remaining);
    return second.served;
  }
}
