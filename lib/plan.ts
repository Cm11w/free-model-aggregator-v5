// The plan: which free models may serve which requests, in which order, and
// what the entry has learned about each provider so far.
//
// The candidate universe is built once from the probes and the quality table:
// a model is a candidate only when the table calls it free and the provider's
// listing names it (or the listing never answered, in which case the table's
// rows stand in). A paid model is never a candidate, whatever its rating.
//
// Ranking is by rating first — quality is 60% of the score — with ties broken
// by spreading load: among equal ratings, the provider the entry has called
// least goes first, so no single provider's unknown budget is drained while an
// equal alternative sits idle. The final tie-breaks are provider id and model
// id, so the order is total and deterministic.
//
// Health is a small state machine per provider, driven only by classified
// faults the entry has actually observed:
//   unknown  — nothing seen yet; the provider is fully trusted.
//   strained — transient refusals seen; still trusted, still ranked by rating.
//   dead     — a permanent refusal seen, or too many transient refusals in a
//              row for a failure schedule to explain; never called again.
// Dead providers are skipped outright: calling one again could only waste a
// call the tie-break counts, and could never serve.

import { isFree, type ListedModel, type ProbeOutcome } from './catalog.ts';
import type { Brief, Needs, ProviderEntry, RatingRow } from './brief.ts';
import { ratingRowFor } from './brief.ts';
import type { FaultClass } from './faults.ts';

/** One routable model: its provider, its rating, and what it can do. */
export interface Candidate {
  provider: ProviderEntry;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export type ProviderHealth = 'unknown' | 'strained' | 'dead';

interface ProviderState {
  health: ProviderHealth;
  /** Calls this entry has sent the provider (every attempt, success or refusal). */
  callsMade: number;
  /** Transient refusals in a row; a success resets it. */
  transientStreak: number;
  deadModels: Set<string>;
}

/** More consecutive transient refusals than any failure schedule can explain. */
const MAX_TRANSIENT_STREAK = 3;

export class Plan {
  readonly candidates: Candidate[] = [];
  private readonly states = new Map<string, ProviderState>();

  constructor(brief: Brief, probes: readonly ProbeOutcome[]) {
    for (const provider of brief.providers) {
      this.states.set(provider.id, { health: 'unknown', callsMade: 0, transientStreak: 0, deadModels: new Set() });
    }
    for (const probe of probes) {
      const provider = probe.provider;
      if (probe.reached) {
        for (const listed of probe.models) {
          const row = ratingRowFor(brief.ratings, provider.id, listed.id);
          if (row === null || !row.free) continue; // No free row: paid or unknown, never routed.
          void listed;
          this.candidates.push(candidateOf(provider, row));
        }
      } else {
        // The listing never answered: the table's free rows stand in.
        for (const row of brief.ratings) {
          if (row.provider === provider.id && row.free) this.candidates.push(candidateOf(provider, row));
        }
      }
    }
    // De-duplicate defensively: the same (provider, model) must appear once.
    const seen = new Set<string>();
    this.candidates = this.candidates.filter((c) => {
      const key = `${c.provider.id}/${c.model}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  private stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { health: 'unknown', callsMade: 0, transientStreak: 0, deadModels: new Set() };
      this.states.set(providerId, state);
    }
    return state;
  }

  isDead(providerId: string): boolean {
    return this.stateOf(providerId).health === 'dead';
  }

  isModelDead(providerId: string, model: string): boolean {
    return this.stateOf(providerId).deadModels.has(model);
  }

  callsMade(providerId: string): number {
    return this.stateOf(providerId).callsMade;
  }

  /** Record that one attempt was sent. Called before the call, so counts stay ordered. */
  noteAttempt(providerId: string): void {
    this.stateOf(providerId).callsMade += 1;
  }

  /** Record a success: the provider is alive and its transient streak is over. */
  noteSuccess(providerId: string): void {
    const state = this.stateOf(providerId);
    state.transientStreak = 0;
    if (state.health === 'strained') state.health = 'unknown';
  }

  /** Record a classified fault and move the provider's health accordingly. */
  noteFault(providerId: string, model: string, fault: FaultClass): void {
    const state = this.stateOf(providerId);
    if (fault === 'dead-provider') {
      state.health = 'dead';
      return;
    }
    if (fault === 'dead-model') {
      state.deadModels.add(model);
      return;
    }
    state.transientStreak += 1;
    if (state.health === 'unknown') state.health = 'strained';
    if (state.transientStreak >= MAX_TRANSIENT_STREAK) state.health = 'dead';
  }

  /**
   * The candidates that may serve one request, best first. `needs` is the
   * manifest's requirements; `jsonWire` is true when the client's request
   * carries JSON mode on a wire the provider must be able to echo — an
   * anthropic-wire provider cannot carry it, so it is excluded then. Dead
   * providers and dead models are excluded. When `strict` is false the
   * requirements are ignored: the degradation ladder's last rung, where any
   * free model beats no answer at all.
   */
  ranked(needs: Needs, jsonWire: boolean, strict: boolean): Candidate[] {
    const pool = this.candidates.filter((c) => {
      if (this.isDead(c.provider.id) || this.isModelDead(c.provider.id, c.model)) return false;
      if (jsonWire && c.provider.wire === 'anthropic') return false;
      if (!strict) return true;
      if (needs.tools && !c.tools) return false;
      if (needs.vision && !c.vision) return false;
      if (needs.json && !c.json) return false;
      if (needs.minContext !== null && (c.contextLength === null || c.contextLength < needs.minContext)) return false;
      return true;
    });
    return pool.sort((a, b) => {
      if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
      const load = this.callsMade(a.provider.id) - this.callsMade(b.provider.id);
      if (load !== 0) return load;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
  }
}

function candidateOf(provider: ProviderEntry, row: RatingRow): Candidate {
  return {
    provider,
    model: row.model,
    ratingMilli: row.ratingMilli,
    contextLength: row.contextLength,
    tools: row.tools,
    vision: row.vision,
    json: row.json,
  };
}

/** The free model ids the catalog registered, de-duplicated, in catalog order. */
export function registeredFreeIds(catalog: { providers: readonly { models: readonly { id: string; free: boolean }[] }[] }): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  return ids;
}

export type { ListedModel };
export { isFree };
