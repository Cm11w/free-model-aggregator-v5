// The catalog: what each provider lists, probed in its own wire, with the
// quality table as the fallback when a probe cannot reach its provider.
//
// Probing is the entry's first contact with the providers, and a probe that
// fails must not take its provider out of the trial: a failed probe is retried
// (a refusal or a torn answer is usually a momentary thing), and when the
// retries are spent the provider is still catalogued from the brief's quality
// table, which is the sealed truth about its free models. A provider is only
// left out of the catalog when neither its listing nor the brief says anything
// about it.
//
// The one catalog object built here is used for `--list` and for the register
// body alike, so the two documents are byte-identical by construction. Provider
// order follows the brief; models are sorted by id; nothing here reads a clock.

import {
  authHeaders,
  dialBase,
  isRecord,
  ratingRowFor,
  type Brief,
  type Json,
  type ProviderEntry,
  type RatingRow,
  type Wire,
} from './brief.ts';
import { sleep } from './faults.ts';

/** One model as a catalog route listed it. Null members are what the listing did not say. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The outcome of probing one provider's catalog route. */
export interface ProbeOutcome {
  provider: ProviderEntry;
  reached: boolean;
  models: ListedModel[];
  problem: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const PROBE_TIMEOUT_MS = 5000;
const PROBE_ATTEMPTS = 3;
const PROBE_RETRY_WAIT_MS = 50;

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asFinite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** Whether a listing's pricing block prices the model at zero everywhere it speaks. */
function zeroPriced(row: Json): boolean | null {
  const pricing = isRecord(row.pricing) ? row.pricing : null;
  if (pricing === null) return null;
  const amounts = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (amounts.length === 0) return null;
  return amounts.every((n) => Number.isFinite(n) && n === 0);
}

/** One listing row as a model, in whichever wire's spelling the listing used. */
function listedModel(wire: Wire, row: unknown): ListedModel | null {
  if (!isRecord(row)) return null;
  const id = wire === 'gemini' ? asString(row.name).replace(/^models\//, '') : asString(row.id);
  if (id === '') return null;
  return {
    id,
    displayName: asString(row.display_name) || asString(row.displayName) || id,
    free: asBool(row.free) ?? zeroPriced(row),
    contextLength: asFinite(row.context_length) ?? asFinite(row.context_window) ?? asFinite(row.inputTokenLimit),
    tools: asBool(row.tools),
    vision: asBool(row.vision),
    json: asBool(row.json),
  };
}

/** The models one catalog answer listed, in its wire's shape. */
function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const rows = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((row) => {
    const model = listedModel(wire, row);
    return model === null ? [] : [model];
  });
}

async function probeOnce(envUrl: string, provider: ProviderEntry): Promise<ProbeOutcome> {
  const base = dialBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, reached: false, models: [], problem: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, reached: true, models: listedModels(provider.wire, body), problem: null };
  } catch (err) {
    return { provider, reached: false, models: [], problem: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** Probe one provider, retrying a failed probe on a fixed ladder before giving up. */
export async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<ProbeOutcome> {
  let last: ProbeOutcome | null = null;
  for (let attempt = 0; attempt < PROBE_ATTEMPTS; attempt++) {
    if (attempt > 0) await sleep(PROBE_RETRY_WAIT_MS);
    const outcome = await probeOnce(envUrl, provider);
    if (outcome.reached) return outcome;
    last = outcome;
  }
  return last ?? { provider, reached: false, models: [], problem: 'the probe never ran' };
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 * A model is only ever routed when the table calls it free.
 */
export function isFree(listed: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return listed.free ?? false;
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog document: every provider the probes reached, plus every provider
 * the probes missed but the brief's table covers, each with the models the
 * listing named (or the table's rows when the listing never answered). The
 * base_url is the directory's own path, never joined to the env origin: the
 * origin differs between a live pass and its replay, and the catalog is part
 * of the state export a replay must reproduce byte for byte.
 */
export function buildCatalog(brief: Brief, probes: readonly ProbeOutcome[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const provider = probe.provider;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const rows = [...probe.models].sort(byModelId);
    if (probe.reached) {
      for (const model of rows) {
        if (seen.has(model.id)) continue;
        seen.add(model.id);
        const row = ratingRowFor(brief.ratings, provider.id, model.id);
        models.push({ id: model.id, free: isFree(model, row), context_length: row?.contextLength ?? model.contextLength ?? null });
      }
    } else {
      // The probe never answered: fall back to the quality table's rows, which
      // name exactly the provider's free models. The catalog stays complete.
      const fallback = brief.ratings
        .filter((r) => r.provider === provider.id && r.free)
        .map((r) => r.model)
        .sort();
      for (const id of fallback) {
        if (seen.has(id)) continue;
        seen.add(id);
        const row = ratingRowFor(brief.ratings, provider.id, id);
        models.push({ id, free: true, context_length: row?.contextLength ?? null });
      }
      if (models.length === 0) continue; // Nothing anywhere says this provider exists: leave it out.
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { providers };
}

/** Probe every provider in the brief's order, one at a time, and build the catalog. */
export async function loadCatalog(envUrl: string, brief: Brief, onProblem: (message: string) => void): Promise<{ probes: ProbeOutcome[]; catalog: Catalog }> {
  const probes: ProbeOutcome[] = [];
  for (const provider of brief.providers) {
    const outcome = await probeProvider(envUrl, provider);
    if (!outcome.reached && outcome.problem !== null) onProblem(outcome.problem);
    probes.push(outcome);
  }
  return { probes, catalog: buildCatalog(brief, probes) };
}
