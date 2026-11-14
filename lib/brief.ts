// The brief: the three JSON documents a trial hands the entry, found by schema.
//
// The brief file is prose with three JSON documents embedded in it. Each document
// carries a `schema` member, and that member is the only thing this parser trusts:
//   providers  free-model-aggregator-01/providers/v1  — the directory of providers
//   ratings    free-model-aggregator-01/ratings/v1     — the quality table (sealed truth)
//   workload   free-model-aggregator-01/workload/v1    — the request manifest
// Everything else in the file is ignored. A document that is absent or malformed
// degrades to an empty list rather than aborting the trial: the entry must keep
// working with whatever it could read, and say so on stderr.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** A JSON object, the only shape this entry treats as a document, a body or a row. */
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** One provider as the directory lists it. `base` is its path under the env, or an absolute URL. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table: the sealed truth about a model. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Needs {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

/** One row of the workload manifest, in the order the client sends the requests. */
export interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  needs: Needs;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
  /** Non-null when the brief could not be read at all; the lists above are then empty. */
  problem: string | null;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asFinite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The index of the `}` that closes the object opening at `start`, skipping string contents, or -1. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`, in file order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // A lone `{` in prose: keep scanning.
    }
  }
  return found;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none' ? v : 'bearer';
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = wireOf(v.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path`; accept the absolute spellings too, `path` first.
  const base = asString(v.path) || asString(v.base_path) || asString(v.base_url) || asString(v.base);
  return {
    id,
    name: asString(v.name) || id,
    wire,
    auth: authOf(v.auth),
    base,
    keyEnv: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: asFinite(v.rating_milli) ?? 0,
    contextLength: asFinite(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestOf(v: unknown): ManifestRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const needs = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    needs: {
      tools: needs.tools === true,
      vision: needs.vision === true,
      json: needs.json === true,
      minContext: asFinite(needs.min_context),
    },
  };
}

/** Read and parse the brief at `path`. Never throws: failure is reported in `problem`. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined || !Array.isArray(directory.providers)) {
    return { ...empty, problem: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directory.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings = ratingsDoc === undefined || !Array.isArray(ratingsDoc.ratings)
    ? []
    : ratingsDoc.ratings.flatMap((v) => {
        const r = ratingOf(v);
        return r === null ? [] : [r];
      });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest = workloadDoc === undefined || !Array.isArray(workloadDoc.manifest)
    ? []
    : workloadDoc.manifest.flatMap((v) => {
        const m = manifestOf(v);
        return m === null ? [] : [m];
      });
  return { providers, ratings, manifest, problem: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowFor(ratings: readonly RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** The headers that carry one provider's key, per its own auth scheme. The wire never decides this. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A provider's base joined to the env origin; an absolute base is used as it stands. */
export function dialBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return origin;
  return `${origin}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}
