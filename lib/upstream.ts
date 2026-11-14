// The upstream call: one request to one provider, on its own wire, with a
// timeout, through the gate that keeps at most one provider call in flight.
//
// The gate matters: a provider serves one inference call at a time, and a
// second call sent while the first is in flight is refused outright. The
// client sends one request at a time, so the gate is almost never contended —
// but "almost never" is not "never", and a refusal spent on concurrency is a
// refusal the entry chose. Every attempt goes through the same promise chain,
// and every attempt carries its own timeout, so a hung provider can delay the
// chain by at most one timeout, never wedge it.
//
// A call returns either a validated answer or a classified fault. A 2xx whose
// body cannot be parsed, or that carries no text and no tool call, is a fault,
// not an answer: the client must never be served a hollow reply when another
// candidate may still answer whole.

import { authHeaders, dialBase, type Json, type ProviderEntry, type Wire } from './brief.ts';
import { parseAnswer, upstreamBody, type Answer, type NormalizedRequest } from './codec.ts';
import { classify, evidenceOf, transportFault, unusableFault, type Fault } from './faults.ts';
import type { Candidate } from './plan.ts';

/** How long one provider call may take before it is abandoned as a fault. */
const CALL_TIMEOUT_MS = 10000;

export type CallOutcome = { ok: true; answer: Answer } | { ok: false; fault: Fault };

/** The upstream gate: a promise chain every attempt joins, one at a time. */
let gate: Promise<void> = Promise.resolve();

function throughGate<T>(work: () => Promise<T>): Promise<T> {
  const run = gate.then(work, work);
  gate = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

function callUrl(base: string, wire: Wire, model: string): string {
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${model}:generateContent`;
}

async function attempt(envUrl: string, candidate: Candidate, request: NormalizedRequest): Promise<CallOutcome> {
  const provider: ProviderEntry = candidate.provider;
  const base = dialBase(envUrl, provider.base);
  const url = callUrl(base, provider.wire, candidate.model);
  const body: Json = upstreamBody(provider.wire, request, candidate.model);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    const detail = (err as Error).name === 'TimeoutError' ? `the call to ${provider.id} timed out after ${CALL_TIMEOUT_MS} ms` : `the call to ${provider.id} failed: ${(err as Error).message}`;
    return { ok: false, fault: transportFault(detail) };
  }
  let text: string;
  try {
    text = await res.text();
  } catch (err) {
    return { ok: false, fault: transportFault(`the answer from ${provider.id} could not be read: ${(err as Error).message}`) };
  }
  if (!res.ok) {
    return { ok: false, fault: classify(evidenceOf(res.status, text)) };
  }
  const answer = parseAnswer(provider.wire, text);
  if (answer === null) {
    return { ok: false, fault: unusableFault(`the answer from ${provider.id} carried no usable text, tool call or usage`) };
  }
  return { ok: true, answer };
}

/** One attempt at one candidate, through the gate. Never throws. */
export async function callCandidate(envUrl: string, candidate: Candidate, request: NormalizedRequest): Promise<CallOutcome> {
  try {
    return await throughGate(() => attempt(envUrl, candidate, request));
  } catch (err) {
    return { ok: false, fault: transportFault(`the call machinery failed: ${(err as Error).message}`) };
  }
}
