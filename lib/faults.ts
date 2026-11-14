// The fault taxonomy: every way a provider call can fail, classified so the
// serving pipeline can pick the right response to each one.
//
// A provider refusal is never just "an error". The wire envelopes carry a status,
// a code and a message, and the message is what separates the failures that will
// never heal from the ones that heal on the very next call:
//
//   budget spent   "…free-tier budget is spent"  — permanent: every later call refuses.
//   outage         "…is out"                    — permanent: every call from the outage on refuses.
//   schedule       "…refused this call"         — transient: the NEXT call number usually serves.
//   concurrency    code concurrent_requests     — transient: one call at a time; wait a beat, retry.
//   auth           401                          — permanent: the key never changes mid-trial.
//   bad model      400 invalid_model            — permanent for that model; the request shape is fine.
//   anything else  5xx, network, timeout, junk  — ambiguous: retry once, then move on.
//
// The taxonomy reads the error the way each wire spells it (OpenAI's error.code,
// Anthropic's error.type, Gemini's error.status), so a refusal is classified the
// same no matter which wire carried it. Classification is a pure function of the
// failure observed: no clock, no randomness, identical on a replay.

/** The classes of provider failure, each with its own policy. */
export type FaultClass =
  | 'dead-provider' // The provider will refuse every later call. Never retry it.
  | 'dead-model' // This model will never serve. Other models on the provider may.
  | 'schedule' // A failure-schedule refusal: the next call number usually serves.
  | 'concurrency' // A second call while one was in flight: wait, then retry.
  | 'ambiguous'; // A 5xx, a network fault, a timeout, or a body that makes no sense.

export interface Fault {
  kind: FaultClass;
  /** Human-readable detail for the log; never sent anywhere. */
  detail: string;
  /** The HTTP status when the provider answered one, else 0. */
  status: number;
}

/** What a failed provider call looked like, in wire-neutral form. */
export interface FailureEvidence {
  status: number;
  /** The wire's error code, when the envelope carried one. */
  code: string;
  /** The wire's error message, when the envelope carried one. */
  message: string;
}

/**
 * Pull the status, code and message out of one provider error body, in any of the
 * three wires' envelopes. A body that is not JSON, or an envelope in an unknown
 * shape, yields empty strings: the classifier below then falls back to the status.
 */
export function evidenceOf(status: number, bodyText: string): FailureEvidence {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    parsed = null;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { status, code: '', message: '' };
  }
  const root = parsed as Record<string, unknown>;
  const error = root.error;
  if (error === null || typeof error !== 'object' || Array.isArray(error)) {
    return { status, code: '', message: typeof root.message === 'string' ? root.message : '' };
  }
  const envelope = error as Record<string, unknown>;
  const code =
    typeof envelope.code === 'string'
      ? envelope.code
      : typeof envelope.type === 'string'
        ? envelope.type
        : typeof envelope.status === 'string'
          ? envelope.status
          : '';
  return { status, code, message: typeof envelope.message === 'string' ? envelope.message : '' };
}

/** Classify one provider failure from its evidence. */
export function classify(evidence: FailureEvidence): Fault {
  const { status, code, message } = evidence;
  const detail = `HTTP ${status} ${code} ${message}`.trim();
  // Concurrency first: it is the only 429 that is about timing, not budget.
  if (code === 'concurrent_requests' || /one call at a time|serves \d+ call/i.test(message)) {
    return { kind: 'concurrency', detail, status };
  }
  // The two permanent refusals, told apart from a schedule refusal by the message.
  if (/budget is spent/i.test(message)) return { kind: 'dead-provider', detail, status };
  if (/\bis out\b/i.test(message)) return { kind: 'dead-provider', detail, status };
  if (/refused this call/i.test(message)) return { kind: 'schedule', detail, status };
  // Auth and model faults are permanent: retrying them can only waste calls.
  if (status === 401 || code === 'invalid_api_key' || code === 'authentication_error' || code === 'UNAUTHENTICATED') {
    return { kind: 'dead-provider', detail, status };
  }
  if (status === 400) return { kind: 'dead-model', detail, status };
  // A 402/429 in an unfamiliar shape: the schedule is the likeliest story, but the
  // streak valve in the health tracker will kill the provider if it keeps refusing.
  if (status === 402 || status === 429) return { kind: 'schedule', detail, status };
  return { kind: 'ambiguous', detail, status };
}

/** A failure with no provider answer at all: a network fault, a timeout, a torn body. */
export function transportFault(detail: string): Fault {
  return { kind: 'ambiguous', detail, status: 0 };
}

/** A 2xx answer whose body cannot be used: unparseable, or missing the answer itself. */
export function unusableFault(detail: string): Fault {
  return { kind: 'ambiguous', detail, status: 200 };
}

/**
 * The retry ladder: fixed waits between attempts on the same candidate, in
 * milliseconds. Fixed waits are deterministic — a wait is never written into
 * anything sent, and the same failures take the same path on a replay. The
 * ladder is short on purpose: the client allows 30 s a request, and a local
 * provider answers in milliseconds, so a long sleep can only hurt.
 */
export const RETRY_WAITS_MS = [0, 30, 60] as const;

/** How many attempts one candidate gets for one request, per fault class. */
export function attemptsAllowed(fault: FaultClass): number {
  switch (fault) {
    case 'dead-provider':
    case 'dead-model':
      return 1; // Never again: the next attempt would fail the same way.
    case 'schedule':
      return 3; // At most two consecutive schedule refusals exist per provider.
    case 'concurrency':
      return 3; // A beat of waiting clears the in-flight slot.
    case 'ambiguous':
      return 2; // One second chance, then move on: the fault may be the candidate.
  }
}

/** The wait before attempt `n` (0-based) on the same candidate. */
export function waitBefore(attempt: number): number {
  return RETRY_WAITS_MS[Math.min(attempt, RETRY_WAITS_MS.length - 1)];
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
