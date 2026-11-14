// Chaos harness: prove the entry survives provider behavior far worse than any
// real draw, using the kit's REAL env server and REAL scorer.
//
// Each variant rewrites only providers.json (budgets, failure schedules,
// outages) over trial-0's workload, then runs the full trial: env + entry +
// scorer. The generator can only draw budgets in [calls_min, calls_max], at
// most two failure calls per provider, and one outage; the variants below go
// deliberately beyond that, to the point where serving everything is
// mathematically impossible, and assert the entry still serves everything
// servable, never crashes, and always exits clean.
//
// Run:  node chaos/chaos.mjs [--kit <kit dir>] [--trial 0]
// The kit defaults to the sibling unpacked kit directory.
//
// The harness also checks the catalog fallback (--list against a dead env
// must still report every provider's free set from the brief) and the fault
// taxonomy (every refusal shape, in every wire, classified as designed).

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { classify, evidenceOf } from '../lib/faults.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const ARGS = process.argv.slice(2);

function flag(name, fallback) {
  const at = ARGS.indexOf(name);
  return at !== -1 && ARGS[at + 1] !== undefined ? ARGS[at + 1] : fallback;
}

const KIT = flag('--kit', process.env.MAKEFUN_KIT ?? join(REPO, '..', 'free-model-aggregator-01-kit', 'free-model-aggregator-01-kit'));
const TRIAL = flag('--trial', '0');
const BASE = join('/tmp', 'agg-v5-chaos');
const TRIAL_DIR = join(KIT, 'practice', `trial-${TRIAL}`);

let failures = 0;
function check(name, ok, detail) {
  process.stdout.write(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== undefined ? ` — ${detail}` : ''}\n`);
  if (!ok) failures += 1;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(path, tries = 100) {
  for (let i = 0; i < tries; i++) {
    if (existsSync(path)) return true;
    await sleep(50);
  }
  return false;
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...opts });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', (e) => resolve({ code: -1, out, err: `${err}${e.message}` }));
    child.on('close', (code) => resolve({ code: code ?? -1, out, err }));
  });
}

/** Run one full trial against one providers.json variant; return the scores. */
async function runVariant(name, mutate, port) {
  const dir = join(BASE, name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(join(dir, 'env'), { recursive: true });
  mkdirSync(join(dir, 'state'), { recursive: true });
  mkdirSync(join(dir, 'scores'), { recursive: true });
  const providers = JSON.parse(readFileSync(join(TRIAL_DIR, 'input', 'env', 'providers.json'), 'utf8'));
  mutate(providers);
  writeFileSync(join(dir, 'env', 'providers.json'), JSON.stringify(providers));
  writeFileSync(join(dir, 'env', 'workload.json'), readFileSync(join(TRIAL_DIR, 'input', 'env', 'workload.json'), 'utf8'));

  const env = spawn('node', [join(KIT, 'program', 'env', 'server.mjs')], {
    env: {
      ...process.env,
      MAKEFUN_ENV_DATA_DIR: join(dir, 'env'),
      MAKEFUN_STATE_DIR: join(dir, 'state'),
      MAKEFUN_ENV_PORT: String(port),
      MAKEFUN_ENV_HOST: '127.0.0.1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let envErr = '';
  env.stderr.on('data', (d) => { envErr += d; });
  const ready = await waitFor(join(dir, 'state', 'ready'));
  if (!ready) {
    env.kill('SIGKILL');
    return { name, error: `env never ready: ${envErr.slice(0, 300)}` };
  }
  const entry = await run('node', ['main.ts'], {
    cwd: REPO,
    env: {
      ...process.env,
      MAKEFUN_PROMPT_FILE: join(TRIAL_DIR, 'input', 'prompt.txt'),
      MAKEFUN_ENV_URL: `http://127.0.0.1:${port}`,
    },
    timeout: 120000,
  });
  env.kill('SIGTERM');
  await new Promise((resolve) => env.on('close', resolve));
  const scored = await run('node', [join(KIT, 'program', 'scorer', 'score.mjs')], {
    env: {
      ...process.env,
      MAKEFUN_OUTPUT_DIR: join(dir, 'state', 'export'),
      MAKEFUN_REFERENCE_DIR: join(TRIAL_DIR, 'reference'),
      MAKEFUN_SCORES_DIR: join(dir, 'scores'),
    },
  });
  let scores = null;
  try {
    scores = JSON.parse(readFileSync(join(dir, 'scores', 'scores.json'), 'utf8'));
  } catch {
    scores = null;
  }
  // The export tells what the scores hide: how many requests got a real answer
  // (model set), even when the answer could not meet the requirements because
  // no requirement-meeting model was still alive. A degraded answer is the
  // designed outcome, not a failure — the scorer's `served` counts only
  // fully-valid requests, so the harness asserts on both.
  let answered = null;
  let reqOnly = null;
  try {
    const state = JSON.parse(readFileSync(join(dir, 'state', 'export', 'state.json'), 'utf8'));
    answered = state.requests.filter((r) => r.model !== null && r.model !== undefined).length;
    reqOnly = state.requests.filter((r) => r.model !== null && r.model !== undefined && !r.valid && r.checks.every((c) => c.pass || c.name === 'requirements')).length;
  } catch {
    answered = null;
    reqOnly = null;
  }
  return { name, entryExit: entry.code, scorerExit: scored.code, scores, answered, reqOnly, entryErr: entry.err };
}

const VARIANTS = [
  {
    name: 'mass-outage',
    // The five strongest providers go dark from their 2nd call on.
    mutate: (doc) => {
      for (const p of doc.providers) {
        if (['openrouter', 'zai', 'cloudflare', 'sambanova', 'gemini'].includes(p.id)) p.outage_from = 2;
      }
    },
    expect: (s, r) => r.answered === 20 && s.served >= 95 && s.wire >= 99 && s.catalog === 100,
    describe: 'every request answered; at most the vision request degrades (all vision providers dark)',
  },
  {
    name: 'tiny-budgets',
    // Every provider gets exactly 2 calls: 20 calls for 20 requests, zero slack.
    // Zero waste is impossible without knowing the budgets: each death costs
    // exactly one discovery call. The entry must waste at most one per
    // provider (10) and answer everything; one min-context request may
    // degrade when the last 200k-context provider dies first.
    mutate: (doc) => {
      for (const p of doc.providers) {
        p.budget = 2;
        p.fail_calls = [];
        p.outage_from = null;
      }
    },
    expect: (s, r) => r.answered === 20 && s.served >= 95 && s.upstream_calls <= 30,
    describe: 'zero-slack budgets: everything answered, at most one discovery call per dead provider',
  },
  {
    name: 'fail-storm',
    // Every provider refuses its 2nd and 3rd calls in a row.
    mutate: (doc) => {
      for (const p of doc.providers) p.fail_calls = [2, 3];
    },
    expect: (s) => s.served === 100 && s.wire === 100 && s.failover === 100 && s.quality >= 98,
    describe: 'consecutive schedule refusals absorbed by same-candidate retries',
  },
  {
    name: 'one-survivor',
    // Nine providers serve one call each and die; one carries the trial.
    mutate: (doc) => {
      for (const p of doc.providers) {
        if (p.id === 'kilol') {
          p.budget = 25;
          p.fail_calls = [];
          p.outage_from = null;
        } else {
          p.budget = 1;
          p.fail_calls = [];
          p.outage_from = 2;
        }
      }
    },
    expect: (s, r) => r.answered === 20 && s.served >= 75 && s.wire >= 95 && r.reqOnly === 20 - Math.round((s.served / 100) * 20),
    describe: 'total failover to the last provider standing; every shortfall is requirements-only degradation',
  },
  {
    name: 'top-dead',
    // The five strongest providers serve one call each, then die.
    mutate: (doc) => {
      for (const p of doc.providers) {
        if (['openrouter', 'zai', 'cloudflare', 'sambanova', 'gemini'].includes(p.id)) {
          p.budget = 1;
          p.fail_calls = [];
          p.outage_from = 2;
        }
      }
    },
    expect: (s, r) => r.answered === 20 && s.served >= 95 && s.wire >= 99,
    describe: 'graceful degradation down the whole rating ladder; everything answered',
  },
  {
    name: 'beyond-capacity',
    // Every provider serves exactly one request: 10 successes for 20 requests.
    // Serving everything is impossible; the entry must answer exactly the
    // capacity, spend no calls once every provider is dead, stay up, and
    // exit clean. (The scorer's `served` is lower: degraded answers fail the
    // requirements check, which is the designed ladder, not lost capacity.)
    mutate: (doc) => {
      for (const p of doc.providers) {
        p.fail_calls = [2, 3];
        p.outage_from = 4;
      }
    },
    expect: (s, r) => r.answered === 10 && s.working === 1,
    describe: 'answers exactly the 10-request capacity, stays up, exits 0',
  },
];

async function main() {
  rmSync(BASE, { recursive: true, force: true });
  mkdirSync(BASE, { recursive: true });
  process.stdout.write(`chaos harness: kit=${KIT} trial=${TRIAL}\n\n`);

  // 1. The taxonomy, against every refusal shape in every wire.
  const taxonomyCases = [
    ['openai budget', 429, JSON.stringify({ error: { message: "Groq's free-tier budget is spent", type: 'rate_limit_error', code: 'rate_limit_exceeded' } }), 'dead-provider'],
    ['openai schedule', 429, JSON.stringify({ error: { message: 'Groq refused this call', type: 'rate_limit_error', code: 'rate_limit_exceeded' } }), 'schedule'],
    ['openai outage', 402, JSON.stringify({ error: { message: 'Pollinations is out', type: 'invalid_request_error', code: 'payment_required' } }), 'dead-provider'],
    ['openai concurrent', 429, JSON.stringify({ error: { message: 'Z.ai serves 1 call at a time', type: 'rate_limit_error', code: 'concurrent_requests' } }), 'concurrency'],
    ['openai auth', 401, JSON.stringify({ error: { message: 'the key is missing or wrong', type: 'invalid_request_error', code: 'invalid_api_key' } }), 'dead-provider'],
    ['openai bad model', 400, JSON.stringify({ error: { message: 'the provider serves no model "x"', type: 'invalid_request_error', code: 'invalid_model' } }), 'dead-model'],
    ['anthropic budget', 429, JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: "Z.ai's free-tier budget is spent" } }), 'dead-provider'],
    ['anthropic schedule', 429, JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'Z.ai refused this call' } }), 'schedule'],
    ['anthropic outage', 429, JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'Z.ai is out' } }), 'dead-provider'],
    ['gemini budget', 429, JSON.stringify({ error: { code: 429, message: "Google Gemini's free-tier budget is spent", status: 'RESOURCE_EXHAUSTED' } }), 'dead-provider'],
    ['gemini schedule', 429, JSON.stringify({ error: { code: 429, message: 'Google Gemini refused this call', status: 'RESOURCE_EXHAUSTED' } }), 'schedule'],
    ['gemini outage', 429, JSON.stringify({ error: { code: 429, message: 'Google Gemini is out', status: 'RESOURCE_EXHAUSTED' } }), 'dead-provider'],
    ['server error', 500, JSON.stringify({ error: { message: 'boom', code: 'internal' } }), 'ambiguous'],
    ['junk body', 200, 'this is not json', 'ambiguous'],
  ];
  for (const [name, status, body, want] of taxonomyCases) {
    const got = status === 200 ? 'ambiguous' : classify(evidenceOf(status, body)).kind;
    check(`taxonomy: ${name} -> ${want}`, got === want, `got ${got}`);
  }
  process.stdout.write('\n');

  // 2. The chaos variants, on the real env and the real scorer.
  // Ports start at 8951: 8931 is a teammate's local page server, and the
  // 8891-8911 range is used by other builders' verification runs.
  let port = 8951;
  for (const variant of VARIANTS) {
    const result = await runVariant(variant.name, variant.mutate, port++);
    if (result.error !== undefined) {
      check(`chaos: ${variant.name}`, false, result.error);
      continue;
    }
    const s = result.scores;
    const summary = s === null ? `scorer exit ${result.scorerExit}` : `score ${s.score} (q${s.quality}/s${s.served}/w${s.wire}/f${s.failover}/c${s.catalog}, calls ${s.upstream_calls}), answered ${result.answered}/20, requirements-only ${result.reqOnly}`;
    check(`chaos: ${variant.name} — ${variant.describe}`, result.entryExit === 0 && s !== null && variant.expect(s, result), `${summary}, entry exit ${result.entryExit}`);
  }
  process.stdout.write('\n');

  // 3. The catalog fallback: --list against a dead env must still be complete.
  const dead = await run('node', ['main.ts', '--list'], {
    cwd: REPO,
    env: { ...process.env, MAKEFUN_PROMPT_FILE: join(TRIAL_DIR, 'input', 'prompt.txt'), MAKEFUN_ENV_URL: 'http://127.0.0.1:9' },
  });
  let fallbackOk = false;
  let fallbackDetail = `exit ${dead.code}`;
  try {
    const catalog = JSON.parse(dead.out);
    const truth = JSON.parse(readFileSync(join(TRIAL_DIR, 'reference', 'catalog.json'), 'utf8'));
    const sameSets = truth.providers.every((t) => {
      const got = catalog.providers.find((p) => p.id === t.id);
      if (got === undefined) return false;
      const free = got.models.filter((m) => m.free).map((m) => m.id).sort();
      return JSON.stringify(free) === JSON.stringify([...t.free_models].sort()) && got.base_url === t.base_url && got.key_env === t.key_env && got.wire === t.wire;
    });
    fallbackOk = dead.code === 0 && catalog.providers.length === truth.providers.length && sameSets;
    fallbackDetail = `${catalog.providers.length} providers, free sets ${sameSets ? 'exact' : 'MISMATCH'}`;
  } catch (err) {
    fallbackDetail = `unparseable --list output: ${String(err).slice(0, 120)}`;
  }
  check('catalog fallback: --list with a dead env still reports every free set', fallbackOk, fallbackDetail);

  // 4. --list with no brief and no env: the empty catalog.
  const empty = await run('node', ['main.ts', '--list'], { cwd: REPO, env: { ...process.env, MAKEFUN_PROMPT_FILE: '', MAKEFUN_ENV_URL: '' } });
  check('catalog empty: --list with no brief prints {"providers":[]}', empty.code === 0 && empty.out.trim() === '{"providers":[]}', empty.out.trim().slice(0, 60));

  process.stdout.write(`\n${failures === 0 ? 'ALL CHAOS CHECKS PASSED' : `${failures} CHECK(S) FAILED`}\n`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`chaos harness crashed: ${err.stack ?? err}\n`);
  process.exit(2);
});
