// v5 — the most resilient free-model aggregator: one server, two client wires,
// ten providers, and a failover strategy that treats every provider failure as
// a classified, recoverable event rather than an answer.
//
// Shape of a trial:
//   1. Read the brief (providers, quality table, workload manifest).
//   2. Probe every provider's catalog route, retrying failures, falling back
//      to the quality table for any provider the probes never reach. The one
//      catalog object built here is printed by `--list` and posted to
//      /register, byte for byte the same document.
//   3. Listen on an ephemeral port, register it with the env, and serve the
//      client's twenty requests, one at a time, each answered by the best
//      free model that meets its requirements and still answers.
//   4. Poll GET /work and exit 0 the moment the env says the workload is done.
//
// Resilience, the entry's whole point:
//   - Every upstream call is non-streaming and buffered whole; the answer is
//     validated before anything is sent, and streams are synthesized from the
//     buffered answer. A torn provider stream can never reach the client.
//   - Every failure is classified (dead provider, dead model, schedule,
//     concurrency, ambiguous) and answered with the policy that fits it:
//     permanent faults kill their target at once, schedule refusals are
//     retried on the same candidate, ambiguous faults get one second chance.
//   - When no model meeting the requirements answers, the entry degrades to
//     any free model rather than failing the request outright.
//   - Nothing here reads the wall clock or unseeded randomness, so a replay
//     sees the same choices and leaves the same state export, byte for byte.
//   - The server never crashes for a request: every handler is guarded, and
//     process-level guards log and continue instead of dying.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  isRecord,
  readBrief,
  type Brief,
  type Json,
  type Needs,
} from './lib/brief.ts';
import { loadCatalog, type Catalog } from './lib/catalog.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
} from './lib/codec.ts';
import { sleep } from './lib/faults.ts';
import { Plan, registeredFreeIds } from './lib/plan.ts';
import { Server as ServingPipeline } from './lib/serve.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const REGISTER_ATTEMPTS = 5;
const REGISTER_WAIT_MS = 200;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator-v5: ${message}\n`);
}

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function needsFor(brief: Brief, index: number, body: Json): Needs {
  const row = brief.manifest[index];
  if (row !== undefined) return row.needs;
  return needsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function needsFromBody(body: Json): Needs {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
    minContext: null,
  };
}

/** One client request body as a JSON object, or null when it is not one or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendClientError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError('overloaded_error', message));
  else sendJson(res, status, openAIError(message, 'no_free_model'));
}

interface Runtime {
  brief: Brief;
  catalog: Catalog;
  plan: Plan;
  pipeline: ServingPipeline;
  modelIds: string[];
  owners: Map<string, string>;
  displayNames: Map<string, string>;
  arrivals: number;
}

function buildRuntime(brief: Brief, catalog: Catalog, probes: Parameters<typeof Plan>[1]): Runtime {
  const plan = new Plan(brief, probes);
  const owners = new Map<string, string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!owners.has(model.id)) owners.set(model.id, provider.id);
    }
  }
  const displayNames = new Map<string, string>();
  for (const probe of probes) {
    for (const model of probe.models) {
      if (!displayNames.has(model.id)) displayNames.set(model.id, model.displayName);
    }
  }
  return {
    brief,
    catalog,
    plan,
    pipeline: new ServingPipeline(ENV_URL, plan),
    modelIds: registeredFreeIds(catalog),
    owners,
    displayNames,
    arrivals: 0,
  };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(rt: Runtime, req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(rt.modelIds, rt.displayNames));
  else sendJson(res, 200, openAIModelList(rt.modelIds, rt.owners));
}

/** POST /v1/chat/completions or POST /v1/messages: serve, then render in the client's wire. */
async function inferRoute(rt: Runtime, wire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendClientError(res, wire, 400, 'the request body is not one JSON object');
    return;
  }
  const arrival = rt.arrivals;
  rt.arrivals += 1;
  const served = await rt.pipeline.serve(wire, body, needsFor(rt.brief, arrival, body));
  if (served === null) {
    sendClientError(res, wire, 503, 'no provider could serve this request');
    return;
  }
  const model = served.candidate.model;
  if (body.stream === true) {
    const stream = wire === 'openai' ? openAIStreamAnswer(served.answer, model) : anthropicStreamAnswer(served.answer, model);
    sendText(res, 200, 'text/event-stream; charset=utf-8', stream);
    return;
  }
  sendJson(res, 200, wire === 'openai' ? openAIJsonAnswer(served.answer, model) : anthropicJsonAnswer(served.answer, model));
}

async function handle(rt: Runtime, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(rt, req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await inferRoute(rt, 'openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await inferRoute(rt, 'anthropic', req, res);
    return;
  }
  sendClientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog, retrying a refused registration. */
async function register(port: number, catalog: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt++) {
    if (attempt > 0) await sleep(REGISTER_WAIT_MS);
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
    }
  }
  throw lastError ?? new Error('registration never ran');
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.problem !== null) log(brief.problem);
  const { probes, catalog } = await loadCatalog(ENV_URL, brief, log);
  if (process.argv.includes('--list')) {
    // The same document the register body carries: print the one built object
    // and stop before any server starts. Wait for stdout to flush.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const rt = buildRuntime(brief, catalog, probes);
  const server = createServer((req, res) => {
    void handle(rt, req, res).catch((err: unknown) => {
      // A request must never take the server down with it.
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        try {
          sendClientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
        } catch {
          res.destroy();
        }
      } else {
        res.destroy();
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} catalogued providers and ${rt.plan.candidates.length} routable free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

process.on('uncaughtException', (err) => log(`uncaught: ${err.stack ?? String(err)}`));
process.on('unhandledRejection', (reason) => log(`unhandled rejection: ${String(reason)}`));

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
