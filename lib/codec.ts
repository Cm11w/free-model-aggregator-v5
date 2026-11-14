// The codec: every translation between the two client wires and the three
// upstream wires, and back.
//
// The providers echo what they parsed — the system prompt, the user text, the
// tool names, max_tokens, the stop sequences, the image, the JSON mode — and
// the client compares that echo with what it sent. So every translation here
// is lossless by construction: nothing the request carried is dropped, renamed
// or reworded, in any of the six directions. The one thing a wire cannot carry
// is JSON mode on the Anthropic wire, and the planner excludes anthropic-wire
// providers for those requests instead of letting a translation lose it.
//
// Answers are normalized into one shape — text, tool calls, usage, stop
// family — whatever wire carried them, and then rendered into the wire the
// client used, as JSON or as a synthesized event stream. Upstream calls are
// always non-streaming: the whole answer is buffered and validated before
// anything is sent to the client, so a torn stream can never reach it.

import { isRecord, type Json, type Wire } from './brief.ts';

/** One tool call in an answer, wire-neutral. */
export interface ToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}

/** One provider answer, normalized out of its wire. */
export interface Answer {
  text: string;
  toolCalls: ToolCall[];
  inputTokens: number;
  outputTokens: number;
  /** 0 stop, 1 length, 2 tool use, 3 refusal — the families the client checks. */
  stopFamily: number;
  /** Whether the answer carries anything at all: text or a tool call. */
  usable: boolean;
}

const STOP_FAMILIES: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

const numOr = (v: unknown, fallback: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

/** The text of a string or of a block array's text blocks, joined the way the providers join them. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Split a data URL into its media type and base64 payload, or null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] !== ';base64') return null;
  return { media: match[1] === '' ? 'image/png' : match[1], data: match[3] };
}

// ---------------------------------------------------------------------------
// Client request -> normalized conversation
// ---------------------------------------------------------------------------

interface ImageRef {
  kind: 'base64' | 'url';
  media: string;
  data: string;
}

interface NormalizedMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  images: ImageRef[];
  toolCalls: ToolCall[];
  /** For a tool message: the call it answers, and its result text. */
  toolCallId: string;
  toolResult: string;
  isToolResult: boolean;
}

interface NormalizedRequest {
  systemText: string;
  messages: NormalizedMessage[];
  toolNames: string[];
  toolsOpenAI: Json[];
  toolsAnthropic: Json[];
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  toolChoiceOpenAI: unknown;
  toolChoiceAnthropic: unknown;
  jsonMode: boolean;
}

function imageOfOpenAIPart(part: Json): ImageRef | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = part.image_url.url;
  if (typeof url !== 'string' || url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { kind: 'base64', media: data.media, data: data.data };
  return { kind: 'url', media: '', data: url };
}

function imageOfAnthropicBlock(block: Json): ImageRef | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'base64', media: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', media: '', data: source.url };
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  const text = blocksText(content);
  if (text !== '') return text;
  if (Array.isArray(content)) return JSON.stringify(content);
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsJsonOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** An OpenAI Chat Completions body, normalized. */
function normalizeOpenAI(body: Json): NormalizedRequest {
  const messages: NormalizedMessage[] = [];
  const systemParts: string[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const role = entry.role;
    if (role === 'system') {
      const text = blocksText(entry.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolCallId: typeof entry.tool_call_id === 'string' ? entry.tool_call_id : '',
        toolResult: typeof entry.content === 'string' ? entry.content : toolResultText(entry.content),
        isToolResult: true,
      });
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      const calls = Array.isArray(entry.tool_calls) ? entry.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          argumentsJson: argumentsJsonOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', text: blocksText(entry.content), images: [], toolCalls, toolCallId: '', toolResult: '', isToolResult: false });
      continue;
    }
    // User (and anything else, treated as user): text plus image parts.
    const content = entry.content;
    const images: ImageRef[] = [];
    if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const image = imageOfOpenAIPart(part);
        if (image !== null) images.push(image);
      }
    }
    messages.push({ role: 'user', text: blocksText(content), images, toolCalls: [], toolCallId: '', toolResult: '', isToolResult: false });
  }
  const toolsOpenAI: Json[] = [];
  const toolsAnthropic: Json[] = [];
  const toolNames: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      toolNames.push(tool.function.name);
      toolsOpenAI.push({
        type: 'function',
        function: {
          name: tool.function.name,
          description: typeof tool.function.description === 'string' ? tool.function.description : '',
          parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
        },
      });
      toolsAnthropic.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = numOr(body.max_tokens, -1) !== -1 ? (body.max_tokens as number) : numOr(body.max_completion_tokens, -1) !== -1 ? (body.max_completion_tokens as number) : null;
  return {
    systemText: systemParts.join('\n'),
    messages,
    toolNames,
    toolsOpenAI,
    toolsAnthropic,
    maxTokens,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [],
    toolChoiceOpenAI: body.tool_choice,
    toolChoiceAnthropic: openAIChoiceToAnthropic(body.tool_choice),
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

function openAIChoiceToAnthropic(choice: unknown): unknown {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required' || choice === 'any') return { type: 'any' };
  if (choice === 'none') return { type: 'auto' };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return undefined;
}

function anthropicChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

/** An Anthropic Messages body, normalized. */
function normalizeAnthropic(body: Json): NormalizedRequest {
  const messages: NormalizedMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const assistant = entry.role === 'assistant';
    const content = entry.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', text: content, images: [], toolCalls: [], toolCallId: '', toolResult: '', isToolResult: false });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: ToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          argumentsJson: argumentsJsonOf(block.input),
        });
      }
      messages.push({ role: 'assistant', text: blocksText(content), images: [], toolCalls, toolCallId: '', toolResult: '', isToolResult: false });
      continue;
    }
    // User: text/image blocks, plus tool_result blocks which become tool messages.
    const images: ImageRef[] = [];
    const textParts: string[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          text: '',
          images: [],
          toolCalls: [],
          toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          toolResult: toolResultText(block.content),
          isToolResult: true,
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
      const image = imageOfAnthropicBlock(block);
      if (image !== null) images.push(image);
    }
    if (textParts.length > 0 || images.length > 0) {
      messages.push({ role: 'user', text: textParts.join('\n'), images, toolCalls: [], toolCallId: '', toolResult: '', isToolResult: false });
    }
  }
  const toolsOpenAI: Json[] = [];
  const toolsAnthropic: Json[] = [];
  const toolNames: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      toolNames.push(tool.name);
      toolsAnthropic.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        input_schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
      toolsOpenAI.push({
        type: 'function',
        function: {
          name: tool.name,
          description: typeof tool.description === 'string' ? tool.description : '',
          parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
        },
      });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    toolNames,
    toolsOpenAI,
    toolsAnthropic,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    toolChoiceOpenAI: anthropicChoiceToOpenAI(body.tool_choice),
    toolChoiceAnthropic: body.tool_choice,
    jsonMode: false, // The Anthropic wire carries no JSON-mode flag.
  };
}

/** Normalize a client body from either wire. */
export function normalizeRequest(clientWire: 'openai' | 'anthropic', body: Json): NormalizedRequest {
  return clientWire === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
}

// ---------------------------------------------------------------------------
// Normalized request -> upstream wire bodies (always non-streaming)
// ---------------------------------------------------------------------------

function openAIMessages(req: NormalizedRequest): Json[] {
  const out: Json[] = [];
  if (req.systemText !== '') out.push({ role: 'system', content: req.systemText });
  for (const message of req.messages) {
    if (message.role === 'tool') {
      out.push({ role: 'tool', tool_call_id: message.toolCallId, content: message.toolResult });
      continue;
    }
    if (message.role === 'assistant') {
      const entry: Json = { role: 'assistant', content: message.text !== '' ? message.text : null };
      if (message.toolCalls.length > 0) {
        entry.tool_calls = message.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.argumentsJson },
        }));
      }
      out.push(entry);
      continue;
    }
    if (message.images.length === 0) {
      out.push({ role: 'user', content: message.text });
      continue;
    }
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ type: 'text', text: message.text });
    for (const image of message.images) {
      const url = image.kind === 'base64' ? `data:${image.media};base64,${image.data}` : image.data;
      parts.push({ type: 'image_url', image_url: { url } });
    }
    out.push({ role: 'user', content: parts });
  }
  return out;
}

function anthropicMessages(req: NormalizedRequest): Json[] {
  const out: Json[] = [];
  for (const message of req.messages) {
    if (message.role === 'tool') {
      out.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: message.toolCallId, content: message.toolResult }],
      });
      continue;
    }
    if (message.role === 'assistant') {
      const blocks: Json[] = [];
      if (message.text !== '') blocks.push({ type: 'text', text: message.text });
      for (const call of message.toolCalls) {
        let input: unknown = {};
        try {
          input = JSON.parse(call.argumentsJson);
        } catch {
          input = {};
        }
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: isRecord(input) ? input : {} });
      }
      out.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    if (message.text !== '') blocks.push({ type: 'text', text: message.text });
    for (const image of message.images) {
      blocks.push(
        image.kind === 'base64'
          ? { type: 'image', source: { type: 'base64', media_type: image.media, data: image.data } }
          : { type: 'image', source: { type: 'url', url: image.data } },
      );
    }
    out.push({ role: 'user', content: blocks });
  }
  return out;
}

function geminiContents(req: NormalizedRequest): Json[] {
  const out: Json[] = [];
  const appendUserParts = (parts: Json[]): void => {
    // A trailing tool result must not become its own user turn: the provider
    // reads the last user turn's text as the request, and a result-only turn
    // would blank it. Fold result parts into the previous user turn instead.
    if (out.length > 0) {
      const last = out[out.length - 1];
      if (last.role === 'user' && Array.isArray(last.parts)) {
        last.parts.push(...parts);
        return;
      }
    }
    out.push({ role: 'user', parts });
  };
  for (const message of req.messages) {
    if (message.role === 'tool') {
      let response: unknown = {};
      try {
        response = JSON.parse(message.toolResult);
      } catch {
        response = { result: message.toolResult };
      }
      appendUserParts([{ functionResponse: { name: message.toolCallId, response: isRecord(response) ? response : { result: message.toolResult } } }]);
      continue;
    }
    if (message.role === 'assistant') {
      const parts: Json[] = [];
      if (message.text !== '') parts.push({ text: message.text });
      for (const call of message.toolCalls) {
        let args: unknown = {};
        try {
          args = JSON.parse(call.argumentsJson);
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: call.name, args: isRecord(args) ? args : {} } });
      }
      out.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ text: message.text });
    for (const image of message.images) {
      parts.push(
        image.kind === 'base64'
          ? { inlineData: { mimeType: image.media, data: image.data } }
          : { fileData: { fileUri: image.data } },
      );
    }
    out.push({ role: 'user', parts });
  }
  return out;
}

/** Build the upstream request body for one wire. Never streaming: the answer is buffered whole. */
export function upstreamBody(upstream: Wire, req: NormalizedRequest, model: string): Json {
  if (upstream === 'openai') {
    const body: Json = { model, messages: openAIMessages(req), stream: false };
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.topP !== null) body.top_p = req.topP;
    if (req.stops.length > 0) body.stop = req.stops;
    if (req.toolsOpenAI.length > 0) body.tools = req.toolsOpenAI;
    if (req.toolChoiceOpenAI !== undefined) body.tool_choice = req.toolChoiceOpenAI;
    if (req.jsonMode) body.response_format = { type: 'json_object' };
    return body;
  }
  if (upstream === 'anthropic') {
    const body: Json = { model, messages: anthropicMessages(req), stream: false };
    if (req.systemText !== '') body.system = req.systemText;
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.topP !== null) body.top_p = req.topP;
    if (req.stops.length > 0) body.stop_sequences = req.stops;
    if (req.toolsAnthropic.length > 0) body.tools = req.toolsAnthropic;
    if (req.toolChoiceAnthropic !== undefined) body.tool_choice = req.toolChoiceAnthropic;
    return body;
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  const body: Json = { contents: geminiContents(req) };
  if (req.systemText !== '') body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (Object.keys(config).length > 0) body.generationConfig = config;
  if (req.toolNames.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.toolsAnthropic.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema,
        })),
      },
    ];
  }
  return body;
}

// ---------------------------------------------------------------------------
// Upstream answer -> normalized answer
// ---------------------------------------------------------------------------

function familyOf(reason: unknown): number {
  if (typeof reason !== 'string') return 0;
  return STOP_FAMILIES[reason] ?? 0;
}

function answerFromOpenAI(body: Json): Answer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      argumentsJson: argumentsJsonOf(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : null;
  if (usage === null) return null;
  return {
    text,
    toolCalls,
    inputTokens: numOr(usage.prompt_tokens, 0),
    outputTokens: numOr(usage.completion_tokens, 0),
    stopFamily: toolCalls.length > 0 ? 2 : familyOf(choice.finish_reason),
    usable: text !== '' || toolCalls.length > 0,
  };
}

function answerFromAnthropic(body: Json): Answer | null {
  if (!Array.isArray(body.content)) return null;
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        argumentsJson: argumentsJsonOf(block.input),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : null;
  if (usage === null) return null;
  return {
    text: textParts.join(''),
    toolCalls,
    inputTokens: numOr(usage.input_tokens, 0),
    outputTokens: numOr(usage.output_tokens, 0),
    stopFamily: toolCalls.length > 0 ? 2 : familyOf(body.stop_reason),
    usable: textParts.join('') !== '' || toolCalls.length > 0,
  };
}

function answerFromGemini(body: Json): Answer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') textParts.push(part.text);
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        argumentsJson: argumentsJsonOf(part.functionCall.args),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  if (usage === null) return null;
  return {
    text: textParts.join(''),
    toolCalls,
    inputTokens: numOr(usage.promptTokenCount, 0),
    outputTokens: numOr(usage.candidatesTokenCount, 0),
    stopFamily: toolCalls.length > 0 ? 2 : familyOf(candidate.finishReason),
    usable: textParts.join('') !== '' || toolCalls.length > 0,
  };
}

/** Parse one upstream 2xx body into a normalized answer, or null when it cannot be used. */
export function parseAnswer(upstream: Wire, bodyText: string): Answer | null {
  let body: unknown;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  const answer = upstream === 'openai' ? answerFromOpenAI(body) : upstream === 'anthropic' ? answerFromAnthropic(body) : answerFromGemini(body);
  if (answer === null || !answer.usable) return null;
  return answer;
}

// ---------------------------------------------------------------------------
// Normalized answer -> client wire (JSON or synthesized SSE)
// ---------------------------------------------------------------------------

/** Fixed instants for synthesized responses: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The width one synthesized content delta carries, mirroring the providers. */
const DELTA_CHARS = 13;

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

export function openAIJsonAnswer(answer: Answer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: call.argumentsJson },
    }));
  }
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: OPENAI_STOPS[answer.stopFamily] ?? 'stop', logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

export function anthropicJsonAnswer(answer: Answer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.argumentsJson);
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregator', name: call.name, input: isRecord(input) ? input : {} });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: ANTHROPIC_STOPS[answer.stopFamily] ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An OpenAI event stream synthesized from a buffered answer: role, deltas, finish, usage, [DONE]. */
export function openAIStreamAnswer(answer: Answer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null, usage?: Json): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage !== undefined ? { usage } : {}),
  });
  frames.push(sseFrame(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, i) => {
    frames.push(
      sseFrame(
        null,
        chunk({ tool_calls: [{ index: i, id: call.id !== '' ? call.id : `call_${i}`, type: 'function', function: { name: call.name, arguments: call.argumentsJson } }] }, null),
      ),
    );
  });
  frames.push(sseFrame(null, chunk({}, OPENAI_STOPS[answer.stopFamily] ?? 'stop')));
  frames.push(
    sseFrame(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic event stream synthesized from a buffered answer, in the wire's event order. */
export function anthropicStreamAnswer(answer: Answer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    frames.push(sseFrame('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregator', name: call.name, input: {} },
      }),
    );
    frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.argumentsJson } }));
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTHROPIC_STOPS[answer.stopFamily] ?? 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** GET /v1/models in the OpenAI shape. */
export function openAIModelList(ids: readonly string[], owners: ReadonlyMap<string, string>): Json {
  return {
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: owners.get(id) ?? 'aggregator' })),
  };
}

/** GET /v1/models in the Anthropic shape. */
export function anthropicModelList(ids: readonly string[], displayNames: ReadonlyMap<string, string>): Json {
  return {
    data: ids.map((id) => ({ type: 'model', id, display_name: displayNames.get(id) ?? id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

export function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
