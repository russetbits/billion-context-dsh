/**
 * `textOverrides`: id-addressed FULL REPLACEMENT of model-visible copy.
 *
 * Covers the six acceptance cases (plus one engine-level wiring case):
 *  1. an override actually reaches the rendered text;
 *  2. no override → byte-identical to the defaults;
 *  3. an unknown id is ignored with a warning, never an exception;
 *  4. `{pct}`-style placeholders still render from the override;
 *  5. EVERY id in `ableTextIds()` is really override-able (id table vs reality);
 *  6. an override >1.5x the default warns and still works.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import { createCore } from 'acp-kernel'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import { compressParameters, compressParametersFor, makeTools, type ToolEnvironment } from '../src/tools.ts'
import {
  DEFAULT_PROMPTS,
  DEFAULT_RESOLVED,
  ableTextIds,
  applyTextOverrides,
  estimateTextTokens,
  renderTemplate,
  resolvePrompts,
  textById,
  textOverrideWarningLines,
} from '../src/prompts.ts'
import { AcpStateStore } from '../src/state.ts'
import { buildTextSession } from './helpers.ts'

function fakeExec(session: Session): ToolRunContext {
  const agent = {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx: new Context(),
  } as unknown as Agent
  return {
    callId: 'call-status',
    name: 'acp_status',
    arguments: {},
    signal: new AbortController().signal,
    agent,
  } as unknown as ToolRunContext
}

async function statusText(env: ToolEnvironment): Promise<string> {
  const session = buildTextSession(6)
  const status = makeTools(env).find((definition) => definition.name === 'acp_status')
  assert.ok(status, 'acp_status registered')
  const result = (await status.execute({}, fakeExec(session))) as { text: string }
  return result.text
}

test('textOverrides 1: an override replaces the default text', () => {
  const report = applyTextOverrides(DEFAULT_RESOLVED, { 'nudge.normal': 'lean now ({pct}%)' })
  assert.equal(report.prompts.nudge.normal, 'lean now ({pct}%)')
  assert.equal(report.unknown.length, 0)
  assert.equal(report.oversized.length, 0)
  assert.equal(renderTemplate(report.prompts.nudge.normal, { pct: 42 }), 'lean now (42%)')
  // Untouched ids keep the shipped bytes.
  assert.equal(report.prompts.nudge.emergency, DEFAULT_PROMPTS.nudge.emergency)
})

test('textOverrides 2: no override is byte-identical to the defaults', () => {
  for (const input of [undefined, null, {}]) {
    const report = applyTextOverrides(DEFAULT_RESOLVED, input)
    assert.equal(JSON.stringify(report.prompts), JSON.stringify(DEFAULT_PROMPTS))
    assert.deepEqual(report.unknown, [])
    assert.deepEqual(report.oversized, [])
  }
})

test('textOverrides 3: unknown ids are ignored with a warning, never thrown', () => {
  const report = applyTextOverrides(DEFAULT_RESOLVED, {
    'nudge.bogus': 'x',
    'tool.nope': 'y',
    'noDotAtAll': 'z',
  })
  assert.deepEqual(report.unknown, ['nudge.bogus', 'tool.nope', 'noDotAtAll'])
  assert.equal(JSON.stringify(report.prompts), JSON.stringify(DEFAULT_PROMPTS), 'nothing applied')
  const lines = textOverrideWarningLines(report)
  assert.equal(lines.length, 1)
  assert.match(lines[0]!, /ignored 3 unknown id\(s\)/)
  assert.match(lines[0]!, /nudge\.bogus/)
})

test('textOverrides 4: placeholders inside an override render normally', () => {
  const report = applyTextOverrides(DEFAULT_RESOLVED, {
    'rangeTable.title': 'Ranges: {count} row(s) — fold the oldest',
  })
  assert.equal(
    renderTemplate(report.prompts.rangeTable.title, { count: 7 }),
    'Ranges: 7 row(s) — fold the oldest',
  )
  // A value that OMITS the default's placeholder is legal plain text.
  const plain = applyTextOverrides(DEFAULT_RESOLVED, { 'rangeTable.title': 'Ranges' })
  assert.equal(renderTemplate(plain.prompts.rangeTable.title, { count: 7 }), 'Ranges')
})

test('textOverrides 5: every id in ableTextIds() is actually override-able', () => {
  const ids = ableTextIds()
  assert.ok(ids.includes('nudge.normal'))
  assert.ok(ids.includes('nudge.cacheCost'))
  assert.ok(ids.includes('rangeTable.footer'))
  assert.ok(ids.includes('tool.compress'))
  assert.ok(ids.includes('tool.condense'))
  assert.ok(ids.includes('tool.recall'))
  assert.ok(ids.includes('tool.acpStatus'))
  assert.ok(ids.includes('systemPrompt'))
  const overrides: Record<string, string> = {}
  for (const [index, id] of ids.entries()) overrides[id] = `MARKER-${index}-${id}`
  const report = applyTextOverrides(DEFAULT_RESOLVED, overrides)
  assert.deepEqual(report.unknown, [], 'id table matches the shipped slots')
  assert.deepEqual(report.oversized, [], 'a short marker never trips the length guard')
  for (const [index, id] of ids.entries()) {
    assert.equal(textById(report.prompts, id), `MARKER-${index}-${id}`, `${id} did not apply`)
  }
})

test('textOverrides 6: an override over 1.5x the default warns but still applies', () => {
  const doubled = DEFAULT_PROMPTS.nudge.growth + DEFAULT_PROMPTS.nudge.growth
  assert.ok(estimateTextTokens(doubled) > estimateTextTokens(DEFAULT_PROMPTS.nudge.growth) * 1.5)
  const report = applyTextOverrides(DEFAULT_RESOLVED, { 'nudge.growth': doubled })
  assert.equal(report.prompts.nudge.growth, doubled, 'applied anyway')
  assert.equal(report.oversized.length, 1)
  assert.equal(report.oversized[0]!.id, 'nudge.growth')
  const lines = textOverrideWarningLines(report)
  assert.equal(lines.length, 1)
  assert.match(lines[0]!, /nudge\.growth is ~\d+ tokens vs ~\d+ default \(2\.00x\)/)
  assert.match(lines[0]!, /makes every request heavier/)
})

test('textOverrides 7: engine wiring — tool descriptions and acp_status self-check', () => {
  const env: ToolEnvironment = {
    kernel: createCore({}),
    store: new AcpStateStore(),
    modelContextLimit: 200000,
    prompts: applyTextOverrides(DEFAULT_RESOLVED, {
      'tool.compress': 'OVERRIDDEN-COMPRESS',
      'tool.condense': 'OVERRIDDEN-CONDENSE',
      'nudge.oops': 'typo',
    }).prompts,
    textWarnings: ['textOverrides: ignored 1 unknown id(s): nudge.oops'],
  }
  const tools = makeTools(env)
  const byName = new Map(tools.map((tool) => [tool.name, tool]))
  assert.equal(byName.get('compress')!.description, 'OVERRIDDEN-COMPRESS')
  const schemaOf = (tools[0]!.parameters as { properties: Record<string, { description: string }> }).properties
  assert.equal(schemaOf.condense!.description, 'OVERRIDDEN-CONDENSE')
  // The shipped sheet is untouched for every other caller (see tests/g5-governance).
  const shipped = makeTools({ ...env, prompts: DEFAULT_RESOLVED }).find((tool) => tool.name === 'compress')!
  assert.equal(
    (shipped.parameters as { properties: Record<string, { description: string }> }).properties.condense!.description,
    DEFAULT_PROMPTS.tools.condense,
  )
})

test('textOverrides 8: structured config.prompts still wins less than textOverrides', () => {
  const viaPrompts = resolvePrompts({ nudge: { growth: '+{growth}K from structured' } })
  const report = applyTextOverrides(viaPrompts, { 'nudge.growth': '+{growth}K flat override' })
  assert.equal(report.prompts.nudge.growth, '+{growth}K flat override')
  // …and the structured value survives when the id is not overridden.
  const untouched = applyTextOverrides(viaPrompts, { 'nudge.tip': 'x' })
  assert.equal(untouched.prompts.nudge.growth, '+{growth}K from structured')
})

test('textOverrides 9: acp_status prints the self-check line, and nothing when clean', async () => {
  const base: ToolEnvironment = {
    kernel: createCore({}),
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    compressCallIdsToHide: new Set(),
  }
  const warned = await statusText({
    ...base,
    textWarnings: ['textOverrides: ignored 1 unknown id(s): nudge.oops'],
  })
  assert.match(warned, /Text overrides: textOverrides: ignored 1 unknown id\(s\): nudge\.oops/)
  assert.match(warned, /Surface: /, 'the rest of the report still renders')
  const clean = await statusText(base)
  assert.ok(!clean.includes('Text overrides:'), 'no extra bytes on a clean config')
})

/** 参数级 id → schema 路径(与 src/tools.ts 的 *_PARAM_PATCHES 互为校验)。 */
const PARAM_PATH: Readonly<Record<string, readonly (string | number)[]>> = {
  'compress.arguments': ['arguments'],
  'compress.topic': ['topic'],
  'compress.content': ['content'],
  'compress.startSeq': ['content', 'items', 'properties', 'startSeq', 'oneOf', 0],
  'compress.startSeq.text': ['content', 'items', 'properties', 'startSeq', 'oneOf', 1],
  'compress.endSeq': ['content', 'items', 'properties', 'endSeq', 'oneOf', 0],
  'compress.endSeq.text': ['content', 'items', 'properties', 'endSeq', 'oneOf', 1],
  'compress.summary': ['content', 'items', 'properties', 'summary'],
  'compress.item.topic': ['content', 'items', 'properties', 'topic'],
  'compress.verifiedReadings': ['content', 'items', 'properties', 'verifiedReadings'],
  'compress.force': ['force'],
  'compress.condense.topic': ['condense', 'properties', 'topic'],
  'compress.condense.force': ['condense', 'properties', 'force'],
  'compress.condense.verifiedReadings': ['condense', 'properties', 'verifiedReadings'],
  'recall.query': ['query'],
  'recall.blockId': ['blockId'],
  'recall.seq': ['seq'],
  'recall.chars': ['chars'],
  'recall.maxResults': ['maxResults'],
  'recall.offset': ['offset'],
  'recall.pageLimit': ['pageLimit'],
  'acpStatus.detail': ['detail'],
  'acpStatus.tool': ['tool'],
  'acpStatus.sort': ['sort'],
  'acpStatus.limit': ['limit'],
}

function envFor(prompts: ToolEnvironment['prompts']): ToolEnvironment {
  return {
    kernel: createCore({}),
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    compressCallIdsToHide: new Set(),
    prompts,
  }
}

function paramDescription(tools: ReturnType<typeof makeTools>, id: string): string | undefined {
  const toolName = id.startsWith('compress.') ? 'compress' : id.startsWith('recall.') ? 'recall' : 'acp_status'
  // defineTool wraps the flat parameter sheet as { type: 'object', properties: { … } }.
  let node: unknown = (tools.find((tool) => tool.name === toolName)?.parameters as { properties?: unknown })?.properties
  for (const key of PARAM_PATH[id]!) node = (node as Record<string | number, unknown> | undefined)?.[key as never]
  return (node as { description?: string } | undefined)?.description
}

test('textOverrides 10: every parameter-level id reaches the schema built by makeTools', () => {
  const ids = Object.keys(PARAM_PATH)
  assert.equal(ids.length, 25, '25 parameter slots are wired (the 26th, condense.summary, is the note skeleton)')
  // Every wired id is advertised by ableTextIds(), so a caller can discover it.
  const advertised = new Set(ableTextIds())
  for (const id of ids) assert.ok(advertised.has(`tool.param.${id}`), `tool.param.${id} not advertised`)
  for (const id of ids) {
    const overrides = { [`tool.param.${id}`]: `MARKER-${id}` }
    const report = applyTextOverrides(DEFAULT_RESOLVED, overrides)
    assert.deepEqual(report.unknown, [], `${id} is a known id`)
    const tools = makeTools(envFor(report.prompts))
    assert.equal(paramDescription(tools, id), `MARKER-${id}`, `${id} did not reach the schema`)
    // Overriding ONE id leaves the other 24 alone.
    const other = ids.find((candidate) => candidate !== id)!
    assert.equal(paramDescription(tools, other), DEFAULT_PROMPTS.toolParamTexts[other], `${other} leaked`)
  }
})

test('textOverrides 11: no override hands back the shipped constant object (zero allocation)', () => {
  const tools = makeTools(envFor(DEFAULT_RESOLVED))
  const compress = tools.find((tool) => tool.name === 'compress')!
  // Zero allocation: the builder hands back the shipped constant itself, so the
  // only object defineTool sees is the one that always existed.
  assert.equal(compressParametersFor(DEFAULT_RESOLVED), compressParameters, 'same reference, not a rebuild')
  for (const id of Object.keys(PARAM_PATH)) {
    assert.equal(paramDescription(tools, id), DEFAULT_PROMPTS.toolParamTexts[id], `${id} ships its default`)
  }
  // The wrapped sheet defineTool builds is what the model sees; its parameter
  // copy must equal the shipped defaults one id at a time (done above).
})

test('textOverrides 12: an override never changes type / required / enum / oneOf structure', () => {
  const strip = (value: unknown): unknown =>
    JSON.parse(JSON.stringify(value, (key, node) => (key === 'description' ? undefined : node)))
  const overrides = Object.fromEntries(Object.keys(PARAM_PATH).map((id) => [`tool.param.${id}`, 'X']))
  const overridden = makeTools(envFor(applyTextOverrides(DEFAULT_RESOLVED, overrides).prompts))
  const shipped = makeTools(envFor(DEFAULT_RESOLVED))
  for (const name of ['compress', 'recall', 'acp_status']) {
    const after = overridden.find((tool) => tool.name === name)
    const before = shipped.find((tool) => tool.name === name)
    assert.deepEqual(strip(after!.parameters), strip(before!.parameters), `${name} schema shape changed`)
    assert.equal(after!.description, before!.description, `${name} tool description moved`)
  }
})
