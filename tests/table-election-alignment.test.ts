/**
 * Table-election alignment (2026-11-27, live session seq4356→4361→4365→4376).
 *
 * The bug: the nudge range table (buildCompressibleSeqRanges geometry, host
 * guards applied) listed only the NEW plain rows (e.g. 4014..4232) while the
 * condense election's timeline also sees the OLD ACTIVE block carriers, which
 * the guards deliberately keep out of the table (checkpoints are their own
 * class, rule 3). The election then picked a stretch made of old carriers plus
 * a thin new tail — a tier-2 distillation the table never offered. The model's
 * note, written for the table rows, landed on carrier distillation (note
 * mismatch) while the listed rows stayed untouched.
 *
 * Fix, two directions:
 * 1. the election REFUSES a pick whose span has NO positional overlap with the
 *    table's rows, with executable-hint copy (rule 9): explicit content range
 *    or explicit block-id distill;
 * 2. the range table ADVERTISES the distill candidates (`distill: b1..bN`
 *    line) so rule 7's "distillation is an explicit act" has a visible handle.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { allLogMessages } from '../src/messages.ts'
import { blockRegistry } from '../src/region.ts'
import { makeTools, pickCondenseSpan, type CondensePick, type ToolEnvironment } from '../src/tools.ts'
import { rangeTable } from '../src/nudge.ts'
import { appendAssistant, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'

const bigText = (label: string, seed: number): string => longText(label, seed) + longText(label + 'b', seed + 100)
const NOTE = '## Task\nFold the consumed stretch.\n## Now\nNext step is to verify the table alignment.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    foldFloorPct: 0,
    maxPaybackRequests: null,
    maxFoldCostTokens: null,
    compressCallIdsToHide: new Set(),
  }
}

function turnOf(env: ToolEnvironment, session: Session) {
  return env.kernel.processTurn({
    messages: allLogMessages(session),
    state: env.store.stateFor(session),
    config: kernelConfigFor({ modelContextLimit: 128000 }),
    tokenCount: 300000,
  })
}

function execStub(session: Session, callId: string): never {
  return {
    callId,
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent: { id: session.id, session, options: { provider: 'test-provider', model: 'test-model' }, ctx: { tokenMeter: undefined } },
  } as never
}

function compressTool(env: ToolEnvironment) {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  return definition
}

async function call(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const out = await compressTool(env).execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}

/** Append a wall-free group of rows; returns its surface span. */
function segment(session: Session, label: string, seed: number, rows: number, scale = 2): { start: number; end: number } {
  const before = session.surface.nodes.length
  for (let index = 0; index < rows; index += 1) {
    const text = Array.from({ length: scale }, (_, copy) => bigText(`${label}${index}c${copy}`, seed + index)).join('')
    if (index % 2 === 0) appendUser(session, text)
    else appendAssistant(session, text)
  }
  const nodes = session.surface.nodes.slice(before)
  return { start: nodes[0]!, end: nodes[nodes.length - 1]! }
}

async function fold(session: Session, env: ToolEnvironment, range: { start: number; end: number }, callId: string): Promise<string> {
  const text = await call(session, env, callId, { content: [{ startSeq: range.start, endSeq: range.end, summary: NOTE }] })
  assert.match(text, /Compressed 1 block/, 'fixture fold lands: ' + text)
  return text
}

function activeCarriers(session: Session) {
  return blockRegistry(session).filter((entry) => entry.active && entry.summarySeq !== null)
}

/**
 * The live-session shape: contiguous work segments (NO gaps between them — the
 * carrier checkpoints then sit positionally BEFORE the newer rows) each folded
 * into a block, then a NEW tail segment the table offers as its own row.
 */
async function blocksPlusTail(id: string, blocks: number, tailRows = 3, tailScale = 2): Promise<{ session: Session; env: ToolEnvironment; tail: { start: number; end: number } }> {
  const session = Session.create(id)
  const env = makeEnv()
  appendTurn(session, 1)
  // Append EVERYTHING first, then fold (tests/prefix-fold.test.ts fixture rule:
  // folding mid-build puts the newest rows inside the protected window).
  const works: Array<{ start: number; end: number }> = []
  for (let index = 0; index < blocks; index += 1) {
    works.push(segment(session, 'w' + index, 100 + index * 7, 2))
  }
  const tail = segment(session, 'tail', 900, tailRows, tailScale)
  for (const [index, work] of works.entries()) await fold(session, env, work, id + '-b' + index)
  return { session, env, tail }
}

// ---------------------------------------------------------------- A-1
test('A-1: an election whose stretch is pure old-carrier distillation is refused with executable guidance', async () => {
  // The live shape: MANY old checkpoint carriers whose summaries alone clear the
  // capacity gate, ZERO table-worthy plain rows (the thin tail is below the note
  // minimum). The election's timeline sees the carriers (the table does not —
  // host guards keep checkpoints out of the kernel geometry), so the cheapest
  // stretch is the OLD-CARRIER DISTILLATION. The alignment guard must refuse it
  // and name the explicit forms.
  const { session, env } = await blocksPlusTail('align-refuse', 30, 1, 1)
  const table = rangeTable(session, wholeSurfaceRangeView(session) as never, undefined)
  assert.doesNotMatch(table, /seq \d+\.\.\d+ · \d+msg/, 'the fixture really has no plain table row: ' + table.slice(0, 200))
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok('reason' in pick, 'a pure-carrier pick must be refused: ' + JSON.stringify('reason' in pick ? pick.reason : (pick as CondensePick).spans))
  if (!('reason' in pick)) return
  assert.match(pick.reason, /OUTSIDE the current range table/)
  assert.match(pick.reason, /compress\(\{ content:/, 'the guidance names an executable call shape')
  assert.match(pick.reason, /acp_status/, 'and points back at the latest table')
})

// ---------------------------------------------------------------- A-2
test('A-2: a table-row pick still passes (plain condense regression)', async () => {
  const { session, env, tail } = await blocksPlusTail('align-ok', 3, 8, 2)
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok(!('reason' in pick), 'a pick is elected: ' + JSON.stringify(pick))
  const elected = pick as CondensePick
  const overlapsTail = elected.spans.some((span) => span.start <= tail.end && span.end >= tail.start)
  assert.ok(overlapsTail, 'the elected stretch overlaps the table-offered tail row')
})

// ---------------------------------------------------------------- A-3
test('A-3: the range table advertises the distill candidates', async () => {
  const { session } = await blocksPlusTail('align-table', 4)
  const carriers = activeCarriers(session)
  assert.ok(carriers.length >= 2, 'the fixture carries multiple active blocks')
  const table = rangeTable(session, wholeSurfaceRangeView(session) as never, undefined)
  assert.match(table, /distill: b1\.\./, 'the distill line names the oldest block: ' + table)
  assert.match(table, /compress\(\{ content: \[\{ startSeq: 'b1', endSeq: 'b/, 'the line carries an executable call')
  assert.match(table, /tier-2/, 'and says what tier 2 costs')
})

// ---------------------------------------------------------------- A-4
test('A-4: the distill line rides the FRAGMENTED table (every plain row below the note minimum)', async () => {
  // 8 contiguous blocks folded one by one: every remaining plain stretch (thin
  // gap runs plus a short tail inside the protected window) stays below the
  // note-carrying minimum, so the range table offers ZERO plain rows — and the
  // explicit distill call must STILL be advertised there (it is the only
  // working move on a fragmented surface).
  const { session, env } = await blocksPlusTail('align-empty', 8)
  const carriers = activeCarriers(session)
  assert.equal(carriers.length, 8, 'the fixture really carries 8 active blocks')
  const table = rangeTable(session, wholeSurfaceRangeView(session) as never, undefined)
  assert.match(table, /distill: b1\.\./, 'the distill line is present on the fragmented table: ' + table)
  assert.match(table, /compress\(\{ content: \[\{ startSeq: 'b1', endSeq: 'b8'/, 'executable call shape')
  void env
})
