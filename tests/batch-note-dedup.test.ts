/**
 * Batch note dedup (B 方案, 2026-11-26): ONE compress call that lands SEVERAL
 * blocks from ONE shared note stores the full note in the FIRST block only;
 * every later block writes a one-line pointer (batch position + seqs range +
 * the first block's compactionId + an executable recall hint). Contracts:
 *   BD-1  a multi-span condense batch: first block full note, later blocks pointers
 *   BD-2  split-around-protected: same (covered in protected-split.test.ts ①)
 *   BD-3  an EXPLICIT content array with distinct summaries is untouched
 *   BD-4  shadowedTokenCount is unchanged (the ledger debt stays priced on
 *         the shadowed originals, never on the note text)
 *   BD-5  recall({ blockId: pointer block }) reads the pointer line back
 * The frame prefix rides the SAME framed channel for both shapes (written by
 * runCompactionTransaction's prefixSummaryBlocks), so projection and the
 * engine-written exemption stay byte-consistent.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { SUMMARY_FRAME_PREFIX } from '../src/messages.ts'
import { readCompactionSummary, blockRegistry } from '../src/region.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const NOTE = '## Task\nFold the consumed prefix and keep the ledger honest.\n'
  + '## Now\nRe-measure the election; done when every gate is named in the result.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session, callId: string): never {
  return {
    callId,
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent: {
      id: session.id,
      session,
      options: { provider: 'test-provider', model: 'test-model' },
      ctx: { tokenMeter: undefined },
    },
  } as never
}

async function compressCall(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  const out = await definition.execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}

function summaryEvents(session: Session) {
  return session.snapshotEvents().filter((event) => event.type === 'compaction/summary')
}

function notesOf(session: Session): string[] {
  return summaryEvents(session).map((event) => {
    const data = readCompactionSummary(event as never)
    return (data.summary ?? []).map((block) => block.text ?? '').join('')
  })
}

function withoutFrame(note: string): string {
  return note.startsWith(SUMMARY_FRAME_PREFIX) ? note.slice(SUMMARY_FRAME_PREFIX.length + 1) : note
}

/** A plain walled surface: two foldable runs separated by a guarded wall, so a
 *  condense election returns a MULTI-SPAN batch. Head/body big enough that each
 *  run clears MIN_CONDENSE_SPAN_TOKENS on its own (longText ~4 chars/token). */
function walledFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  for (let index = 1; index <= 4; index += 1) {
    if (index % 2 === 1) appendUser(session, longText('a' + index, index) + longText('a' + index + 'x', index + 50))
    else appendAssistant(session, longText('a' + index, index) + longText('a' + index + 'x', index + 50))
  }
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Instructions vwall: ' + longText('instrwall', 77) }],
    source: {
      kind: 'agent-instructions',
      form: 'instructions',
      baseline: true,
      changes: [{ action: 'set', scope: './\u0000AGENTS.md', path: 'AGENTS.md', digest: 'wall' }],
    },
  }), { surfaceOp: 'append' })
  for (let index = 1; index <= 4; index += 1) {
    if (index % 2 === 1) appendUser(session, longText('b' + index, 90 + index) + longText('b' + index + 'x', 140 + index))
    else appendAssistant(session, longText('b' + index, 90 + index) + longText('b' + index + 'x', 140 + index))
  }
  for (let index = 1; index <= 5; index += 1) {
    if (index % 2 === 1) appendUser(session, 'short tail ' + index)
    else appendAssistant(session, 'short tail ' + index)
  }
  return session
}

// ---------------------------------------------------------------- BD-1
test('BD-1: a multi-span condense batch keeps the full note in the FIRST block; later blocks carry the pointer', async () => {
  const env = makeEnv()
  const session = walledFixture('bd1-multi-span')
  const text = await compressCall(session, env, 'bd1', { condense: { summary: NOTE } })
  const notes = notesOf(session).map(withoutFrame)
  const events = summaryEvents(session)
  const blockCount = events.length
  assert.ok(blockCount >= 2, 'the batch lands more than one block: ' + text)
  // First block: the full note verbatim (frame stripped).
  assert.ok(notes[0]!.includes(NOTE), 'the FIRST block carries the full note: ' + notes[0]!.slice(0, 120))
  const firstId = (events[0]!.data as unknown as { compactionId: string }).compactionId
  assert.ok(firstId, 'the first event carries a compactionId')
  // Every later block: the pointer line ONLY (no full note), naming the first id.
  for (let index = 1; index < notes.length; index += 1) {
    const pointer = notes[index]!
    // Segment notes (2026-11-27): the pointer line STAYS the first line, but a
    // mechanical metadata line may follow it (the no-spans fallback).
    assert.match(pointer, /^Batch block \d+\/\d+ covering seqs \d+\.\.\d+ — full handoff note lives in block [0-9a-f-]+; recall\(\{ blockId: "[0-9a-f-]+" \}\) to read it back\.(|\nSegment sketch \(mechanical, from surface metadata\): seqs \d+\.\.\d+ — \d+ message\(s\), ~\d+t, \d+ tool message\(s\)\.)$/,
      `block ${index + 1} carries the pointer line: ${JSON.stringify(pointer)}`)
    assert.ok(pointer.includes(firstId), `pointer ${index + 1} names the FIRST block id: ${pointer}`)
    assert.doesNotMatch(pointer, /## Task/, `pointer ${index + 1} does NOT carry the note body`)
  }
  // The result reports the dedup.
  assert.match(text, /note dedup: full note in block [0-9a-f]+, \d+ pointer block\(s\) reference it instead of storing the note \(~\d+ tokens saved resident\)/,
    'the result line reports the dedup saving: ' + text)
})

// ---------------------------------------------------------------- BD-3
test('BD-3: an EXPLICIT content array with distinct summaries is untouched — each block stores its own summary', async () => {
  const env = makeEnv()
  const session = walledFixture('bd3-explicit')
  // Fold the two runs EXPLICITLY, each with its OWN distinct summary text.
  const text = await compressCall(session, env, 'bd3', {
    content: [
      { startSeq: 2, endSeq: 4, summary: NOTE },
      { startSeq: 6, endSeq: 8, summary: NOTE.replace('ledger honest', 'ledger honest — SECOND span') },
    ],
  })
  const notes = notesOf(session).map(withoutFrame)
  assert.ok(notes.length >= 2, 'both explicit ranges land: ' + text)
  for (const note of notes) {
    assert.doesNotMatch(note, /^Batch block \d+\/\d+ covering seqs /, 'no pointer line on an explicit call: ' + note.slice(0, 120))
    assert.ok(note.includes('## Task'), 'every explicit block keeps its full summary: ' + note.slice(0, 120))
  }
})

// ---------------------------------------------------------------- BD-4
test('BD-4: shadowedTokenCount is unchanged by the dedup — priced on the shadowed originals', async () => {
  const env = makeEnv()
  const session = walledFixture('bd4-shadowed')
  const before = summaryEvents(session).length
  await compressCall(session, env, 'bd4', { condense: { summary: NOTE } })
  const events = summaryEvents(session)
  assert.equal(before, 0, 'no earlier compaction in the fixture')
  for (const event of events) {
    const count = (event.data as unknown as { shadowedTokenCount: number }).shadowedTokenCount
    assert.ok(count > 0, 'every block prices its shadowed span, not the note: ' + String(count))
  }
  // The dedup only changed the summary TEXT — the ledger entries stay priced.
  const registry = blockRegistry(session)
  assert.ok(registry.length >= 2, 'the registry lists every landed block')
})

// ---------------------------------------------------------------- BD-5
test('BD-5: recall({ blockId: pointer block }) reads the pointer line back', async () => {
  const env = makeEnv()
  const session = walledFixture('bd5-recall')
  const text = await compressCall(session, env, 'bd5', { condense: { summary: NOTE } })
  const events = summaryEvents(session)
  const firstId = (events[0]!.data as unknown as { compactionId: string }).compactionId
  const pointerEvent = events[1]!
  const pointerId = (pointerEvent.data as unknown as { compactionId: string }).compactionId
  const recall = makeTools(env).find((candidate) => candidate.name === 'recall')
  assert.ok(recall, 'recall tool registered')
  const read = await recall.execute({ blockId: pointerId } as never, execStub(session, 'bd5-read'))
  const body = typeof read === 'string' ? read : (read as { text: string }).text
  void text
  assert.ok(!/no such block|unknown block/i.test(body), 'the pointer block resolves: ' + body.slice(0, 200))
  assert.match(body, /Batch block \d+\/\d+ covering seqs \d+\.\.\d+ — full handoff note lives in block /,
    'the recall output carries the pointer line: ' + body.slice(0, 300))
  assert.ok(body.includes(firstId), 'the recalled pointer names the first block id: ' + body.slice(0, 300))
})
