/**
 * Segment notes（段级 note, 2026-11-27）：批内 note dedup 之后指针块只留一行
 * 全局 note 指针，各段的细节没有区分。两个补法：
 *   a. 模型 `condense.spans` 逐段写 note（主路径）——指针块追加模型写的段级 note；
 *   b. 模型没写时指针块自动带一行机械描述（范围+行数+体积+工具构成，从 surface
 *      元数据生成，引擎绝不自己摘要）。
 * Contracts:
 *   SN-1  a multi-span condense batch WITH spans: pointer blocks carry the
 *         segment note whose span they cover (intersection match)
 *   SN-2  WITHOUT spans: pointer blocks carry the mechanical metadata line
 *   SN-3  segment notes NEVER trip the note-weight gate (they are new
 *         information on top of the global note, not part of its budget)
 *   SN-4  a single-span batch is unaffected (regression: no pointer, no
 *         mechanical line, the global note lands verbatim)
 * Weight relation: noteWeightRatio / MAX_HANDOFF_NOTE_TOKENS constrain ONLY the
 * global handoff note; segment notes carry an independent MAX_SEGMENT_NOTE_CHARS
 * cap and never feed the weight gate.
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
import { appendAssistant, appendToolCall, appendToolResult, appendTurn, appendUser, longText } from './helpers.ts'

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

/** [start, end] seq range each landed summary covers (from the event itself). */
function spansOf(session: Session): Array<{ start: number; end: number; note: string; id: string }> {
  return summaryEvents(session).map((event) => {
    const data = readCompactionSummary(event as never)
    return {
      start: data.start,
      end: data.end,
      note: (data.summary ?? []).map((block) => block.text ?? '').join(''),
      id: (event.data as unknown as { compactionId: string }).compactionId,
    }
  })
}

function withoutFrame(note: string): string {
  return note.startsWith(SUMMARY_FRAME_PREFIX) ? note.slice(SUMMARY_FRAME_PREFIX.length + 1) : note
}

function toolBody(text: string, length: number): string {
  return longText(text, length)
}

/** A walled surface (like BD's walledFixture) whose SECOND run carries real
 *  tool-call/result pairs so the mechanical line has a tool share to report.
 *  Head/body sized so each run clears MIN_CONDENSE_SPAN_TOKENS on its own. */
function walledFixture(id: string, withTools = false): Session {
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
  if (withTools) {
    for (let index = 1; index <= 2; index += 1) {
      appendToolCall(session, toolBody('tw' + index, 60), 'call-' + id + '-' + index, 10 + index)
      appendToolResult(session, toolBody('tr' + index, 60), 'call-' + id + '-' + index, 10 + index)
      appendUser(session, longText('b' + index, 90 + index) + longText('b' + index + 'x', 140 + index))
    }
  } else {
    for (let index = 1; index <= 4; index += 1) {
      if (index % 2 === 1) appendUser(session, longText('b' + index, 90 + index) + longText('b' + index + 'x', 140 + index))
      else appendAssistant(session, longText('b' + index, 90 + index) + longText('b' + index + 'x', 140 + index))
    }
  }
  for (let index = 1; index <= 5; index += 1) {
    if (index % 2 === 1) appendUser(session, 'short tail ' + index)
    else appendAssistant(session, 'short tail ' + index)
  }
  return session
}

/** A one-run surface (no wall) — the condense election folds ONE span. */
function plainFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  for (let index = 1; index <= 4; index += 1) {
    if (index % 2 === 1) appendUser(session, longText('a' + index, index) + longText('a' + index + 'x', index + 50))
    else appendAssistant(session, longText('a' + index, index) + longText('a' + index + 'x', index + 50))
  }
  for (let index = 1; index <= 5; index += 1) {
    if (index % 2 === 1) appendUser(session, 'short tail ' + index)
    else appendAssistant(session, 'short tail ' + index)
  }
  return session
}

// ---------------------------------------------------------------- SN-1
test('SN-1: a multi-span batch WITH condense.spans — pointer blocks carry the model segment note for their span', async () => {
  const env = makeEnv()
  const session = walledFixture('sn1-segments')
  const text = await compressCall(session, env, 'sn1', {
    condense: {
      summary: NOTE,
      spans: [
        { startSeq: 0, endSeq: 3, note: 'SEG-HEAD: this run held the fixture bootstrap only.' },
        { startSeq: 0, endSeq: 999999, note: 'SEG-WIDE: covers every seq in this surface.' },
      ],
    },
  })
  const landed = spansOf(session)
  assert.ok(landed.length >= 2, 'the batch lands more than one block: ' + text)
  // First block: the full global note, verbatim, WITHOUT any segment/mechanical tail.
  assert.ok(withoutFrame(landed[0]!.note).includes(NOTE), 'the FIRST block carries the full note')
  assert.doesNotMatch(withoutFrame(landed[0]!.note), /SEG-|Segment sketch/, 'the first block gains no segment tail')
  // Every pointer block: the pointer line + the segment note covering ITS span.
  const firstId = landed[0]!.id
  for (const block of landed.slice(1)) {
    const body = withoutFrame(block.note)
    assert.match(body, /^Batch block \d+\/\d+ covering seqs \d+\.\.\d+ — full handoff note lives in block [0-9a-f-]+; recall\(\{ blockId: "[0-9a-f-]+" \}\) to read it back\.\n/,
      `pointer ${block.start}..${block.end} first line intact: ${JSON.stringify(body.slice(0, 160))}`)
    assert.ok(body.includes(firstId), 'pointer names the first block id')
    // The segment note for THIS span is appended (SEG-WIDE always intersects).
    assert.ok(body.includes('SEG-WIDE:'), `pointer ${block.start}..${block.end} carries the wide segment note: ${body}`)
    if (block.end <= 3) {
      assert.ok(body.includes('SEG-HEAD:'), `pointer over seqs ≤3 also carries SEG-HEAD: ${body}`)
    }
    assert.doesNotMatch(body, /Segment sketch/, 'a model segment note replaces the mechanical fallback')
    assert.doesNotMatch(body, /## Task/, 'pointer does NOT carry the global note body')
  }
})

// ---------------------------------------------------------------- SN-2
test('SN-2: WITHOUT spans — pointer blocks carry the mechanical metadata line (tool share included)', async () => {
  const env = makeEnv()
  const session = walledFixture('sn2-mechanical', true)
  const text = await compressCall(session, env, 'sn2', { condense: { summary: NOTE } })
  const landed = spansOf(session)
  assert.ok(landed.length >= 2, 'the batch lands more than one block: ' + text)
  assert.ok(withoutFrame(landed[0]!.note).includes(NOTE), 'the first block is the full note')
  let sawToolLine = false
  for (const block of landed.slice(1)) {
    const body = withoutFrame(block.note)
    const line = body.split('\n').find((row) => row.startsWith('Segment sketch'))
    assert.ok(line, `pointer ${block.start}..${block.end} carries a mechanical line: ${body}`)
    assert.match(line!, /^Segment sketch \(mechanical, from surface metadata\): seqs \d+\.\.\d+ — \d+ message\(s\), ~\d+t, \d+ tool message\(s\)\.$/, 'mechanical line shape: ' + line)
    const count = Number((line!.match(/(\d+) tool message/) ?? [])[1])
    if (count > 0) sawToolLine = true
    assert.doesNotMatch(body, /## Task/, 'mechanical fallback does NOT copy the global note')
  }
  assert.ok(sawToolLine, 'at least one pointer block reports tool messages (fixture carries tool pairs)')
})

// ---------------------------------------------------------------- SN-3
test('SN-3: segment notes never trip the note-weight gate — heavy spans land, the fold is not refused', async () => {
  const env = makeEnv()
  const session = walledFixture('sn3-weight')
  // A segment note far beyond MAX_SEGMENT_NOTE_CHARS (normalized to 600 + ellipsis)
  // would blow any weight cap if it counted against the global note's budget.
  const longSegment = 'SEG-HEAVY: ' + 'x'.repeat(4000)
  const text = await compressCall(session, env, 'sn3', {
    condense: {
      summary: NOTE,
      spans: [{ startSeq: 0, endSeq: 999999, note: longSegment }],
    },
  })
  assert.doesNotMatch(text, /refused/, 'no refusal: ' + text.slice(0, 200))
  const landed = spansOf(session)
  assert.ok(landed.length >= 2, 'the batch landed: ' + text)
  for (const block of landed.slice(1)) {
    const body = withoutFrame(block.note)
    assert.ok(body.includes('SEG-HEAVY:'), 'the segment note rides the pointer block')
    // The normalization cap: 600 chars of note body + the ellipsis.
    const segment = body.split('\n').find((row) => row.startsWith('SEG-HEAVY:'))!
    assert.ok(segment.length <= ('SEG-HEAVY: '.length + 600 + 1), 'segment note truncated to the 600-char cap: ' + segment.length)
  }
})

// ---------------------------------------------------------------- SN-4
test('SN-4: single-span regression — spans accepted, no pointer, no mechanical line', async () => {
  const env = makeEnv()
  const session = plainFixture('sn4-single')
  const text = await compressCall(session, env, 'sn4', {
    condense: {
      summary: NOTE,
      spans: [{ startSeq: 0, endSeq: 999999, note: 'SEG-ONLY: one-span note.' }],
    },
  })
  const landed = spansOf(session)
  assert.equal(landed.length, 1, 'a single-span election lands ONE block: ' + text)
  const body = withoutFrame(landed[0]!.note)
  assert.ok(body.includes(NOTE), 'the single block carries the full note verbatim')
  assert.doesNotMatch(body, /Batch block /, 'no pointer on a single-span batch')
  assert.doesNotMatch(body, /Segment sketch/, 'no mechanical line on a single-span batch')
  assert.doesNotMatch(body, /SEG-ONLY/, 'no segment tail on the first block (it IS the full note)')
  assert.ok(blockRegistry(session).length === 1, 'exactly one block registered')
})

// ---------------------------------------------------------------- SN-5
test('SN-5: spans normalize — string seqs accepted, empty notes dropped; junk entries rejected by the schema', async () => {
  const env = makeEnv()
  const session = walledFixture('sn5-normalize')
  const text = await compressCall(session, env, 'sn5', {
    condense: {
      summary: NOTE,
      spans: [
        { startSeq: '0', endSeq: '3', note: 'SEG-STR: string seqs parse.' },
        { startSeq: 0, endSeq: 5, note: '   ' },
        { startSeq: 0, endSeq: 999999, note: 'SEG-BACKUP: wide entry still matches.' },
      ],
    },
  })
  assert.doesNotMatch(text, /refused/, 'the call lands: ' + text.slice(0, 200))
  const landed = spansOf(session)
  assert.ok(landed.length >= 2, 'the batch landed: ' + text)
  for (const block of landed.slice(1)) {
    const body = withoutFrame(block.note)
    assert.ok(body.includes('SEG-BACKUP:'), 'the wide entry matches every pointer')
    if (block.end <= 3) assert.ok(body.includes('SEG-STR:'), 'string-seq entry matched by intersection')
    assert.doesNotMatch(body, /Segment sketch/, 'a matching segment note replaces the mechanical fallback')
  }
  // A non-object entry is NOT silently ignored — the schema gate rejects the
  // whole call with a named property (fail loud, not fail quiet).
  await assert.rejects(
    compressCall(session, env, 'sn5-junk', { condense: { summary: NOTE, spans: ['not an object'] } }),
    /spans\[0\].*must be an object/,
  )
})
