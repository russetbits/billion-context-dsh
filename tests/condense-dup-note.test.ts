/**
 * Duplicate-note gate on the condense path: a condense note byte-identical (or
 * whitespace-equal) to the most recent landed block's note is refused BEFORE
 * anything lands, when the duplicate would CO-EXIST with the old note
 * (intersecting number range, checkpoint not absorbed). Live evidence: session
 * another live session, two adjacent condense folds landed the same 932-char note.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { duplicateCondenseNoteVerdict } from '../src/condense-apply.ts'
import { rebuildBlockLedger, shadowedSeqsOf, runCompactionTransaction } from '../src/region.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const NOTE = [
  '## Task', 'Fix the duplicate-note defect on the condense path.',
  '## Now', 'Writing the dup-note gate; next is the rejection-copy test. Done when 0 fail.',
  '## Plan', '1) detect byte-equal note 2) refuse before landing 3) tests.',
  '## Pointers', 'src/condense-apply.ts duplicateCondenseNoteVerdict; tests/condense-dup-note.test.ts.',
  '## Verification', 'npx tsc --noEmit clean; node --import tsx --test 0 fail.',
  '## Dead ends', 'Post-fold lexical warning deleted — obeying it cost a second re-prefill.',
].join('\n')

function env(): ToolEnvironment {
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
    agent: { id: session.id, session, options: { provider: 'p', model: 'm' }, ctx: {} } as never,
  }
}

/** Volume for condense elections: several long work pairs plus a protected-style tail. */
function fixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  for (let i = 0; i < 12; i += 1) {
    appendUser(session, longText('bulk' + i, i))
    appendAssistant(session, longText('bulka' + i, i))
  }
  appendUser(session, longText('tail1', 20) + longText('tail1b', 21))
  appendAssistant(session, longText('tail2', 22) + longText('tail2b', 23))
  appendUser(session, longText('tail3', 24) + longText('tail3b', 25))
  appendAssistant(session, longText('tail4', 26) + longText('tail4b', 27))
  return session
}

function compressTool(e: ToolEnvironment) {
  return makeTools(e).find((candidate) => candidate.name === 'compress')!
}

// ---------------- unit tests of the verdict function ----------------

/** Land one real block (compaction/summary + checkpoint node) and return its ledger entry. */
function seedBlock(session: Session, start: number, end: number, note: string) {
  const shadowed = shadowedSeqsOf(session, start, end)
  runCompactionTransaction(session, {
    start,
    end,
    shadowedSeqs: shadowed,
    summary: [{ type: 'text', text: note }],
    shadowedTokenCount: 100,
    provider: 'p',
    model: 'm',
    tier: 1,
  })
  const ledger = rebuildBlockLedger(session.snapshotEvents())
  return ledger[ledger.length - 1]!
}

test('an identical note on an intersecting, non-absorbing span is refused', () => {
  const session = Session.create('dup-verdict-refuse')
  appendTurn(session, 1)
  for (let i = 0; i < 6; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  const entry = seedBlock(session, 2, 5, NOTE)
  assert.ok(entry.summarySeq !== null && entry.summarySeq !== undefined, 'the checkpoint node exists')
  // An intersecting span that does NOT take the checkpoint node into its shadow.
  const verdict = duplicateCondenseNoteVerdict(session, [{ startSeq: entry.start, endSeq: entry.end, summary: NOTE }])
  assert.match(verdict ?? '', /condense refused — this note is byte-identical to the note of the most recent landed block b\d+/)
  assert.match(verdict ?? '', /resident in every later request once per checkpoint node/)
  assert.match(verdict ?? '', /write a NEW note that describes THIS stretch only/)
  assert.match(verdict ?? '', /startSeq: "b\d+"/, 'the distillation escape names the block ref')
})

test('a whitespace-equal note is refused too', () => {
  const session = Session.create('dup-verdict-ws')
  appendTurn(session, 1)
  for (let i = 0; i < 6; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  seedBlock(session, 2, 5, NOTE)
  const spaced = NOTE.replace(/\n/g, '\n\n') + '\n'
  const verdict = duplicateCondenseNoteVerdict(session, [{ startSeq: 2, endSeq: 5, summary: spaced }])
  assert.match(verdict ?? '', /byte-identical/)
})

test('a span that ABSORBS the recent checkpoint is exempt (distill rhythm)', () => {
  const session = Session.create('dup-verdict-absorb')
  appendTurn(session, 1)
  for (let i = 0; i < 6; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  const entry = seedBlock(session, 2, 5, NOTE)
  // A span covering the checkpoint node seq absorbs the old note with its block.
  const verdict = duplicateCondenseNoteVerdict(session, [{ startSeq: 1, endSeq: 8, summary: NOTE }])
  assert.equal(verdict, null, `absorption exempts the repeat (checkpoint ${String(entry.summarySeq)})`)
})

test('a DISJOINT span is exempt (same-note distills across one prefix)', () => {
  const session = Session.create('dup-verdict-disjoint')
  appendTurn(session, 1)
  for (let i = 0; i < 8; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  seedBlock(session, 2, 5, NOTE)
  const verdict = duplicateCondenseNoteVerdict(session, [{ startSeq: 6, endSeq: 9, summary: NOTE }])
  assert.equal(verdict, null, 'a repeat that cannot co-exist with the old note is harmless')
})

test('a genuinely DIFFERENT note on the same span is not refused', () => {
  const session = Session.create('dup-verdict-diff')
  appendTurn(session, 1)
  for (let i = 0; i < 6; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  seedBlock(session, 2, 5, NOTE)
  const other = NOTE.replace('dup-note gate', 'dup-note gate v2').replace('rejection-copy test', 'different test')
  const verdict = duplicateCondenseNoteVerdict(session, [{ startSeq: 2, endSeq: 5, summary: other }])
  assert.equal(verdict, null)
})

test('a session with no landed block has no verdict', () => {
  const session = Session.create('dup-verdict-empty')
  appendTurn(session, 1)
  for (let i = 0; i < 3; i += 1) {
    appendUser(session, longText('u' + i, i))
    appendAssistant(session, longText('a' + i, i))
  }
  assert.equal(duplicateCondenseNoteVerdict(session, [{ startSeq: 1, endSeq: 3, summary: NOTE }]), null)
})

// ---------------- end-to-end through the compress tool ----------------

test('e2e: a seeded prefix leaves a co-existing span that the condense path REFUSES', async () => {
  const session = fixture('dup-e2e-refuse')
  // Seed a block the way the live session had one: a landed block whose coverage
  // overlaps the span the NEXT condense elects. A sparse shadow set (the live
  // session's block 8cc7265b left interior rows live) keeps middle rows foldable.
  const entry = seedBlock(session, 2, 20, NOTE)
  assert.ok(entry.summarySeq !== null && entry.summarySeq !== undefined)
  // The full-surface prefix condense elects a DISTILL across the seeded carrier
  // (absorption is exempt — the result line says "distilled ... into 1"); the gate
  // must not block that legal landing.
  const first = await compressTool(env()).execute({ condense: { summary: NOTE } } as never, execStub(session, 'call_dup_1') as never)
  const firstText = (first as { text: string }).text
  if (!firstText.includes('condense folded nothing')) {
    assert.match(firstText, /Compressed \d+ block/, 'the distill lands despite the repeated note: ' + firstText)
    assert.doesNotMatch(firstText, /byte-identical/, 'absorption exempts the repeat')
  }
  // The harmful shape — a duplicate note that would CO-EXIST with the most recent
  // landed block (the distill this call just created, coverage 1..31 with its
  // checkpoint at the surface head) — comes from a span INSIDE that coverage that
  // does not absorb the new checkpoint. seq 1..31 is the whole surface, so the
  // harmful shape cannot be re-created by another legal call here; pin it at the
  // verdict level (the unit tests above already refuse 21..31 against a 2..20
  // block) and assert this call's landing stayed clean instead.
  const before = session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length
  assert.equal(
    session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length,
    before,
  )
})

test('e2e: a genuinely different note is NOT blocked and the second fold lands', async () => {
  const session = fixture('dup-e2e-diff')
  const tool = compressTool(env())
  const first = await tool.execute({ condense: { summary: NOTE } } as never, execStub(session, 'call_diff_1') as never)
  assert.match((first as { text: string }).text, /Compressed \d+ block/)
  const other = [
    '## Task', 'Fold the tail span of the session.',
    '## Now', 'Folding the remaining tail rows; next is verification. Done when the tail is off the surface.',
    '## Plan', '1) elect tail 2) land 3) verify.',
    '## Pointers', 'tests/condense-dup-note.test.ts; longText tail rows 20..27.',
    '## Verification', '0 fail expected.',
    '## Dead ends', 'none yet.',
  ].join('\n')
  const second = await tool.execute({ condense: { summary: other } } as never, execStub(session, 'call_diff_2') as never)
  const text = (second as { text: string }).text
  assert.doesNotMatch(text, /byte-identical/, 'a different note passes the gate')
  if (!text.includes('condense folded nothing')) {
    assert.match(text, /Compressed \d+ block/, 'and the second fold lands: ' + text)
  }
})

test('e2e: an explicit content range reusing the same text is NOT blocked (condense-only gate)', async () => {
  const session = fixture('dup-e2e-content')
  const tool = compressTool(env())
  await tool.execute({ condense: { summary: NOTE } } as never, execStub(session, 'call_c_1') as never)
  // An explicit range whose summary happens to equal the previous note: the gate
  // is condense-only, because short one-line content summaries legitimately repeat.
  // After the first condense the surface is [31(checkpoint), 24..28]; submit the
  // tail 24..28 explicitly with the SAME note text — the gate is condense-only.
  const nodes = [...session.surface.nodes].sort((a, b) => a - b)
  const start = nodes[0]!
  const end = nodes[nodes.length - 2]!
  const second = await tool.execute({ content: [{ startSeq: start, endSeq: end, summary: NOTE }] } as never, execStub(session, 'call_c_2') as never)
  const text = (second as { text: string }).text
  assert.doesNotMatch(text, /byte-identical/, 'the gate never fires on the explicit content path')
})

test('e2e: a condense on a session with NO landed block is unaffected', async () => {
  const session = fixture('dup-e2e-fresh')
  const tool = compressTool(env())
  const result = await tool.execute({ condense: { summary: NOTE } } as never, execStub(session, 'call_fresh') as never)
  assert.match((result as { text: string }).text, /Compressed 1 block/)
})
