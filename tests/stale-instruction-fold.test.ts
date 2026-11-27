/**
 * foldStaleInstructionRows 线上开关 (rule 16 stale-instruction fold opening).
 *
 * Stale instruction rows (superseded AGENTS.md injections, superseded skill
 * catalogs) are opt-in foldable: folding one while its newest copy stays
 * visible never re-triggers an injection (rule 16, live-measured). This file
 * pins that the opt-in is wired on the three paths that should have it:
 *
 * 1. the nudge range table (src/nudge.ts rangeTable) — a stale row no longer
 *    splits a span, so the table offers a continuous span ACROSS it;
 * 2. the newest copy stays a wall on every path — `guardedSurfaceSeqsOf` and
 *    `protectedSurfaceSeqs` pin exactly the newest copies, so no offered span
 *    covers the CURRENT instruction row even when stale rows fold;
 * 3. the emergency overflow pool (src/index.ts compactForOverflow) — its
 *    largest-first election can now select a span containing a stale row.
 *
 * Fixtures reuse the real host injection shape from instruction-barrier.test.ts
 * (source { kind: 'agent-instructions', changes: [{ action:'set', scope, path,
 * digest }] }).
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { buildCompressibleSeqRanges, staleInstructionRowsOf } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { appendTurn, appendUser, appendAssistant, longText, wholeSurfaceRangeView } from './helpers.ts'
import { sessionEventsOf } from '../src/session-events.ts'

/** The exact source shape the host stamps on injected AGENTS.md rows. */
function instructionSource(scope: string, version: string): Record<string, unknown> {
  return {
    kind: 'agent-instructions',
    form: 'instructions',
    baseline: true,
    baselineIdentity: { projectRoot: '', projectRootMarkers: ['.git'], maxBytes: 65536 },
    changes: [{ action: 'set', scope, path: 'AGENTS.md', digest: `digest-${version}` }],
  }
}

function appendInstruction(session: Session, scope: string, version: string): number {
  session.append('user/message', { content: [{ type: 'text', text: `Instructions for ${scope} (${version}): keep tests green; docs in sync.` }], source: instructionSource(scope, version) } as never, { surfaceOp: 'append' })
  return sessionEvents(session).length - 1
}

function sessionEvents(session: Session) {
  return sessionEventsOf(session)
}

/**
 * Surface layout: work ... stale row ... work ... NEWEST row ... work.
 * preserveRecent: 0 keeps the recent tail out of the protection so the rows
 * under test are the only walls.
 */
function buildSession(id: string): { session: Session; staleSeq: number; newestSeq: number } {
  const session = Session.create(id)
  appendTurn(session, 1)
  // Each work segment must clear MIN_CONDENSE_SPAN_TOKENS (2572 since 2026-11-27,
  // 900/0.35) to survive the range-table row filter the assertions read.
  appendUser(session, longText('q0', 0) + longText('q0x', 10))            // seq 1
  appendAssistant(session, longText('a0', 1) + longText('a0x', 11), 1, 1) // seq 2
  const staleSeq = appendInstruction(session, '.\u0000AGENTS.md', 'v1')
  appendUser(session, longText('q1', 2) + longText('q1x', 12))
  appendAssistant(session, longText('a1', 3) + longText('a1x', 13), 1, 3)
  const newestSeq = appendInstruction(session, '.\u0000AGENTS.md', 'v2')
  appendUser(session, longText('q2', 4) + longText('q2x', 14))
  appendAssistant(session, longText('a2', 5) + longText('a2x', 15), 1, 5)
  return { session, staleSeq, newestSeq }
}

test('nudge table: a stale instruction row no longer splits a span — the table offers a continuous span across it', () => {
  const { session, staleSeq, newestSeq } = buildSession('nudge-stale-continuous')
  assert.ok(staleInstructionRowsOf(session).has(staleSeq), 'the superseded copy is stale')
  assert.ok(!staleInstructionRowsOf(session).has(newestSeq), 'the newest copy is not stale')

  const table = rangeTable(session, wholeSurfaceRangeView(session))
  assert.ok(table.length > 0, 'the table is not empty')
  // Only the `  - seq N..M · …` range rows count — the `Surface: seqs 1..N`
  // anchor line would otherwise look like a span covering everything.
  const rows = [...table.matchAll(/- seq (\d+)\.\.(\d+)/g)]
  assert.ok(rows.length > 0, `the table carries range rows: ${table.slice(0, 200)}`)
  const covered = (seq: number): boolean =>
    rows.some((match) => Number(match[1]) <= seq && seq <= Number(match[2]))
  assert.ok(covered(staleSeq), `some table span covers the stale row (seq ${staleSeq}) — it no longer splits`)
  assert.ok(!covered(newestSeq), `no table span covers the newest row (seq ${newestSeq}) — the wall stands`)
})

test('newest copy stays a wall even with foldStaleInstructionRows: true — no offered span covers it', () => {
  const { session, staleSeq, newestSeq } = buildSession('newest-still-wall')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    preserveRecent: 0,
    foldStaleInstructionRows: true,
  })
  assert.ok(ranges.length > 0, 'ranges exist')
  const covered = (seq: number): boolean => ranges.some((range) => range.start <= seq && seq <= range.end)
  assert.ok(covered(staleSeq), `the stale row (seq ${staleSeq}) is inside some offered span`)
  assert.ok(!covered(newestSeq), `the newest row (seq ${newestSeq}) is never offered — the guard does not loosen`)
})

test('emergency pool: largest-first election can select a span containing a stale instruction row', () => {
  const { session, staleSeq } = buildSession('emergency-stale')
  // The emergency path (src/index.ts foldPass) builds ONE kernel range spanning
  // the whole surface with preserveRecent: 5 and the same option; mirror it here.
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    preserveRecent: 5,
    mediaPriceOf: undefined,
    foldStaleInstructionRows: true,
  })
  assert.ok(ranges.length > 0, 'the emergency pool is not empty')
  // Mirror the largest-first election of compactForOverflow.
  const aboveFloor = ranges.filter((range) => !range.belowFloor)
  const pool = aboveFloor.length > 0 ? aboveFloor : ranges
  const best = pool.reduce((largest, range) => (range.tokens > largest.tokens ? range : largest))
  assert.ok(
    best.start <= staleSeq && staleSeq <= best.end,
    `the elected emergency span ${best.start}..${best.end} covers the stale row (seq ${staleSeq})`,
  )
})
