/**
 * Working-set floor (gap G1, arXiv 2611.05306 Appendix D).
 *
 * The paper’s open-weight band is this machine’s biography: at the 16K/12K
 * regime Qwen 3 Coder Next fell 53.0 → 3.6, Gemma 4 31B 65.2 → 58.0, Step 3.7
 * Flash 70.0 → 56.6, and EVERY model lost — diagnosed as too little room to make
 * progress between compactions, not summary quality. So a fold that leaves a
 * shallow working set is refused (advisory in the table, hard in condense), and the
 * refusal points at the row that STILL LEAVES the floor: a fold leaves exactly as
 * much room as it does not remove, so the widest row is the one that strands it.
 * When no row can keep the floor, condense degrades instead of deadlocking.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { condenseFloorRefusal, makeTools, type ToolEnvironment } from '../src/tools.ts'
import {
  foldPricingFor,
  WORKING_SET_FLOOR_MAX_TOKENS,
  WORKING_SET_FLOOR_MIN_TOKENS,
} from '../src/config.ts'
import { buildCompressibleSeqRanges, DEFAULT_FOLD_FLOOR_PCT } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { appendAssistant, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'

const NOTE = ['## Task', 'Guard the fold floor against a stranded working set.', '## Goal', 'Finish the fold-floor guard.', '## Next', 'Run the suite; 0 fail is the completion condition.'].join('\n')

function envWith(limit: number, foldFloorPct?: number): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: limit,
    ...(foldFloorPct === undefined ? {} : { foldFloorPct }),
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session): never {
  return {
    callId: 'call_1_condense',
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent: { id: session.id, session, options: { provider: 'p', model: 'm' }, ctx: {} } as never,
  }
}

/** A surface big enough to clear the character gate but small in tokens. */
function workFixture(id: string): Session {
  const session = Session.create(id)
// condense protects the recent tail (5) and the last real user turn, so the
// foldable stretch needs its own nodes AHEAD of that window, each big enough
// that the run clears the 5000-CHARACTER kernel gate.
  appendTurn(session, 1)
  appendUser(session, longText('task', 3) + longText('task2', 3))
  appendAssistant(session, longText('work', 3) + longText('work2', 3))
  appendUser(session, longText('ask', 3) + longText('ask2', 3))
  appendAssistant(session, longText('mid', 3) + longText('mid2', 3))
  appendUser(session, longText('tail1', 2))
  appendAssistant(session, longText('tail2', 2))
  appendUser(session, longText('tail3', 2))
  appendAssistant(session, longText('tail4', 2))
  return session
}

test('a surface smaller than the floor is never guarded — the floor needs room to keep', async () => {
  const session = workFixture('floor-room')
  // Real window, default pct ⇒ floor 39322 tokens, far above this 12392-token
  // surface. Guarding here would refuse every compress on a short session; the
  // floor protects a working set that could have been kept, not a wish.
  const definition = makeTools(envWith(262144)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: NOTE } } as never, execStub(session) as never)
  assert.match((result as { text: string }).text, /condense folds \d+t/, 'no refusal when the floor exceeds the whole surface')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1)
})

test('a model-named range is guarded by the same floor as an elected one', async () => {
  const session = workFixture('floor-explicit')
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ content: [{ startSeq: 1, endSeq: 3, summary: NOTE }] } as never, execStub(session) as never)
  const text = (result as { text: string }).text
  assert.match(text, /under the working-set floor/, 'the explicit range cannot slip past the guard')
  assert.match(text, /widen blindly/i, 'refusal explains that widening is not automatically valid')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 0, 'nothing written')
})

test('the floor is window × foldFloorPct, clamped to 8K–16K tokens', () => {
  // 0.15 of 262144 is 39322 — above the 16K ceiling (one KV block), because a bigger forced
  // working set squeezes the foldable window against the guarded tail.
  assert.equal(foldPricingFor({ modelContextLimit: 262144 }).floorTokens, WORKING_SET_FLOOR_MAX_TOKENS)
  assert.equal(foldPricingFor({ modelContextLimit: 262144, foldFloorPct: 0.25 }).floorTokens, WORKING_SET_FLOOR_MAX_TOKENS)
  // Inside the band the percentage rules.
  assert.equal(foldPricingFor({ modelContextLimit: 100000, foldFloorPct: DEFAULT_FOLD_FLOOR_PCT })?.floorTokens, 15000)
  // A small window still keeps a workable working set (the paper’s 7–9K peak band).
  assert.equal(foldPricingFor({ modelContextLimit: 32768 }).floorTokens, WORKING_SET_FLOOR_MIN_TOKENS)
  assert.equal(foldPricingFor({ modelContextLimit: 262144, foldFloorPct: 0 }).floorTokens, undefined, 'pct 0 disables the guard')
})

test('a row that would strand the working set is marked in the table', () => {
  const session = workFixture('floor-row')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    // The fixture surface totals 12392 tokens and the elected fold leaves 6197, so
    // 8000 is under the surface (guardable) and over the remainder (below floor).
    // A floor ABOVE the whole surface must NOT guard — see belowWorkingSetFloor.
    foldPricing: { floorTokens: 8000 },
  })
  assert.ok(ranges.length > 0, 'the fixture yields a range')
  assert.ok(ranges.every((range) => range.belowFloor), `every row lands under the floor: ${JSON.stringify(ranges.map((r) => r.liveAfterFoldTokens))}`)
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, { floorTokens: 8000 })
  assert.match(table, /below floor/, 'the row warns before the model picks it')
  assert.match(table, /STILL LEAVES the floor/, 'and points at the row that is safe, not the widest one')
})

test('a below-floor condense is refused with the branch that fits the row set', async () => {
  const session = workFixture('floor-refuse')
  // M-3: before the payback gate was removed this fixture was refused as a WHOLE
  // (the below-floor shape could not be elected at all). Now the engine elects the
  // one row that still clears BOTH capacity conditions and refuses the rest row by
  // row on the content path — so the below-floor refusal copy is pinned by feeding
  // the elected pick straight to `condenseFloorRefusal` (the branch test above) and
  // the tool-level behavior is pinned by the explicit-range guard instead.
  // The tool-level guard for a below-floor range stays: an EXPLICIT range that
  // would strand the floor is refused with the compliant moves, and force escapes it.
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const nodes = session.surface.nodes as readonly number[]
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { foldPricing: { floorTokens: 8192 } })
  const belowRow = ranges.find((range) => range.belowFloor)!
  const result = await definition!.execute(
    { content: [{ startSeq: belowRow.start, endSeq: belowRow.end, summary: NOTE }] } as never,
    execStub(session) as never,
  )
  const text = (result as { text: string }).text
  assert.match(text, /under the working-set floor/, 'the explicit range is refused for the floor')
  assert.match(text, /force:true/, 'the escape hatch is named')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 0, 'nothing written')
  void nodes
})

test('force folds it anyway — overflow pressure is a legitimate override', async () => {
  const session = workFixture('floor-force')
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute(
    { condense: { summary: NOTE, force: true } } as never,
    execStub(session) as never,
  )
  const text = (result as { text: string }).text
  assert.match(text, /condense folds \d+t/, 'the forced fold lands')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1, 'one summary written')
})

test('the refusal names WHICH gate failed and its arithmetic', () => {
  // 2026-11-25 (M-3): ONE gate. The refusal names capacity and its numbers — a
  // refusal that does not name the gate and its arithmetic leaves the model
  // guessing, and guessing means a retry of the same shape. Payback/cost stay
  // REPORTED in the head line but gate nothing, so no gate branch can be invented.
  const base = {
    spans: [{ start: 1, end: 4, endBlock: null, carriers: [], tokens: 9000 }],
    tokens: 9000, noteTokens: 900, keptTokens: 2000, floorTokens: 8000,
    belowFloor: true, aboveFloorAvailable: false, guardedRowCount: 3, eligibleRangeCount: 2,
    carrierCount: 0, foldCostTokens: 9000, paybackRequests: 1,
    gates: { capacity: false },
  }
  const floor = condenseFloorRefusal(base as never, 8000)
  assert.match(floor, /floor gate/, 'the capacity gate is named')
  assert.match(floor, /short by ~6000/, 'and the shortfall is printed')
  assert.match(floor, /force: ?true/, 'the only compliant escape, with the copyable call shape')
  assert.doesNotMatch(floor, /payback gate|cost gate/, 'the removed gates are gone entirely')

  // A pick that only REPORTS a slow payback (no gate can fail but capacity) is
  // refused for another reason only via the every-gate-passes branch.
  const passing = condenseFloorRefusal({
    ...base, gates: { capacity: true }, belowFloor: false,
    keptTokens: 12000, foldCostTokens: 27000, paybackRequests: 3,
  } as never, 8000)
  assert.match(passing, /every gate passes/, 'a capacity-passing shape is not invented a floor failure')
})
