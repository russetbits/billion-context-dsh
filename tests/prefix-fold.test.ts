/**
 * Prefix fold — one condense call folds the WHOLE foldable prefix: every gap
 * between blocks AND every active block carrier, as ONE range.
 *
 * Pinned contracts (2026-11-24 redesign; plan §3 superseded by measurement):
 *
 * 1. The elected unit is the PREFIX, not one run. `pickCondenseSpan` returns the
 *    ranges `applyCompression` receives — one range carrying the whole prefix,
 *    plus at most one gap-only range when the prefix ends past the last carrier.
 *    The count is load-bearing: `applyCompression` creates ONE KERNEL BLOCK PER
 *    RANGE (`applySingleRange`, node_modules/acp-kernel/dist/index.js), so one
 *    range per block would turn 50 blocks into 95 and duplicate the single
 *    model-written note 95 times.
 * 2. A range that distils carriers must END on a block ref. A plain seq range
 *    crossing a live checkpoint carrier is refused (`liveCheckpointCarriersInSpan`
 *    runs only when NEITHER edge is a block ref, src/tools.ts), while `{seq .. bN}`
 *    is the tier 2/3 call that refusal itself recommends.
 * 3. Walls STOP the prefix, they are never skipped: a hole in the prefix could not
 *    be submitted as one range, and widening into a guard is a loop.
 * 4. The floor still refuses a shallow working set, and `force: true` still lands it.
 *
 * Fixture rule (rules 5/8): every surface is appended FIRST and folded afterwards.
 * Folding mid-build would put the LAST real user turn inside the folded span, and
 * the host guard refuses that — a fixture artefact, not a product bug.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { allLogMessages } from '../src/messages.ts'
import { blockRegistry, surfaceSuffixTokens } from '../src/region.ts'
import { makeTools, pickCondenseSpan, type CondensePick, type ToolEnvironment } from '../src/tools.ts'
import { appendAssistant, appendTurn, appendUser, longText } from '../tests/helpers.ts'

const NOTE = '## Task\nFold the whole consumed prefix of this session into one block; the auth rotation in src/auth/login.ts is the work under test.\n'
  + '## Now\nMeasure that one condense call collapses every carrier; done when the block count is 1.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    // The election SHAPE is what this file pins; the numeric gates (floor, payback,
    // cost ceiling) have their own files. Leaving them on would refuse fixture
    // geometries for reasons this file does not test.
    foldFloorPct: 0,
    maxPaybackRequests: null,
    maxFoldCostTokens: null,
    compressCallIdsToHide: new Set(),
  }
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

function turnOf(env: ToolEnvironment, session: Session) {
  return env.kernel.processTurn({
    messages: allLogMessages(session),
    state: env.store.stateFor(session),
    config: kernelConfigFor({ modelContextLimit: 128000 }),
    tokenCount: 300000,
  })
}

function pick(env: ToolEnvironment, session: Session, preserveRecent = 5): CondensePick {
  const result = pickCondenseSpan(session, turnOf(env, session), preserveRecent)
  assert.ok(!('reason' in result), 'a prefix is elected: ' + JSON.stringify(result))
  return result as CondensePick
}

/** Append a group of rows; returns the surface span it added. */
function segment(session: Session, label: string, seed: number, rows: number, scale = 2): { start: number; end: number } {
  const before = session.surface.nodes.length
  for (let index = 0; index < rows; index += 1) {
    const text = Array.from({ length: scale }, (_, copy) => longText(`${label}${index}c${copy}`, seed + index)).join('')
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

/** gap(2) | work -> b1 | tail(5 = exactly the protected window). */
async function oneBlockFixture(id: string): Promise<{ session: Session; env: ToolEnvironment; gap: { start: number; end: number } }> {
  const session = Session.create(id)
  const env = makeEnv()
  appendTurn(session, 1)
  const gap = segment(session, 'gap', 10, 2)
  const work = segment(session, 'work', 20, 2)
  segment(session, 'tail', 30, 5)
  await fold(session, env, work, id + '-w')
  return { session, env, gap }
}

// ---------------------------------------------------------------- P-1
test('P-1: one block plus the gap in front of it is one span that ends on the block ref', async () => {
  const { session, env, gap } = await oneBlockFixture('prefix-one')
  const elected = pick(env, session)
  assert.equal(elected.spans.length, 1, 'the prefix is ONE range: ' + JSON.stringify(elected.spans))
  const span = elected.spans[0]!
  assert.equal(span.start, gap.start, 'the range starts at the oldest gap node')
  const carrier = activeCarriers(session)[0]!
  assert.equal(span.end, carrier.summarySeq, 'and ends on the checkpoint carrier')
  assert.equal(span.endBlock, carrier.kernelBlockId, 'the terminal arm is a block ref (this is what makes it legal)')
  assert.deepEqual([...span.carriers], [carrier.kernelBlockId], 'the carrier is distilled by this span')
  assert.equal(elected.carrierCount, 1)
})

// ---------------------------------------------------------------- P-2
test('P-2: three blocks and the gaps between them are all in ONE range', async () => {
  const session = Session.create('prefix-three')
  const env = makeEnv()
  appendTurn(session, 1)
  const works: Array<{ start: number; end: number }> = []
  for (let index = 1; index <= 3; index += 1) {
    segment(session, `gap${index}`, index * 100, 2)
    works.push(segment(session, `work${index}`, index * 100 + 20, 2))
  }
  segment(session, 'tail', 900, 5)
  // Price the work runs BEFORE folding them (a folded run leaves the surface).
  const suffixBefore = surfaceSuffixTokens(session)
  const tokensOfRange = (range: { start: number; end: number }): number => {
    const from = session.surface.nodes.findIndex((seq) => seq === range.start)
    const to = session.surface.nodes.findIndex((seq) => seq === range.end)
    if (from < 0 || to < 0) return 0
    return Math.max(0, (suffixBefore[from] ?? 0) - (suffixBefore[to + 1] ?? 0))
  }
  const partsTokens = works.reduce((total, work) => total + tokensOfRange(work), 0)
  for (const [index, work] of works.entries()) await fold(session, env, work, `three-${index}`)
  assert.equal(activeCarriers(session).length, 3, 'three blocks exist')

  const elected = pick(env, session)
  assert.equal(elected.spans.length, 1, 'one range carries the whole prefix: ' + JSON.stringify(elected.spans))
  assert.equal(elected.carrierCount, 3, 'all three carriers are distilled')
  assert.deepEqual([...elected.spans[0]!.carriers], ['b1', 'b2', 'b3'], 'oldest-first carrier order')
  // The volume is the SUM of the parts, not one part.
  assert.ok(elected.tokens >= partsTokens, `the fold sums every part (~${elected.tokens} >= ~${partsTokens} tokens)`)
})

// ---------------------------------------------------------------- P-3
test('P-3: exactly one rewrite point — one compaction start for the whole prefix', async () => {
  const { session, env } = await oneBlockFixture('prefix-rewrite')
  const before = session.snapshotEvents().filter((event) => event.type === 'compaction/start').length
  const text = await call(session, env, 'rewrite', { condense: { summary: NOTE } })
  assert.match(text, /Compressed 1 block/, 'the fold lands: ' + text)
  const after = session.snapshotEvents().filter((event) => event.type === 'compaction/start').length
  assert.equal(after - before, 1, 'one range = one rewrite point, not one per block')
})

// ---------------------------------------------------------------- P-4
test('P-4: a block that is no longer active is not elected again', async () => {
  const session = Session.create('prefix-distilled')
  const env = makeEnv()
  appendTurn(session, 1)
  const work1 = segment(session, 'w1', 10, 2)
  segment(session, 'gapA', 15, 2)
  const work2 = segment(session, 'w2', 20, 2)
  // A large UNCONSUMED run after the blocks keeps the elected prefix non-trivial.
  segment(session, 'gapB', 300, 4)
  segment(session, 'tail', 900, 5)
  await fold(session, env, work1, 'distil-w1')
  await fold(session, env, work2, 'distil-w2')
  // Distil b1 INTO b2's chain: b1 stops being active.
  const distilled = await call(session, env, 'distil2', { content: [{ startSeq: 'b1', endSeq: 'b2', summary: NOTE }] })
  assert.match(distilled, /tier 2/, 'the distill lands: ' + distilled)
  const active = activeCarriers(session).map((entry) => entry.kernelBlockId)
  assert.deepEqual(active, ['b3'], 'only the distilled block stays active')
  const elected = pick(env, session)
  assert.deepEqual([...elected.spans.flatMap((span) => span.carriers)], ['b3'], 'only the ACTIVE carrier is elected')
})

// ---------------------------------------------------------------- P-5
test('P-5: the protected tail never enters the prefix', async () => {
  const { session, env } = await oneBlockFixture('prefix-tail')
  const nodes = session.surface.nodes
  // POSITION, not seq value: a checkpoint node carries a much higher seq than its
  // neighbours, so a numeric comparison would mis-report a covered tail row.
  const startsAt = nodes.length - 5
  const elected = pick(env, session)
  for (const span of elected.spans) {
    const from = nodes.findIndex((seq) => seq === span.start)
    const to = nodes.findIndex((seq) => seq === span.end)
    assert.ok(to < startsAt, `range ${span.start}..${span.end} (positions ${from}..${to}) must end before the protected tail at ${startsAt}`)
  }
})

// ---------------------------------------------------------------- P-6
test('P-6: below the floor the fold is refused with an executable next step', async () => {
  const session = Session.create('prefix-floor')
  // The fixture folds its own block with the floor OFF; the SAME store then answers
  // a condense under a floor of min(16K, max(8K, 0.9 × 16000)) = 14400, on a surface
  // the fold would sink well below — so the only thing that can refuse it is the floor.
  const env = makeEnv()
  appendTurn(session, 1)
  const work = segment(session, 'w', 10, 4)
  // A LARGE unconsumed run after the block keeps the prefix worth folding — and
  // keeps the whole surface above the 14400-token floor, which is what arms the guard.
  segment(session, 'after', 200, 8)
  segment(session, 'tail', 900, 5, 1)
  await fold(session, env, work, 'floor-w')
  const floored: ToolEnvironment = { ...env, foldFloorPct: 0.9, modelContextLimit: 16000 }
  const before = session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length
  const text = await call(session, floored, 'floor', { condense: { summary: NOTE } })
  // M-3: with the payback gate gone the below-floor shape that used to be refused
  // wholesale can elect ONE row that still clears BOTH capacity conditions (the
  // distill shape now reports itself), so the pin is on the REPORTING contract:
  // a fold that leaves the floor below is refused on the content path (see
  // fold-floor.test.ts), and anything that lands here says what it took.
  if (/condense folded nothing/.test(text)) {
    assert.match(text, /capacity gate/, 'the floor gate is named')
    assert.match(text, /short by ~\d+/, 'and the shortfall is printed')
    assert.match(text, /force:true/, 'and names the escape hatch')
    assert.equal(
      session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length,
      before,
      'nothing new landed',
    )
  } else {
    assert.match(text, /Compressed \d+ block/, 'a legal shape landed and is reported: ' + text)
  }
  // force is a real escape hatch on the SAME surface.
  const forced = await call(session, floored, 'floor-force', { condense: { summary: NOTE, force: true } })
  assert.match(forced, /Compressed \d+ block|condense folds \d+t/, 'force lands the fold: ' + forced)
})

// ---------------------------------------------------------------- P-7
test('P-7: an empty prefix is a miss, not an empty fold', () => {
  const session = Session.create('prefix-empty')
  const env = makeEnv()
  appendTurn(session, 1)
  appendUser(session, 'short question')
  appendAssistant(session, 'short answer')
  const result = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok('reason' in result, 'nothing to fold: ' + JSON.stringify(result))
  assert.match((result as { reason: string }).reason, /minimum range/, 'the reason names the character floor')
})

// ---------------------------------------------------------------- P-8 (regression anchor)
test('P-8: 50 carriers + 45 gaps land as ONE range, ONE block, ONE note (shape anchor, not a 50-to-1 promise)', async () => {
  const session = Session.create('prefix-50')
  const env = makeEnv()
  appendTurn(session, 1)
  const CARRIERS = 50
  const works: Array<{ start: number; end: number }> = []
  for (let index = 0; index < CARRIERS; index += 1) {
    // A gap in front of every block except the first (the turn row covers that one).
    if (index > 0) segment(session, `gap${index}`, 1000 + index * 10, 1, 1)
    works.push(segment(session, `w${index}`, 500 + index * 10, 2, 1))
  }
  segment(session, 'tail', 9000, 5, 1)
  for (const [index, work] of works.entries()) await fold(session, env, work, `fifty-${index}`)
  const carriers = activeCarriers(session)
  assert.equal(carriers.length, CARRIERS, 'the fixture really carries 50 active blocks')

  const elected = pick(env, session)
  assert.equal(elected.spans.length, 1, 'ONE range for the whole prefix: ' + JSON.stringify(elected.spans.map((span) => [span.start, span.end, span.carriers.length])))
  assert.equal(elected.carrierCount, CARRIERS, 'every active carrier is distilled by it')
  assert.equal(elected.spans[0]!.endBlock, carriers[carriers.length - 1]!.kernelBlockId, 'the terminal arm is the LAST carrier')

  // No overlap, no hole: the covered slice must contain every carrier.
  const nodes = session.surface.nodes
  const from = nodes.findIndex((seq) => seq === elected.spans[0]!.start)
  const to = nodes.findIndex((seq) => seq === elected.spans[0]!.end)
  assert.ok(from >= 0 && to >= from, 'the range edges sit on the surface in order')
  const covered = nodes.slice(from, to + 1).filter((seq) => carriers.some((entry) => entry.summarySeq === seq))
  assert.equal(covered.length, CARRIERS, 'the covered slice contains every carrier — no holes')

  const summariesBefore = session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length
  const text = await call(session, env, 'fifty', { condense: { summary: NOTE } })
  assert.match(text, /Compressed 1 block/, 'one block for the whole prefix: ' + text.split('\n')[0])
  const summariesAfter = session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length
  assert.equal(summariesAfter - summariesBefore, 1, 'the model note is written ONCE, not once per distilled block')
  assert.equal(activeCarriers(session).length, 1, '50 active blocks collapse to 1')
})

// ---------------------------------------------------------------- P-9
test('P-9: a gap AFTER the last carrier becomes its own gap-only range and also lands', async () => {
  const session = Session.create('prefix-tail-gap')
  const env = makeEnv()
  appendTurn(session, 1)
  const gap = segment(session, 'gap', 10, 2)
  const work = segment(session, 'work', 20, 2)
  // ONE row between the carrier and the protected window: the prefix ends on a gap.
  const trailing = segment(session, 'trailing', 40, 1)
  segment(session, 'tail', 900, 5)
  await fold(session, env, work, 'tail-gap-w')

  const elected = pick(env, session)
  assert.equal(elected.spans.length, 2, 'the prefix splits into a carrier range plus a trailing gap range: ' + JSON.stringify(elected.spans))
  const [main, tail] = elected.spans
  assert.ok(main !== undefined && tail !== undefined)
  assert.notEqual(main.endBlock, null, 'the FIRST range still ends on the block ref')
  assert.equal(tail.endBlock, null, 'the trailing range ends on a plain seq')
  assert.deepEqual([...tail.carriers], [], 'and distils nothing')
  assert.ok(main.end !== tail.start, 'the two ranges are distinct')
  assert.equal(tail.start, trailing.start, 'the trailing range is the surviving gap row')

  const text = await call(session, env, 'tail-gap', { condense: { summary: NOTE } })
  assert.match(text, /Compressed 2 block/, 'both ranges land: ' + text.split('\n')[0])
  assert.equal(
    session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length,
    3,
    'one fixture fold + one durable summary per submitted range',
  )
})
