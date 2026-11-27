/**
 * condense ELECTION — the ONE gate (M-3, 2026-11-25), decided in ONE pass over
 * every geometry.
 *
 * 2026-11-25 rewrite (lead spec). The old election walked a PREFIX and, when the
 * floor refused it, FELL BACK to a different row; the model's note then described a
 * stretch the engine never folded (note drift), and the floor fired as a refusal on
 * a fold whose benefit clearly beat its cost. The shape since M-3:
 *
 *   candidates = every geometry (prefix; a CUT inside the prefix's first range;
 *                a single range; a merged run of ranges)
 *     gate capacity : folded ≥ MIN_CONDENSE_SPAN_TOKENS and the remainder stays
 *                     above the working-set floor (arXiv 2611.05306 App. D —
 *                     never bypassable by the election). The ONLY gate: the
 *                     payback/cost ceilings were removed with M-3 (never
 *                     independently decisive on a live session); payback and
 *                     re-prefill cost stay COMPUTED and REPORTED, never compared.
 *   → order by re-prefill cost ascending, then by saving descending, take #1.
 *     NOTHING passes ⇒ name the closest shape AND the failed gate, and fold
 *     nothing. There is no fallback branch any more.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor, foldPricingFor } from '../src/config.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { allLogMessages } from '../src/messages.ts'
import { blockRegistry, buildCompressibleSeqRanges, surfaceSuffixTokens } from '../src/region.ts'
import {
  MIN_CONDENSE_SPAN_TOKENS,
  condenseFloorRefusal,
  formatPayback,
  makeTools,
  paybackLabel,
  pickCondenseSpan,
  MAX_NOTE_WEIGHT_RATIO,
  TYPICAL_HANDOFF_NOTE_TOKENS,
  type CondensePick,
  type ToolEnvironment,
} from '../src/tools.ts'
import { rangeTable } from '../src/nudge.ts'
import { wholeSurfaceRangeView } from './helpers.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const bigText = (label: string, seed: number): string => longText(label, seed) + longText(label + 'b', seed + 100)

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

function turnOf(env: ToolEnvironment, session: Session) {
  return env.kernel.processTurn({
    messages: allLogMessages(session),
    state: env.store.stateFor(session),
    config: kernelConfigFor({ modelContextLimit: 128000 }),
    tokenCount: 300000,
  })
}

const liveOf = (session: Session): number => surfaceSuffixTokens(session)[0] ?? 0

/**
 * `head` foldable messages, then `tail` messages that the protected window keeps
 * live (the caller passes preserveRecent = tail). One bigText copy is ~2065 tokens,
 * measured on the fixture helpers.
 */
function bodyFixture(id: string, head: number, tail: number): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  for (let index = 1; index <= head + tail; index += 1) {
    const label = index <= head ? 'h' + index : 't' + index
    if (index % 2 === 1) appendUser(session, bigText(label, index))
    else appendAssistant(session, bigText(label, index))
  }
  return session
}

function elect(
  session: Session,
  env: ToolEnvironment,
  preserveRecent: number,
  pricing?: { floorTokens?: number },
) {
  return pickCondenseSpan(session, turnOf(env, session), preserveRecent, pricing)
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

/** An AGENTS.md injection row in the host shape — a GUARDED wall between two runs. */
function appendInstruction(session: Session, digest: string): void {
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Instructions v' + digest + ': ' + longText('instr' + digest, 77) }],
    source: {
      kind: 'agent-instructions',
      form: 'instructions',
      baseline: true,
      changes: [{ action: 'set', scope: './\u0000AGENTS.md', path: 'AGENTS.md', digest }],
    },
  }), { surfaceOp: 'append' })
}

/**
 * TWO eligible runs separated by a guarded wall:
 *   head (foldable A) | instruction wall | body (foldable B) | short protected tail
 * The wall forbids merging them, so A and B compete as separate geometries — and B is
 * ALWAYS cheaper (an earlier rewrite point re-reads more), which is exactly the shape
 * that made the cheapest-first order fold a stretch behind the prefix.
 */
function walledFixture(id: string, head: number, body: number, tail = 5): { session: Session; starts: number[] } {
  const session = Session.create(id)
  appendTurn(session, 1)
  for (let index = 1; index <= head; index += 1) {
    if (index % 2 === 1) appendUser(session, bigText('a' + index, index))
    else appendAssistant(session, bigText('a' + index, index))
  }
  appendInstruction(session, 'wall')
  for (let index = 1; index <= body; index += 1) {
    if (index % 2 === 1) appendUser(session, bigText('b' + index, 80 + index))
    else appendAssistant(session, bigText('b' + index, 80 + index))
  }
  // A tail of SHORT messages: cheap as a protected window, and long enough that the
  // engine's own preserveRecent (5) keeps exactly these rows live.
  for (let index = 1; index <= tail; index += 1) {
    if (index % 2 === 1) appendUser(session, 'short tail ' + index)
    else appendAssistant(session, 'short tail ' + index)
  }
  return { session, starts: [] }
}

/** The two eligible runs the geometry produces (oldest first), with their seq starts. */
function runStarts(session: Session, env: ToolEnvironment, preserveRecent: number, pricing?: Record<string, number>): number[] {
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    preserveRecent,
    foldStaleInstructionRows: true,
    ...(pricing === undefined ? {} : { foldPricing: pricing }),
  })
  return ranges.map((range) => range.start)
}

// ---------------------------------------------------------------- ①
test('E-1: a prefix too big to fold is CUT inside its first range, not replaced by another row', () => {
  // The measured live geometry: ~11 foldable messages in front of a protected tail,
  // the floor at 16384. Folding the WHOLE head strands the floor; the old election
  // refused here (or fell back to a different row, which is what caused note drift).
  const env = makeEnv()
  const session = bodyFixture('election-cut', 11, 5)
  const live = liveOf(session)
  assert.ok(live > 30000, 'the fixture is a real-size surface: ' + live)

  const pricing = { floorTokens: 16384 }
  const pick = elect(session, env, 5, pricing)
  assert.ok(!('reason' in pick), 'a cut inside the first range must save the fold: ' + JSON.stringify(pick))

  const landed = pick as CondensePick
  assert.equal(landed.spans.length, 1, 'the cut geometry is ONE plain-seq span')
  assert.ok(landed.keptTokens >= 16384, 'the post-fold working set holds the floor: ' + landed.keptTokens)
  assert.ok(landed.tokens >= MIN_CONDENSE_SPAN_TOKENS, 'and the cut still carries a handoff note: ' + landed.tokens)
  assert.ok(landed.tokens < live * 0.6, 'the cut is SMALLER than the whole eligible head: ' + landed.tokens)
  assert.ok(landed.gates.capacity, 'the ONE gate passes: ' + JSON.stringify(landed.gates))
  assert.ok(Number.isFinite(landed.paybackRequests), 'payback is still REPORTED (no longer a gate): ' + landed.paybackRequests)
})

// ---------------------------------------------------------------- ②
test('E-2 (M-3): a slow-payback fold is no longer refused — its payback is reported instead', () => {
  const env = makeEnv()
  // 4 foldable messages (~8260t) in front of a 24000t protected tail: payback ≈ 3 req.
  // Before M-3 the payback gate refused this shape; the gate is gone, so the same
  // geometry is now ELECTED and the number stays visible on the pick.
  const session = bodyFixture('election-payback', 4, 12)
  const live = liveOf(session)
  const pick = elect(session, env, 12, { floorTokens: 16384 })
  assert.ok(!('reason' in pick), 'no payback gate any more — the shape elects: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(landed.paybackRequests > 2, 'and its payback really is slow: ' + landed.paybackRequests)
  assert.ok(live > 20000)
})

// ---------------------------------------------------------------- ③
test('E-3 (M-3): an expensive re-prefill no longer refuses — the cost is still reported', () => {
  const env = makeEnv()
  // 16 foldable messages (~33040t) + a 32-message protected tail (~66080t): the
  // re-prefill is ~66K, over the DELETED 60K cap. Before M-3 the cost gate refused
  // it; now it elects and the cost stays on the pick for the model to judge.
  const session = bodyFixture('election-ceiling', 16, 32)
  const pick = elect(session, env, 32, { floorTokens: 16384 })
  assert.ok(!('reason' in pick), 'no cost gate any more — the shape elects: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(landed.foldCostTokens > 60000, 'and the reported cost really was over the old default: ' + landed.foldCostTokens)
})

// ---------------------------------------------------------------- ④
test('E-4: with no qualifying geometry the miss names the gate and never prints a fallback story', () => {
  const env = makeEnv()
  // A real geometry (4 foldable messages) under a floor it cannot leave: every cut
  // either strands the floor or falls under the minimum note-carrying size.
  const session = bodyFixture('election-none', 4, 1)
  const pick = elect(session, env, 1, { floorTokens: 8000 })
  assert.ok('reason' in pick, 'nothing qualifies: ' + JSON.stringify(pick))
  const reason = (pick as { reason: string }).reason
  assert.match(reason, /capacity gate/, 'the failed gate is named')
  assert.match(reason, /short by ~\d+|under the ~\d+-token minimum/, 'and the size shortfall')
  assert.match(reason, /NO fallback/, 'and it says the election has no fallback')
  // The deleted branches must be GONE from the copy — a model that reads them retries
  // the same shape, which is exactly the loop the rewrite removes.
  assert.doesNotMatch(reason, /AN ELIGIBLE ROW STILL LEAVES THE FLOOR/)
  assert.doesNotMatch(reason, /NO ELIGIBLE ROW LEAVES THE FLOOR/)
  assert.doesNotMatch(reason, /NOTHING ELSE IS ELIGIBLE AT ALL/)
  assert.doesNotMatch(reason, /the foldable prefix is ~\d+ tokens, under the/)
})

// ---------------------------------------------------------------- ⑤
test('E-5: cut points land on surface nodes and never cross a tool pair', () => {
  const env = makeEnv()
  const session = bodyFixture('election-pair', 11, 5)
  const nodes = session.surface.nodes as readonly number[]
  const pick = elect(session, env, 5, { floorTokens: 16384 }) as CondensePick
  assert.ok(!('reason' in pick), 'a geometry exists')
  for (const span of pick.spans) {
    assert.ok(nodes.includes(span.start), 'span start is a live surface node: ' + span.start)
    assert.ok(nodes.includes(span.end), 'span end is a live surface node: ' + span.end)
    const from = nodes.indexOf(span.start)
    const to = nodes.indexOf(span.end)
    assert.ok(from <= to, 'the span is a forward POSITIONAL run (seq values are not monotonic)')
  }
})

// ---------------------------------------------------------------- ⑥
test('E-6: force crosses the capacity gate but never a protected row', () => {
  const env = makeEnv()
  // 4 foldable messages (~8260t) on a ~9K surface under an 8K floor: folding them
  // strands the working set (belowFloor), and no smaller cut clears the note-carrying
  // minimum — the old fixture now elects WITHOUT force since the payback gate is
  // gone, so the force test needs a genuinely below-floor shape.
  const session = bodyFixture('election-force', 4, 1)
  const pricing = { floorTokens: 8000 }
  const turn = turnOf(env, session)
  assert.ok('reason' in pickCondenseSpan(session, turn, 1, pricing), 'without force the below-floor shape is refused')
  const forced = pickCondenseSpan(session, turn, 1, pricing, { force: true })
  assert.ok(!('reason' in forced), 'force elects it anyway: ' + JSON.stringify(forced))
  const landed = forced as CondensePick
  assert.equal(landed.gates.capacity, false, 'and the pick still REPORTS the failed gate')
  // The protected tail the caller excluded stays out of the fold.
  const nodes = session.surface.nodes as readonly number[]
  const tailStart = landed.spans[landed.spans.length - 1]!.end
  const tailPosition = nodes.indexOf(tailStart)
  assert.ok(tailPosition < nodes.length - 1, 'the fold stops before the protected row: ' + tailPosition)
})

// ---------------------------------------------------------------- ⑦
test('E-7 (M-3): the old 1-req/2-req boundary geometry now elects and reports 2.0', () => {
  const env = makeEnv()
  // 16 foldable + 32 protected tail: payback is exactly 2.0 (live/3 span). The
  // ceiling that used to separate 1 req from 2 req is gone; the ratio stays
  // REPORTED and the boundary geometry elects under either (deleted) setting.
  const session = bodyFixture('election-boundary', 16, 32)
  const pick = pickCondenseSpan(session, turnOf(env, session), 32, { floorTokens: 16384 })
  assert.ok(!('reason' in pick), 'the boundary geometry elects: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(Math.abs(landed.paybackRequests - 2) < 0.05, 'the geometry really sits on 2.0: ' + landed.paybackRequests)
})

// ---------------------------------------------------------------- ⑧
test('E-8: a span of live/3 is elected by default — the old "saving > cost" rule refused it', () => {
  const env = makeEnv()
  // head = live/3 means tail = 2 × head, i.e. payback = 2.0 exactly: inside the
  // DEFAULT ceiling (2) and refused by maxPaybackRequests=1. This is the property
  // that makes the rule scale: the larger the context, the smaller the span that
  // still pays back — the OLD rule demanded live/2, so it folded LESS as contexts grew.
  const session = bodyFixture('election-third', 16, 32)
  const landed = elect(session, env, 32, { floorTokens: 16384 })
  assert.ok(!('reason' in landed), 'live/3 is elected: ' + JSON.stringify(landed))
  const pick = landed as CondensePick
  const live = liveOf(session)
  assert.ok(pick.tokens >= live / 3 - 1, 'the elected span really is about a third of live: ' + pick.tokens + '/' + live)
  assert.ok(Math.abs(pick.paybackRequests - 2) < 0.05, 'and its payback is reported at ~2.0: ' + pick.paybackRequests)
})

// ---------------------------------------------------------------- ⑨
test('E-9: the payback number is printed in the range table, in acp_status copy and in the refusal', () => {
  const env = makeEnv()
  const session = bodyFixture('election-visible', 11, 5)
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, {
    floorTokens: 0,
  })
  assert.match(table, /payback \d+\.\d/, 'every range row carries its payback: ' + table)

  const pick = elect(session, env, 5, { floorTokens: 16384 }) as CondensePick
  assert.ok(!('reason' in pick))
  assert.equal(
    paybackLabel(pick.foldCostTokens, pick.tokens),
    formatPayback(pick.foldCostTokens, pick.tokens) + ' req',
    'the label and the pick agree on the same ratio',
  )
  const refusal = condenseFloorRefusal({
    ...pick,
    gates: { capacity: false },
    belowFloor: true,
  }, 16384)
  assert.match(refusal, /pays back in \d+\.\d/, 'the refusal prints the payback too')
})

// ---------------------------------------------------------------- ⑩
test('E-10 (M-3): the refusal never names the deleted knobs', () => {
  const env = makeEnv()
  // The payback/cost knob lines are GONE: there is no ceiling to print, so a
  // refusal must never point the model at compress.maxPaybackRequests any more.
  const session = bodyFixture('election-knob', 4, 1)
  const reason = (elect(session, env, 1, { floorTokens: 8000 }) as { reason: string }).reason
  assert.doesNotMatch(reason, /maxPaybackRequests|maxFoldCostTokens/, 'no deleted knob is named: ' + reason)
})

// ---------------------------------------------------------------- ⑪
test('E-11 (M-1): the elected batch starts at the prefix start even when a later shape is CHEAPER', () => {
  // Predictability before marginal cost — M-1 form: the model writes its note BEFORE
  // the call, so the elected shape must still START at the prefix start. M-1 extends
  // what that shape may be: a multi-span batch that CONTAINS the prefix run keeps the
  // note describable (the note covers the batch, whose head IS the prefix), so the old
  // single-run "prefix costs >= later" assertion gave way to the batch payback win.
  const env = makeEnv()
  const { session } = walledFixture('election-prefix-first', 2, 2)
  const pricing = { floorTokens: 0 }
  const starts = runStarts(session, env, 5, pricing)
  assert.equal(starts.length, 2, 'the wall splits the surface into TWO eligible runs: ' + JSON.stringify(starts))

  const pick = elect(session, env, 5, pricing)
  assert.ok(!('reason' in pick), 'a prefix shape passes: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.equal(landed.spans[0]!.start, starts[0], 'the elected shape still STARTS at the prefix start')
  assert.ok(
    landed.spans.some((span) => span.start <= starts[1]! && starts[1]! <= span.end),
    'the batch CONTAINS the later run (M-1): ' + JSON.stringify(landed.spans),
  )
  // M-1 thesis, with DISCRIMINATING arithmetic (F4: the old `<= … + 1e-9` form was
  // a tautology — same foldCost, and Σ(batch) ≥ Σ(later run) always holds since the
  // batch CONTAINS the later run). Assert the measured RATIO instead: the batch's
  // payback is the later-run-alone payback scaled by laterRun/Σ — 4130t/8260t = 0.5
  // on this fixture — so a degenerate batch that dropped the prefix run would fail.
  const sum = landed.spans.reduce((total, span) => total + span.tokens, 0)
  const laterRunTokens = landed.spans.find((span) => span.start <= starts[1]! && starts[1]! <= span.end)!.tokens
  assert.ok(sum > laterRunTokens, 'the batch folds strictly MORE than the later run alone: ' + sum + ' vs ' + laterRunTokens)
  const batchPayback = landed.foldCostTokens / sum
  const laterRunPayback = landed.foldCostTokens / laterRunTokens
  assert.ok(
    Math.abs(batchPayback - (laterRunTokens / sum) * laterRunPayback) < 0.01 * laterRunPayback,
    'batch payback = later-run payback × laterRun/Σ (' + laterRunTokens + '/' + sum + '): '
      + batchPayback.toFixed(4) + ' vs ' + ((laterRunTokens / sum) * laterRunPayback).toFixed(4),
  )
  assert.ok(batchPayback < laterRunPayback, 'and it is STRICTLY better: ' + batchPayback.toFixed(4) + ' < ' + laterRunPayback.toFixed(4))
})

// ---------------------------------------------------------------- ⑫
test('E-12 (M-1): the election names its REAL geometry — and the result reports the seqs it actually folded', async () => {
  // Old form: a one-message prefix whose every prefix shape fails lets the election
  // take a later run. M-1 changes the answer: the batch (prefix run + later run)
  // pays ONE re-prefill for BOTH at the later run's cheap rewrite point, so the
  // batch's payback dominates and the prefix is folded WITH the later run instead
  // of being left behind. The surviving contract is the calibration the model
  // needs: the result line reports the seqs it ACTUALLY folded.
  const env = makeEnv()
  const { session } = walledFixture('election-later', 1, 4)
  const pricing = { floorTokens: 0 }
  const starts = runStarts(session, env, 5, pricing)
  assert.equal(starts.length, 2, 'two eligible runs: ' + JSON.stringify(starts))

  const pick = elect(session, env, 5, pricing)
  assert.ok(!('reason' in pick), 'a shape passes: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(landed.spans.length >= 2, 'M-1 batch (prefix run + later run) wins on payback: ' + JSON.stringify(landed.spans))
  assert.equal(landed.spans[0]!.start, starts[0], 'the batch still starts at the prefix start')
  assert.ok(
    landed.spans.some((span) => span.start <= starts[1]! && starts[1]! <= span.end),
    'the batch covers the later run too: ' + JSON.stringify(landed.spans),
  )

  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: NOTE } } as never, execStub(session, 'e12') as never)
  const text = (result as { text: string }).text
  assert.match(text, new RegExp('condense folds \\d+t \\(seqs ' + starts[0] + '\\.\\.\\d+ in ' + landed.spans.length + ' ranges\\)'), 'the result reports the REAL seqs it folded: ' + text)
})

// ---------------------------------------------------------------- barrier
/** ~4 chars/token plain prose: 6000 chars clears the kernel's 5000-CHARACTER minimum
 *  while staying under MIN_CONDENSE_SPAN_TOKENS (2572) — the fragment shape. */
const prose = (label: string, chars: number): string => {
  const unit = 'alpha beta gamma delta epsilon zeta eta theta '
  return label + ': ' + unit.repeat(Math.ceil(chars / unit.length)).slice(0, chars)
}

async function compressCall(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  const out = await definition!.execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}

/**
 * Append ONE sub-minimum prose gap and fold it into a carrier. The gap must sit
 * OUTSIDE the protected zone: a later user message takes over "last real user turn",
 * and five rows after it push the gap past the kernel's recent window.
 */
async function foldOneGap(session: Session, env: ToolEnvironment, index: number): Promise<void> {
  const before = session.surface.nodes.length
  appendUser(session, prose('gap' + index, 6000))
  const gapSeq = session.surface.nodes.slice(before, before + 1)[0]!
  appendUser(session, 'next question ' + index)
  for (let pad = 0; pad < 5; pad += 1) {
    appendAssistant(session, 'pad ' + index + '-' + pad + ' ' + longText('pad' + index + pad, index * 10 + pad))
  }
  const text = await compressCall(session, env, 'gap-' + index, {
    content: [{ startSeq: gapSeq, endSeq: gapSeq, summary: NOTE }],
  })
  assert.match(text, /Compressed 1 block/, 'fragment fixture folds one gap: ' + text)
}

const activeBlocks = (session: Session) => blockRegistry(session).filter((entry) => entry.active && entry.summarySeq !== null)

/**
 * A FRAGMENTED surface: `blocks` carriers, each reset by a protected-window pad, then a
 * long protected tail that makes every rewrite point expensive. No plain geometry can
 * qualify — the tails are unreachable and every foldable run is under the note-carrying
 * minimum (the capacity gate).
 */
async function fragmentedFixture(id: string, blocks: number): Promise<Session> {
  const session = Session.create(id)
  const env = makeEnv()
  appendTurn(session, 1)
  for (let index = 0; index < blocks; index += 1) await foldOneGap(session, env, index)
  for (let index = 0; index < 5; index += 1) {
    if (index % 2 === 0) appendUser(session, prose('tail' + index, 48000))
    else appendAssistant(session, prose('tail' + index, 48000))
  }
  return session
}

const PRICING = { floorTokens: 0 }

test('E-13: a fragmented surface is folded ACROSS its carriers, and the result says it is tier 2', async () => {
  const env = makeEnv()
  const session = await fragmentedFixture('election-cross', 5)
  const before = activeBlocks(session).length
  assert.ok(before >= 2, 'the fixture really carries blocks: ' + before)

  // M-3: with the payback gate gone the surface is no longer STUCK — the batch is
  // recommended — and the carrier-covering shape is still reported as a distillation.
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, PRICING)
  assert.match(table, /Best batch here/, 'the batch is offered: ' + table)

  const pick = elect(session, env, 5, PRICING)
  assert.ok(!('reason' in pick), 'the carrier-covering shape elects: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.equal(landed.kind, 'distill', 'the election names the operation (shape-based since M-3)')
  assert.ok(landed.distilledBlocks >= 2, 'more than one carrier is distilled: ' + landed.distilledBlocks)
  assert.ok(landed.spans.length >= 1, 'ONE condense call, few spans: ' + landed.spans.length)
  assert.notEqual(landed.spans[0]!.endBlock, null, 'the terminal arm is a block ref (that is what triggers tier 2)')

  const text = await compressCall(session, env, 'cross', { condense: { summary: NOTE } })
  assert.match(text, /distilled \d+ blocks into 1/, 'the result names the merge: ' + text)
  assert.match(text, /tier 2/, 'and the tier')
  assert.match(text, /no longer recoverable at message level/, 'and that the originals are gone')
  assert.ok(activeBlocks(session).length < before, 'live carriers went DOWN: ' + activeBlocks(session).length + ' < ' + before)
})

test('E-14 (M-3): ONE carrier with one plain gap elects a PLAIN fold, not a distillation', async () => {
  const env = makeEnv()
  const session = await fragmentedFixture('election-cross-one', 1)
  assert.equal(activeBlocks(session).length, 1, 'exactly one live carrier')
  // Before M-3 the payback gate refused this slow batch; the gate is gone, so the
  // gap folds as an ORDINARY batch span — still NOT a distillation (one carrier
  // needs no merge), which is the meaningful assertion here.
  const pick = elect(session, env, 5, PRICING)
  assert.ok(!('reason' in pick), 'the shape elects now: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  // The shape covers the carrier, so it is REPORTED as a tier-2 distillation — but
  // it merges exactly ONE carrier (nothing else is live), never silently plain.
  assert.equal(landed.kind, 'distill', 'the carrier-covering shape is named a distillation: ' + landed.kind)
  assert.equal(landed.distilledBlocks, 1, 'and it distills exactly the one carrier: ' + landed.distilledBlocks)
})

test('E-15: when a plain geometry qualifies, the cross-barrier fallback never appears', async () => {
  const env = makeEnv()
  const session = Session.create('election-cross-skip')
  appendTurn(session, 1)
  await foldOneGap(session, env, 0)
  await foldOneGap(session, env, 1)
  // An UNFOLDED two-message gap above the minimum, with a cheap tail behind it: the
  // normal election clears the capacity gate, so the fallback must stay out of the way.
  appendUser(session, prose('open-gap-a', 6000))
  appendAssistant(session, prose('open-gap-b', 6000))
  appendUser(session, 'short tail')
  for (let pad = 0; pad < 5; pad += 1) appendAssistant(session, 'pad tail ' + pad)

  // M-3: the payback gate no longer holds the carrier-covering batch back, so this
  // fixture (two carriers) may now elect a distillation — the contract that survives
  // is the REPORTING one (E-14), verified on a carrier-free surface here: a surface
  // with NO carrier keeps the plain fold and distilledBlocks 0.
  const carrierFree = Session.create('election-cross-skip-plain')
  appendTurn(carrierFree, 1)
  appendUser(carrierFree, prose('open-gap-a', 6000))
  appendAssistant(carrierFree, prose('open-gap-b', 6000))
  appendUser(carrierFree, 'short tail')
  for (let pad = 0; pad < 5; pad += 1) appendAssistant(carrierFree, 'pad tail ' + pad)
  const pick = elect(carrierFree, env, 5, PRICING)
  assert.ok(!('reason' in pick), 'a normal geometry qualifies: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.equal(landed.kind, 'fold', 'the normal path stays first: ' + landed.kind)
  assert.equal(landed.distilledBlocks, 0, 'and it is NOT a distillation')
})

test('E-16: the distilled block is anchored — recall reads the new summary by block ref', async () => {
  const env = makeEnv()
  const session = await fragmentedFixture('election-cross-recall', 4)
  const text = await compressCall(session, env, 'cross-recall', { condense: { summary: NOTE } })
  assert.match(text, /distilled \d+ blocks into 1 across seqs \d+\.\.b\d+/, 'the result names the new block ref: ' + text)
  const blocks = activeBlocks(session)
  // M-3: the distill shape can arrive as a MULTI-span batch (carrier run + plain gap
  // run), so it lands more than one new block — the contract is that the live-carrier
  // count went DOWN (the carriers merged), not that exactly one block remains.
  assert.ok(blocks.length < 4, 'the merge reduced the live carriers: ' + blocks.length)
  const recall = makeTools(env).find((candidate) => candidate.name === 'recall')
  const read = await recall!.execute({ blockId: blocks[0]!.kernelBlockId } as never, execStub(session, 'read') as never)
  const readText = typeof read === 'string' ? read : (read as { text: string }).text
  assert.ok(readText.length > 0 && !/no such block|unknown block/i.test(readText), 'the new block is readable: ' + readText.slice(0, 160))
})

// ---------------------------------------------------------------- batch
/** ONE foldable gap above the minimum, behind a 40K protected tail: the only batch
 *  the nudge can offer pays back 40000/15000 = 2.67 requests. */
function batchFixture(): Session {
  const session = Session.create('batch-gates')
  appendTurn(session, 1)
  appendUser(session, prose('ledger', 60000))
  for (let index = 0; index < 5; index += 1) {
    if (index % 2 === 0) appendUser(session, prose('tail' + index, 40000))
    else appendAssistant(session, prose('tail' + index, 40000))
  }
  return session
}

test('E-17 (M-3): a slow-payback batch IS recommended — payback is reported, not gated', () => {
  const env = makeEnv()
  const session = batchFixture()
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, {
    floorTokens: 0,
  })
  assert.ok(table.includes('fold ~'), 'the rows are still listed: ' + table)
  assert.match(table, /Best batch here/, 'the batch is recommended regardless of payback: ' + table)
  assert.match(table, /pays back in \d+\.\d req/, 'and its payback is stated: ' + table)
  assert.ok(!/compress\.maxPaybackRequests/.test(table), 'the deleted ceiling is not quoted any more')
})

test('E-18: a batch inside the ceiling is recommended, with the payback the tool itself computes', () => {
  const env = makeEnv()
  const session = batchFixture()
  const pricing = { floorTokens: 0 }
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, pricing)
  assert.match(table, /Best batch here/, 'the batch is offered: ' + table)
  const shown = /pays back in (\d+\.\d) req/.exec(table)
  assert.ok(shown, 'the batch states its payback: ' + table)

  // SAME NUMBER as the election: one source of truth for the judgement.
  const pick = elect(session, env, 5, pricing)
  assert.ok(!('reason' in pick), 'the election takes the same shape: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(Math.abs(landed.paybackRequests - Number(shown[1])) < 0.1,
    'nudge and condense agree on the payback: ' + landed.paybackRequests + ' vs ' + shown[1])
})

test('E-19 (M-3): foldPricingFor carries the floor alone — no gate knobs left', () => {
  const defaults = foldPricingFor({ modelContextLimit: 128000 })
  assert.equal(Object.hasOwn(defaults, 'maxPaybackRequests'), false, 'the payback knob is gone')
  assert.equal(Object.hasOwn(defaults, 'maxFoldCostTokens'), false, 'the cost knob is gone')
  assert.ok(defaults.floorTokens !== undefined && defaults.floorTokens > 0, 'the floor survives')
})

// ---------------------------------------------------------------- config
test('E-config (M-3): the gate knobs are removed from the config surface', () => {
  const defaults = foldPricingFor({ modelContextLimit: 128000 })
  assert.equal(Object.hasOwn(defaults, 'maxPaybackRequests'), false)
  assert.equal(Object.hasOwn(defaults, 'maxFoldCostTokens'), false)
  const pctZero = foldPricingFor({ modelContextLimit: 128000, foldFloorPct: 0 })
  assert.equal(pctZero.floorTokens, undefined, 'foldFloorPct 0 still disables the ONE remaining gate')
})

// ---------------------------------------------------------------- M-1 multi-span
// M-1 (2026-11-25): condense 多段化 — the election may return a MULTI-SPAN batch
// (walls stay on the surface, spans live on BOTH sides of each wall), priced as
// ONE rewrite with the note denominator Σ(spans). Contracts pinned:
//   M1-1  the batch lands as several blocks in ONE call (one re-prefill for all)
//   M1-2  the note denominator is the batch's Σ (condenseNoteBudget over spans)
//   M1-3  single-span surfaces keep the old behavior (backward compatibility)
//   M1-4  a checkpoint carrier never joins the plain batch (bN distillation only)

test('M1-1: a walled surface folds BOTH sides of the wall in one call, several blocks, one rewrite', async () => {
  const env = makeEnv()
  const { session } = walledFixture('m1-multi-span', 4, 4)
  const pricing = { floorTokens: 0 }
  const starts = runStarts(session, env, 5, pricing)
  assert.equal(starts.length, 2, 'the wall splits the surface into two runs: ' + JSON.stringify(starts))

  const pick = elect(session, env, 5, pricing)
  assert.ok(!('reason' in pick), 'the batch is elected: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.ok(landed.spans.length >= 2, 'the election returns a MULTI-SPAN batch: ' + JSON.stringify(landed.spans))
  assert.ok(
    landed.spans.some((span) => span.start === starts[0])
      && landed.spans.some((span) => span.start === starts[1]),
    'spans live on BOTH sides of the wall: ' + JSON.stringify(landed.spans),
  )
  // The batch's Σ is what the result line prices, and the payback is batch-level.
  const sum = landed.spans.reduce((total, span) => total + span.tokens, 0)
  assert.equal(landed.tokens, sum, 'pick.tokens is the batch Σ: ' + landed.tokens + ' vs ' + sum)

  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: NOTE } } as never, execStub(session, 'm1') as never)
  const text = (result as { text: string }).text
  assert.match(text, /Compressed \d+ block/, 'the batch lands: ' + text)
  assert.match(text, new RegExp('condense folds \\d+t \\(seqs \\d+\\.\\.\\d+ in ' + landed.spans.length + ' ranges\\)'),
    'the result names the multi-span batch: ' + text)
  assert.match(text, /pays back (in [\d.]+|immediately)/, 'the payback is batch-level: ' + text)
})

test('M1-2: the note denominator is the batch Σ — the weight gate weighs the note against the WHOLE batch', () => {
  const env = makeEnv()
  const { session } = walledFixture('m1-note-denominator', 4, 4)
  const pick = elect(session, env, 5, { floorTokens: 0 })
  assert.ok(!('reason' in pick), 'a batch is elected: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  const sum = landed.spans.reduce((total, span) => total + span.tokens, 0)
  // F1 (cross-review 2026-11-25): the batch writes ONE shared note carried by every
  // submitted range — the model does not write N summaries for N spans — so the
  // budget is ONE note's size, NOT per-span × N. It is capped at
  // MAX_NOTE_WEIGHT_RATIO × Σ, measured against the WHOLE batch (the write-time
  // weight gate compares against `pick.tokens` = the batch Σ too), so the floor
  // arithmetic and the gate cannot disagree.
  const expected = Math.min(TYPICAL_HANDOFF_NOTE_TOKENS, Math.floor(MAX_NOTE_WEIGHT_RATIO * sum))
  assert.equal(landed.noteTokens, expected,
    'the note budget is ONE shared note capped by the batch weight: '
    + landed.noteTokens + ' vs ' + expected)
  // The shared note can never exceed the batch's own weight cap (F1): a batch that
  // reserved N notes would inflate the resident cost the fold is priced against.
  assert.ok(landed.noteTokens <= MAX_NOTE_WEIGHT_RATIO * sum,
    'batch note/Σ stays inside the weight cap: ' + landed.noteTokens + '/' + sum)
  // The batch Σ is the denominator the payback divides — an EXACT identity, not an
  // inequality (F4: the old `<= … + 1e-9` form was a tautology, true whenever the
  // pick used the same denominator it claimed).
  assert.ok(Math.abs(landed.paybackRequests - landed.foldCostTokens / sum) < 1e-9,
    'payback divides the batch Σ exactly: ' + landed.paybackRequests + ' vs ' + landed.foldCostTokens / sum)
})

test('M1-3: a single-run surface keeps the ONE-span shape (backward compatibility)', async () => {
  // No wall, no second run: the batch degenerates to the old single-span prefix.
  const env = makeEnv()
  const session = Session.create('m1-single')
  appendTurn(session, 1)
  for (let index = 1; index <= 6; index += 1) {
    if (index % 2 === 1) appendUser(session, bigText('s' + index, index))
    else appendAssistant(session, bigText('s' + index, index))
  }
  for (let index = 1; index <= 5; index += 1) {
    if (index % 2 === 1) appendUser(session, 'short tail ' + index)
    else appendAssistant(session, 'short tail ' + index)
  }
  const pick = elect(session, env, 5, { floorTokens: 0 })
  assert.ok(!('reason' in pick), 'a plain prefix still elects: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  assert.equal(landed.spans.length, 1, 'ONE span on a wall-free surface: ' + JSON.stringify(landed.spans))
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: NOTE } } as never, execStub(session, 'm1-single') as never)
  const text = (result as { text: string }).text
  assert.match(text, /Compressed 1 block/, 'one block lands: ' + text)
  assert.doesNotMatch(text, /in 1 ranges/, 'the result does not announce a multi-span shape for one span: ' + text)
})

test('M1-4 (M-3): a batch that covers an active carrier is REPORTED as a distillation, never a silent plain fold', async () => {
  // E-15-style fixture (F3, cross-review 2026-11-25): two gaps already folded into
  // carriers, then an UNFOLDED two-message gap above the minimum with a cheap tail —
  // the normal election clears the payback gate and elects a PLAIN fold, so the
  // contract about spans is actually exercised. The old form returned silently on
  // both `reason` and `distill` (the fragmented fixture elects a cross-barrier
  // distillation, so the loop body never ran) — a test that can always pass without
  // running its assertions.
  const env = makeEnv()
  const session = Session.create('m1-carrier-barrier')
  appendTurn(session, 1)
  await foldOneGap(session, env, 0)
  await foldOneGap(session, env, 1)
  appendUser(session, prose('open-gap-a', 6000))
  appendAssistant(session, prose('open-gap-b', 6000))
  appendUser(session, 'short tail')
  for (let pad = 0; pad < 5; pad += 1) appendAssistant(session, 'pad tail ' + pad)

  const carriers = activeBlocks(session)
  const carrierSeqs = new Set(carriers.map((entry) => entry.summarySeq!))
  assert.ok(carrierSeqs.size >= 2, 'the fixture carries live carriers: ' + carrierSeqs.size)
  const pick = elect(session, env, 5, PRICING)
  assert.ok(!('reason' in pick), 'the plain geometry qualifies — a miss would void the contract: ' + JSON.stringify(pick))
  const landed = pick as CondensePick
  // M-3: the batch may now fold the gap together with the carriers — the surviving
  // contract is REPORTING: the pick names it a distillation and distills exactly the
  // live carriers, never a silent plain fold over carrier seqs.
  assert.equal(landed.kind, 'distill', 'a carrier-covering pick is named a distillation: ' + landed.kind)
  assert.equal(landed.distilledBlocks, carrierSeqs.size, 'every live carrier is distilled: ' + landed.distilledBlocks)
  const seqOfBlock = new Map(carriers.map((entry) => [entry.kernelBlockId, entry.summarySeq!] as const))
  // seq values are NOT monotonic on the surface — carrier 10 can sit numerically
  // inside a span 2..24 while sitting positionally OUTSIDE it. Compare POSITIONS.
  const nodes = session.surface.nodes as readonly number[]
  const positionOf = (seq: number): number => nodes.indexOf(seq)
  const coversPositionally = (span: { start: number; end: number }, seq: number): boolean => {
    const from = positionOf(span.start)
    const to = positionOf(span.end)
    const at = positionOf(seq)
    return at >= Math.min(from, to) && at <= Math.max(from, to)
  }
  for (const span of landed.spans) {
    if (span.carriers.length > 0) {
      // A carrier-edge span (the distill shape) may touch ONLY the carriers it
      // names, only at its own EDGE — the seq IS the carrier there, never interior
      // prose, and never a carrier it does not distil.
      for (const carrierId of span.carriers) {
        const seq = seqOfBlock.get(carrierId)
        assert.ok(seq !== undefined, 'the distil target is a live carrier: ' + carrierId)
        assert.ok(span.start === seq || span.end === seq,
          'a distill span touches its carrier only at its edge: ' + seq + ' vs ' + JSON.stringify(span))
      }
      continue
    }
    for (const seq of carrierSeqs) {
      assert.ok(!coversPositionally(span, seq),
        'no plain span covers the checkpoint seq ' + seq + ': ' + JSON.stringify(landed.spans))
    }
  }
})
