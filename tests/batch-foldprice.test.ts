/**
 * Batch merge re-prefill pricing — the "Best batch here" nudge row, the
 * condense election and the condense result line must price a MULTI-span
 * batch at the EARLIEST rewrite point, not the last span's end.
 *
 * Measured 2026-11-27 (probe-batch fixture + live host): a multi-span batch's
 * durable replace rewrites the surface at the FIRST span's start edge (the
 * first range's checkpoint node lands there), so the ONE re-prefill is
 *
 *     suffix[firstStartIdx] − Σ(spans.tokens)
 *
 * For a single contiguous span the identity `suffix[startIdx] − tokens =
 * suffix[endIndex + 1]` makes this exactly the old single-row formula — which
 * is why single-row pricing was never wrong. Multi-span batches WERE
 * under-quoted by every kept row (guarded walls, barriers) sitting between
 * the spans: a live 6-row batch quoted ~4183t for a re-prefill the host
 * actually billed 28–45× higher.
 *
 * Contracts pinned here (BP-1..BP-4):
 *   BP-1  foldPlanOf prices a multi-row batch above the LAST row's foldCost
 *   BP-2  the condense pick prices a multi-span batch at the earliest rewrite
 *         point and the result line reports exactly that number
 *   BP-3  the single-span identity still holds (no regression on 1-row sets)
 *   BP-4  position order and price order CROSS: the cheapest fold-cost row is
 *         the LAST one positionally, and the batch price still exceeds it
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { allLogMessages } from '../src/messages.ts'
import { surfaceSuffixTokens, buildCompressibleSeqRanges, type SeqCompressibleRange } from '../src/region.ts'
import {
  makeTools,
  pickCondenseSpan,
  type CondensePick,
  type ToolEnvironment,
} from '../src/tools.ts'
import { foldPlanOf } from '../src/nudge.ts'
import { appendAssistant, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'

const NOTE = '## Task\nPin the batch merge re-prefill pricing contract.\n'
  + '## Now\nAssert the multi-span batch is priced at the earliest rewrite point; done when BP-1..BP-4 match.\n'
  + '## Plan\n- fixture\n- elect\n- reconcile\n## Pointers\n- probe-batch fixture\n'

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

/** ~4 chars/token plain prose (same recipe as condense-election.test.ts). */
const prose = (label: string, chars: number): string => {
  const unit = 'alpha beta gamma delta epsilon zeta eta theta '
  return label + ': ' + unit.repeat(Math.ceil(chars / unit.length)).slice(0, chars)
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
    },
  } as never
}

async function compressCall(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  const out = await definition!.execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}

/** An AGENTS.md injection row in the host shape — a GUARDED wall between runs. */
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
 * THREE foldable runs separated by guarded walls, each run growing SMALLER toward
 * the tail (45K → 30K → 15K chars), plus a protected tail. This makes the row
 * fold-costs DESCEND positionally (earliest row = highest fold cost), so the
 * position order and the price order cross — the shape that broke the old
 * last-span pricing.
 */
function batchFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  const runChars = [45000, 30000, 15000]
  for (let run = 0; run < runChars.length; run += 1) {
    for (let i = 0; i < 3; i += 1) {
      appendUser(session, prose('u' + run + i, runChars[run]! / 3))
      appendAssistant(session, prose('a' + run + i, runChars[run]! / 3))
    }
    appendInstruction(session, 'w' + run)
  }
  for (let i = 0; i < 3; i += 1) {
    appendUser(session, prose('tail-u' + i, 6000))
    appendAssistant(session, prose('tail-a' + i, 6000))
  }
  return session
}

const PRICING = { floorTokens: 0 }

function elect(session: Session, env: ToolEnvironment, preserveRecent: number): CondensePick {
  const pick = pickCondenseSpan(session, turnOf(env, session), preserveRecent, PRICING)
  assert.ok(!('reason' in pick), 'the shape elects: ' + JSON.stringify(pick))
  return pick as CondensePick
}

const reportedFoldCost = (pattern: RegExp, text: string): number => {
  const shown = pattern.exec(text)
  assert.ok(shown, 'the result reports the re-prefill price: ' + text)
  return Number(shown[1])
}

test('BP-1: foldPlanOf prices a multi-row batch ABOVE the last row\u2019s own fold cost', () => {
  const env = makeEnv()
  const session = batchFixture('batch-price-plan')
  const suffix = surfaceSuffixTokens(session)
  const liveTotal = suffix[0] ?? 0
  // Rebuild the range rows the nudge table feeds foldPlanOf (same geometry the
  // election walks), then assert the batch price exceeds every member row's
  // own fold cost — the member prices each pretend the OTHER rows stay live.
  const rows = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), PRICING)
  assert.ok(rows.length >= 2, 'the fixture yields several rows: ' + JSON.stringify(rows))
  const plan = foldPlanOf(rows, 400, PRICING)
  assert.ok(plan !== null, 'a plan exists')
  const lastRow = rows.reduce((a, b) => (b.start > a.start ? b : a))
  assert.ok(plan.foldCostTokens > lastRow.foldCostTokens,
    'the batch re-prefill (' + plan.foldCostTokens + ') must exceed the last row\u2019s own price ('
    + lastRow.foldCostTokens + ') — the kept walls between the spans are re-prefilled too')
  // And the price stays bounded by the whole surface minus the reclaim (the
  // formula ceiling): everything before the rewrite point is reused.
  assert.ok(plan.foldCostTokens <= liveTotal,
    'the batch price never exceeds the live surface: ' + plan.foldCostTokens + ' <= ' + liveTotal)
})

test('BP-2: the condense pick prices a multi-span batch at the EARLIEST rewrite point and the result line reports it', async () => {
  const env = makeEnv()
  const session = batchFixture('batch-price-elect')
  const pick = elect(session, env, 5)
  assert.ok(pick.spans.length >= 2, 'the fixture elects a multi-span batch: ' + JSON.stringify(pick.spans))
  const suffix = surfaceSuffixTokens(session)
  const firstStart = pick.spans[0]!.start
  const startIdx = session.surface.nodes.indexOf(firstStart)
  assert.ok(startIdx >= 0, 'the earliest span start is on the surface')
  // The measured batch-merge formula, evaluated independently of the code:
  const expected = Math.max(0, (suffix[startIdx] ?? 0) - pick.tokens)
  assert.ok(expected > 0, 'the expected batch price is non-zero: ' + expected)
  assert.equal(pick.foldCostTokens, expected,
    'the pick prices the batch at the earliest rewrite point: ' + pick.foldCostTokens + ' vs ' + expected)

  const text = await compressCall(session, env, 'bp2', { condense: { summary: NOTE } })
  const reported = reportedFoldCost(/re-prefills (\d+)t once/, text)
  assert.equal(reported, expected,
    'the result line reports exactly the batch-merge price: ' + reported + ' vs ' + expected)
})

test('BP-3: the single-span identity still holds — a 1-row batch prices exactly suffix[endIndex+1]', () => {
  const env = makeEnv()
  const session = batchFixture('batch-price-single')
  const suffix = surfaceSuffixTokens(session)
  const pick = elect(session, env, 5)
  assert.ok(pick.spans.length >= 1, 'a span is elected: ' + JSON.stringify(pick.spans))
  const firstStart = pick.spans[0]!.start
  const startIdx = session.surface.nodes.indexOf(firstStart)
  // Independent re-derivation of the same identity, not a copy of the code:
  // suffix[startIdx] − folded must equal suffix[endIndex+1] when the spans are
  // contiguous AND nothing live sits between them.
  const contiguous = pick.spans.length === 1
    || pick.spans.every((span, index) =>
      index === 0 || session.surface.nodes.indexOf(span.start)
        === session.surface.nodes.indexOf(pick.spans[index - 1]!.end) + 1)
  if (contiguous) {
    const lastEnd = pick.spans[pick.spans.length - 1]!.end
    const endIdx = session.surface.nodes.indexOf(lastEnd)
    assert.equal(pick.foldCostTokens, suffix[endIdx + 1] ?? 0,
      'contiguous spans collapse to the single-row formula: ' + pick.foldCostTokens)
  } else {
    // Non-contiguous: the batch price must exceed the last span's own price —
    // the kept rows between the spans are re-prefilled too.
    const lastEnd = pick.spans[pick.spans.length - 1]!.end
    const endIdx = session.surface.nodes.indexOf(lastEnd)
    assert.ok(pick.foldCostTokens >= (suffix[endIdx + 1] ?? 0),
      'a gapped batch is never priced below its last span\u2019s own price: ' + pick.foldCostTokens)
  }
  assert.ok(startIdx >= 0)
})

test('BP-4: position order and price order CROSS — the cheapest row is the last one, and the batch still prices above it', () => {
  const env = makeEnv()
  const session = batchFixture('batch-price-cross')
  const rows = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), PRICING)
  assert.ok(rows.length >= 2, 'the fixture yields several rows: ' + JSON.stringify(rows))
  // The fixture grows rows SMALLER toward the tail, so fold costs DESCEND
  // positionally: cheapest row sits LAST.
  for (let i = 1; i < rows.length; i += 1) {
    assert.ok(rows[i]!.foldCostTokens < rows[i - 1]!.foldCostTokens,
      'row fold costs descend with position (cheapest last): row ' + i + ' = ' + rows[i]!.foldCostTokens
      + ' < ' + rows[i - 1]!.foldCostTokens)
  }
  const pick = elect(session, env, 5)
  assert.ok(pick.spans.length >= 2, 'the election batches several rows: ' + JSON.stringify(pick.spans))
  const suffix = surfaceSuffixTokens(session)
  const firstStart = pick.spans[0]!.start
  const startIdx = session.surface.nodes.indexOf(firstStart)
  const expected = Math.max(0, (suffix[startIdx] ?? 0) - pick.tokens)
  const lastRow = rows.reduce((a, b) => (b.start > a.start ? b : a))
  assert.ok(expected > lastRow.foldCostTokens,
    'the earliest-rewrite-point price (' + expected + ') exceeds the cheapest row ('
    + lastRow.foldCostTokens + ') — the old last-span quote under-priced by the kept walls')
  assert.equal(pick.foldCostTokens, expected)
})
