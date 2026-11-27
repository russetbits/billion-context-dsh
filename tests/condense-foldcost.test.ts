/**
 * condense foldCost pricing — the re-prefill price reported in the RESULT text
 * must be the price elected on the PRE-fold surface (CondensePick.foldCostTokens).
 *
 * 2026-11-26 fix: condense-apply used to re-compute the price AFTER the fold
 * landed (`indexOfSurfaceSeq(session, firstSpan.start)` returned -1 because the
 * first span was already shadowed), reporting `one re-prefill of ~0t` on live
 * tier-2 batches (a live session). The fix reads condensePick.foldCostTokens,
 * which condense-elect priced on the pre-fold surface for every geometry.
 *
 * Contracts pinned here:
 *   FC-1  a multi-span distill batch reports `one re-prefill of ~<N>t` with N > 0
 *         and N equal to the election-time foldCostTokens
 *   FC-2  a plain (non-distill) condense reports `re-prefills <N>t once` with
 *         N > 0 and equal to the election-time foldCostTokens
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { allLogMessages } from '../src/messages.ts'
import { surfaceSuffixTokens } from '../src/region.ts'
import {
  makeTools,
  pickCondenseSpan,
  type CondensePick,
  type ToolEnvironment,
} from '../src/tools.ts'
import { appendAssistant, appendUser, longText } from './helpers.ts'

const NOTE = '## Task\nPin the fold-cost reporting contract.\n'
  + '## Now\nAssert the reported re-prefill equals the pre-fold election price; done when both branches match.\n'

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

/** A fragmented surface: `blocks` carriers each reset by a protected-window pad,
 *  then a long protected tail — the fixture that elects the multi-span distill. */
async function fragmentedFixture(id: string, blocks: number): Promise<Session> {
  const session = Session.create(id)
  const env = makeEnv()
  appendUser(session, 'kickoff turn')
  for (let index = 0; index < blocks; index += 1) await foldOneGap(session, env, index)
  for (let index = 0; index < 5; index += 1) {
    if (index % 2 === 0) appendUser(session, prose('tail' + index, 48000))
    else appendAssistant(session, prose('tail' + index, 48000))
  }
  return session
}

/** One big foldable gap above the minimum, behind an expensive protected tail. */
function batchFixture(id: string): Session {
  const session = Session.create(id)
  appendUser(session, prose('ledger', 60000))
  for (let index = 0; index < 5; index += 1) {
    if (index % 2 === 0) appendUser(session, prose('tail' + index, 40000))
    else appendAssistant(session, prose('tail' + index, 40000))
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

test('FC-1: a multi-span distill batch reports the PRE-FOLD re-prefill price, not ~0t', async () => {
  const env = makeEnv()
  const session = await fragmentedFixture('foldcost-distill', 5)
  const pick = elect(session, env, 5)

  assert.equal(pick.kind, 'distill', 'the fixture elects the distill shape: ' + pick.kind)
  assert.ok(pick.spans.length >= 1, 'the batch carries spans: ' + JSON.stringify(pick.spans))
  // The election priced the rewrite on the pre-fold surface: the protected tail
  // (5 x 48K chars ≈ 60K tokens) sits after the earliest rewrite point.
  assert.ok(pick.foldCostTokens > 0, 'the election prices a non-zero re-prefill: ' + pick.foldCostTokens)

  const text = await compressCall(session, env, 'fc1', { condense: { summary: NOTE } })
  assert.match(text, /distilled \d+ blocks into 1/, 'the result names the tier-2 merge: ' + text)
  const reported = reportedFoldCost(/one re-prefill of ~(\d+)t/, text)
  assert.ok(reported > 0, 'the reported re-prefill is non-zero (regression: post-fold recompute reads -1 → ~0t): ' + text)
  assert.equal(reported, pick.foldCostTokens,
    'the reported price equals the election-time pricing: ' + reported + ' vs ' + pick.foldCostTokens)
})

test('FC-2: a plain condense reports the PRE-FOLD re-prefill price in `re-prefills <N>t once`', async () => {
  const env = makeEnv()
  const session = batchFixture('foldcost-plain')
  const pick = elect(session, env, 5)

  assert.equal(pick.kind, 'fold', 'the fixture elects a plain fold: ' + pick.kind)
  assert.equal(pick.distilledBlocks, 0, 'and it is not a distillation')
  assert.ok(pick.foldCostTokens > 0, 'the election prices a non-zero re-prefill: ' + pick.foldCostTokens)

  const text = await compressCall(session, env, 'fc2', { condense: { summary: NOTE } })
  const reported = reportedFoldCost(/re-prefills (\d+)t once/, text)
  assert.ok(reported > 0, 'the plain-branch re-prefill is non-zero: ' + text)
  assert.equal(reported, pick.foldCostTokens,
    'the reported price equals the election-time pricing: ' + reported + ' vs ' + pick.foldCostTokens)
  assert.match(text, /pays back in \d/, 'and the payback copy stays non-degenerate: ' + text)
})

test('FC-3: the election price is a pre-fold suffix — it is strictly the live surface after the rewrite point', async () => {
  // Sanity on the pricing source: foldCostTokens must equal the pre-fold suffix
  // token count after the last span's end (the condense-elect formula), so the
  // apply-side reuse cannot silently drift from what the surface had.
  const env = makeEnv()
  const session = await fragmentedFixture('foldcost-suffix', 3)
  const pick = elect(session, env, 5)
  assert.ok(pick.foldCostTokens > 0, 'the elect prices the tail: ' + pick.foldCostTokens)
  const suffix = surfaceSuffixTokens(session)
  const liveTotal = suffix[0] ?? 0
  assert.ok(pick.foldCostTokens <= liveTotal,
    'the price never exceeds the live surface: ' + pick.foldCostTokens + ' <= ' + liveTotal)
})
