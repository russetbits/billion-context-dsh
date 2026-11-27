/**
 * 方案 D — the no-action emergency degrade (compaction-basic passive fallback
 * replacement). Three layers, pinned here:
 *   1. the emergency nudge carries an ACTION INSTRUCTION rider ("compress NOW,
 *      this step"), not just advice;
 *   2. K consecutive emergency-nudged turns without a successful compress fold
 *      the largest eligible span ONCE per session, EMERGENCY PROVENANCE
 *      stamped, ledger row written, no overflow retry spent;
 *   3. a successful compress resets the counter; emergencyFoldAfterTurns: 0
 *      never degrades.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import { createCore, type CompressionCore, type NudgeDecision } from 'acp-kernel'
import AcpCompactionEngine, { resolveAcpConfig } from '../src/index.ts'
import { buildNudge, emergencyActionRider } from '../src/nudge.ts'
import { rebuildBlockLedger } from '../src/region.ts'
import { eventsToCoreMessages, surfaceEventsOf, SUMMARY_FRAME_PREFIX } from '../src/messages.ts'
import { sessionEventsOf } from '../src/session-events.ts'
import { AcpStateStore } from '../src/state.ts'
import { buildTextSession } from './helpers.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { NudgeGate } from '../src/nudge.ts'

function fakeAgent(session: Session, ctx: Context): Agent {
  return {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx,
  } as unknown as Agent
}

interface LogRow { type: string; message: string }

async function mountEngine(
  root: Context,
  config: Record<string, unknown>,
): Promise<{ fiber: { dispose: () => Promise<void> }; engine: AcpCompactionEngine }> {
  let engine: AcpCompactionEngine | undefined
  const fiber = root.plugin((ctx) => {
    engine = new AcpCompactionEngine(ctx, config as never)
  })
  await fiber
  if (engine === undefined) throw new Error('engine did not mount')
  return { fiber, engine }
}

/** Drive one pre-step through the same bus the host uses. */
function preStep(root: Context, agent: Agent): Promise<{ kind: string; messages?: unknown[] } | undefined> {
  return root.waterfall(
    'agent/pre-step' as never,
    { agent, turn: 1, step: 1 } as never,
    async () => ({ kind: 'enter', messages: [] }) as never,
  ) as Promise<{ kind: string; messages?: unknown[] } | undefined>
}

function compactionSummaryCount(session: Session): number {
  return sessionEventsOf(session).filter((event) => event.type === 'compaction/summary').length
}

// ── layer 1: the hard nudge rider ────────────────────────────────────────────

const NUDGE_ENV_BASE = {
  kernel: createCore({}) as CompressionCore,
  store: new AcpStateStore(),
  // Tests here exercise rider copy and thresholds, not fold pricing: an 8K
  // window would clamp the working-set floor to its minimum and starve the
  // range table the emergency example needs.
  foldFloorPct: 0,
}

test('D: the emergency nudge carries the ACTION REQUIRED rider (snapshot assertion)', () => {
  // 12 messages ≈ 12.4K of a 13000 window ≈ 95% — above the 85% emergency line.
  const env = { ...NUDGE_ENV_BASE, modelContextLimit: 13000 }
  const session = buildTextSession(12)
  const outcome = buildNudge(fakeAgent(session, new Context()), env, new Map<string, NudgeGate>(), new Map())
  assert.ok(outcome !== null, 'a nudge is produced at 90% pressure')
  assert.equal(outcome!.emergency, true, 'the pressure crosses the emergency threshold')
  const text = outcome!.message.content.map((block) => (block as { text?: string }).text ?? '').join('')
  // The rider must be an action instruction, not advice:
  assert.match(text, /ACTION REQUIRED THIS STEP/, 'the rider names the action deadline')
  assert.match(text, /compress\(\{ condense: \{ summary \} \}\) NOW/, 'the rider names the exact call to make')
  assert.match(text, /before doing anything else in this turn/, 'the rider forbids deferring the action')
})

test('D: the non-emergency nudge does NOT carry the action rider', () => {
  // 12 messages ≈ 12.4K of a 15000 window ≈ 83% — hard tier, below emergency.
  const env = { ...NUDGE_ENV_BASE, modelContextLimit: 15000 }
  const session = buildTextSession(12)
  const outcome = buildNudge(fakeAgent(session, new Context()), env, new Map<string, NudgeGate>(), new Map())
  assert.ok(outcome !== null, 'a nudge is produced at 83% pressure')
  assert.equal(outcome!.emergency, false)
  const text = outcome!.message.content.map((block) => (block as { text?: string }).text ?? '').join('')
  assert.doesNotMatch(text, /ACTION REQUIRED THIS STEP/, 'advisory tiers stay advisory')
})

test('D: the PTC rider names the run_code call shape', () => {
  assert.match(emergencyActionRider('ptc'), /await tools\.compress\(\{ condense: \{ summary \} \}\) from inside `run_code`/)
  assert.doesNotMatch(emergencyActionRider('native'), /await tools\./)
})

// ── layer 2/3: the no-action degrade fold ────────────────────────────────────

const DEGRADE_LIMIT = 13000 // 12.4K surface ≈ 95% of this window — every pre-step lands an emergency nudge

test('D: after K emergency nudges without a compress, the engine folds once (ledger + EMERGENCY PROVENANCE)', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber, engine } = await mountEngine(root, { modelContextLimit: DEGRADE_LIMIT, emergencyFoldAfterTurns: 3 })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const session = buildTextSession(30)
    const agent = fakeAgent(session, root)
    const surfaceBefore = session.surface.nodes.length

    await preStep(root, agent) // reminder 1
    await preStep(root, agent) // reminder 2
    assert.equal(compactionSummaryCount(session), 0, 'no fold before the counter reaches K')
    await preStep(root, agent) // reminder 3 → degrade
    assert.equal(compactionSummaryCount(session), 1, 'exactly one durable compaction after K reminders')
    assert.ok(session.surface.nodes.length < surfaceBefore, 'the surface shrank')

    const ledger = rebuildBlockLedger(sessionEventsOf(session))
    assert.equal(ledger.length, 1, 'the fold wrote one ledger row ($dshAcpBlockLedger)')
    const block = ledger[0]!
    assert.equal(block.topic, 'no-action emergency degrade', 'the block names the degrade topic')
    assert.match(block.summary, /EMERGENCY PROVENANCE: no-action degrade/, 'the marker is EMERGENCY-stamped')
    assert.match(block.summary, /reminded 3 turn/, 'the marker names the reminder count')
    assert.ok(!block.summary.includes(SUMMARY_FRAME_PREFIX), 'engine marker is not mislabeled as model-written')
    assert.ok(block.shadowedSeqs.length > 0, 'the block shadowed a non-empty range')
    const projected = eventsToCoreMessages(surfaceEventsOf(session))
    assert.ok(projected.some((message) => message.text === block.summary), 'stored == projected bytes')

    // The degrade logged WHY, in plain terms, including the loss notice. The
    // cordis exporter puts the message in `args[0]` on this host line.
    const warns = logs
      .filter((row) => row.type === 'warn')
      .map((row) => (typeof row.message === 'string' ? row.message : (row as unknown as { args?: string[] }).args?.join(' ') ?? ''))
    assert.ok(
      warns.some((m) => /EMERGENCY no-action degrade.*reminded 3 turn/.test(m)),
      `expected the degrade warn, got: ${JSON.stringify(warns)}`,
    )
    assert.ok(
      warns.some((m) => /Semantic detail .* summarized by the marker/.test(m)),
      'the log states the semantic loss explicitly',
    )

    // One-time guard: a fourth reminder at high pressure must NOT fold again.
    await preStep(root, agent)
    assert.equal(compactionSummaryCount(session), 1, 'the degrade fires at most once per session')
    void engine
  } finally {
    await fiber.dispose()
  }
})

test('D: a successful compress resets the counter — the degrade never fires behind a working model', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber, engine } = await mountEngine(root, { modelContextLimit: DEGRADE_LIMIT, emergencyFoldAfterTurns: 3 })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const session = buildTextSession(30)
    const agent = fakeAgent(session, root)

    await preStep(root, agent)
    await preStep(root, agent) // counter at 2
    // Simulate a successful compress the way handleCompress signals it: the
    // call id is queued for pair-hiding, then the tool/result event lands.
    const callId = 'call-reset'
    engine.compressCallIdsToHide.add(callId)
    session.append('tool/result', {
      turn: 1,
      step: 1,
      message: {
        id: `res-${callId}`,
        role: 'user',
        content: [{ type: 'tool-result', toolCallId: callId, content: [{ type: 'text', text: 'ok' }] }],
        source: { kind: 'tool', callId },
      },
    }, { surfaceOp: 'append' })
    root.emit('session/event', session, sessionEventsOf(session).at(-1))
    // Two more nudges would have crossed K=3 without the reset.
    await preStep(root, agent)
    await preStep(root, agent)
    assert.equal(compactionSummaryCount(session), 0, 'the counter reset prevented the degrade')
    assert.ok(
      !logs.some((row) => /EMERGENCY no-action degrade/.test(row.message)),
      'no degrade log line',
    )
  } finally {
    await fiber.dispose()
  }
})

test('D: emergencyFoldAfterTurns: 0 disables the degrade (hard nudge only)', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber } = await mountEngine(root, { modelContextLimit: DEGRADE_LIMIT, emergencyFoldAfterTurns: 0 })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const session = buildTextSession(30)
    const agent = fakeAgent(session, root)
    for (let i = 0; i < 6; i += 1) await preStep(root, agent)
    assert.equal(compactionSummaryCount(session), 0, 'no fold at any reminder count')
    assert.ok(!logs.some((row) => /EMERGENCY no-action degrade/.test(row.message)))
  } finally {
    await fiber.dispose()
  }
})

test('D: the degrade does not spend the overflow retry budget', async () => {
  const root = new Context()
  const { fiber, engine } = await mountEngine(root, { modelContextLimit: DEGRADE_LIMIT, emergencyFoldAfterTurns: 3 })
  try {
    const session = buildTextSession(30)
    const agent = fakeAgent(session, root)
    await preStep(root, agent)
    await preStep(root, agent)
    await preStep(root, agent) // the degrade fires
    assert.equal(compactionSummaryCount(session), 1, 'the degrade landed')
    // The overflow budget lives on the overflowRetries map keyed by agent; the
    // degrade path never writes it, so a fresh overflow still gets its full
    // default budget of 1.
    const retries = (engine as unknown as { overflowRetries: Map<Agent, number> }).overflowRetries.get(agent)
    assert.equal(retries, undefined, 'overflow retry budget untouched by the degrade')
  } finally {
    await fiber.dispose()
  }
})

test('D: resolveAcpConfig normalizes emergencyFoldAfterTurns (undefined → default 3, invalid → throw)', () => {
  assert.equal(resolveAcpConfig({}).emergencyFoldAfterTurns, 3, 'absent key reads the default')
  assert.equal(resolveAcpConfig({ emergencyFoldAfterTurns: 0 }).emergencyFoldAfterTurns, 0, 'explicit 0 is honored (disable)')
  assert.equal(resolveAcpConfig({ emergencyFoldAfterTurns: 5 }).emergencyFoldAfterTurns, 5)
  assert.throws(() => resolveAcpConfig({ emergencyFoldAfterTurns: -1 }), /non-negative integer/)
  assert.throws(() => resolveAcpConfig({ emergencyFoldAfterTurns: 1.5 }), /non-negative integer/)
})
