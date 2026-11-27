/**
 * B-1（2026-11-23）：`agent/pre-step` 的两段自有逻辑必须 fail-open。
 *
 * 事故形态：housekeeping（prune / strip / 校准 / 设置热应用 / 注册自愈）与
 * nudge 构建都跑在同一个 async pre-step 回调里，任何一处抛错都会让宿主整步
 * 拒绝 —— 代价是一整个回合，而收益只是少一条提示或一次清理。
 *
 * 这里用两个注入点各钉一次：
 *   1. session.surface 访问抛错 → housekeeping 段被 catch，step 继续
 *   2. windowFor（窗口解析）抛错 → nudge 段被 catch，返回未加 nudge 的 decision
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import type { Session } from '@deepseek-ai/dsh-session'
import { AcpCompactionEngine } from '../src/index.ts'
import { buildTextSession } from './helpers.ts'

interface LogRow {
  type: string
  message: string
}

async function mountEngine(
  root: Context,
  config: Record<string, unknown> = {},
): Promise<{ fiber: { dispose: () => Promise<void> }; engine: AcpCompactionEngine }> {
  let engine: AcpCompactionEngine | undefined
  const fiber = root.plugin((ctx) => {
    engine = new AcpCompactionEngine(ctx, config as never)
  })
  await fiber
  if (engine === undefined) throw new Error('engine did not mount')
  return { fiber, engine }
}

/** A session whose `surface` access THROWS — the housekeeping段 runs first on it. */
function sessionWithBrokenSurface(session: Session): Session {
  return new Proxy(session, {
    get(target, prop, receiver) {
      if (prop === 'surface') throw new Error('boom: surface unavailable')
      return Reflect.get(target, prop, receiver)
    },
  }) as Session
}

test('B-1: housekeeping 抛错时 pre-step 仍返回 enter（打 warn，不拒绝整步）', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber, engine } = await mountEngine(root, { autoNudge: true })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const agent = { session: sessionWithBrokenSurface(buildTextSession(4)) }
    const decision = await root.waterfall(
      'agent/pre-step' as never,
      { agent } as never,
      async () => ({ kind: 'enter', messages: [] }) as never,
    )
    assert.equal((decision as { kind: string }).kind, 'enter', 'a broken housekeeping pass must not reject the step')
    const warns = logs.filter((row) => row.type === 'warn').map((row) => JSON.stringify(row))
    assert.ok(
      warns.some((m) => /pre-step housekeeping failed \(fail-open\)/.test(m)),
      `expected the housekeeping fail-open warn, got: ${JSON.stringify(warns)}`,
    )
  } finally {
    await fiber.dispose()
  }
})

test('B-1: nudge 构建抛错时 pre-step 返回未加 nudge 的 decision', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber, engine } = await mountEngine(root, { autoNudge: true, modelContextLimit: 300000 })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    // 窗口解析是 nudge 段的第一步；让它抛错，等于「拿不到压力/窗口」。
    ;(engine as unknown as { windowFor: () => Promise<never> }).windowFor = () => {
      throw new Error('boom: window unavailable')
    }
    const baseMessages = [{ role: 'user', content: [] }]
    const agent = { session: buildTextSession(4) }
    const decision = await root.waterfall(
      'agent/pre-step' as never,
      { agent } as never,
      async () => ({ kind: 'enter', messages: baseMessages }) as never,
    ) as { kind: string; messages: unknown[] }
    assert.equal(decision.kind, 'enter')
    assert.equal(decision.messages.length, baseMessages.length, 'no nudge row may be appended when the build failed')
    const warns = logs.filter((row) => row.type === 'warn').map((row) => JSON.stringify(row))
    assert.ok(
      warns.some((m) => /nudge build failed \(fail-open\)/.test(m)),
      `expected the nudge fail-open warn, got: ${JSON.stringify(warns)}`,
    )
  } finally {
    await fiber.dispose()
  }
})
