/**
 * `disableBasic` — ACP takes the host's own auto-compaction engine down.
 *
 * 事故形态（2026-11-25 实测）：宿主 composition 同时加载 `compaction-basic`，
 * 它自己的 `agent/pre-step` 自动压缩按 `auto: true` 默认在跑，把 121 条消息 /
 * 77,280 tokens 折成一条 32 字符摘要，区段里的用户约束全丢。ACP 的
 * `compactIfNeeded` 返回 null 挡不住它 —— 那根本不是宿主的 seam 契约。
 *
 * 这里钉的是 `disableBasicOnce()` 的五个行为面：
 *   1. 找到 `compaction-basic` → 调 `update({ disabled: true })`
 *   2. 幂等：第二次调用不再动 entry
 *   3. entry 缺失 → warn，不抛
 *   4. 宿主没有 loader 服务 → warn，不抛
 *   5. `disableBasic: false` → 完全不碰 entry
 *   6. `update` 自己抛错 → warn，不抛（挂载绝不能因它失败）
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import { AcpCompactionEngine } from '../src/index.ts'

interface LogRow {
  type: string
  message: string
}

interface FakeEntry {
  options?: { id?: string }
  update(options: Record<string, unknown>): Promise<unknown> | unknown
}

/** Mount the engine with a fake loader hanging off its context. */
async function mountWithLoader(
  root: Context,
  loader: { entries(): Iterable<FakeEntry> } | undefined,
  config: Record<string, unknown> = {},
): Promise<{ fiber: { dispose: () => Promise<void> }; engine: AcpCompactionEngine }> {
  let engine: AcpCompactionEngine | undefined
  const fiber = root.plugin((ctx) => {
    // `loader` is a cordis SERVICE, not a plain property: assigning it through
    // the context proxy throws "cannot set property … without provide", so the
    // fake has to arrive through the same `ctx.provide` the real loader uses
    // (`cordis-plugin-loader/lib/index.js`, `ctx.reflect.provide("loader", …)`).
    if (loader !== undefined) ctx.provide('loader', loader)
    engine = new AcpCompactionEngine(ctx, config as never)
  })
  await fiber
  if (engine === undefined) throw new Error('engine did not mount')
  return { fiber, engine }
}

function warnTexts(logs: LogRow[]): string[] {
  // The exporter hands back its OWN row shape, not a `{message}` envelope —
  // existing tests regex over `JSON.stringify(row)`, so match that here too.
  return logs.filter((row) => (row as { type?: string }).type === 'warn').map((row) => JSON.stringify(row))
}

test('disableBasic: 命中 compaction-basic 时调 update({disabled:true})，且只调它的', async () => {
  const root = new Context()
  const touched: string[] = []
  const loader = {
    *entries(): Iterable<FakeEntry> {
      yield { options: { id: 'compaction-acp' }, update() { touched.push('acp') } }
      yield { options: { id: 'compaction-basic' }, update(options) { touched.push(`basic:${JSON.stringify(options)}`) } }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader)
  try {
    ;(engine as unknown as { disableBasicOnce(): void }).disableBasicOnce()
    assert.deepEqual(touched, ['basic:{"disabled":true}'], '只有 compaction-basic 该被下线')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: 幂等——第二次调用不再碰 entry', async () => {
  const root = new Context()
  let calls = 0
  const loader = {
    *entries(): Iterable<FakeEntry> {
      yield { options: { id: 'compaction-basic' }, update() { calls += 1 } }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader)
  try {
    const once = (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce.bind(engine)
    once()
    once()
    once()
    assert.equal(calls, 1, '重复调用必须短路（dispose 是一次性动作）')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: entry 缺失时不置 handled——下回合重试（晚加载可被追上）', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const loader = {
    *entries(): Iterable<FakeEntry> {
      yield { options: { id: 'compaction-acp' }, update() { throw new Error('不该被碰') } }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader)
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const once = (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce.bind(engine)
    assert.doesNotThrow(once)
    assert.deepEqual(warnTexts(logs), [], '首回合 entry 未出现是正常态，不该有告警')
    // 第二回合补上 entry：还能被禁用，说明 handled 没被提前置位。
    const mutable = loader as { entries(): Iterable<FakeEntry> }
    ;(mutable as { rows?: FakeEntry[] }).rows = [
      { options: { id: 'compaciton-basic-placeholder' }, update() { return undefined } },
      { options: { id: 'compaction-basic' }, update() { return undefined } },
    ]
    Object.defineProperty(mutable, 'entries', { value: function* () { yield* (mutable as { rows: FakeEntry[] }).rows } })
    assert.doesNotThrow(once)
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: entries() 本身抛异常 → warn 一次，下一回合重试', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  let throws = true
  const loader = {
    *entries(): Iterable<FakeEntry> {
      if (throws) throw new Error('registry mid-rebuild')
      yield { options: { id: 'compaction-basic' }, update() { return undefined } }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader)
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const once = (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce.bind(engine)
    assert.doesNotThrow(once, 'entries() 抛错不能穿透到 pre-step housekeeping')
    assert.ok(
      warnTexts(logs).some((m) => /compaction-basic lookup failed: registry mid-rebuild/.test(m)),
      '应 warn 一次 lookup failed',
    )
    // 恢复后下回合成功禁用。
    throws = false
    assert.doesNotThrow(once)
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: 宿主没有 loader 服务时静默跳过，不抛', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const { fiber, engine } = await mountWithLoader(root, undefined)
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    assert.doesNotThrow(() => (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce())
    assert.deepEqual(warnTexts(logs), [], '没有 loader 插件是正常环境差异，不该有告警')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: false 时完全不碰 entry', async () => {
  const root = new Context()
  let calls = 0
  const loader = {
    *entries(): Iterable<FakeEntry> {
      yield { options: { id: 'compaction-basic' }, update() { calls += 1 } }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader, { disableBasic: false })
  try {
    ;(engine as unknown as { disableBasicOnce(): void }).disableBasicOnce()
    assert.equal(calls, 0, 'disableBasic: false 表示保留宿主原生行为')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: update 抛错只 warn，不抛（挂载绝不能因它失败）', async () => {
  const logs: LogRow[] = []
  const root = new Context()
  const loader = {
    *entries(): Iterable<FakeEntry> {
      yield {
        options: { id: 'compaction-basic' },
        async update() { throw new Error('dispose exploded') },
      }
    },
  }
  const { fiber, engine } = await mountWithLoader(root, loader)
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    assert.doesNotThrow(() => (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce())
    // 拒绝发生在微任务里，让出一拍再断言
    await new Promise((resolve) => setImmediate(resolve))
    assert.ok(
      warnTexts(logs).some((m) => /could not disable compaction-basic: dispose exploded/.test(m)),
      `expected the dispose-failure warn, got: ${JSON.stringify(warnTexts(logs))}`,
    )
  } finally {
    await fiber.dispose()
  }
})
