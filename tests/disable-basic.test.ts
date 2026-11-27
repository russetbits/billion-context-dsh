/**
 * `disableBasic` — ACP takes the host's own auto-compaction engine down.
 *
 * 事故形态（2026-11-25 实测）：宿主 composition 同时加载 `compaction-basic`，
 * 它自己的 `agent/pre-step` 自动压缩按 `auto: true` 默认在跑，把 121 条消息 /
 * 77,280 tokens 折成一条 32 字符摘要，区段里的用户约束全丢。ACP 的
 * `compactIfNeeded` 返回 null 挡不住它 —— 那根本不是宿主的 seam 契约。
 *
 * 这里钉的是 `disableBasicOnce()` 的七个行为面：
 *   1. 找到 `compaction-basic` → 调 `update({ disabled: true })`
 *   2. 幂等：第二次调用不再动 entry
 *   3. entry 缺失 → warn，不抛
 *   4. 宿主没有 loader 服务 → warn，不抛
 *   5. `disableBasic: false` → 完全不碰 entry
 *   6. `update` 自己抛错 → warn，不抛（挂载绝不能因它失败）
 *   7. 隔离 realm（retired）：realm 扫描已退役，disableBasicOnce 不碰 realm
 *      的 compaction-basic，根平面提供者也不受影响
 *
 * realm 路径（2026-11-27 退役）：实机验证 disposing realm basic 导致四个 preset
 * 「waiting for compaction」全部加载失败——realm 内 command-compact 等的
 * compaction 服务只有 basic 一个提供者，`isolate:{compaction:true}` 又挡住根平面
 * 的 acp。src/index.ts 已改为 inert stub（`disableRealmBasics` 只置
 * `realmCompactionHandled` + 打一条 debug log，不再扫描/禁用），下面的 realm
 * 测试全部改为「退役口径」：断言扫描不碰 realm。
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

// ---------------------------------------------------------------------------
// 隔离 realm（isolate）平面 — 退役口径（2026-11-27）。
//
// 退役原因：实机验证 disposing realm basic 导致四个 preset「waiting for
// compaction」全部加载失败（realm 内 compaction 服务只有 basic 一个提供者，
// isolate:{compaction:true} 又挡住根平面 acp）——见 src/index.ts
// `disableRealmBasics` 的 alignment 注释。扫描已从 src 移除，stub 只置
// `realmCompactionHandled` + 打一条 debug log。
//
// 测试 fixture 直接把假 impl 塞进全局 reflect.store 的 realm Symbol 键下，
// 断言的是「退役后扫描不碰 realm」：entry.update 从未被调用、提供者完好、
// stub 自身的幂等/不炸语义成立。
// ---------------------------------------------------------------------------

interface StoreImpl {
  name: string
  ctx: { fiber?: { entry?: { options?: { id?: string }; update(options: Record<string, unknown>): unknown } } }
}

const ISOLATE = Symbol.for('cordis.isolate')

function disableBasicOnceOf(engine: AcpCompactionEngine): () => void {
  return (engine as unknown as { disableBasicOnce(): void }).disableBasicOnce.bind(engine)
}

/** The retired realm scan is now an inert stub; reach it through the class-private name. */
function disableRealmBasicsOf(engine: AcpCompactionEngine): () => void {
  return (engine as unknown as { disableRealmBasics(): void }).disableRealmBasics.bind(engine)
}

function realmCompactionHandledOf(engine: AcpCompactionEngine): boolean {
  return (engine as unknown as { realmCompactionHandled: boolean }).realmCompactionHandled
}

test('disableBasic: 隔离 realm 的 compaction-basic 不再被碰（扫描已退役）', async () => {
  // 退役口径（2026-11-27）：disposing realm basic 导致四个 preset「waiting for
  // compaction」全部加载失败（realm 内 compaction 服务只有 basic 一个提供者，
  // isolate:{compaction:true} 又挡住根平面 acp），扫描从 src/index.ts 移除，
  // 只留 inert stub——见 src/index.ts `disableRealmBasics` 的 alignment 注释。
  const root = new Context()
  let realmUpdateCalls = 0
  const { fiber, engine } = await mountWithLoader(root, {
    *entries(): Iterable<FakeEntry> {
      yield { options: { id: 'compaction-acp' }, update() { throw new Error('不该被碰') } }
    },
  })
  try {
    const realmKey = Symbol('compaction#agent')
    ;(root.reflect.store as Record<symbol, StoreImpl | undefined>)[realmKey] = {
      name: 'compaction',
      ctx: { fiber: { entry: { options: { id: 'compaction-basic' }, update() { realmUpdateCalls += 1 } } } },
    }
    disableBasicOnceOf(engine)()
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(realmUpdateCalls, 0, '退役后的 disableBasicOnce 绝不能碰 realm 的 compaction-basic')
    // The stub itself is the only thing left of the scan: it flags the plane
    // handled (its sole remaining semantic) without touching the entry.
    disableRealmBasicsOf(engine)()
    assert.equal(realmCompactionHandledOf(engine), true, 'stub 应把 realmCompactionHandled 置位（退役标记）')
    assert.equal(realmUpdateCalls, 0, 'stub 也不得调 entry.update')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: realm stub 幂等——重复调用只打一次 debug log，entry 始终不被碰', async () => {
  // 退役口径（2026-11-27）：扫描已退役（四个 preset「waiting for compaction」，
  // 见 src/index.ts `disableRealmBasics` 的 alignment 注释）。stub 保留的
  // once-guard 语义仍有意义：重复调用不重复打日志。
  const logs: LogRow[] = []
  const root = new Context()
  let realmUpdateCalls = 0
  const { fiber, engine } = await mountWithLoader(root, {
    *entries(): Iterable<FakeEntry> { return },
  })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const realmKey = Symbol('compaction#agent')
    ;(root.reflect.store as Record<symbol, StoreImpl | undefined>)[realmKey] = {
      name: 'compaction',
      ctx: { fiber: { entry: { options: { id: 'compaction-basic' }, update() { realmUpdateCalls += 1 } } } },
    }
    const stub = disableRealmBasicsOf(engine)
    stub()
    stub()
    stub()
    assert.equal(realmUpdateCalls, 0, 'stub 无论如何不碰 realm entry')
    assert.equal(realmCompactionHandledOf(engine), true, '首次调用即置位')
    const retireLogs = logs
      .filter((row) => (row as { type?: string }).type === 'debug')
      .map((row) => JSON.stringify(row))
      .filter((m) => /isolated compaction-basic scan retired/.test(m))
    assert.equal(retireLogs.length, 1, `退役 debug 日志只打一次，got: ${JSON.stringify(retireLogs)}`)
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: 根平面与 realm 的 compaction 提供者都完好（退役后无任何扫描）', async () => {
  // 退役口径（2026-11-27）：扫描已退役（四个 preset「waiting for compaction」，
  // 见 src/index.ts `disableRealmBasics` 的 alignment 注释）。方向反转——旧
  // 断言是「根平面不被 realm 扫描误伤」，现在没有任何扫描，两个平面的
  // compaction 提供者都必须原样保留。
  const root = new Context()
  let rootImplTouched = false
  let realmDisabled = false
  const { fiber, engine } = await mountWithLoader(root, {
    *entries(): Iterable<FakeEntry> { return },
  })
  try {
    const rootKey = (root as unknown as Record<symbol, Record<string, symbol>>)[ISOLATE].compaction
    assert.notEqual(rootKey, undefined, '前置：根平面 compaction symbol 已存在')
    ;(root.reflect.store as Record<symbol, StoreImpl | undefined>)[rootKey] = {
      name: 'compaction',
      ctx: { fiber: { entry: { options: { id: 'compaction-basic' }, update(options) { rootImplTouched = options.disabled === true } } } },
    }
    const realmKey = Symbol('compaction#agent')
    ;(root.reflect.store as Record<symbol, StoreImpl | undefined>)[realmKey] = {
      name: 'compaction',
      ctx: { fiber: { entry: { options: { id: 'compaction-basic' }, update(options) { realmDisabled = options.disabled === true } } } },
    }
    disableBasicOnceOf(engine)()
    disableRealmBasicsOf(engine)()
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(rootImplTouched, false, '根平面的 compaction 提供者必须完好（disabled 从未置 true）')
    assert.equal(realmDisabled, false, 'realm 的 compaction 提供者必须完好——它是 realm 内唯一的提供者')
  } finally {
    await fiber.dispose()
  }
})

test('disableBasic: 退役路径不触碰 store——store 里塞抛错对象也不炸、不 warn', async () => {
  // 退役口径（2026-11-27）：扫描已退役（四个 preset「waiting for compaction」，
  // 见 src/index.ts `disableRealmBasics` 的 alignment 注释）。旧测试断言的
  // 「store 抛错 → warn 一次，下一回合重试」已无对应代码路径：扫描不存在，
  // store 即使坏了也不该被读到。
  const logs: LogRow[] = []
  const root = new Context()
  const realmKey = Symbol('compaction#agent')
  const { fiber, engine } = await mountWithLoader(root, {
    *entries(): Iterable<FakeEntry> { return },
  })
  try {
    root.logger.exporter({ levels: { default: 3 }, export: (row) => { logs.push(row as LogRow) } })
    const store = root.reflect.store as Record<symbol, StoreImpl | undefined>
    Object.defineProperty(store, realmKey, {
      get() { throw new Error('store mid-rebuild') },
      configurable: true,
    })
    const once = disableBasicOnceOf(engine)
    const stub = disableRealmBasicsOf(engine)
    assert.doesNotThrow(once, 'store 抛错不能穿透到 pre-step housekeeping')
    assert.doesNotThrow(stub, '退役 stub 同样不能穿透')
    assert.doesNotThrow(once)
    await new Promise((resolve) => setImmediate(resolve))
    assert.deepEqual(warnTexts(logs), [], '退役路径不读 store，任何 store 异常都不该产生告警')
  } finally {
    delete (root.reflect.store as Record<symbol, unknown>)[realmKey]
    await fiber.dispose()
  }
})

test('disableBasic: false 时 realm 扫描同样不跑', async () => {
  const root = new Context()
  let realmDisabled = false
  const { fiber, engine } = await mountWithLoader(root, {
    *entries(): Iterable<FakeEntry> { return },
  }, { disableBasic: false })
  try {
    const realmKey = Symbol('compaction#agent')
    ;(root.reflect.store as Record<symbol, StoreImpl | undefined>)[realmKey] = {
      name: 'compaction',
      ctx: { fiber: { entry: { options: { id: 'compaction-basic' }, update() { realmDisabled = true } } } },
    }
    disableBasicOnceOf(engine)()
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(realmDisabled, false, 'disableBasic: false 表示保留宿主原生行为（含 realm）')
  } finally {
    await fiber.dispose()
  }
})
