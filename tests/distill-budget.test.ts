/**
 * 蒸馏产物截断预算（2026-11-27）：显式 content 蒸馏路径的大小约束。
 *
 * 背景：condense 合成批过 weight gate（distill 以 max(span, MIN) 为基准、35%
 * 上限 + 1500t 绝对顶），但模型显式 `content: [{ startSeq: "bN", … }]` 蒸馏
 * 完全没有大小约束——T2/T3 产物可以超过它替换的源摘要，膨胀的 checkpoint 节点
 * 常驻每个后续请求。
 *
 * Contracts:
 *   DB-1  T2（被吸收块全是 tier1）：超 0.6×基准截断 + advisory（保头砍尾）
 *   DB-2  T3（吸收含 tier2 块）：超 0.4×基准截断 + advisory
 *   DB-3  基准不可得（span 不吸收任何 active 块 checkpoint）→ fail-open 不截断
 *   DB-4  截断后剩余 <200t → 拒绝该项 + 引导文案（2026-11-27 三段下限后：
 *         band-2 120t..200t 放行原 note、band-3 <120t 拒绝，见 DB-7/DB-8）
 *   DB-5  普通 content（不吸收 checkpoint）不受预算影响
 *   DB-6  骨架 heading 在文件头部 → 保头砍尾保住 Task/Now
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { defaultCountTokens } from '../src/kernel/index.ts'
import { withSummaryFramePrefix, SUMMARY_FRAME_PREFIX } from '../src/messages.ts'
import { readCompactionSummary, blockRegistry } from '../src/region.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const NOTE = '## Task\nFold the consumed prefix and keep the ledger honest — the consumed tool rows, the decisions taken and the state left behind.\n'
  + '## Now\nRe-measure the election against the shorter surface; done when every gate is named in the compress result and the remaining candidates are re-priced.\n'

/** 长 note：每个 fixture 块的源摘要 ~400 tokens，让 0.6× 预算足够容纳一次
 *  成功的截断（截断后 ≥200t）。DB-4（拒绝路径）单独用短 NOTE。 */
const NOTE_BIG = '## Task\nFold the consumed prefix and keep the ledger honest — record every consumed tool row, the decisions taken, the verification readings already green and the exact state the next step inherits.\n'
  + '## Now\nRe-measure the election against the shorter surface; done when every gate is named in the compress result, the remaining candidates are re-priced, and the ledger reconciles against the block accounting.\n'
  + '## Pointers\nThe election geometry lives in condense-elect; the weight gate in condense-note; the transaction in region. Each path was measured on a live session before it shipped, and every knob it carries was benchmarked against the fallback it replaced.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
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

async function compressCall(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  const out = await definition.execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}

const bigText = (label: string, seed: number): string => longText(label, seed) + longText(label + 'b', seed + 100)

function summaryEvents(session: Session) {
  return session.snapshotEvents().filter((event) => event.type === 'compaction/summary')
}

/** 每 tier：源块摘要的 stored tokens（frame 计价）与 checkpoint seq。 */
function tierOf(session: Session, order: number): number {
  const entry = blockRegistry(session)[order]
  return entry?.tier ?? 0
}

function checkpointSeqOf(session: Session, order: number): number {
  const entry = blockRegistry(session)[order]
  assert.ok(entry?.summarySeq !== null && entry?.summarySeq !== undefined, `block ${order} has a checkpoint seq`)
  return entry.summarySeq!
}

function activeCheckpointSeqs(session: Session): number[] {
  return blockRegistry(session)
    .filter((entry) => entry.active && entry.summarySeq !== null)
    .map((entry) => entry.summarySeq!)
}

/**
 * 建 N 个 tier-1 块：每块 = 一段 foldable prose + 5 行 pad（走出 protected 窗口）
 * 后独立折叠。返回会话。
 */
async function tierOneBlocks(session: Session, env: ToolEnvironment, blocks: number, note: string = NOTE_BIG): Promise<void> {
  for (let index = 0; index < blocks; index += 1) {
    const before = session.surface.nodes.length
    appendUser(session, 'gap ' + index + ': ' + longText('gap' + index, 6000) + longText('gap' + index + 'x', 6000))
    const gapSeq = session.surface.nodes.slice(before, before + 1)[0]!
    appendUser(session, 'next question ' + index)
    for (let pad = 0; pad < 5; pad += 1) {
      appendAssistant(session, 'pad ' + index + '-' + pad + ' ' + longText('pad' + index + pad, index * 10 + pad))
    }
    const text = await compressCall(session, env, 'gap-' + index, {
      content: [{ startSeq: gapSeq, endSeq: gapSeq, summary: note }],
    })
    assert.match(text, /Compressed 1 block/, 'fixture folds one gap: ' + text)
  }
}

// ---------------------------------------------------------------- DB-1
test('DB-1: a T2 distill whose note exceeds 0.6× the absorbed tier-1 summary base is truncated with an advisory', async () => {
  const env = makeEnv()
  const session = Session.create('db1-t2-truncate')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 2)
  // 吸收两个 tier1 checkpoint：edges 都钉在 checkpoint seq 上。
  const first = checkpointSeqOf(session, 0)
  const second = checkpointSeqOf(session, 1)
  // 写一条远超基准的长 distill note：骨架 heading 在头部（保头验证用），后面是
  // 大段叙事填充。
  const filler = Array.from({ length: 200 }, (_, i) => `Filler line ${i}: the narrative this test wants truncated, alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const oversized = NOTE + filler + '\n'
  const text = await compressCall(session, env, 'db1', {
    content: [{ startSeq: String(first), endSeq: String(second), summary: oversized }],
  })
  assert.match(text, /Compressed 1 block/, 'the distill lands (truncated, not refused): ' + text)
  assert.match(text, /distill note for seqs .*truncated to/, 'advisory names the truncation: ' + text)
  assert.match(text, /budget = source Σ/, 'advisory names the budget arithmetic: ' + text)
  // 落账的 summary 必须在预算内：先吸收两个 tier1 块的源摘要 tokens 作基准，
  // 新块 tier 应为 2，且 stored tokens ≤ 0.6×基准。
  const events = summaryEvents(session)
  const newEntry = blockRegistry(session)[2]
  assert.equal(newEntry?.tier, 2, 'the new block is tier 2')
  const sourceTokens = [0, 1].reduce((total, order) => {
    const entry = blockRegistry(session)[order]
    return total + defaultCountTokens(SUMMARY_FRAME_PREFIX + (readCompactionSummary(summaryEvents(session)[order] as never).summary ?? []).map((b: { text?: string }) => b.text ?? '').join(''))
  }, 0)
  const storedText = (readCompactionSummary(events[2] as never).summary ?? []).map((b: { text?: string }) => b.text ?? '').join('')
  const storedTokens = defaultCountTokens(withSummaryFramePrefix(storedText))
  assert.ok(
    storedTokens <= Math.floor(sourceTokens * 0.6),
    `stored ~${storedTokens}t must fit the 0.6× budget of ~${sourceTokens}t`,
  )
  // 保头：Task/Now heading 必须还在。
  assert.ok(storedText.includes('## Task'), 'the head (## Task) survives the tail cut')
  assert.ok(storedText.includes('## Now'), 'the head (## Now) survives the tail cut')
  // 蒸馏后旧 checkpoint 消失：active 块只剩新块。
  const stillActive = blockRegistry(session).filter((entry) => entry.active)
  assert.equal(stillActive.length, 1, 'the two tier-1 blocks were absorbed')
})

// ---------------------------------------------------------------- DB-2
test('DB-2: a T3 distill absorbing a tier-2 block is capped at 0.4× the base', async () => {
  const env = makeEnv()
  const session = Session.create('db2-t3-truncate')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 3, NOTE_BIG + ' ' + Array.from({ length: 120 }, (_, i) => `Detail line ${i}: the decisions, file paths and open objectives this block carries forward into the next turn, with the verification readings already green.`).join(' '))
  // t1 摘要必须够大：t3 预算 = 0.4×(0.6×t1Σ)，要容 ≥200t 截断 → t1Σ ≥ ~1250t。
  // 先蒸馏前两个 tier1 块 → tier2 块。t2 note 用 NOTE_BIG：base = t2 源摘要，
  // 0.4×base 必须容得下一次 ≥200t 的截断（NOTE 太短会让 budget <200 走拒绝分支）。
  const c1 = checkpointSeqOf(session, 0)
  const c2 = checkpointSeqOf(session, 1)
  const t2Text = await compressCall(session, env, 'db2-t2', {
    content: [{ startSeq: String(c1), endSeq: String(c2), summary: NOTE_BIG }],
  })
  assert.match(t2Text, /Compressed 1 block/, 'the tier-2 distill lands: ' + t2Text)
  const t2Entry = blockRegistry(session).find((entry) => entry.tier === 2)
  assert.ok(t2Entry !== undefined, 'a tier-2 block landed')
  // 再蒸馏 tier2 块 → tier3。T3 必须 block-ref 边（bN）——kernel 的 tier 判定
  // 要求 boundaryKind==='block'（纯 seq 边即使覆盖 checkpoint 也产出 tier2）；
  // 且 outputTier = min(3, minTier(nested)+1)，混入 tier1 会降到 tier2，所以
  // 这里单蒸 tier2 块（b4）。bN = registry 下标 + 1（本 fixture 按序落账）。
  // T3 预算 = 0.4×tier2 基准：本 fixture 的 tier2 基准 ~197t → 预算 ~79t
  // < 200t 可用下限 → 拒绝路径（截断路径已由 DB-1 的 tier2 覆盖）。断言拒绝
  // 文案点名 0.4 ratio（即吸收了 tier≥2 块）。
  const registryBefore = blockRegistry(session)
  // blockRegistry 每次调用都新建 entry 对象，必须按 blockId 定位而不是 indexOf。
  const t2Index = registryBefore.findIndex((entry) => entry.blockId === t2Entry.blockId)
  const filler = Array.from({ length: 60 }, (_, i) => `Filler line ${i}: the narrative this test wants truncated, alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const oversized = NOTE + filler + '\n'
  const before = summaryEvents(session).length
  const text = await compressCall(session, env, 'db2-t3', {
    content: [{ startSeq: 'b' + (t2Index + 1), endSeq: 'b' + (t2Index + 1), summary: oversized }],
  })
  assert.match(text, /REJECTED/, 'the T3 item is rejected (budget too small for a briefing): ' + text)
  assert.match(text, /× 0\.4\)/, 'the ratio names 0.4 (a tier-2 source was absorbed): ' + text)
  // 没有新块落账。
  assert.equal(summaryEvents(session).length, before, 'nothing landed for the rejected T3 item')
})

// ---------------------------------------------------------------- DB-3
test('DB-3: a span with no absorbed active checkpoint is fail-open (no truncation)', async () => {
  const env = makeEnv()
  const session = Session.create('db3-failopen')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 1)
  // 一段普通长 prose（不覆盖任何 checkpoint seq）：无基准 → 不截断。
  const before = session.surface.nodes.length
  appendUser(session, 'plain work ' + longText('plain', 6000) + longText('plainx', 6000))
  appendUser(session, 'next question')
  for (let pad = 0; pad < 5; pad += 1) appendAssistant(session, 'pad plain-' + pad + ' ' + longText('pp' + pad, pad))
  const startSeq = session.surface.nodes.slice(before, before + 1)[0]!
  const filler = Array.from({ length: 60 }, (_, i) => `Filler line ${i}: alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const text = await compressCall(session, env, 'db3', {
    content: [{ startSeq, endSeq: startSeq, summary: NOTE + '\n' + filler + '\n' }],
  })
  assert.match(text, /Compressed 1 block/, 'the plain fold lands: ' + text)
  assert.doesNotMatch(text, /distill note for seqs .*truncated/, 'no budget advisory on a plain span')
  // 落账 summary 完整保留（含 filler）。
  const last = summaryEvents(session)[summaryEvents(session).length - 1]!
  const storedText = (readCompactionSummary(last as never).summary ?? []).map((b: { text?: string }) => b.text ?? '').join('')
  assert.ok(storedText.includes('Filler line 59'), 'the note is stored verbatim (fail-open)')
})

// ---------------------------------------------------------------- DB-4
test('DB-4: a truncation that would leave <200 usable tokens REJECTS the item with guidance', async () => {
  const env = makeEnv()
  const session = Session.create('db4-reject')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 1)
  const checkpoint = checkpointSeqOf(session, 0)
  // 源摘要很小（~NOTE 量级，几十 token），note 塞 2000t 叙事 → 0.6×基准后
  // 剩余远低于 200t → 拒绝。
  const filler = Array.from({ length: 200 }, (_, i) => `Filler line ${i}: the narrative this test wants rejected, alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const before = summaryEvents(session).length
  const text = await compressCall(session, env, 'db4', {
    content: [{ startSeq: String(checkpoint), endSeq: String(checkpoint), summary: NOTE + filler + '\n' }],
  })
  assert.match(text, /REJECTED/, 'the item is rejected: ' + text)
  assert.match(text, /below the ~120t minimum/, 'the guidance names the floor: ' + text)
  assert.match(text, /Distill FEWER source blocks/, 'the guidance names the compliant move: ' + text)
  // 没有新块落账。
  assert.equal(summaryEvents(session).length, before, 'nothing landed for the rejected item')
  // 旧 checkpoint 仍在 surface 上（未被动过）。
  assert.ok(activeCheckpointSeqs(session).includes(checkpoint), 'the source block stays active')
})

// ---------------------------------------------------------------- DB-5
test('DB-5: an ordinary content range is unaffected by the budget', async () => {
  const env = makeEnv()
  const session = Session.create('db5-plain')
  appendTurn(session, 1)
  // 无任何已折叠块：普通 content 折叠，即使 note 很长也原样落账。
  const before = session.surface.nodes.length
  appendUser(session, 'work ' + longText('w', 6000) + longText('wx', 6000))
  appendUser(session, 'next question')
  for (let pad = 0; pad < 5; pad += 1) appendAssistant(session, 'pad ' + pad + ' ' + longText('p' + pad, pad))
  const startSeq = session.surface.nodes.slice(before, before + 1)[0]!
  const filler = Array.from({ length: 120 }, (_, i) => `Filler line ${i}: alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const text = await compressCall(session, env, 'db5', {
    content: [{ startSeq, endSeq: startSeq, summary: NOTE + filler + '\n' }],
  })
  assert.match(text, /Compressed 1 block/, 'the plain fold lands: ' + text)
  assert.doesNotMatch(text, /distill note for seqs .*truncated/, 'no budget advisory')
  const last = summaryEvents(session)[summaryEvents(session).length - 1]!
  const storedText = (readCompactionSummary(last as never).summary ?? []).map((b: { text?: string }) => b.text ?? '').join('')
  assert.ok(storedText.includes('Filler line 119'), 'stored verbatim')
})

// ---------------------------------------------------------------- DB-6
test('DB-6: the head-of-file skeleton headings survive the tail cut', async () => {
  const env = makeEnv()
  const session = Session.create('db6-headings')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 1)
  const checkpoint = checkpointSeqOf(session, 0)
  // 源摘要 ≈ NOTE（~30t），预算 0.6× ≈ 十几 t——任何截断都只剩头部一两行。
  // 用 Task/Now + 大 filler：截断后 Task 行必须幸存。
  const filler = Array.from({ length: 100 }, (_, i) => `Filler line ${i}: alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const text = await compressCall(session, env, 'db6', {
    content: [{ startSeq: String(checkpoint), endSeq: String(checkpoint), summary: '## Task\nFold the prefix.\n## Now\nWait.\n' + filler + '\n' }],
  })
  // 剩余 <200t → 拒绝路径（DB-4 覆盖）；这里验证拒绝文案点名 budget。
  assert.match(text, /REJECTED/, 'a tiny budget rejects: ' + text)
  assert.match(text, /source summaries Σ/, 'the arithmetic is named: ' + text)
})

// ---------------------------------------------------------------- DB-7
// 2026-11-27 lead 裁决的三段下限：band-2（120t..200t）接受原 note 不截断，
// band-3（<120t）REJECT 且文案改为 120t 语义。
test('DB-7: a ceiling in the 120t..200t band accepts the note UNTRUNCATED with an advisory', async () => {
  const env = makeEnv()
  const session = Session.create('db7-band2')
  appendTurn(session, 1)
  // 源摘要 ~260t（0.6× → ceiling ~156t，落在 band-2）；note 超 ceiling 但
  // 截断无意义（产物≈源大小），必须原样放行。
  const sourceNote = NOTE_BIG + Array.from({ length: 8 }, (_, i) => `Source line ${i}: decision ${i} recorded with its path and its verification reading.`).join(' ')
  await tierOneBlocks(session, env, 1, sourceNote)
  const checkpoint = checkpointSeqOf(session, 0)
  // 需要知道确切 ceiling：读账里源摘要的 stored tokens。
  const { readCompactionSummary: readSummary } = await import('../src/region.ts')
  const events0 = session.snapshotEvents()
  const sourceStored = readSummary(events0.filter((e: { type?: string }) => (e as { type?: string }).type === 'compaction/summary').at(-1) as never).summary ?? []
  const baseTokens = sourceStored.map((b: { text?: string }) => b.text ?? '').join('').length
  const filler = Array.from({ length: 60 }, (_, i) => `Band2 line ${i}: the narrative this test wants kept whole, alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const note = NOTE + filler + '\n'
  const before = summaryEvents(session).length
  const text = await compressCall(session, env, 'db7', {
    content: [{ startSeq: String(checkpoint), endSeq: String(checkpoint), summary: note }],
  })
  void baseTokens
  assert.match(text, /left as-is/, 'band-2 advisory names the no-truncation path: ' + text)
  assert.match(text, /below the ~200t truncation floor/, 'the advisory names the floor: ' + text)
  // 落账且原样（含 filler 尾行，未截断）。
  assert.equal(summaryEvents(session).length, before + 1, 'the band-2 item landed')
  const last = summaryEvents(session)[summaryEvents(session).length - 1]!
  const storedText = (readCompactionSummary(last as never).summary ?? []).map((b: { text?: string }) => b.text ?? '').join('')
  assert.ok(storedText.includes('Band2 line 59'), 'the note is stored untruncated')
})

test('DB-8: a ceiling below 120t REJECTS with the band-3 guidance', async () => {
  const env = makeEnv()
  const session = Session.create('db8-band3')
  appendTurn(session, 1)
  await tierOneBlocks(session, env, 1)
  const checkpoint = checkpointSeqOf(session, 0)
  const filler = Array.from({ length: 200 }, (_, i) => `Filler line ${i}: the narrative this test wants rejected, alpha beta gamma delta epsilon zeta eta theta ${i}.`).join('\n')
  const before = summaryEvents(session).length
  const text = await compressCall(session, env, 'db8', {
    content: [{ startSeq: String(checkpoint), endSeq: String(checkpoint), summary: NOTE + filler + '\n' }],
  })
  assert.match(text, /REJECTED/, 'band-3 rejects: ' + text)
  assert.match(text, /below the ~120t minimum/, 'the guidance names the 120t minimum: ' + text)
  assert.match(text, /distill at a phase close/, 'the guidance names the phase-close move: ' + text)
  assert.equal(summaryEvents(session).length, before, 'nothing landed for the rejected item')
})
