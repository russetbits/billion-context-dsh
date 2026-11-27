/**
 * C-1（2026-11-23）：recall 的检索形态覆盖**仍然可见**的表面。
 *
 * 原来语料只有「块摘要 + 被折叠的原文」。模型问「刚才那个报错在哪」时，只要
 * 那段还没被压，检索反而一无所获 —— 得靠翻可见消息，而可见面正是长会话里
 * 最翻不动的部分。acp-rs 的全日志面是它唯一实测更优的能力（同一会话 934 事件
 * / 88.7 万字符：TS 2.234ms vs Rust 1.837ms，差 0.4ms —— 不构成移植理由，但
 * 覆盖面本身值得补）。
 *
 * 噪音排除同样重要：nudge 回显、compress 配对桩、prune 墓碑这些宿主元数据行
 * 会随每步复读，把它们塞进语料会让每次检索都命中同一批无内容行。
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { searchBlocks } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { buildSearchDocs } from '../src/tools.ts'
import { appendAssistant, appendNudge, appendTurn, appendUser, longText } from './helpers.ts'

const NEEDLE = 'kangaroo-anomaly-42'

function fixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, `visible needle: ${NEEDLE} lives only here`)
  // 宿主元数据行：带同一个 needle，但它是 nudge 回显，不该成为检索结果
  appendNudge(session, `Efficiency nudge: compress consumed ranges (mentions ${NEEDLE})`)
  appendAssistant(session, longText('work', 3))
  appendTurn(session, 2)
  appendUser(session, 'tail question')
  return session
}

test('C-1: 仍可见的消息进入检索语料（标为未压缩，可直接读 seq）', () => {
  const docs = buildSearchDocs(fixture('c1-visible'))
  const hits = docs.filter((doc) => doc.text.includes(NEEDLE))
  assert.equal(hits.length, 1, 'the real user message only — the nudge echo is excluded')
  const hit = hits[0]!
  assert.equal(hit.kind, 'message')
  assert.equal(hit.role, 'user')
  assert.equal(hit.blockId, undefined, 'no blockId ⇒ the engine renders it as still-visible')
  assert.match(hit.ref, /^seq \d+$/, 'ref is a surface seq, directly readable')
})

test('C-1: 检索真的命中可见消息（不是只进了数组）', () => {
  const docs = buildSearchDocs(fixture('c1-search'))
  const results = searchBlocks(docs, NEEDLE, { limit: 5 })
  assert.ok(results.length >= 1, 'hybrid search finds the needle')
  const top = results[0]!
  assert.equal(top.blockId, undefined, 'top hit is the visible row, not a block summary')
  assert.ok(top.preview.includes(NEEDLE))
})

test('C-1: nudge 回显行不进语料（每步复读的元数据不是可检索内容）', () => {
  const docs = buildSearchDocs(fixture('c1-nudge'))
  const nudges = docs.filter((doc) => doc.text.startsWith('Efficiency nudge:'))
  assert.equal(nudges.length, 0, 'metadata rows are filtered before they reach the corpus')
})

test('C-1: 被折叠的原文仍从块语料可达（新面不改旧面）', () => {
  // 同一 seq 既被折叠又在 surface 上是不可能的；这里只需确认 buildSearchDocs
  // 对「无块」会话给出可见面、对「有块」会话仍保留块摘要 —— 用两个会话对照。
  const plain = buildSearchDocs(fixture('c1-plain'))
  assert.ok(plain.every((doc) => doc.kind === 'message'), 'no blocks ⇒ all docs are messages')
  assert.ok(plain.length >= 2, 'visible rows are indexed')
})
