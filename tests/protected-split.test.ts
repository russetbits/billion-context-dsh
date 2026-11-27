/**
 * content 形态：跨「受保护行」的 range 自动拆分（2026-11-25）。
 *
 * 证据（真实会话 s2115，50 次 compress）：失败 10 次里有 8 次是模型把合法区间连着
 * 当前用户消息 / AGENTS.md 注入行 / skill catalog 一起框进去，被工具整段拒绝；
 * 而且这 8 次都在**紧邻的下一步换范围就成功了**——模型能自己拆，代价是多花一轮。
 * 现在这一刀在工具侧、在同一次调用里做掉。
 *
 * 不变量（这四条是这套拆分的全部语义）：
 *   1. 受保护行本身永不被压缩，也永不被跨过（它仍在 surface 上）。
 *   2. 某一侧为空（受保护行贴边）就不生成那块。
 *   3. 整段都是受保护行 → 仍然拒绝，且文案说清「这一段全是受保护内容」。
 *   4. 没有受保护行 → 行为与拆分引入前逐字一致（回归锚）。
 * 另外：一个 range 拆成 N 段就是 N 块（既有 batch 语义）；condense 路径不参与拆分。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { makeTools, splitAroundProtected, type ToolEnvironment } from '../src/tools.ts'
import { SUMMARY_FRAME_PREFIX } from '../src/messages.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const SUMMARY = '## Task\nRotate the auth token store and prove the split keeps the live turn on the surface.\n'
  + '## Now\nAsserting the cut-around semantics; done when every protected row is still visible.\n'

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
    agent: { id: session.id, session, options: { provider: 'test-provider', model: 'test-model' }, ctx: { tokenMeter: undefined } },
  } as never
}

async function call(session: Session, env: ToolEnvironment, callId: string, args: unknown): Promise<string> {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  const out = await definition.execute(args as never, execStub(session, callId))
  return typeof out === 'string' ? out : (out as { text: string }).text
}


/** The stored note of every durable block, in write order. */
function blockNotes(session: Session): string[] {
  return session
    .snapshotEvents()
    .filter((event) => event.type === 'compaction/summary')
    .map((event) => {
      const blocks = (event.data as { summary?: Array<{ type: string; text?: string }> }).summary ?? []
      return blocks.map((block) => block.text ?? '').join('')
    })
}

/**
 * The note with the engine's provenance frame peeled off. The frame is prepended by
 * the EXISTING `withSummaryFramePrefix` at creation, so a coverage line is necessarily
 * the second line of the stored text — the assertion targets "the model's own note
 * now opens with the mechanical line", which is the property that matters.
 */
function withoutFrame(note: string): string {
  return note.startsWith(SUMMARY_FRAME_PREFIX) ? note.slice(SUMMARY_FRAME_PREFIX.length + 1) : note
}

/** Big enough to clear the kernel's 5000-CHARACTER minimum range on its own. */
function big(label: string, seed: number): string {
  return Array.from({ length: 3 }, (_, copy) => longText(`${label}${copy}`, seed)).join('')
}

/**
 * q0 a0 | q1 (ACTIVE user turn) | a1..a6
 *
 * q1 is the newest real user turn, so it is guarded; six assistant rows after it
 * keep it OUT of the protected recent window, which is what makes it a row in the
 * MIDDLE of a span rather than a tail.
 */
function middleFixture(id: string): { session: Session; seqs: { q0: number; a0: number; q1: number; last: number } } {
  const session = Session.create(id)
  // Surface seqs are NOT array indices (a `turn/start` event consumes a seq without
  // landing on the surface), so every anchor is read back from the surface tail.
  const tail = (): number => session.surface.nodes[session.surface.nodes.length - 1]!
  appendTurn(session, 1)
  appendUser(session, big('q0', 1))
  const q0 = tail()
  appendAssistant(session, big('a0', 2), 1, 2)
  const a0 = tail()
  appendUser(session, big('q1', 3))
  const q1 = tail()
  for (let index = 1; index <= 6; index += 1) appendAssistant(session, big(`a${index}`, 10 + index), 1, 10 + index)
  return { session, seqs: { q0, a0, q1, last: tail() } }
}

// ---------------------------------------------------------------- ①
test('split ①: a protected row in the MIDDLE yields one block per side, in this call', async () => {
  const env = makeEnv()
  const { session, seqs } = middleFixture('split-middle')
  const text = await call(session, env, 'split-middle-1', {
    content: [{ startSeq: seqs.q0, endSeq: seqs.last, summary: SUMMARY }],
  })
  assert.match(text, /Compressed 2 block\(s\) from 1 range \(split around 1 protected line\)/, text)
  assert.ok(session.surface.nodes.includes(seqs.q1 as never), 'the ACTIVE user turn is still visible')
  // Both sides landed as their OWN blocks: the durable summaries are two.
  const summaries = session.snapshotEvents().filter((event) => event.type === 'compaction/summary')
  assert.equal(summaries.length, 2, 'one block per surviving side')

  // (a) The shared note is EXPLICIT about covering more than its block: each cut
  // block opens with the mechanical coverage line, numbered 1/2 and 2/2.
  const notes = blockNotes(session).map(withoutFrame)
  assert.equal(notes.length, 2)
  assert.ok(notes[0]!.startsWith('[split 1/2 of seqs '), `first cut note opens with the coverage line: ${notes[0]!.slice(0, 80)}`)
  assert.ok(notes[1]!.startsWith('[split 2/2 of seqs '), `second cut note opens with the coverage line: ${notes[1]!.slice(0, 80)}`)
  assert.match(notes[0]!, /the note below covers the WHOLE span/, 'the note says it describes the whole span')
  assert.match(notes[0]!, /this block holds only the part BEFORE it/, 'side 1 is the part BEFORE the protected row')
  assert.match(notes[1]!, /this block holds only the part AFTER it/, 'side 2 is the part AFTER the protected row')
  // (b) Both declarations point at the SAME requested span — that is what makes the
  // two blocks readable as one record split in two.
  const spans = notes.map((note) => /of seqs (\d+)\.\.(\d+)/.exec(note)?.slice(1).join('..'))
  assert.deepEqual(spans, [`${seqs.q0}..${seqs.last}`, `${seqs.q0}..${seqs.last}`], 'both pieces name the SAME requested span')
  // (c) The note itself is stored in BOTH blocks, unchanged after the coverage line.
  for (const note of notes) assert.ok(note.includes(SUMMARY), 'the model\'s whole note is carried verbatim')
})

// ---------------------------------------------------------------- ②
test('split ②: a protected row at an EDGE yields only the one surviving side', async () => {
  const env = makeEnv()
  const { session, seqs } = middleFixture('split-edge')
  // Start the span ON the guarded turn: there is no left side to fold.
  const text = await call(session, env, 'split-edge-1', {
    content: [{ startSeq: seqs.q1, endSeq: seqs.last, summary: SUMMARY }],
  })
  assert.match(text, /Compressed 1 block\(s\) from 1 range \(split around 1 protected line\)/, text)
  assert.ok(session.surface.nodes.includes(seqs.q1 as never), 'the active turn stays visible')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1, 'only the right side landed')
})

// ---------------------------------------------------------------- ③
test('split ③: a span that IS the protected row is still refused, and says so', async () => {
  const env = makeEnv()
  const { session, seqs } = middleFixture('split-only')
  const before = session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length
  const text = await call(session, env, 'split-only-1', {
    content: [{ startSeq: seqs.q1, endSeq: seqs.q1, summary: SUMMARY }],
  })
  assert.match(text, /Compressed 0 block/, text)
  assert.match(text, /no part of this span is compressible/, 'the note says the span IS the protected content')
  assert.match(text, /every row in it is protected content that must stay live/, 'and says why nothing can fold')
  assert.equal(
    session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length,
    before,
    'nothing landed',
  )
  assert.ok(session.surface.nodes.includes(seqs.q1 as never), 'the row itself is untouched')
})

// ---------------------------------------------------------------- ④
test('split ④: with no protected row inside, the result is byte-identical to before', async () => {
  const env = makeEnv()
  const session = Session.create('split-clean')
  const tail = (): number => session.surface.nodes[session.surface.nodes.length - 1]!
  appendTurn(session, 1)
  appendUser(session, big('q0', 1))
  const q0 = tail()
  appendAssistant(session, big('a0', 2), 1, 2)
  const a0 = tail()
  appendUser(session, big('q1', 3))
  for (let index = 1; index <= 6; index += 1) appendAssistant(session, big(`a${index}`, 20 + index), 1, 20 + index)
  const text = await call(session, env, 'split-clean-1', {
    content: [{ startSeq: q0, endSeq: a0, summary: SUMMARY }],
  })
  assert.match(text, /^Compressed 1 block\(s\), ~\d+ tokens reclaimed\./, 'the ORIGINAL success line, with no split clause')
  assert.doesNotMatch(text, /split around/)
  // Byte-level regression anchor: a range with NO protected row inside stores the
  // model's note exactly as written — not one byte of coverage line.
  const stored = blockNotes(session).map(withoutFrame)
  assert.equal(stored.length, 1)
  assert.doesNotMatch(stored[0]!, /\[split /)
  assert.ok(stored[0]!.startsWith(SUMMARY), 'the stored note IS the model\'s note, from its first byte')
})

// ---------------------------------------------------------------- unit
test('splitAroundProtected: positional runs, edges, and the all-protected case', () => {
  assert.deepEqual(splitAroundProtected([10, 11, 12, 20, 30, 31], [20]), [[10, 11, 12], [30, 31]])
  assert.deepEqual(splitAroundProtected([1, 2, 3], [1]), [[2, 3]], 'a leading hit leaves one side')
  assert.deepEqual(splitAroundProtected([1, 2, 3], [3]), [[1, 2]], 'a trailing hit leaves one side')
  assert.deepEqual(splitAroundProtected([1, 2, 3], [1, 3]), [[2]], 'two edge hits still leave the middle')
  assert.deepEqual(splitAroundProtected([5], [5]), [], 'a span that is only the hit has no side')
  assert.deepEqual(splitAroundProtected([1, 2], []), [[1, 2]], 'no hits: one piece, unchanged')
  // Positional, not numeric: a checkpoint node carries a far higher seq than its
  // neighbours, so an interval scan would cut the wrong side.
  assert.deepEqual(splitAroundProtected([100, 7, 8], [7]), [[100], [8]], 'order comes from the slice, never from the value')
})
