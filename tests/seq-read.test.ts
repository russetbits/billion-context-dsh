/**
 * C-2（2026-11-23）：recall 按 seq 读一条消息。
 *
 * C-1 让 recall 的检索形态命中仍可见的消息并报出 seq，但模型拿到 seq 后无路可走：
 * 按块读取只认 blockId，而可见消息不属于任何块。这条路径把「检索命中 → 读到
 * 内容」接上，也能读任何历史 seq（日志保留原文），因此折叠之后同样有效。
 *
 * 合并之后（recall = decompress + search_context）这里还钉住「二选一」语义：
 * query 与 blockId/seq 同给要拒，都不给要给出两种形态。
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { AcpStateStore } from '../src/state.ts'
import { runCompactionTransaction } from '../src/region.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    compressCallIdsToHide: new Set(),
  }
}

function fakeExec(session: Session): ToolRunContext {
  const agent = {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx: new Context(),
  } as unknown as Agent
  return { callId: 'call-read', name: 'recall', arguments: {}, signal: new AbortController().signal, agent } as unknown as ToolRunContext
}

async function recall(args: Record<string, unknown>, session: Session): Promise<string> {
  const tool = makeTools(makeEnv()).find((definition) => definition.name === 'recall')
  assert.ok(tool, 'decompress tool is registered')
  const result = await tool.execute(args, fakeExec(session))
  return (result as { text: string }).text
}

function fixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, 'read me by seq: needle-in-visible-row')
  appendAssistant(session, longText('work', 3))
  appendTurn(session, 2)
  appendUser(session, 'tail question')
  return session
}

test('C-2: 按 seq 读到可见消息原文，并标注它仍可见', async () => {
  const session = fixture('c2-visible')
  // surface 第 1 个节点就是那条 user 消息（turn/start 不在 surface 上）
  const text = await recall({ seq: 1 }, session)
  assert.match(text, /needle-in-visible-row/, 'the bytes are returned')
  assert.match(text, /\[seq 1 · user\/message · visible\]/, 'header names seq, type and visibility')
})

test('C-2: 折叠之后同一 seq 仍可读（日志保留原文）', async () => {
  const session = Session.create('c2-folded')
  appendTurn(session, 1)
  appendUser(session, 'soak-target-alpha ' + longText('a', 3))
  appendAssistant(session, longText('b', 3))
  appendUser(session, 'third ' + longText('c', 3))
  appendAssistant(session, longText('d', 3))
  runCompactionTransaction(session, {
    start: 1,
    end: 3,
    shadowedSeqs: [1, 2, 3],
    summary: [{ type: 'text', text: 'folded summary' }],
    shadowedTokenCount: 500,
    provider: 'p',
    model: 'm',
  })
  // 原 seq 2 已被折叠：不再是可见节点，但日志有原文
  const text = await recall({ seq: 2 }, session)
  assert.match(text, /Authentication uses JWT/, 'the original bytes are still there')
  assert.match(text, /folded \(original kept in the log\)/, 'the header says so instead of pretending it is visible')
})

test('C-2: chars 上限截断并给出提示', async () => {
  const session = fixture('c2-truncate')
  // seq 2 = 那条长 assistant 消息（longText 是几千字符）
  const text = await recall({ seq: 2, chars: 1000 }, session)
  assert.match(text, /truncated at 1000 of \d+ chars/, 'truncation is explicit')
})

test('C-2: 不存在的 seq 与缺参数各有明确报错', async () => {
  const session = fixture('c2-errors')
  assert.match(await recall({ seq: 999999 }, session), /is not in this session log/)
  assert.match(await recall({}, session), /pass exactly one form/)
  assert.match(await recall({ blockId: 'no-such-block' }, session), /not found/)
})

test('recall: query 与 blockId/seq 同给时拒绝，并指出该去掉哪个', async () => {
  const session = fixture('recall-exclusive')
  const withBlock = await recall({ query: 'needle', blockId: 'b1' }, session)
  assert.match(withBlock, /mutually exclusive/)
  assert.match(withBlock, /Drop `query` to read the original, or drop `blockId` to run the search/)
  const withSeq = await recall({ query: 'needle', seq: 1 }, session)
  assert.match(withSeq, /drop `seq` to run the search/)
  const withBoth = await recall({ query: 'needle', blockId: 'b1', seq: 1 }, session)
  assert.match(withBoth, /`blockId`\/`seq`/)
  // blockId + seq 同给也是歧义（两个都指名一个 READ 目标），此前 seq 会静默胜出
  const readBoth = await recall({ blockId: 'b1', seq: 1 }, session)
  assert.match(readBoth, /mutually exclusive/, 'blockId+seq is refused too')
  assert.match(readBoth, /`blockId`\/`seq`/, 'and the note names both keys')
})

test('recall: 两种形态都不给时列出可用写法', async () => {
  const session = fixture('recall-noform')
  const text = await recall({}, session)
  assert.match(text, /pass exactly one form/)
  assert.match(text, /recall\(\{ query: /)
  assert.match(text, /recall\(\{ blockId: "b1" \}\)/)
  assert.match(text, /recall\(\{ seq: 42 \}\)/)
})

test('recall: query 形态命中后给出的读法在同一个工具里成立', async () => {
  const session = Session.create('recall-roundtrip')
  appendTurn(session, 1)
  appendUser(session, 'needle-token-xyz appears in the visible row')
  appendAssistant(session, longText('work', 9))
  const hit = await recall({ query: 'needle-token-xyz' }, session)
  assert.match(hit, /Matches for "needle-token-xyz"/)
  assert.match(hit, /Read a hit with: recall\(\{ blockId \}\) or recall\(\{ seq \}\)/)
  const seq = Number(/message seq (\d+)/.exec(hit)?.[1])
  assert.ok(Number.isInteger(seq), 'the hit names the seq to read')
  assert.match(await recall({ seq }, session), /needle-token-xyz/, 'the same tool reads it back')
})
