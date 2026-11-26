/**
 * 非单调表面上的值倒挂区间（2026-11-26 回归，live 会话 a live session）。
 *
 * 事故形态：一次 compaction replace 把 checkpoint 节点 splice 进原位置——checkpoint
 * 的 seq（新写入，数值大）与它后面的残留节点（旧 seq，数值小）形成【相邻的值倒挂对】。
 * 位置序上它们递增，是合法的可压缩区间；但 resolveSurfaceRange 与
 * runCompactionTransaction 的三处值序 throw（`start > end`）把它当成倒置调用拒绝。
 * 真实会话里所有 compress 调用都报 `reversed range 368..326`（常数，与输入无关——
 * 报的是 split-around-protected 切出的 piece 的端点，不是模型传的边）。
 *
 * 修复：区间语义统一为【位置序】——surface.nodes 的 indexOf 判序，值序不再作错误。
 * 真正的倒置调用（两 edge 位置倒挂）仍报 reversed。
 *
 * 不变量：
 *   1. 位置递增、值倒挂的 span 被 resolveSurfaceRange 正常返回（含 stale 恢复路径）。
 *   2. runCompactionTransaction 对这种 span 正常落盘（shadowed 位置 slice 一致）。
 *   3. 两 edge 位置倒挂（真倒置）仍报 `reversed range`。
 *   4. 端到端：handleCompress 在这样的表面上不再抛 reversed，成功落块。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import {
  isExactSurfaceSpan,
  resolveSurfaceRange,
  runCompactionTransaction,
  shadowedSeqsOf,
} from '../src/region.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const SUMMARY = '## Task\nPinned regression: a value-inverted positional span must not read as a reversed call.\n'
  + '## Now\nAsserting positional-order semantics; done when the span resolves and lands.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
}

/** Big enough to clear the kernel's 5000-CHARACTER minimum on its own. */
function big(label: string, seed: number): string {
  return Array.from({ length: 3 }, (_, copy) => longText(`${label}${copy}`, seed)).join('')
}

/**
 * 压掉会话开头一段，让 checkpoint（新 seq，值大）splice 进表面中部，其后紧跟
 * 旧 seq 的残留节点——真实会话  的表面形状（相邻值倒挂对）。
 */
function foldedFixture(): { session: Session; checkpointSeq: number; residual: number[] } {
  const session = Session.create('test-nonmonotonic')
  appendTurn(session, 1)
  appendUser(session, big('q0', 0))
  for (let index = 0; index < 8; index += 1) {
    appendAssistant(session, big(`a${index}`, index + 1), 1, index)
  }
  const live = shadowedSeqsOf(session, 1, session.snapshotEvents().length - 1)
  const shadowed = live.slice(1, live.length - 3)
  const { seqs } = runCompactionTransaction(session, {
    start: shadowed[0]!,
    end: shadowed[shadowed.length - 1]!,
    shadowedSeqs: shadowed,
    summary: [{ type: 'text', text: SUMMARY }],
    shadowedTokenCount: 10_000,
    provider: 'test-provider',
    model: 'test-model',
    tier: 1,
  })
  // seqs = [compaction/start, compaction/summary, user/message(checkpoint node), compaction/end]。
  // 表面上的 checkpoint 节点是第三个 append（user/message, surfaceOp replace）。
  const checkpointSeq = seqs[2]! // user/message checkpoint node (append AFTER compaction/summary)
  // replace 之后：checkpoint 位置在原 slice 处，其后是 seq 更小的残留节点。
  const residual = shadowedSeqsOf(session, checkpointSeq, checkpointSeq)
  return { session, checkpointSeq, residual }
}

test('non-monotonic: value-inverted positional span resolves (regression for "reversed range 368..326")', () => {
  const { session, checkpointSeq, residual } = foldedFixture()
  const nodes = (session.surface as { nodes: number[] }).nodes
  // 前提：表面确实非单调（checkpoint 之后紧跟更小的 seq）。
  const checkpointIdx = nodes.indexOf(checkpointSeq)
  assert.ok(checkpointIdx >= 0, 'checkpoint on surface')
  assert.ok(nodes[checkpointIdx + 1]! < checkpointSeq, `fixture is value-inverted: ${checkpointSeq} then ${nodes[checkpointIdx + 1]}`)

  // 修复前：resolveSurfaceRange(checkpoint, 后继小值节点) 抛 "reversed range 高..低"。
  // 修复后：位置递增 → 合法区间，端点原样返回。
  const next = nodes[checkpointIdx + 1]!
  const resolved = resolveSurfaceRange(session, checkpointSeq, next)
  assert.equal(resolved.start, checkpointSeq)
  assert.equal(resolved.end >= next, true)
})

test('non-monotonic: stale recovery over a value-inverted span stays positional', () => {
  const { session, checkpointSeq } = foldedFixture()
  // 起点 checkpoint 之前的一个已被 shadow 的 seq → 走 recoverStaleRange；
  // 值窗口 [min,max] 修复前按 [start..end] 计算为空 → 误报 already-compressed；
  // 修复后恢复出仍活着的非 checkpoint 内容。
  const nodes = (session.surface as { nodes: number[] }).nodes
  const last = nodes[nodes.length - 1]!
  const resolved = resolveSurfaceRange(session, checkpointSeq - 1, last)
  assert.equal(resolved.recovered, true)
  // checkpoint 节点绝不被 stale 恢复折叠（规则 7），但残余活节点在：
  assert.notEqual(resolved.start, checkpointSeq)
  assert.equal(nodes.indexOf(resolved.start)! <= nodes.indexOf(resolved.end)!, true)
})

test('non-monotonic: runCompactionTransaction lands a value-inverted positional span', () => {
  const { session, checkpointSeq } = foldedFixture()
  const nodes = (session.surface as { nodes: number[] }).nodes
  const checkpointIdx = nodes.indexOf(checkpointSeq)
  const end = nodes[nodes.length - 1]!
  assert.ok(nodes[checkpointIdx]! > end, 'fixture inverted')
  const shadowed = shadowedSeqsOf(session, checkpointSeq, end)
  assert.ok(isExactSurfaceSpan(session, checkpointSeq, end, shadowed), 'positional slice matches')
  const { seqs } = runCompactionTransaction(session, {
    start: checkpointSeq,
    end,
    shadowedSeqs: shadowed,
    summary: [{ type: 'text', text: `${SUMMARY}second fold` }],
    shadowedTokenCount: 10_000,
    provider: 'test-provider',
    model: 'test-model',
    tier: 1,
  })
  assert.equal(seqs.length > 0, true)
  assert.equal(nodes.includes(checkpointSeq), false, 'shadowed off the surface')
})

test('non-monotonic: a genuinely reversed call (position-inverted edges) still throws', () => {
  const { session } = foldedFixture()
  const nodes = (session.surface as { nodes: number[] }).nodes
  const a = nodes[2]!
  const b = nodes[nodes.length - 1]!
  assert.throws(
    () => resolveSurfaceRange(session, b, a),
    (error: Error) => error.message.includes(`reversed range ${b}..${a}`),
  )
})

test('non-monotonic: handleCompress lands blocks on a value-inverted surface (live end-to-end)', async () => {
  const env = makeEnv()
  const session = Session.create('test-nonmonotonic-e2e')
  appendTurn(session, 1)
  appendUser(session, big('q0', 0))
  for (let index = 0; index < 8; index += 1) {
    appendAssistant(session, big(`a${index}`, index + 1), 1, index)
  }
  const live = shadowedSeqsOf(session, 1, session.snapshotEvents().length - 1)
  const shadowed = live.slice(1, live.length - 3)
  runCompactionTransaction(session, {
    start: shadowed[0]!,
    end: shadowed[shadowed.length - 1]!,
    shadowedSeqs: shadowed,
    summary: [{ type: 'text', text: SUMMARY }],
    shadowedTokenCount: 10_000,
    provider: 'test-provider',
    model: 'test-model',
    tier: 1,
  })
  // 在非单调表面上再压一段（端到端走 makeTools 的 compress）：
  const nodes = (session.surface as { nodes: number[] }).nodes
  const checkpointSeq = nodes[1]!
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  const exec = {
    callId: 'call-nonmonotonic',
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent: { id: session.id, session, options: { provider: 'test-provider', model: 'test-model' }, ctx: { tokenMeter: undefined } },
  } as never
  const startSeq = checkpointSeq
  const endSeq = nodes[nodes.length - 1]!
  const out = await definition.execute({ content: [{ startSeq, endSeq, summary: SUMMARY }] } as never, exec)
  const text = typeof out === 'string' ? out : (out as { text: string }).text
  assert.match(text, /Compressed \d+ block\(s\)/)
  assert.doesNotMatch(text, /reversed range/)
})
