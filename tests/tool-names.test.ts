/**
 * A-3（2026-11-23）：区间行的工具名指纹。
 *
 * `condense`（A-1 的默认动作）由引擎选段，模型看不见候选里是什么，只能盲写
 * 摘要；区间表原来只有体积、tool/text 比例和 fold 价，回答不了「这段是读代码、
 * 跑命令还是查文档」。工具名是内容最便宜的指纹，让模型判断该不该折、摘要里该
 * 留住什么。
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { createAssistantMessage } from '@deepseek-ai/dsh-llm'
import { buildCompressibleSeqRanges, TOOL_NAME_LIMIT } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { appendAssistant, appendToolCall, appendToolResult, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'

/** An assistant node whose content is a tool call with an arbitrary name. */
function appendNamedCall(session: Session, text: string, callId: string, name: string): void {
  session.append('assistant/message', {
    turn: 1,
    step: 1,
    stream: [],
    message: createAssistantMessage({
      content: [
        { type: 'text', text },
        { type: 'tool-call', id: callId, name, arguments: '{}' },
      ],
      provider: 'test-provider',
      model: 'test-model',
    }),
  }, { surfaceOp: 'append' })
}

/** One foldable stretch carrying three different tools, two of them twice. */
function toolFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('task', 3))
  appendToolCall(session, longText('w1', 3), 'c1')
  appendToolResult(session, longText('r1', 3), 'c1')
  appendNamedCall(session, longText('w2', 3), 'c2', 'read')
  appendToolResult(session, longText('r2', 3), 'c2')
  appendToolCall(session, longText('w3', 3), 'c3')
  appendToolResult(session, longText('r3', 3), 'c3')
  appendNamedCall(session, longText('w4', 3), 'c4', 'read')
  appendToolResult(session, longText('r4', 3), 'c4')
  appendAssistant(session, longText('close', 3))
  appendTurn(session, 2)
  appendUser(session, 'tail question')
  return session
}

test('A-3: 区间行收集工具名并按次数降序（多工具去重）', () => {
  const session = toolFixture('a3-collect')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session) as never, { preserveRecent: 1 })
  assert.ok(ranges.length >= 1, 'the stretch is foldable')
  const tools = ranges[0]!.tools
  assert.deepEqual(
    tools.map((tool) => `${tool.name}×${tool.count}`),
    ['bash×2', 'read×2'],
    'both tools counted, ties broken by name',
  )
})

test('A-3: 没有工具调用的区间 tools 为空数组（渲染不出噪音）', () => {
  const session = Session.create('a3-empty')
  appendTurn(session, 1)
  appendUser(session, longText('task', 3))
  appendAssistant(session, longText('prose', 3))
  appendToolResult(session, longText('orphan', 1), 'never-called') // 会被 orphan 清理，不贡献名字
  appendTurn(session, 2)
  appendUser(session, 'tail question')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session) as never, { preserveRecent: 1 })
  for (const range of ranges) assert.deepEqual(range.tools, [], 'no tool call in this stretch')
})

test('A-3: 名称清单封顶 TOOL_NAME_LIMIT（每个名字都随每条请求计费）', () => {
  const session = Session.create('a3-cap')
  appendTurn(session, 1)
  appendUser(session, longText('task', 3))
  for (let index = 0; index < TOOL_NAME_LIMIT + 2; index += 1) {
    appendNamedCall(session, longText(`w${index}`, 3), `cap${index}`, `tool${index}`)
    appendToolResult(session, longText(`r${index}`, 3), `cap${index}`)
  }
  appendAssistant(session, longText('close', 3))
  appendTurn(session, 2)
  appendUser(session, 'tail question')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session) as never, { preserveRecent: 1 })
  assert.ok(ranges.length >= 1)
  assert.equal(ranges[0]!.tools.length, TOOL_NAME_LIMIT)
})

test('A-3: 工具名渲染进区间行（空则整段消失）', () => {
  // rangeTable 走默认 preserveRecent(5)：尾部补 4 个节点，让工具段落在保护窗外。
  const session = toolFixture('a3-render')
  for (let index = 0; index < 4; index += 1) appendAssistant(session, longText(`pad${index}`, 3))
  const withTools = rangeTable(session, wholeSurfaceRangeView(session) as never)
  assert.match(withTools, /tools: bash×2, read×2/, 'the row carries the fingerprint')

  const plain = Session.create('a3-render-plain')
  appendTurn(plain, 1)
  appendUser(plain, longText('task', 3))
  appendAssistant(plain, longText('prose', 3))
  for (let index = 0; index < 5; index += 1) appendAssistant(plain, longText(`pad${index}`, 3))
  const noTools = rangeTable(plain, wholeSurfaceRangeView(plain) as never)
  assert.doesNotMatch(noTools, /tools:/, 'no suffix when the stretch called nothing')
})
