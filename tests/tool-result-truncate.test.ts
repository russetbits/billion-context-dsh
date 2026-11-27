/**
 * Tool-result middle truncation (src/tool-result-truncate.ts) — a thin adapter
 * that BORROWS the host `ToolResultPruner` algorithm via prototype delegation.
 *
 * Pinned here:
 *  - the pruned shape is the host's own: head + one marker + tail;
 *  - code-point slicing never splits a surrogate pair;
 *  - rows the host already SPILLED to a file are skipped (bytes already out);
 *  - media (image/file) blocks are kept verbatim;
 *  - a processed seq is never re-truncated (dedup);
 *  - `toolResultPruneChars: 0` disables the pass entirely;
 *  - the replacement is a 1:1 replace — surface length and node seqs do not
 *    shift, and the `compaction/prune` shadow-price event precedes it.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session, type SessionEvent } from '@deepseek-ai/dsh-session'
import { appendAssistant, appendToolCall, appendToolResult, appendTurn } from './helpers.ts'
import { PRUNE_MARKER } from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import { TRUNCATE_MARKER, truncateToolResults } from '../src/tool-result-truncate.ts'
import { eventAtOf } from '../src/session-events.ts'

const BUDGET = 8192

function longResult(label: string, length = BUDGET * 3): string {
  const unit = 'x'.repeat(64) + ` [${label}] `
  return unit.repeat(Math.ceil(length / unit.length)).slice(0, length)
}

function liveToolResultText(session: Session, callId: string): string {
  for (const seq of session.surface.nodes) {
    const event = eventAtOf(session, seq as number)
    if (event === undefined || event.type !== 'tool/result') continue
    const message = (event.data as { message?: { content?: Array<{ type: string; toolCallId?: string; content?: Array<{ type: string; text?: string }> }> } }).message
    const block = message?.content?.find((b) => b.type === 'tool-result' && b.toolCallId === callId)
    const text = block?.content?.map((c) => c.text ?? '').join('') ?? ''
    if (text !== '') return text
  }
  return ''
}

function appendRichToolResult(session: Session, callId: string, blocks: ReadonlyArray<unknown>, turn = 1, step = 1): void {
  session.append('tool/result', {
    turn,
    step,
    message: {
      id: `res-${callId}`,
      role: 'user',
      content: [{ type: 'tool-result', toolCallId: callId, content: [...blocks] }],
      source: { kind: 'tool', callId },
    },
  }, { surfaceOp: 'append' })
}

test('over-budget text is pruned into the host shape: head + one marker + tail', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  appendToolResult(session, longResult('big'), 'call-1', 1, 1)
  const result = truncateToolResults(session, new Set<number>(), BUDGET)
  assert.equal(result.pruned.length, 1)
  assert.equal(result.pruned[0]!.callId, 'call-1')
  const text = liveToolResultText(session, 'call-1')
  const first = text.indexOf(PRUNE_MARKER)
  assert.notEqual(first, -1, 'the host marker must be present')
  assert.equal(text.indexOf(PRUNE_MARKER, first + 1), -1, 'the marker must appear exactly once')
  assert.equal(text.length, 4096 + PRUNE_MARKER.length + 1024, 'head 4096 + marker + tail 1024 (host arithmetic)')
  assert.equal(TRUNCATE_MARKER, PRUNE_MARKER)
})

test('code-point slicing never splits a surrogate pair', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  // '😀' is 2 UTF-16 code units / 1 code point: a naive 4096-CODE-UNIT cut
  // would land mid-pair. The host slices by code POINT — the head it writes
  // must end on a code-point boundary.
  const text = 'a'.repeat(4090) + '😀'.repeat(200) + 'b'.repeat(5000)
  appendToolResult(session, text, 'call-1', 1, 1)
  truncateToolResults(session, new Set<number>(), BUDGET)
  const truncated = liveToolResultText(session, 'call-1')
  const head = truncated.slice(0, 4096)
  for (let i = 0; i < head.length; i++) {
    const code = head.charCodeAt(i)!
    if (code >= 0xd800 && code <= 0xdbff) {
      assert.ok(i + 1 < head.length && head.charCodeAt(i + 1)! >= 0xdc00, 'no high surrogate without its low half')
      i++
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      assert.fail('lone low surrogate in the head slice')
    }
  }
  // Encoding must not throw (clean UTF-8 = no lone surrogates).
  Buffer.from(head, 'utf8')
})

test('spilled rows are skipped — their bytes are already in a file', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  appendToolCall(session, '', 'call-2', 1, 1)
  // A PROSE spill notice pointing inside a real spill directory (the host form).
  const spilled = 'a'.repeat(9000) + '\n[output truncated; full output: /tmp/dsh-spill-abc/1-stdout.log]'
  appendToolResult(session, spilled, 'call-1', 1, 1)
  appendToolResult(session, longResult('normal'), 'call-2', 1, 1)
  const result = truncateToolResults(session, new Set<number>(), BUDGET)
  assert.equal(result.pruned.length, 1, 'only the non-spilled row is truncated')
  assert.equal(result.pruned[0]!.callId, 'call-2')
  assert.ok(liveToolResultText(session, 'call-1').includes('dsh-spill-abc'), 'the spilled row is untouched')
})

test('media blocks (image/file) are kept verbatim', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  const media = { type: 'image', mimeType: 'image/png', data: 'iVBORw0KGgo=' }
  appendRichToolResult(session, 'call-1', [media, { type: 'text', text: longResult('with-media') }])
  truncateToolResults(session, new Set<number>(), BUDGET)
  let image: unknown
  for (const seq of session.surface.nodes) {
    const event = eventAtOf(session, seq as number)
    if (event === undefined || event.type !== 'tool/result') continue
    const message = (event.data as { message?: { content?: Array<{ content?: Array<{ type: string }> }> } }).message
    const found = message?.content?.[0]?.content?.find((b) => b.type === 'image')
    if (found !== undefined) { image = found; break }
  }
  assert.deepEqual(image, media, 'the image block must survive the truncation verbatim')
})

test('a processed seq is never re-truncated (dedup)', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  appendToolResult(session, longResult('first'), 'call-1', 1, 1)
  const processed = new Set<number>()
  const first = truncateToolResults(session, processed, BUDGET)
  assert.equal(first.pruned.length, 1)
  const afterFirst = liveToolResultText(session, 'call-1')
  // The processed set already carries the seq from the first pass: the second
  // pass must be a no-op.
  const second = truncateToolResults(session, processed, BUDGET)
  assert.equal(second.pruned.length, 0)
  assert.equal(second.charsRemoved, 0)
  assert.equal(liveToolResultText(session, 'call-1'), afterFirst, 'no second marker / no rework')
})

test('toolResultPruneChars: 0 disables the pass entirely', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  appendToolResult(session, longResult('huge'), 'call-1', 1, 1)
  const result = truncateToolResults(session, new Set<number>(), 0)
  assert.equal(result.pruned.length, 0)
  assert.equal(result.charsRemoved, 0)
  assert.ok(liveToolResultText(session, 'call-1').startsWith('x'), 'the original bytes are still live')
})

test('the replacement is a 1:1 replace — slot count unchanged, the processed set carries the live seq', () => {
  const session = Session.create('detached')
  appendTurn(session, 1)
  appendAssistant(session, 'go', 1, 1)
  appendToolCall(session, '', 'call-1', 1, 1)
  appendToolResult(session, longResult('to-truncate'), 'call-1', 1, 1)
  const lengthBefore = session.surface.nodes.length
  const resultSeq = session.surface.nodes[lengthBefore - 1] as number
  const processed = new Set<number>()
  const result = truncateToolResults(session, processed, BUDGET)
  assert.equal(session.surface.nodes.length, lengthBefore, '1:1 replace: the surface keeps exactly one node per replaced slot')
  assert.equal(result.pruned[0]!.originalSeq, resultSeq)
  // The replace takes a NEW log seq for the SAME slot — the processed set must
  // remember the live one or a later pass re-truncates the already-pruned node.
  const replacementSeq = result.pruned[0]!.replacementSeq
  assert.notEqual(replacementSeq, resultSeq, 'a surfaceOp replace lands a fresh event seq in the slot')
  assert.ok(processed.has(replacementSeq), 'the processed set carries the LIVE replacement seq')
  const event: SessionEvent | undefined = eventAtOf(session, replacementSeq)
  assert.equal(event?.type, 'tool/result', 'the slot still holds a tool/result event')
  const pruned = (event!.data as { message?: { content?: Array<{ content?: Array<{ type: string; text?: string }> }> } }).message
  const text = (pruned?.content?.[0]?.content ?? []).map((b) => b.text ?? '').join('')
  assert.ok(text.includes(PRUNE_MARKER))
})
