/**
 * Model-free tool-result middle truncation — THIN ADAPTER over the HOST's own
 * pruner algorithm (`@deepseek-ai/dsh-compaction-tool-result-pruner`).
 *
 * The truncation arithmetic itself is NEVER re-implemented here: the host
 * `ToolResultPruner.pruneContent` (head 4096 + `\n\n[... tool result middle
 * pruned ...]\n\n` + tail 1024, code-point slicing, marker inserted once, empty
 * text blocks dropped, replacement asserted smaller and within threshold) is
 * BORROWED through a prototype-delegate that never instantiates the cordis
 * Service (`Object.create(ToolResultPruner.prototype)` + a `resolveConfig`
 * budget object). `PRUNE_MARKER` / `codePointLength` / `resolveConfig` are
 * imported from the same package, so the marker bytes and the slicing stay
 * byte-identical to the host by construction.
 *
 * What the adapter owns (and the host `pruneSession` cannot do for us —
 * verified: its service is realm-isolated, unreachable from the root plane,
 * and its scan is whole-surface with no spill/processed/threshold awareness):
 *  - runs at `agent/pre-step` AFTER the orphan strip and BEFORE the nudge
 *    build, only on NEW over-budget nodes (a per-session processed set dedups;
 *    no full-surface re-scan rework);
 *  - rows the host already SPILLED to a file (`spillPathsOfEvent`,
 *    src/messages.ts) are skipped — their bytes are already out of the
 *    context, truncating them again is pure loss;
 *  - the replacement is written as a 1:1 surface `replace` of the single node
 *    (len == 1, cache-invalidation minimal shape) plus the `compaction/prune`
 *    shadow-price event in the host token vocabulary (rule 12), the same
 *    protocol `hideSurfaceSeqs` uses (src/region.ts);
 *  - media blocks (image/file) never enter the text branch (host `pruneContent`
 *    keeps every non-text block verbatim);
 *  - `toolResultPruneChars: 0` disables the pass.
 * @module billion-context-dsh/tool-result-truncate
 */

import type { Session } from '@deepseek-ai/dsh-session'
import { PRUNE_MARKER, ToolResultPruner, codePointLength, resolveConfig } from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import { eventAtOf } from './session-events.ts'
import { spillPathsOfEvent, toolCallIdOfResultEvent } from './messages.ts'
import { hostPriceEvent } from './host-tokens.ts'

/** Marker written into the truncated middle — re-exported verbatim from the host package. */
export { PRUNE_MARKER as TRUNCATE_MARKER }
/** Default per-result text budget in Unicode code points (host parity). */
export const DEFAULT_TRUNCATE_CHARS = 8192

/** One pruned node, as the host's `PrunedEntry` reports it. */
export interface TruncateEntry {
  readonly originalSeq: number
  readonly replacementSeq: number
  readonly callId: string
  readonly charsBefore: number
  readonly charsAfter: number
}

export interface TruncateResult {
  readonly pruned: readonly TruncateEntry[]
  readonly charsRemoved: number
}

/** Host budget object carried by the prototype delegate (the Service fields stay unset). */
type PrunerDelegate = ToolResultPruner & { config: ReturnType<typeof resolveConfig> }

/**
 * A stateless instance of the HOST algorithm: prototype-delegate, no cordis
 * Service construction (which would register a second `toolResultPruner`
 * provider on this context — only one provider per isolation key may own a
 * name). The borrowed methods (`measureContent`, `pruneContent`) read only
 * `this.config`, so a prototype child carrying just the budget is sufficient.
 */
function hostPrunerDelegate(thresholdChars: number): PrunerDelegate {
  const delegate = Object.create(ToolResultPruner.prototype) as PrunerDelegate
  delegate.config = resolveConfig({ thresholdChars })
  return delegate
}

interface TextBlockShape {
  type?: unknown
  text?: unknown
}

function isTextBlock(block: unknown): block is { type: 'text'; text: string } {
  const b = block as TextBlockShape | null
  return typeof b === 'object' && b !== null
    && b.type === 'text' && typeof b.text === 'string'
}

/**
 * Build the truncated replacement content through the HOST `pruneContent`,
 * or `null` when the result is within budget.
 */
/**
 * Build the truncated replacement through the HOST `pruneContent`.
 *
 * The input layer matters: the host `pruneSession` runs `pruneContent` over
 * `session.deriveEventMessage(event).content` — the PROJECTED flat content
 * (text/image blocks at the top level). The raw `tool/result` event instead
 * nests the text inside its `tool-result` block (`message.content[0].content`),
 * so the adapter prunes EACH nested `tool-result` block's content and splices
 * the replacement back in place; blocks of any other shape pass through
 * untouched. Returns `null` when every nested content stayed within budget.
 */
export function truncateToolResultContent(
  content: ReadonlyArray<unknown>,
  pruner: PrunerDelegate,
): { content: ReadonlyArray<unknown>; charsBefore: number; charsAfter: number } | null {
  let charsBefore = 0
  let charsAfter = 0
  let changed = false
  const out: unknown[] = []
  for (const block of content) {
    const b = block as { type?: unknown; content?: unknown } | null
    if (b === null || typeof b !== 'object' || b.type !== 'tool-result' || !Array.isArray(b.content)) {
      out.push(block)
      continue
    }
    const nested = b.content as ReadonlyArray<unknown>
    for (const inner of nested) {
      if (isTextBlock(inner)) charsBefore += codePointLength(inner.text)
    }
    const pruned = pruner.pruneContent(nested as Parameters<PrunerDelegate['pruneContent']>[0])
    if (pruned === null) {
      out.push(block)
      continue
    }
    for (const inner of pruned) {
      if (isTextBlock(inner)) charsAfter += codePointLength(inner.text)
    }
    out.push({ ...b, content: pruned })
    changed = true
  }
  if (!changed) return null
  return { content: out, charsBefore, charsAfter }
}

/**
 * Run one truncation pass over the session surface.
 *
 * `processed` is the engine's per-session set of already-truncated surface
 * seqs — a node replaced by a previous pass must never be re-truncated (its
 * text already carries the marker and is below budget, but the dedup makes the
 * no-op explicit and O(1)).
 *
 * Writes, per pruned node, the same event pair `hideSurfaceSeqs` uses:
 * `compaction/prune` (shadow-price claim, host vocabulary) then the 1:1
 * `surfaceOp: replace` of the single node with the truncated `tool/result`
 * message.
 */
export function truncateToolResults(
  session: Session,
  processed: Set<number>,
  thresholdChars: number,
): TruncateResult {
  const pruned: TruncateEntry[] = []
  let charsRemoved = 0
  if (thresholdChars <= 0) return { pruned, charsRemoved }
  const pruner = hostPrunerDelegate(thresholdChars)
  for (const seq of session.surface.nodes) {
    if (processed.has(seq)) continue
    const event = eventAtOf(session, seq)
    if (event === undefined || event.type !== 'tool/result') continue
    // Skip rows the host already spilled to a file: their bytes are already
    // out of the context, and the nudge table marks them `N spilled` so the
    // model folds them first — truncating them again is pure loss.
    if (spillPathsOfEvent(event).length > 0) continue
    const rawMessage: unknown = (event.data as { message?: unknown }).message
    if (rawMessage === undefined || typeof rawMessage !== 'object') continue
    const message = rawMessage as { content?: ReadonlyArray<unknown> } & Record<string, unknown>
    if (message === undefined) continue
    const replacement = truncateToolResultContent(message.content ?? [], pruner)
    if (replacement === null) continue
    const callId = toolCallIdOfResultEvent(event) ?? ''
    // Shadow-price claim FIRST (host vocabulary — rule 12), then the 1:1
    // replace. Same order and same protocol as `hideSurfaceSeqs` (src/region.ts).
    session.append('compaction/prune', {
      shadowedRange: { start: seq, end: seq },
      shadowedSeqs: [seq],
      shadowedTokenCount: hostPriceEvent(event),
    })
    const replacementData = { ...event.data } as Record<string, unknown>
    replacementData.message = { ...message, content: replacement.content }
    const replacementEvent = session.append('tool/result', replacementData as typeof event.data, {
      surfaceOp: { op: 'replace', startSeq: seq, endSeq: seq },
      sourceEventSeqs: [seq],
    })
    // A 1:1 replace takes a NEW seq (the slot keeps one node, but the log seq
    // is fresh) — the processed set must remember the LIVE seq, or a later
    // pass would re-truncate the already-pruned node.
    processed.add(replacementEvent.seq)
    pruned.push({ originalSeq: seq, replacementSeq: replacementEvent.seq, callId, charsBefore: replacement.charsBefore, charsAfter: replacement.charsAfter })
    charsRemoved += replacement.charsBefore - replacement.charsAfter
  }
  return { pruned, charsRemoved }
}
