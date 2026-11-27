/** Condense copy: payback formatting, refusal/note text, note weight rules. */

/**
 * M3 — the three model tools: compress / recall / acp_status, registered
 * through `ctx.tools` (defineTool).
 *
 * compress is the heart of ACP: the model writes the summary and the tool
 * lands it as a durable surface replacement (no second LLM summarization
 * call). recall is the ONE way back into the log — either a SEARCH over the
 * block summaries, the shadowed originals and the still-visible surface
 * (`query`), or a READ of original bytes (`blockId` / `seq`); the two forms
 * are mutually exclusive, so the model face carries one schema instead of two.
 * acp_status reports the block ledger and pressure.
 * @module billion-context-dsh/tools
 */

import { defineTool, ToolArgsError, type ToolDefinition, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import { buildStatusReport, defaultCountTokens, searchBlocks, type CompressionCore, type MessageRole, type SearchDoc } from './kernel/index.ts'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { AcpStateStore } from './state.ts'
import { foldPricingFor, kernelConfigFor, type KernelConfigInput } from './config.ts'
import { resolveTokenCount } from './nudge.ts'
import { routeFor, type AcpWindow } from './window.ts'
import {
  AlreadyCompressedRangeError,
  buildCompressibleSeqRanges,
  blockIdOfKernelRef,
  blockRefForSummarySeq,
  belowWorkingSetFloor,
  MIN_CONDENSE_SPAN_TOKENS,
  blockRegistry,
  compactionIdsOfKernelBlocks,
  expandShadowedSeqs,
  guardedSurfaceSeqsOf,
  type KernelRangeView,
  type SeqCompressibleRange,
  isExactSurfaceSpan,
  rebuildBlockLedger,
  resolveSurfaceRange,
  runCompactionTransaction,
  liveCheckpointCarriersInSpan,
  shadowedSeqsOf,
  stripOrphanedSurfaceToolMessages,
  summarySeqOfKernelBlock,
  openToolCallIds,
  protectedSurfaceSeqs,
  sliceDecompressPage,
  surfaceSuffixTokens,
  surfaceSummary,
  DEFAULT_DECOMPRESS_PAGE,
  DEFAULT_DECOMPRESS_PAGE_CHARS,
  foldSpanPricing,
  type FoldPricing,
  type ResolvedSurfaceRange,
} from './region.ts'
import { allLogMessages, attachmentsOfEvent, buildToolCallIndex, classifySurfaceEvent, eventsToCoreMessages, extractEventText, isAgentInstructionsRow, isCheckpointNode, isRealUserTurn, isSkillCatalogRow, surfaceEventsOf, toolCallsOf, withSummaryFramePrefix } from './messages.ts'
import { shadowedTokensViaMeter } from './host-tokens.ts'
import { lastRequestInput, peakRequestInput, providerUsage, recordCompressCost, totalRequestInput } from './usage.ts'
import { eventAtOf, sessionEventsOf } from './session-events.ts'
import { DEFAULT_PROMPTS, DEFAULT_RESOLVED, DEFAULT_TOOL_PARAM_TEXTS, type ResolvedPrompts } from './prompts.ts'
import { callHint, type ResolvedCallStyle } from './ptc.ts'
import type { SettingsCommandSurface } from './settings.ts'
import type { PresetName } from './presets.ts'
import { CondensePick } from './condense-elect.ts'

/** `1.0` — one decimal, the way every refusal/recommendation prints a payback. */
export function formatPayback(foldCostTokens: number, spanTokens: number): string {
  if (spanTokens <= 0) return '∞'
  const value = foldCostTokens / spanTokens
  if (!Number.isFinite(value)) return '∞'
  return (Math.round(value * 10) / 10).toFixed(1)
}

/** Format a payback ratio the way the model-facing copy prints it (`2.3 req`). */
export function paybackLabel(foldCostTokens: number, spanTokens: number): string {
  if (spanTokens <= 0) return '—'
  const value = foldCostTokens / spanTokens
  if (!Number.isFinite(value)) return '—'
  return `${formatPayback(foldCostTokens, spanTokens)} req`
}

/**
 * Why no shape passed — with the arithmetic, per gate.
 *
 * The model cannot act on "nothing is foldable": it needs to know WHICH gate failed
 * and by how much, because two of the three are numeric thresholds it can influence
 * (fold a bigger stretch, or wait), while the third is a hard capability floor.
 */
export function condenseNoCandidateReason(input: {
  readonly startIndex: number
  readonly endIndex: number
  readonly nearest: {
    readonly startSeq: number
    readonly endSeq: number
    readonly tokens: number
    readonly keptTokens: number
    readonly foldCostTokens: number
    readonly failed: ReadonlyArray<'capacity'>
  }
  readonly surfaceTotal: number
  readonly floorTokens: number
  readonly payback: number | null
  readonly eligibleRangeCount: number
  readonly guardedRowCount: number
}): string {
  const { nearest } = input
  const lines: string[] = []
  // seq VALUES are locally non-monotonic (a checkpoint node carries a much higher seq
  // than its neighbours), so a ""seqs 20..12"" headline reads like a bug. Lead with the
  // positional run and name the seqs only as the anchorable identity.
  lines.push(`the closest shape covers surface rows ${input.startIndex}..${input.endIndex} (seqs ${nearest.startSeq}..${nearest.endSeq}): it removes ~${nearest.tokens} tokens`)
  if (nearest.failed.includes('capacity')) {
    if (nearest.tokens < MIN_CONDENSE_SPAN_TOKENS) {
      lines.push(`  · capacity gate: ~${nearest.tokens} tokens is under the ~${MIN_CONDENSE_SPAN_TOKENS}-token minimum a condense must reach to carry a handoff note at all`)
    } else {
      lines.push(`  · capacity gate: folding it leaves ~${nearest.keptTokens} live, under the working-set floor ~${input.floorTokens} (short by ~${Math.max(0, input.floorTokens - nearest.keptTokens)}). The floor is not bypassable by waiting — only \`force:true\` crosses it`)
    }
  }
  lines.push(`${input.eligibleRangeCount} eligible run(s), ${input.guardedRowCount} guarded row(s) refused by the guards (current instruction copies, the newest skill catalog, the active user turn). This election has NO fallback: it folds a shape that passes the capacity gate, or it folds nothing.`)
  return lines.join('\n')
}

/**
 * Refusal copy for a below-floor condense. The THREE branches ARE the point
 * (cross-review Q4): two-branch copy conflated "nothing wider is above the floor"
 * with "nothing is eligible at all", so a session whose reclaimable area is merely
 * FRAGMENTED by the guards was told to stop compressing — and a session with
 * nothing to fold was told to widen into a guard. Both are loops. Exported so
 * every branch is testable without building three fixtures.
 */
export function condenseFloorRefusal(pick: CondensePick, floor: number): string {
  const first = pick.spans[0]!
  const last = pick.spans[pick.spans.length - 1]!
  const payback = paybackLabel(pick.foldCostTokens, pick.tokens)
  // M-1 (2026-11-25): the refusal says HOW the engine elected, not just what it
  // priced — the span COUNT (a multi-span batch is the M-1 shape; a single run
  // means no batch cleared the gates either) and the batch's Σ, so the model can
  // judge whether forcing it is worth it.
  // F10 (cross-review 2026-11-25): judge the batch shape by the pick's ORIGIN
  // (`fromBatch`), not by span count — a one-run batch and a plain prefix are the
  // same single span, and a distillation is never a batch.
  const shape = pick.fromBatch
    ? `${pick.spans.length} spans (M-1 multi-span batch, walls kept on the surface)`
    : pick.kind === 'distill'
      ? `1 distilled range (cross-barrier tier 2)`
      : `1 span (no multi-span batch cleared the gates either)`
  const head = `compress: condense refused seqs ${first.start}..${last.end} — the engine elected ${shape} totaling ${pick.tokens}t; folding them costs ${pick.foldCostTokens}t of re-prefill (pays back in ${payback}) and leaves ~${pick.keptTokens} live.\n`
  const lines: string[] = []
  if (!pick.gates.capacity) {
    lines.push(`  · floor gate: ~${pick.keptTokens} live is under the working-set floor ~${floor} (short by ~${Math.max(0, floor - pick.keptTokens)}), so folding it strands the working set. Fold a SMALLER span instead — a fold leaves exactly as much room as it does NOT remove, so the widest available span is the one that strands it — or accept a shallow working set with force:true.`)
  }
  if (lines.length === 0) {
    // No gate is actually failing (only reachable if a caller ignored the election):
    // say so plainly instead of inventing a reason.
    lines.push(`  · every gate passes for this shape (~${pick.keptTokens} live, payback ${payback}) — it was not folded for another reason; retry it unchanged.`)
  }
  lines.push(`  Batch only spans you would fold anyway: one call pays ONE re-prefill measured from its EARLIEST rewrite point.`)
  return head + lines.join('\n')
}

/**
 * A condense note replaces a stretch the model never named, so it must carry the
 * handoff the model would have written by hand: what the work IS, and what runs
 * next. A note missing either heading turns one fold into a re-run of consumed
 * exploration, which costs more than the fold saved.
 */
// Tolerant on purpose: `## Task`, `# Now`, `**Now**`, `Now:` and the Chinese
// spellings all pass. A gate that rejects a formatting variant is a production
// break on a model whose note is otherwise correct (cross-review Q1).
// arXiv 2611.05306 **Appendix A** (G is the agent's prompt outline, not this). Its
// control found the best compactions are "cherry-picked briefs for further
// execution", never narratives of the trace, and the FIRST field is a verbatim or
// near-verbatim copy of the input task — we had zero coverage of it before
// 2026-11-24. NOW absorbs what we used to split across Goal / Next / Remaining
// plan (same Appendix A field: current occupation, intentions, planned next
// steps), so a note written to the old shape still passes NOW and only gains the
// one new heading.
export const CONDENSE_TASK = /(^|\n)\s*(?:#{1,4}|\*{1,3})?\s*[{[【]?\s*(task|instruction|request|\u4efb\u52a1|\u7528\u6237\u8981\u6c42|\u9700\u6c42)\s*[}\]】]?\s*[:：]?/i
export const CONDENSE_NOW = /(^|\n)\s*(?:#{1,4}|\*{1,3})?\s*[{[【]?\s*(now|current|status|goal|objective|next|next step|remaining plan|\u5f53\u524d|\u73b0\u72b6|\u76ee\u6807|\u4efb\u52a1\u72b6\u6001|\u4e0b\u4e00\u6b65|\u5269\u4f59\u8ba1\u5212|\u5b8c\u6210\u6761\u4ef6)\s*[}\]】]?\s*[:：]?/i

export function missingCondenseHeadings(note: string): string[] {
  const missing: string[] = []
  if (!CONDENSE_TASK.test(note)) {
    missing.push('## Task \u2014 a verbatim (or near-verbatim) copy of the task this stretch served. Copy the wording; do not retell what you did with it')
  }
  if (!CONDENSE_NOW.test(note)) {
    missing.push('## Now \u2014 what you are doing right now, why, and the immediate next step (done when <criterion>)')
  }
  return missing
}

/**
 * The four advisory fields of arXiv 2611.05306 Appendix A (Task and Now are the
 * two enforced above). ADVISORY, never a gate: a hard six-heading gate costs
 * resident prompt text plus a retry loop, and on an open-weight model that trade
 * measured worse than the missing fields (both cross-reviews rejected the hard
 * gate). A note missing the rest folds with a one-line warning.
 */
export const CONDENSE_FIELDS: ReadonlyArray<readonly [RegExp, string]> = [
  [/^\s*#{1,3}\s*(task|instruction|\u4efb\u52a1|\u7528\u6237\u8981\u6c42|\u9700\u6c42)/im, 'Task'],
  [/^\s*#{1,3}\s*(now|current|status|goal|objective|next|\u5f53\u524d|\u73b0\u72b6|\u76ee\u6807|\u4e0b\u4e00\u6b65)/im, 'Now'],
  [/^\s*#{1,3}\s*(plan|decomposition|\u8ba1\u5212|\u5206\u89e3|\u5b8c\u6210\u6807\u51c6)/im, 'Plan'],
  [/^\s*#{1,3}\s*(pointers?|artifacts?|files?|paths?|\u6307\u9488|\u4ea7\u7269|\u6587\u4ef6|\u8def\u5f84|\u6539\u52a8)/im, 'Pointers'],
  [/^\s*#{1,3}\s*(verification|verified|\u9a8c\u8bc1|\u5df2\u9a8c\u8bc1)/im, 'Verification'],
  [/^\s*#{1,3}\s*(dead ends|failed|\u6b7b\u8def|\u5931\u8d25\u5c1d\u8bd5)/im, 'Dead ends'],
]

export function condenseNoteGaps(note: string): string[] {
  return CONDENSE_FIELDS.filter(([pattern]) => !pattern.test(note)).map(([, name]) => name)
}

/**
 * Does the note actually COMPRESS? Three cross-reviews killed the earlier
 * post-fold duplication warning: it fires AFTER the fold lands, so obeying it
 * means rewriting the note and paying a second full re-prefill — costlier to act
 * on than to ignore — and lexical shingles flag exactly the verbatim identifiers,
 * paths and error strings a good note SHOULD carry.
 *
 * The replacement is deterministic, pre-fold and language-independent: compare
 * the note’s weight against the stretch it replaces. A note weighing a third of
 * the folded span is not a briefing, it is a second copy that stays resident in
 * every later request — the fold never earns its one-time re-prefill back.
 */
export const MAX_NOTE_WEIGHT_RATIO = 0.35

/**
 * What a handoff note should weigh on this host. F13 (cross-review 2026-11-25):
 * 400 is the value the NEW Appendix-A doctrine is EXPECTED to produce once the
 * prompt distribution settles — it sits inside the paper's 150-400t guidance
 * (median 185t). The old 2026-11-15 measurement (~915t) and the 2026-11-25
 * session samples (s2115 n=56, s5e5c n=33, p50 ~480t / p75 ~690t) were taken
 * under the OLD prompt doctrine; they describe a distribution the new prompts
 * are meant to retire, NOT a target. Re-measure on a distribution written under
 * the new doctrine before touching this number again.
 */
export const TYPICAL_HANDOFF_NOTE_TOKENS = 400

/**
 * Hard ceiling on a stored handoff note, independent of the weight ratio.
 *
 * The ratio alone does not bound a note: at `MAX_NOTE_WEIGHT_RATIO` a 84K stretch
 * admits 29K tokens, so it never fires on the folds that matter. Measured on a
 * real 13-block session (2026-11-24) the notes averaged ~1.78K and peaked at 5.2K
 * against a 900-token assumption — the stored area was underestimated by ~4K per
 * block, and 23.1K of a 161K surface was handoff prose. The paper's control
 * harness reports a 185-token median with p97.5 <= 2K, so a note past this
 * ceiling is a narrative, not the "brief for the work ahead" the prompt asks for.
 *
 * Set above the 400-token typical so a good note is never clipped, and low enough
 * to catch the 3-5K outliers that made the stored area grow.
 */
export const MAX_HANDOFF_NOTE_TOKENS = 1500

/**
 * M-1 (2026-11-25, verdict §6): the largest TOTAL a single condense call submits.
 * The offline replay measured a real 16-segment batch at ~235Kt — one note cannot
 * honestly cover that in one response. Above this ceiling the condense path
 * submits the FIRST sub-80Kt batch only and reports the rest as follow-up
 * batches; each later call pays its own re-prefill but writes a note that
 * actually describes what it folded.
 */
export const CONDENSE_BATCH_SPLIT_TOKENS = 80_000

/**
 * F11 (cross-review 2026-11-25): the forced election prefers a multi-span batch
 * over the best single run only when the batch folds STRICTLY more than this
 * multiple of the run's volume (and does not fail more gates). Rationale: a
 * forced fold means the operator accepted a shape risk on purpose, so the batch
 * must buy enough extra volume to justify abandoning the run the gate order
 * would have taken — 2× is the smallest whole multiple that reads as "clearly
 * more"; below it the operator meant the smaller shape.
 */
export const FORCED_BATCH_MIN_ADVANTAGE = 2


/**
 * Cut a note down to a token ceiling, keeping whole LINES from the top.
 *
 * A note written to the skeleton puts its load-bearing headings first (Task,
 * Now, then Pointers), so dropping the tail drops narrative — which is exactly
 * what arXiv 2611.05306 Appendix A says to leave out. Truncating costs nothing;
 * refusing costs the model a full rewrite, and output tokens are priced well
 * above cached input.
 */
export function truncateNoteToCeiling(note: string, maxTokens: number): { note: string; droppedTokens: number } {
  const kept: string[] = []
  let used = 0
  for (const line of note.split('\n')) {
    const lineTokens = defaultCountTokens(line) + 1 // + newline
    if (used + lineTokens > maxTokens) break
    kept.push(line)
    used += lineTokens
  }
  const truncated = kept.join('\n')
  return { note: truncated, droppedTokens: Math.max(0, defaultCountTokens(note) - defaultCountTokens(truncated)) }
}
/**
 * Re-exported from `region.ts` (single source: the NUDGE tests the same number to decide
 * whether every foldable gap is under it). Value: `TYPICAL_HANDOFF_NOTE_TOKENS /
 * MAX_NOTE_WEIGHT_RATIO` — the smallest span that can carry a note this host's models
 * actually write inside the weight cap. The kernel minimum (`minCompressRange`) is a
 * CHARACTER threshold (5000 chars ≈ 1250 tokens) and filters nothing; the real floor is the
 * note's weight (round-13 review Q5: a 750-token election, then the weight gate).
 */
export { MIN_CONDENSE_SPAN_TOKENS }

/**
 * Note weight, priced as the note is STORED: the summary frame prefix the checkpoint node
 * carries is part of what stays resident, so the ratio must count it or a note just under 35%
 * raw would exceed 35% once stored (round-13 review Q1). `storedTokens` lets a caller that
 * already counted the stored form pass it in instead of counting twice.
 */
export function noteWeightRatio(note: string, spanTokens: number, storedTokens?: number): number {
  // A zero-priced span cannot carry ANY note — returning 0 here read as "weightless" and
  // admitted a note onto a span with nothing to save (round-11 Q7).
  if (spanTokens <= 0) return Number.POSITIVE_INFINITY
  return (storedTokens ?? defaultCountTokens(withSummaryFramePrefix(note))) / spanTokens
}

export function noteWeightRejection(
  note: string,
  spanTokens: number,
  storedTokens?: number,
): string {
  const tokens = storedTokens ?? defaultCountTokens(withSummaryFramePrefix(note))
  const ratio = noteWeightRatio(note, spanTokens, tokens)
  if (!Number.isFinite(ratio)) {
    return (
      'compress: condense refused — the elected span holds ~0 tokens, so a note of any size can only ADD resident text.\n'
        + 'Nothing lands. Fold a stretch that actually carries consumed work (see the nudge range table or acp_status), or skip the fold.'
    )
  }
  return (
    'compress: condense refused — your note is ~' + Math.round(ratio * 100) + '% of the ' + spanTokens + ' tokens it replaces (limit ' + Math.round(MAX_NOTE_WEIGHT_RATIO * 100) + '%), counted as it is stored (' + tokens + ' tokens including the summary frame prefix).\n'
      + 'That is a second copy, not a briefing: it stays resident in every later request, so the fold never pays back its one-time re-prefill.\n'
      + 'Cut it to what a resumed session cannot re-derive: ## Task (verbatim), ## Now (next step + done-when), ## Pointers, ## Dead ends. Drop the running commentary. Nothing lands — rewrite the note and send it again.'
  )
}
