/** Condense ELECTION: geometry, ONE capacity gate, pick/miss + refusal assembly. */

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
import { condenseNoCandidateReason, MAX_NOTE_WEIGHT_RATIO, TYPICAL_HANDOFF_NOTE_TOKENS, FORCED_BATCH_MIN_ADVANTAGE, noteWeightRatio } from './condense-note.ts'

export interface CondenseSpan {
  readonly start: number
  readonly end: number
  readonly endBlock: string | null
  readonly carriers: ReadonlyArray<string>
  readonly tokens: number
}

/** The foldable PREFIX a condense submits, priced as ONE rewrite. */
export interface CondensePick {
  /**
   * The ranges to submit, oldest-first. ONE range carries the whole prefix; a
   * SECOND, gap-only range appears when the prefix ends on a gap that sits after
   * the last carrier (`condenseRangesOf` explains why the two cannot merge).
   */
  readonly spans: ReadonlyArray<CondenseSpan>
  readonly tokens: number
  /** Note tokens that stay resident after the fold — one note per submitted range. */
  readonly noteTokens: number
  readonly keptTokens: number
  readonly floorTokens: number
  readonly belowFloor: boolean
  /**
   * Whether ONE eligible row folded ALONE still leaves the floor. A refusal that
   * tells the model to fold bigger when nothing bigger is eligible is a loop
   * generator: every wider span is guarded (cross-review Q3).
   */
  readonly aboveFloorAvailable: boolean
  /**
   * How many surface rows the guards refuse (current instruction copies, the
   * newest skill catalog, the active user turn). A session that is wide but
   * walled is NOT stuck — telling it to fold wider sends it into the guard
   * (cross-review Q4).
   */
  readonly guardedRowCount: number
  /** How many eligible ranges the geometry produced at all (0 = nothing clears the kernel minimum). */
  readonly eligibleRangeCount: number
  /** Active blocks this fold distils (summed over `spans`). */
  readonly carrierCount: number
  /** What ONE fold re-prefills: every token after the LAST span's end. */
  readonly foldCostTokens: number
  /** `foldCostTokens / tokens` — requests needed for the permanent saving to repay the one-time rewrite. */
  readonly paybackRequests: number
  /**
   * What the fold DOES. `fold` replaces consumed prose (the normal case);
   * `distill` is the cross-barrier fallback: it folds the gaps AND distils every
   * active carrier inside the span into one block, which is a TIER 2 operation —
   * the originals behind those carriers stop being recoverable at message level.
   */
  readonly kind: 'fold' | 'distill'
  /** Blocks this fold distils into one (0 for a plain fold). */
  readonly distilledBlocks: number
  /**
   * True when the elected shape came from a `batchGeometryAt` candidate (the M-1
   * multi-span batch path, normal or forced) — NOT from the plain-run election or
   * the cross-barrier distillation. F10 (cross-review 2026-11-25): the refusal
   * copy must judge the batch shape by its ORIGIN, not by `spans.length`, because
   * a one-run batch and a plain prefix are the same single span.
   */
  readonly fromBatch: boolean
  /** The ONE gate verdict (M-3 2026-11-25): capacity. Payback/cost stay computed
   *  for display (`paybackRequests`, `foldCostTokens`) but gate nothing. */
  readonly gates: {
    /** Folded span is big enough to carry a handoff AND the remainder stays above the floor. */
    readonly capacity: boolean
  }
}

export interface CondenseMiss {
  readonly reason: string
  readonly candidates: SeqCompressibleRange[]
  /**
   * The geometry that came CLOSEST to passing, with every gate's numbers. Present
   * whenever any geometry existed: the refusal has to say which gate failed and by
   * how much, or the model is left guessing (and retries the same shape).
   */
  readonly nearest?: {
    readonly start: number
    readonly end: number
    readonly tokens: number
    readonly keptTokens: number
    readonly floorTokens: number
    readonly foldCostTokens: number
    readonly paybackRequests: number
    readonly failed: ReadonlyArray<'capacity'>
  }
}

/** Position of a surface seq, or -1 when it is not on the surface. */
export function indexOfSurfaceSeq(session: Session, seq: number): number {
  const nodes = session.surface.nodes
  for (let index = 0; index < nodes.length; index += 1) if ((nodes[index] as number) === seq) return index
  return -1
}

/** One entry of the prefix timeline: a compressible gap run or one block carrier. */
interface CondenseTimelineItem {
  readonly index: number
  readonly start: number
  readonly end: number
  readonly tokens: number
  /** Kernel block id when this item is an active block's checkpoint carrier. */
  readonly block: string | null
}

/**
 * How many note tokens the fold is assumed to keep resident. The batch writes ONE
 * shared note that every submitted range carries (F1, cross-review 2026-11-25) —
 * the model does not write N summaries for an N-span batch — so the budget is ONE
 * note's size, never ×N. Capped at `MAX_NOTE_WEIGHT_RATIO × Σ` so the floor
 * arithmetic below and the write-time weight gate (which compares against the
 * whole batch Σ) cannot disagree about the same fold.
 */
export function condenseNoteBudget(tokens: number): number {
  return Math.max(0, Math.min(TYPICAL_HANDOFF_NOTE_TOKENS, Math.floor(MAX_NOTE_WEIGHT_RATIO * tokens)))
}

/**
 * Split the elected prefix into the ranges `applyCompression` receives.
 *
 * WHY one range is the cheapest shape — barriers, re-prefills and kept AREA, not
 * prefix-cache reuse. Measured 2026-11-24 on this provider: a live session reported
 * 88,775,441 uncached input tokens with 0 cache read and 0 cache write, so a
 * cache-hit saving has never actually paid out here; do not argue the design from
 * it. The three costs that DO show up are: one kernel block per range (= one barrier
 * each), one re-prefill per submitted range, and the note text staying resident —
 * and all three scale with the range count, not with how much the fold removes.
 *
 * WHY SO FEW RANGES — this is the load-bearing fact of the whole design:
 * `applyCompression` creates ONE BLOCK PER RANGE (`applySingleRange`,
 * node_modules/acp-kernel/dist/index.js), so submitting one range per block and
 * one per gap would turn a 50-block session into 95 blocks AND copy the single
 * model-written note 95 times (~85K resident tokens). The fold would cost more
 * than it saves. ONE range distilling every carrier collapses 50 blocks into 1.
 *
 * A carrier arm is only REQUIRED when a plain seq range would cross a live
 * checkpoint carrier: `src/tools.ts` refuses that (`liveCheckpointCarriersInSpan`
 * runs only when NEITHER edge is a block ref), while a range with a `bN` END arm
 * is exactly the tier 2/3 distillation call the refusal points the model at.
 * Measured 2026-11-24: `{gapStart .. b2}` (mixed edges) distilled two carriers
 * into one block, and `{b1 .. b3}` distilled three into one.
 */
export function condenseRangesOf(items: ReadonlyArray<CondenseTimelineItem>): CondenseSpan[] {
  if (items.length === 0) return []
  const carriersIn = (from: number, to: number): string[] => {
    const ids: string[] = []
    for (let i = from; i <= to; i += 1) {
      const block = items[i]!.block
      if (block !== null) ids.push(block)
    }
    return ids
  }
  const tokensIn = (from: number, to: number): number => {
    let sum = 0
    for (let i = from; i <= to; i += 1) sum += items[i]!.tokens
    return sum
  }
  let lastCarrier = -1
  for (let i = 0; i < items.length; i += 1) if (items[i]!.block !== null) lastCarrier = i
  const last = items[items.length - 1]!
  // No carrier at all: the prefix is pure gap, so a plain seq range is legal
  // (it crosses no live carrier).
  if (lastCarrier < 0) {
    return [{
      start: items[0]!.start,
      end: last.end,
      endBlock: null,
      carriers: [],
      tokens: tokensIn(0, items.length - 1),
    }]
  }
  const mainEnd = items[lastCarrier]!
  const spans: CondenseSpan[] = [{
    start: items[0]!.start,
    end: mainEnd.end,
    endBlock: mainEnd.block,
    carriers: carriersIn(0, lastCarrier),
    tokens: tokensIn(0, lastCarrier),
  }]
  // A gap that sits AFTER the last carrier cannot join the main range: the main
  // range's terminal arm must stay a block ref (that is what makes it legal), and
  // extending past it would leave BOTH arms as plain seqs while still crossing
  // every carrier inside. It is submitted as its own gap-only range instead — it
  // crosses no carrier, so the guard does not apply.
  if (lastCarrier < items.length - 1) {
    const tailFrom = lastCarrier + 1
    spans.push({
      start: items[tailFrom]!.start,
      end: last.end,
      endBlock: null,
      carriers: [],
      tokens: tokensIn(tailFrom, items.length - 1),
    })
  }
  return spans
}

/**
 * The foldable PREFIX of the current surface, for condense.
 *
 * Geometry is the range table's own (buildCompressibleSeqRanges, rule 3), so a
 * condense can never shadow a row the durable surface forbids; the timeline adds
 * the ACTIVE block carriers, which that function deliberately excludes (it
 * flushes on every checkpoint). The election walks the merged timeline
 * oldest-first and stops at the first wall — a guarded row (current instruction
 * copy, newest skill catalog, active user turn) or the working-set floor. It
 * never SKIPS a wall: a prefix with a hole in it could not be submitted as one
 * range, which is the whole point of the design.
 *
 * How long the prefix turns out to be is decided by the guards, not by the block
 * count: on a real 87-node session (4 carriers on the surface, 41K tokens) the
 * oldest foldable prefix crossed exactly ONE carrier, because the last real user
 * turn sat near the front and every later carrier was inside the protected window.
 * A 50-block synthetic surface (tests/prefix-fold.test.ts P-8) crosses all 50. So
 * the useful question is never "how many blocks exist" but "how many of them are on
 * the OLD side of the guards" — `carrierCount` is what the pick reports.
 *
 * What the fold costs is REPORTED (`CondensePick.tokens` against the note), never
 * an election gate: the engine cannot predict how many requests a session has
 * left, and a cost model it cannot validate only removes folds a real session
 * needs. The floor check is `belowWorkingSetFloor` (arXiv 2611.05306 Appendix D),
 * the note-size check happens at write time (`noteWeightRatio`).
 */
export function pickCondenseSpan(
  session: Session,
  turn: ReturnType<CompressionCore['processTurn']>,
  preserveRecent: number,
  foldPricing?: FoldPricing,
  election?: { readonly force?: boolean },
): CondensePick | CondenseMiss {
  const byRaw = turn.state.messageRefs.byRaw
  const nodes = session.surface.nodes
  let firstSeq: number | undefined
  let lastSeq: number | undefined
  let refCount = 0
  for (const seq of nodes) {
    if (byRaw[String(seq)] === undefined) continue
    refCount += 1
    firstSeq ??= seq
    lastSeq = seq
  }
  if (firstSeq === undefined || lastSeq === undefined) {
    return {
      reason: 'the kernel holds no ref for any of the ' + nodes.length + ' live surface nodes \u2014 nothing has been projected yet',
      candidates: [],
    }
  }
  const view: KernelRangeView = {
    ranges: [{ startRef: byRaw[String(firstSeq)]!, endRef: byRaw[String(lastSeq)]! }],
    refs: { byRef: turn.state.messageRefs.byRef },
  }
  const ranges = buildCompressibleSeqRanges(session, view, {
    preserveRecent,
    foldStaleInstructionRows: true,
    ...(foldPricing === undefined ? {} : { foldPricing }),
  })
  const floorTokens = foldPricing?.floorTokens ?? 0
  // M-3 (2026-11-25): ONE gate. The payback/cost ceilings were removed as gates
  // (never independently decisive on a live session; the batch pricing covers the
  // "white fold" risk). Payback stays COMPUTED for display, never compared.
  const suffix = surfaceSuffixTokens(session)
  const surfaceTotal = suffix[0] ?? 0
  const indexOfSeq = new Map<number, number>()
  for (let index = 0; index < nodes.length; index += 1) indexOfSeq.set(nodes[index]!, index)
  const nodeTokens = (index: number): number => Math.max(0, (suffix[index] ?? 0) - (suffix[index + 1] ?? 0))
  const items: CondenseTimelineItem[] = []
  for (const range of ranges) {
    const index = indexOfSeq.get(range.start)
    if (index === undefined) continue
    items.push({ index, start: range.start, end: range.end, tokens: range.tokens, block: null })
  }
  for (const entry of blockRegistry(session)) {
    if (!entry.active || entry.summarySeq === null) continue
    const index = indexOfSeq.get(entry.summarySeq)
    // A carrier that is not on the surface cannot be folded. One numerically inside
    // a gap run CAN happen — seq values are NOT positionally monotonic (a checkpoint
    // node carries a higher seq than its neighbours, 2026-11-25 M1-4 test fixture:
    // carrier seq 21 sat numerically inside the gap run 2..7 while sitting
    // positionally AFTER it) — so the guard must compare POSITIONS, or the carrier
    // is priced twice (once inside the gap's tokens, once as its own item).
    if (index === undefined) continue
    if (items.some((item) => {
      const itemFrom = item.index
      const itemTo = indexOfSeq.get(item.end) ?? item.index
      return index >= Math.min(itemFrom, itemTo) && index <= Math.max(itemFrom, itemTo)
    })) continue
    items.push({
      index,
      start: entry.summarySeq,
      end: entry.summarySeq,
      tokens: nodeTokens(index),
      block: entry.kernelBlockId,
    })
  }
  items.sort((a, b) => a.index - b.index)

  // ---------------------------------------------------------------------------
  // CANDIDATE GEOMETRY
  //
  // Every foldable shape is generated ONCE and judged by the same three gates, so
  // there is no "prefix first, and when that fails try something else" order — that
  // fallback is gone, and with it the note/stretch drift it caused (the model's note
  // described the prefix while the engine folded a different row).
  //
  // A shape is a CONTIGUOUS run of timeline items: it starts at a gap run or a
  // carrier and ends at another one, and it may not cross a wall (a protected node:
  // the recent tail, the last real user turn, the newest instruction/catalog rows,
  // or a guarded row). Positions decide that, never seq values — the surface is
  // locally non-monotonic (a checkpoint node carries a much higher seq than its
  // neighbours), which is exactly the bug `recoverStaleRange` was fixed for.
  // ---------------------------------------------------------------------------
  const guarded = guardedSurfaceSeqsOf(session)
  const protectedSeqs = protectedSurfaceSeqs(session, preserveRecent)
  const wallPositions: number[] = []
  for (const seq of protectedSeqs) {
    const index = indexOfSeq.get(seq)
    if (index !== undefined) wallPositions.push(index)
  }
  for (const seq of guarded) {
    const index = indexOfSeq.get(seq)
    if (index !== undefined) wallPositions.push(index)
  }
  wallPositions.sort((a, b) => a - b)
  /** First wall STRICTLY after `position` (Infinity when none). */
  const firstWallAfter = (position: number): number => {
    let low = 0
    let high = wallPositions.length
    while (low < high) {
      const mid = (low + high) >> 1
      if (wallPositions[mid]! <= position) low = mid + 1
      else high = mid
    }
    return low < wallPositions.length ? wallPositions[low]! : Number.POSITIVE_INFINITY
  }
  const blockedItem = (item: CondenseTimelineItem): boolean => guarded.has(item.start) || guarded.has(item.end)

  interface Geometry {
    readonly startSeq: number
    readonly endSeq: number
    /**
     * Whether this shape STARTS at the compressible prefix's start (a cut inside the
     * first range counts — it folds the oldest bytes, just fewer of them). The model
     * writes its note BEFORE the call, so it can only predict the shape it asked for:
     * a shape that starts anywhere else would fold a stretch the note does not
     * describe (note drift), and that costs far more than a few thousand tokens.
     */
    readonly isPrefix: boolean
    readonly tokens: number
    readonly noteTokens: number
    readonly keptTokens: number
    readonly foldCostTokens: number
    /** Timeline item span, or -1 for a span CUT inside the first range. */
    readonly firstItem: number
    readonly lastItem: number
    /** Pre-built span for a CUT geometry (a cut is always ONE plain-seq span). */
    readonly cutSpan?: CondenseSpan
    readonly failed: ReadonlyArray<'capacity'>
  }

  const gatesOf = (tokens: number, keptTokens: number): {
    capacity: boolean
  } => ({
    // Two capacity conditions in one gate: the stretch must be able to CARRY a handoff
    // note at all (the stored-weight cap would reject a smaller one at write time),
    // and folding it must leave the working set above the floor.
    capacity: tokens >= MIN_CONDENSE_SPAN_TOKENS
      && !belowWorkingSetFloor(surfaceTotal, keptTokens, floorTokens),
    // Payback period, derived from the geometry alone (no session-length forecast):
    // the re-prefill is paid ONCE and the saving applies to every later request, so
    // the fold starts paying for itself after `cost / saving` requests. Compared as a
    // product (`cost <= span × ceiling`) to keep the default-2 boundary exact.
  })
  const failedOf = (gates: { capacity: boolean }): Geometry['failed'] => (
    gates.capacity ? [] : ['capacity']
  )

  // The prefix start is the OLDEST foldable item — the shape a caller can predict.
  const prefixStart = items.find((item) => !blockedItem(item))?.start
  const cumulative: number[] = [0]
  for (const item of items) cumulative.push(cumulative[cumulative.length - 1]! + item.tokens)
  const spanCountOf = (from: number, to: number): number => {
    let lastCarrier = -1
    for (let index = from; index <= to; index += 1) if (items[index]!.block !== null) lastCarrier = index
    if (lastCarrier < 0) return 1
    // A gap AFTER the last carrier cannot join the main range (its terminal arm must
    // stay a block ref), so it becomes its own span — exactly what condenseRangesOf
    // does, and the note budget must match it.
    return lastCarrier === to ? 1 : 2
  }
  const geometries: Geometry[] = []
  for (let first = 0; first < items.length; first += 1) {
    const head = items[first]!
    if (blockedItem(head)) continue
    const limit = firstWallAfter(head.index)
    for (let last = first; last < items.length; last += 1) {
      const tail = items[last]!
      if (blockedItem(tail)) break
      // The run may not reach or cross a wall.
      if (tail.index >= limit) break
      const tokens = cumulative[last + 1]! - cumulative[first]!
      const spanCount = spanCountOf(first, last)
      const noteTokens = condenseNoteBudget(tokens)
      const keptTokens = Math.max(0, surfaceTotal - tokens - noteTokens)
      const endIndex = indexOfSeq.get(tail.end) ?? tail.index
      const foldCostTokens = suffix[endIndex + 1] ?? 0
      const gates = gatesOf(tokens, keptTokens)
      geometries.push({
        startSeq: head.start,
        endSeq: tail.end,
        isPrefix: head.start === prefixStart,
        tokens,
        noteTokens,
        keptTokens,
        foldCostTokens,
        firstItem: first,
        lastItem: last,
        failed: failedOf(gates),
      })
    }
  }

  // CUT INSIDE THE FIRST RANGE (lead's key point 2). When the oldest eligible range
  // is itself too big to fold — it would strand the floor, or blow the cost ceiling —
  // the right shape is a PREFIX OF THAT RANGE, not a different row. Cuts land on
  // node boundaries and go through `resolveSurfaceRange`, so the edges stay
  // tool-pairing balanced and anchorable.
  const cutFirstRange = (): Geometry | undefined => {
    const head = items.find((item) => !blockedItem(item))
    if (head === undefined) return undefined
    const headIndex = indexOfSeq.get(head.start)
    const headEndIndex = indexOfSeq.get(head.end)
    if (headIndex === undefined || headEndIndex === undefined) return undefined
    const limit = firstWallAfter(headIndex)
    const stop = Math.min(headEndIndex, limit - 1)
    let best: Geometry | undefined
    for (let position = headIndex; position <= stop; position += 1) {
      const endSeq = nodes[position] as number
      let resolved: { start: number; end: number }
      try {
        resolved = resolveSurfaceRange(session, head.start, endSeq)
      } catch {
        continue
      }
      const from = indexOfSeq.get(resolved.start)
      const to = indexOfSeq.get(resolved.end)
      if (from === undefined || to === undefined || to < from) continue
      const tokens = Math.max(0, (suffix[from] ?? 0) - (suffix[to + 1] ?? 0))
      const noteTokens = condenseNoteBudget(tokens)
      const keptTokens = Math.max(0, surfaceTotal - tokens - noteTokens)
      const foldCostTokens = suffix[to + 1] ?? 0
      const gates = gatesOf(tokens, keptTokens)
      if (failedOf(gates).length > 0) continue
      // Keep growing: the LARGEST cut that still passes all three gates is the one
      // that reclaims the most while staying legal.
      if (best !== undefined && tokens <= best.tokens) continue
      best = {
        startSeq: resolved.start,
        endSeq: resolved.end,
        // A cut inside the first range IS the prefix, just shorter: it starts where
        // the full prefix starts, so the note the model wrote still describes it.
        isPrefix: true,
        tokens,
        noteTokens,
        keptTokens,
        foldCostTokens,
        firstItem: -1,
        lastItem: -1,
        cutSpan: { start: resolved.start, end: resolved.end, endBlock: null, carriers: [], tokens },
        failed: [],
      }
    }
    return best
  }

  /**
   * CROSS-BARRIER FALLBACK — fold the older side of the surface in ONE transaction,
   * ending on the LAST active carrier's block ref. That single range folds every gap's
   * prose AND distils every carrier inside it into ONE block (kernel `applyCompression`
   * treats a range with a block-ref edge as tier 2/3), which is the only shape that
   * shrinks a surface made of MANY sub-minimum gaps: each gap alone is under
   * `MIN_CONDENSE_SPAN_TOKENS`, so no plain geometry can ever qualify.
   *
   * It runs ONLY when every normal geometry failed, and it is deliberately NOT judged
   * by the three gates: a fragment surface is exactly the case where every foldable
   * stretch is tiny, so the gates that exist to protect the working set would reject
   * the one move that reopens it. The kernels' own carrier rule is the guard here —
   * both edges must resolve (start plain seq, end block ref) or `resolveCompletedSpan`
   * refuses at write time.
   *
   * Cost: tier 2 is a REAL information loss — the originals behind those carriers stop
   * being readable at message level. That is why `kind: 'distill'` makes the result line
   * say so explicitly instead of folding silently.
   */
  const crossBarrierGeometry = (): { geometry: Geometry; spans: CondenseSpan[] } | undefined => {
    const carriers = items
      .map((item, index) => ({ item, index }))
      .filter((entry) => entry.item.block !== null && !blockedItem(entry.item))
    // Two carriers is the floor: one carrier needs no merge, and distilling a single
    // block into itself would only lose its originals for no gain.
    if (carriers.length < 2) return undefined
    const lastCarrier = carriers[carriers.length - 1]!
    const first = items.findIndex((item) => !blockedItem(item))
    if (first < 0 || lastCarrier.index <= first) return undefined
    // The run may not cross a wall (a real user turn, a current instruction copy, the
    // active turn) — nothing here may fold those.
    if (firstWallAfter(items[first]!.index) <= items[lastCarrier.index]!.index) return undefined
    const slice = items.slice(first, lastCarrier.index + 1)
    const spans = condenseRangesOf(slice)
    if (spans.length === 0 || spans[spans.length - 1]!.endBlock === null) return undefined
    const folded = spans.reduce((total, span) => total + span.tokens, 0)
    const noteTokens = condenseNoteBudget(folded)
    const keptTokens = Math.max(0, surfaceTotal - folded - noteTokens)
    const endSeq = spans[spans.length - 1]!.end
    const endIndex = indexOfSeq.get(endSeq)
    const foldCostTokens = endIndex === undefined ? 0 : (suffix[endIndex + 1] ?? 0)
    return {
      geometry: {
        startSeq: spans[0]!.start,
        endSeq,
        isPrefix: spans[0]!.start === prefixStart,
        tokens: folded,
        noteTokens,
        keptTokens,
        foldCostTokens,
        firstItem: first,
        lastItem: lastCarrier.index,
        failed: [],
      },
      spans,
    }
  }

  /**
   * MULTI-SPAN BATCHES (M-1, 2026-11-25): for every candidate lower bound i, the
   * batch folds EVERY unblocked item from i to the surface's foldable end — one
   * span per wall-free run, each run's carriers distilled via `condenseRangesOf`.
   * Verdict `m1-offline-verdict.md` §2/§3: walls + checkpoints are BARRIERS that
   * split the surface into per-segment ranges, and the full batch folds ALL of
   * them in ONE rewrite (~99% of the suffix at ~0.11 payback) — the wall itself
   * never folds (no span covers it), but spans live on BOTH sides of it, which
   * the `content` batch path already supports (M-1b precedent).
   */
  const batchGeometryAt = (first: number): { geometry: Geometry; spans: CondenseSpan[] } | undefined => {
    if (blockedItem(items[first]!)) return undefined
    // Runs of consecutive unblocked items; a run breaks at a blocked item AND at
    // any wall SITTING BETWEEN two items (a wall seq is never an item, so the
    // positional gap is the only signal it is there).
    const runs: Array<{ from: number; to: number }> = []
    let runFrom = -1
    let runTo = -1
    for (let index = first; index < items.length; index += 1) {
      const item = items[index]!
      if (blockedItem(item)) {
        if (runFrom >= 0) {
          runs.push({ from: runFrom, to: runTo })
          runFrom = -1
        }
        continue
      }
      if (runFrom >= 0) {
        // A wall between the previous item and this one splits the run: the wall
        // row itself must stay on the surface (no span may cover it).
        const previous = items[runTo]!
        const hasWallBetween = wallPositions.some((wall) => wall > previous.index && wall < item.index)
        if (hasWallBetween) {
          runs.push({ from: runFrom, to: runTo })
          runFrom = -1
        }
      }
      if (runFrom < 0) runFrom = index
      runTo = index
    }
    if (runFrom >= 0) runs.push({ from: runFrom, to: runTo })
    if (runs.length === 0) return undefined
    const spans: CondenseSpan[] = []
    for (const run of runs) {
      const runItems = items.slice(run.from, run.to + 1)
      if (runItems.some((item) => blockedItem(item))) continue
      const runSpans = condenseRangesOf(runItems)
      // One run that cannot resolve (e.g. its whole body is blocked) is SKIPPED,
      // not fatal: the other wall-free runs still fold (F8/F12, cross-review
      // 2026-11-25 — an empty run must not void the whole batch).
      if (runSpans.length === 0) continue
      spans.push(...runSpans)
    }
    if (spans.length === 0) return undefined
    const folded = spans.reduce((total, span) => total + span.tokens, 0)
    if (folded <= 0) return undefined
    const noteTokens = condenseNoteBudget(folded)
    const keptTokens = Math.max(0, surfaceTotal - folded - noteTokens)
    const endIndex = indexOfSeq.get(spans[spans.length - 1]!.end)
    // Fail-closed (F7, cross-review 2026-11-25): if the last span's end seq has no
    // suffix entry the re-prefill price is UNKNOWN, not zero — a 0 would let the
    // candidate pass the payback and cost gates on a number the engine never priced.
    if (endIndex === undefined) return undefined
    const foldCostTokens = suffix[endIndex + 1] ?? 0
    return {
      geometry: {
        startSeq: spans[0]!.start,
        endSeq: spans[spans.length - 1]!.end,
        isPrefix: spans[0]!.start === prefixStart,
        tokens: folded,
        noteTokens,
        keptTokens,
        foldCostTokens,
        firstItem: first,
        // The batch covers the LAST RUN's tail, not the whole item array (F6,
        // cross-review 2026-11-25): a blocked (guarded) tail after the final run
        // is NOT folded, so `lastItem` must name the last run's end — a geometry
        // record that claims the blocked rows would misstate what the fold took.
        lastItem: runs[runs.length - 1]!.to,
        failed: failedOf(gatesOf(folded, keptTokens)),
      },
      spans,
    }
  }
  const batchCandidates: Array<{ geometry: Geometry; spans: CondenseSpan[] }> = []
  for (let first = 0; first < items.length; first += 1) {
    const candidate = batchGeometryAt(first)
    if (candidate !== undefined) batchCandidates.push(candidate)
  }

  const force = election?.force === true
  const passing = geometries.filter((geometry) => geometry.failed.length === 0)
  // Order: cheapest rewrite point first (the whole point of the change — the old
  // prefix-only order elected a shape whose cost the floor then refused), then the
  // bigger saving, then the older start so the tie is stable across turns.
  // POSITION 1 is predictability, not cost: the model writes its note before the call,
  // so a shape that starts at the prefix start is the only one it can describe. Cost,
  // saving and position only order shapes WITHIN the same class. Measured reason: with
  // the cheapest-first order alone the engine systematically preferred MIDDLE/TAIL
  // rows (an earlier rewrite point always costs more), which is exactly the note drift
  // the rewrite was meant to remove.
  const byCost = (a: Geometry, b: Geometry): number =>
    Number(b.isPrefix) - Number(a.isPrefix)
    || a.foldCostTokens - b.foldCostTokens
    || b.tokens - a.tokens
    || a.startSeq - b.startSeq
  // M-1 (2026-11-25): elect the batch FIRST. For every candidate lower bound the
  // verdict's §3 measured the full batch at ~0.11 payback where the best single run
  // read ~0.20 while folding only the youngest (least consumable) stretch. Ties
  // prefer the larger saving, then FEWER spans (fewer blocks, one note per span).
  // isPrefix first: the model writes its note before the call, so the elected shape
  // must stay predictable — the oldest batch always starts at the prefix start.
  let chosenSpans: CondenseSpan[] | undefined
  let chosenKind: 'fold' | 'distill' = 'fold'
  const paybackOf = (geometry: Geometry): number =>
    geometry.tokens > 0 ? geometry.foldCostTokens / geometry.tokens : Number.POSITIVE_INFINITY
  // SAME order for both classes: isPrefix first (predictability), then the
  // LOWEST PAYBACK (verdict §3: the full batch reads ~0.11 where the best single
  // run reads ~0.20 or worse), then the bigger saving, then fewer spans.
  // The normal batch path inlines the same keys below; `byElection` is what the
  // FORCED path sorts its batch candidates with (F9: previously dead code — the
  // forced path now honors the same marginal comparison once the gate-failure
  // count is settled).
  const byElection = (a: { geometry: Geometry; spans?: ReadonlyArray<unknown> }, b: { geometry: Geometry; spans?: ReadonlyArray<unknown> }): number =>
    Number(b.geometry.isPrefix) - Number(a.geometry.isPrefix)
    || paybackOf(a.geometry) - paybackOf(b.geometry)
    || b.geometry.tokens - a.geometry.tokens
    || (a.spans?.length ?? 0) - (b.spans?.length ?? 0)
  let chosen: Geometry | undefined = passing.sort(byCost)[0]
  // M-1 (2026-11-25): among passing batches pick the LOWEST PAYBACK (verdict §3),
  // then the bigger saving, then fewer spans. The batch wins whenever its payback
  // is strictly better than the elected single run's AND it folds at least as
  // much: the batch contains every run the single shape could fold, so a better
  // payback at an equal-or-larger Σ is a strict improvement (verdict §3 measured
  // ~0.11 vs ~0.20 on a real session). isPrefix is already part of the batch
  // sort — the oldest batch starts at the prefix start, so the note stays
  // describable (the E-11 note-drift concern does not apply to a batch that
  // CONTAINS the prefix).
  const batchPassing = batchCandidates
    .filter((candidate) => candidate.geometry.failed.length === 0)
    .sort((a, b) =>
      // F14 (cross-review 2026-11-25, deliberate, not a bug): `isPrefix` first
      // means a non-prefix batch never competes while a prefix batch exists —
      // even when the non-prefix one pays back better. This is the E-11 note
      // predictability contract kept intact for batches: the model writes its
      // note BEFORE the call from the range table, so the elected shape must
      // start at the prefix start or the note describes a stretch the engine
      // did not fold. Cost, payback and span count only order WITHIN the prefix
      // class.
      Number(b.geometry.isPrefix) - Number(a.geometry.isPrefix)
      || paybackOf(a.geometry) - paybackOf(b.geometry)
      || b.geometry.tokens - a.geometry.tokens
      || a.spans.length - b.spans.length)[0]
  if (batchPassing !== undefined
    && (chosen === undefined
      || (paybackOf(batchPassing.geometry) < paybackOf(chosen)
        && batchPassing.geometry.tokens >= chosen.tokens))) {
    chosen = batchPassing.geometry
    chosenSpans = batchPassing.spans
  }
  if (chosen === undefined) {
    const cut = cutFirstRange()
    if (cut !== undefined) chosen = cut
  }
  if (chosen === undefined && force) {
    // `force` is the operator's ONE override: it may fold a shape that fails the
    // gates (never one that crosses a guard, which is not in `geometries` at all).
    // Ordering: fewer failed gates first, then the SAME election order as the
    // normal path (`byElection`, F9 — the comparator was dead code until the
    // forced path took it up), so the forced pick honors payback/saving/span-count
    // instead of raw size alone.
    const forcedBatch = batchCandidates
      .slice()
      .sort((a, b) => a.geometry.failed.length - b.geometry.failed.length || byElection(a, b))[0]
    const forcedRun = [...geometries].sort((a, b) => a.failed.length - b.failed.length || byCost(a, b))[0]
    // The batch is forced over the single run only when it folds STRICTLY more —
    // at least FORCED_BATCH_MIN_ADVANTAGE × the run's volume — and does not fail
    // MORE gates than the run does (an extra failed gate is a real regression even
    // under force). The ×2 threshold: a forced fold is the operator accepting a
    // shape risk on purpose, so the batch must buy enough extra volume to justify
    // abandoning the run the gate order would have taken; 2× is the smallest whole
    // multiple that reads as "clearly more", and below it the operator meant the
    // smaller shape.
    chosen = forcedBatch !== undefined
      && (forcedRun === undefined
        || (forcedBatch.geometry.failed.length <= forcedRun.failed.length
          && forcedBatch.geometry.tokens > forcedRun.tokens * FORCED_BATCH_MIN_ADVANTAGE))
      ? forcedBatch.geometry
      : forcedRun
    if (chosen !== undefined && forcedBatch !== undefined && chosen === forcedBatch.geometry) chosenSpans = forcedBatch.spans
  }

  if (chosen === undefined) {
    // LAST resort, and only when the normal path has nothing: a fragmented surface
    // (every gap under the note-carrying minimum) is folded across its carriers in ONE
    // transaction. `force` is checked first because an operator forcing a fold means a
    // fold, not a distillation.
    const cross = crossBarrierGeometry()
    if (cross !== undefined) {
      chosen = cross.geometry
      chosenSpans = cross.spans
      chosenKind = 'distill'
    }
  }

  if (chosen === undefined) {
    if (ranges.length === 0) {
      return {
        reason: 'no continuous run clears the kernel minimum range (counted in CHARACTERS, compress.minCompressRange) \u2014 ' + refCount + ' projected nodes, nothing contiguous enough to fold',
        candidates: ranges.slice(0, 6),
      }
    }
    // Name the CLOSEST shape and every gate's arithmetic. The three gates are the
    // only reasons a fold is refused, so the refusal can be actionable without a
    // generic "nothing to fold" story.
    // "Closest" means FEWEST failed gates first, then the BIGGEST saving: a tiny run
    // that fails only the minimum-size gate is not closer to passing than a large run
    // that fails the floor, and leading with the cheapest rewrite point would name a
    // 70-token stub that no amount of waiting turns into a fold.
    const nearest = [...geometries].sort((a, b) => a.failed.length - b.failed.length || b.tokens - a.tokens)[0]
    const payback = nearest === undefined || nearest.tokens === 0
      ? null
      : nearest.foldCostTokens / nearest.tokens
    return {
      reason: nearest === undefined
        ? 'no foldable geometry exists at all \u2014 every eligible run is walled off by a guard or clears the kernel minimum by nothing'
        : condenseNoCandidateReason({
            nearest,
            startIndex: items[nearest.firstItem]?.index ?? 0,
            endIndex: items[nearest.lastItem]?.index ?? 0,
            surfaceTotal,
            floorTokens,
            payback,
            eligibleRangeCount: ranges.length,
            guardedRowCount: guarded.size,
          }),
      candidates: ranges.slice(0, 6),
      ...(nearest === undefined || payback === null
        ? {}
        : {
            nearest: {
              start: nearest.startSeq,
              end: nearest.endSeq,
              tokens: nearest.tokens,
              keptTokens: nearest.keptTokens,
              floorTokens,
              foldCostTokens: nearest.foldCostTokens,
              paybackRequests: payback,
              failed: nearest.failed,
            },
          }),
    }
  }

  const spans: CondenseSpan[] = chosenSpans
    ?? (chosen.cutSpan !== undefined
      ? [chosen.cutSpan]
      : condenseRangesOf(items.slice(chosen.firstItem, chosen.lastItem + 1)))
  // M-3 (2026-11-25): with the payback/cost gates gone, the cross-barrier SHAPE can
  // also arrive through the plain batch path (a run whose spans end on a block ref
  // used to be refused at ~2.3 payback, so only the dedicated fallback ever produced
  // it). The tier-2 semantics follow the SHAPE, not the path that produced it: any
  // span that ends on a carrier — or carries one — is a distillation, and the result
  // line must keep saying so (the originals behind those carriers stop being
  // readable at message level).
  const kind: 'fold' | 'distill' = spans.some((span) => span.endBlock !== null || span.carriers.length > 0)
    ? 'distill'
    : chosenKind
  const folded = spans.reduce((total, span) => total + span.tokens, 0)
  const noteTokens = condenseNoteBudget(folded)
  const kept = Math.max(0, surfaceTotal - folded - noteTokens)
  const endIndex = indexOfSeq.get(spans[spans.length - 1]!.end)
  const foldCostTokens = endIndex === undefined ? chosen.foldCostTokens : (suffix[endIndex + 1] ?? 0)
  const gates = gatesOf(folded, kept)
  return {
    spans,
    tokens: folded,
    noteTokens,
    keptTokens: kept,
    floorTokens,
    belowFloor: belowWorkingSetFloor(surfaceTotal, kept, floorTokens),
    aboveFloorAvailable: passing.length > 0,
    guardedRowCount: guarded.size,
    eligibleRangeCount: ranges.length,
    carrierCount: spans.reduce((total, span) => total + span.carriers.length, 0),
    kind,
    fromBatch: kind === 'fold' && chosenSpans !== undefined,
    distilledBlocks: kind === 'distill'
      ? spans.reduce((total, span) => total + span.carriers.length, 0)
      : 0,
    foldCostTokens,
    paybackRequests: folded > 0 ? foldCostTokens / folded : Number.POSITIVE_INFINITY,
    gates,
  }
}
