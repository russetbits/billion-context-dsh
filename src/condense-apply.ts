/** handleCompress: the compress tool handler — condense branch, batches, advisories, result lines. */

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
import { allLogMessages, attachmentsOfEvent, buildToolCallIndex, classifySurfaceEvent, eventsToCoreMessages, extractEventText, isAgentInstructionsRow, isCheckpointNode, isRealUserTurn, isSkillCatalogRow, surfaceEventsOf, toolCallsOf, withSummaryFramePrefix, SUMMARY_FRAME_PREFIX } from './messages.ts'
import { shadowedTokensViaMeter } from './host-tokens.ts'
import { lastRequestInput, peakRequestInput, providerUsage, recordCompressCost, totalRequestInput } from './usage.ts'
import { eventAtOf, sessionEventsOf } from './session-events.ts'
import { DEFAULT_PROMPTS, DEFAULT_RESOLVED, DEFAULT_TOOL_PARAM_TEXTS, type ResolvedPrompts } from './prompts.ts'
import { callHint, type ResolvedCallStyle } from './ptc.ts'
import type { SettingsCommandSurface } from './settings.ts'
import type { PresetName } from './presets.ts'
import { parseBoundary, CompressArgs, unwrapCompressArgs, validateContentItems } from './compress-args.ts'
import { CondenseSpan, CondensePick, CondenseMiss, indexOfSurfaceSeq, condenseNoteBudget, condenseRangesOf, pickCondenseSpan } from './condense-elect.ts'
import { formatPayback, condenseFloorRefusal, missingCondenseHeadings, condenseNoteGaps, MAX_NOTE_WEIGHT_RATIO, MAX_HANDOFF_NOTE_TOKENS, CONDENSE_BATCH_SPLIT_TOKENS, truncateNoteToCeiling, noteWeightRatio, noteWeightRejection } from './condense-note.ts'
import { ToolEnvironment, TextOutput, requireAgent, resolveEffectiveWindow, guardedRowsInSpan, splitAroundProtected, splitPiecePosition, splitCoverageLine, liveCarrierRejectionNote, protectedRowRejectionNote, edgeRefForSeq } from './shared.ts'

/** Strip the creation-time frame prefix the stored checkpoint node carries (region.ts framedSummary). */
function stripSummaryFrame(text: string): string {
  return text.startsWith(SUMMARY_FRAME_PREFIX)
    ? text.slice(SUMMARY_FRAME_PREFIX.length).replace(/^\n/, '')
    : text
}

/** The most recently landed durable block and its note as the note was WRITTEN (frame stripped). */
interface RecentLandedNote {
  ref: string
  start: number
  end: number
  /** The surface seq of the block's checkpoint summary node (null when gone). */
  summarySeq: number | null
  note: string
}

/**
 * The note of the NEWEST `compaction/summary` in the log (ledger order IS landing
 * order), with its kernel block ref (`bN`) resolved through the registry (which
 * synthesises refs for legacy blocks). Compares against what the ledger STORED —
 * the framed summary — so a caller-side `withSummaryFramePrefix` cannot hide a match.
 */
function recentLandedBlockNote(session: Session): RecentLandedNote | undefined {
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  const last = ledger[ledger.length - 1]
  if (last === undefined) return undefined
  const registryEntry = blockRegistry(session).find((entry) => entry.blockId === last.blockId)
  return {
    ref: registryEntry?.kernelBlockId ?? 'b?',
    start: last.start,
    end: last.end,
    summarySeq: last.summarySeq ?? null,
    note: stripSummaryFrame(last.summary),
  }
}

/** The refusal copy for a condense note byte-identical to the most recent landed block's note. */
function duplicateCondenseNoteRejection(ref: string, start: number, end: number, noteTokens: number): string {
  return 'compress: condense refused — this note is byte-identical to the note of the most recent landed block ' + ref
    + ' (seqs ' + start + '..' + end + ').\n'
    + 'A duplicate note adds nothing: every block stores its note verbatim, so this second copy makes the SAME ~' + noteTokens
    + '-token text resident in every later request once per checkpoint node — pure permanent cost, zero new record.\n'
    + 'Do ONE of:\n'
    + '  - write a NEW note that describes THIS stretch only (what happened here, what to do next, the pointers a resumed session re-derives most expensively) and resend condense; or\n'
    + '  - merge into the previous fold instead — distill through the block id: compress({ content: [{ startSeq: "' + ref
    + '", endSeq: <endSeq>, summary: <a new, shorter note covering the merged span> }] }); naming the block ref on the start edge is what makes the call a tier-2 distillation.'
}

/**
 * Duplicate-note verdict for the CONDENSE submission path. Refuses ONLY the harmful
 * shape measured on the live session (measured 2026-11-26): the submitted note is
 * identical to the most recent landed block's note AND the submitted number range
 * INTERSECTS that block's coverage AND the fold does not absorb that block's
 * checkpoint — i.e. two copies of the same note would co-exist on the surface.
 * Absorption is exempt: when the fold takes the recent block's checkpoint into its
 * shadowed span (what a distillation does), the old note leaves the surface with
 * its block and a repeated text is harmless. Non-intersecting repeats are also
 * exempt: the block-ref distill rhythm legitimately re-runs the same note across
 * disjoint segments of one prefix. Returns null when the submission is acceptable.
 *
 * `spanAbsorbsCheckpoint` is what the CALLER resolved (an endSeq that names a
 * block ref absorbs every older block's checkpoint), because at call time the
 * handler has not yet balanced the edges. `startSeq`/`endSeq` are the raw
 * submitted numbers, used only for the numeric intersect test.
 */
export function duplicateCondenseNoteVerdict(
  session: Session,
  items: ReadonlyArray<{ startSeq: number; endSeq: number; summary: string; spanAbsorbsCheckpoint?: boolean }>,
): string | null {
  const recent = recentLandedBlockNote(session)
  if (recent === undefined) return null
  const normalize = (text: string): string => text.replace(/\s+/g, ' ').trim()
  for (const item of items) {
    const intersects = Math.max(recent.start, item.startSeq) <= Math.min(recent.end, item.endSeq)
    if (!intersects) continue
    if (item.spanAbsorbsCheckpoint === true) continue
    const shadowedSpan = shadowedSeqsOf(session, item.startSeq, item.endSeq)
    const absorbsRecent = recent.summarySeq !== null && shadowedSpan.includes(recent.summarySeq)
    if (absorbsRecent) continue
    const candidate = stripSummaryFrame(item.summary)
    if (candidate === recent.note || normalize(candidate) === normalize(recent.note)) {
      return duplicateCondenseNoteRejection(recent.ref, recent.start, recent.end, defaultCountTokens(withSummaryFramePrefix(candidate)))
    }
  }
  return null
}

export async function handleCompress(env: ToolEnvironment, args: CompressArgs, exec: ToolRunContext): Promise<TextOutput> {
  const agent = requireAgent(exec)
  const session = agent.session
  // Clean orphan tool messages before any range solve: a single orphan result
  // corrupts the pairing balance cache and rejects every large range (issue
  // #18). Every call still in flight — the compress call itself AND any
  // sibling tool called in the same assistant message — must be excluded from
  // orphan pruning: its tool/result lands at the end of the step, and pruning
  // the call now would orphan that result.
  stripOrphanedSurfaceToolMessages(session, openToolCallIds(session))
  const state = env.store.stateFor(session)
  // The kernel gets the FULL log (visible + shadowed): syncBlocks deactivates
  // a block whose consumed messages are absent, and resolveBoundaries refuses
  // to anchor a block ref it cannot find, so tier-2/3 distillation needs the
  // originals present. The token count uses the same priority chain as the
  // nudge (projectedTokens → surfaceTokens → character heuristic).
  const coreMessages = allLogMessages(session)
  const surfaceMessages = eventsToCoreMessages(surfaceEventsOf(session))
  const tokenCount = resolveTokenCount(agent, surfaceMessages)
  const window = await resolveEffectiveWindow(env, agent)
  const config = kernelConfigFor({ ...env, modelContextLimit: window.limit })

  // Assign refs / advance state exactly like a turn would.
  const turn = env.kernel.processTurn({ messages: coreMessages, state, config, tokenCount })
  env.store.set(session, turn.state)
  const byRaw = turn.state.messageRefs.byRaw
  // mN drilldown refs resolve against the CURRENT turn's ref map (not the
  // stored state) — acp_status's turn is never persisted, so its mN rows only
  // exist here; the deterministic re-assignment yields the same mN for the
  // same messages (see parseBoundary).
  const byRef = turn.state.messageRefs.byRef

  // Tolerate the wrapped-arguments forms some models emit (double-nested
  // `{ arguments: "..." }`), which the old DSH validator surfaced as
  // `"arguments" must be an object` and sent the model into a retry loop.
  const unwrapped = unwrapCompressArgs(args)
  if (unwrapped === null) {
    return {
      text: `compress: missing content — pass the content array directly: ${callHint(env.toolCallStyle ?? 'native', 'compress', '{ content: [{ startSeq, endSeq, summary }] }')}`,
    }
  }
  args = unwrapped
  // condense: the engine names the span, the model writes the handoff note. From
  // here on the write is IDENTICAL to an explicit range — guarded rows, live
  // checkpoint carriers, pairing balance and the provenance gate all still apply,
  // because the engine picks from the range table geometry the guards produced.
  let condensePick: CondensePick | undefined
  // True when `args.content` was synthesized by the condense election rather than
  // written by the model. Those ranges are the engine's own choice of geometry and
  // must NOT be cut around carriers — a prefix fold exists precisely to distil every
  // carrier inside its prefix in one call.
  let condenseSynthesizedRange = false
  // Advisories raised by the condense path itself (missing headings, an
  // over-long note that had to be truncated). They land WITH the fold instead
  // of refusing it: every refusal sends the model back to write the note again,
  // and output tokens cost several times what an uncached input token does —
  // a retry loop is the single most expensive failure mode this tool has.
  const condenseNoteAdvisories: string[] = []
  if (args.condense !== undefined) {
    if (args.content !== undefined) {
      return { text: 'compress: condense and content are mutually exclusive — pass one or the other' }
    }
    let note = args.condense.summary ?? ''
    const missing = missingCondenseHeadings(note)
    if (missing.length > 0) {
      // Land the fold anyway, with the gap named. A note missing a heading is
      // still a far better record than the raw stretch it replaces, and the
      // rejection this used to be cost a full note rewrite (~900 output tokens,
      // plus the same again on the retry) to buy a heading we can also just ask
      // for in the NEXT fold.
      condenseNoteAdvisories.push(
        '  condense note is missing ' + missing.join(' and ') + ' — the fold landed anyway; add it to the next note'
        + ' (Task is the one a resumed session cannot reconstruct, Now is what it does next).',
      )
    }
    const forcedElection = args.force === true || args.condense.force === true
    const pick = pickCondenseSpan(session, turn, 5, foldPricingFor(env), { force: forcedElection })
    if ('reason' in pick) {
      const rows = pick.candidates.map((range) =>
        '  - seq ' + range.start + '..' + range.end + ' — ' + range.count + ' messages, ~' + range.tokens + ' tokens — fold costs ~' + range.foldCostTokens + ' once',
      )
      return {
        text: 'compress: condense folded nothing — ' + pick.reason + '.\n'
          + 'Compare against the automatic checkpoint it replaces: a host checkpoint costs a fully cold prefill (~130 s here), so waiting is cheaper only while nothing is consumed.\n'
          + (rows.length > 0
            ? 'Eligible runs you can fold explicitly:\n' + rows.join('\n')
            : 'No eligible run yet — keep working; consumed spans reappear in the next nudge.'),
      }
    }
    // Working-set guard (Appendix D of arXiv 2611.05306): on an open-weight
    // engine a fold that leaves a shallow working set is the regime where
    // benchmark scores collapse, and the failure is not recoverable by a better
    // note. Advisory first, hard only here — the model can force it, because
    // overflow pressure is a legitimate reason to accept a thin context.
    const forced = forcedElection
    const floor = foldPricingFor(env).floorTokens ?? 0

    // Defensive only: without `force` the election never returns a shape that fails
    // a gate, so this can only fire on the forced path (where it is skipped) or if a
    // future gate is added without updating the election. The copy names the failed
    // gate and its arithmetic instead of a generic "below the floor".
    // A cross-barrier distillation is DELIBERATELY outside the three gates (a fragment
    // surface has no shape that could pass them — see `crossBarrierGeometry`), so its
    // own verdict is not a refusal. It is still the only path that reopens the surface.
    // M-3 (2026-11-25): capacity is the ONLY gate, so the distill/plain distinction
    // collapses — a distillation is NOT exempt from `capacity` (the working-set floor
    // is the only gate with experimental support, and tier-2 originals stop being
    // readable at the message layer, so a distillation that lands below the floor is
    // unrecoverable loss, not a slow session).
    const gateFailed = !pick.gates.capacity
    if (gateFailed && !forced) {
      return { text: condenseFloorRefusal(pick, floor) }
    }

    // Priced as the note is STORED: the summary frame prefix the checkpoint node carries
    // (region.ts writes it, messages.ts projects it) is part of what stays resident, so the
    // weight gate counts the same text the transaction writes (round-13 review Q1: a bare
    // `defaultCountTokens(note)` under-priced the resident node by ~16-35 tokens).
    const noteTokens = defaultCountTokens(withSummaryFramePrefix(note))
    // A note that is a third the size of the stretch is a second copy, not a
    // briefing — refuse BEFORE anything lands, so the model can rewrite and resend
    // without paying a second re-prefill (nothing was written).
    const weight = noteWeightRatio(note, pick.tokens, noteTokens)
    // For a distillation the note IS the replacement for every original behind the
    // carriers, so the span it is weighed against is what it overwrites, not the thin
    // residue of carriers it lands on. Weighing it against `pick.tokens` would reject
    // every merge (a 900-token note against ~600t of carrier summaries).
    const weightBase = pick.kind === 'distill'
      ? Math.max(pick.tokens, MIN_CONDENSE_SPAN_TOKENS)
      : pick.tokens
    if (noteWeightRatio(note, weightBase, noteTokens) > MAX_NOTE_WEIGHT_RATIO) {
      return { text: noteWeightRejection(note, pick.tokens, noteTokens) }
    }
    // The ratio above cannot bound a note on a large stretch (35% of 84K is 29K),
    // so the absolute ceiling is what actually keeps the stored area from growing
    // with the fold size. Refused before anything lands, same as the weight gate:
    // the model rewrites and resends without paying a second re-prefill.
    if (noteTokens > MAX_HANDOFF_NOTE_TOKENS) {
      // Truncate instead of refusing. The ceiling exists to stop the STORED area
      // from growing with the fold, and a truncation bounds it just as well as a
      // refusal does — while costing nothing. The headings we care about sit at
      // the TOP of a note written to the skeleton, so the tail that gets dropped
      // is the narrative the paper says to leave out anyway.
      const cut = truncateNoteToCeiling(note, MAX_HANDOFF_NOTE_TOKENS)
      note = cut.note
      args = { ...args, condense: { ...args.condense, summary: cut.note } }
      condenseNoteAdvisories.push(
        '  condense note was ~' + noteTokens + ' tokens, over the ' + MAX_HANDOFF_NOTE_TOKENS
        + '-token ceiling — truncated by ~' + cut.droppedTokens + ' tokens (the tail). Keep the next note to task + next step + pointers.',
      )
    }
    condensePick = pick
    condenseSynthesizedRange = true
    // M-1 batch-size fallback (verdict §6): a full-surface batch can exceed what
    // ONE note can honestly cover (~235Kt measured on a real session). Above
    // `CONDENSE_BATCH_SPLIT_TOKENS` the engine submits only the FIRST batch this
    // call and tells the model the remaining batches — each later condense call
    // pays its own re-prefill but covers a note-sized stretch.
    let spansToSubmit = pick.spans
    if (pick.tokens > CONDENSE_BATCH_SPLIT_TOKENS) {
      const batches: CondenseSpan[][] = []
      let current: CondenseSpan[] = []
      let currentTokens = 0
      for (const span of pick.spans) {
        if (currentTokens > 0 && currentTokens + span.tokens > CONDENSE_BATCH_SPLIT_TOKENS) {
          batches.push(current)
          current = []
          currentTokens = 0
        }
        current.push(span)
        currentTokens += span.tokens
      }
      if (current.length > 0) batches.push(current)
      spansToSubmit = batches[0]!
      const submittedTokens = spansToSubmit.reduce((total, span) => total + span.tokens, 0)
      const remaining = batches.slice(1)
      if (remaining.length > 0) {
        const remainingLines = remaining.map((batch, index) => {
          const batchTokens = batch.reduce((total, span) => total + span.tokens, 0)
          const firstSpan = batch[0]!
          const lastSpan = batch[batch.length - 1]!
          return `  batch ${index + 2}: seqs ${firstSpan.start}..${lastSpan.end} — ${batch.length} span(s), ~${batchTokens}t (call condense again when this batch has landed; each call pays its own re-prefill)`
        })
        condenseNoteAdvisories.push(
          '  condense batch is ~' + pick.tokens + 't, over the ~' + CONDENSE_BATCH_SPLIT_TOKENS
          + 't a single note can cover honestly — submitted batch 1 (seqs ' + spansToSubmit[0]!.start + '..' + spansToSubmit[spansToSubmit.length - 1]!.end + ', ~' + submittedTokens + 't) only. Remaining batches:\n'
          + remainingLines.join('\n')
          + '\n  These seqs are the EXPECTED remainder, not a promise: once batch 1 lands the engine re-elects against the shorter surface, and it may pick a different shape — read the next result line for what actually folds.',
        )
        // F5 (cross-review 2026-11-25): everything the result line and the follow-up
        // logic report must describe what THIS call actually submitted, not the
        // whole batch it declined to submit in one go. Re-price the pick against
        // batch 1: the fold cost is unchanged (the earliest rewrite point —
        // spans[0].start — belongs to batch 1 either way), but the saving, the
        // note denominator, the kept tokens and the payback all change.
        const surfaceTotal = pick.keptTokens + pick.tokens + pick.noteTokens
        const submittedNoteBudget = condenseNoteBudget(submittedTokens)
        const submittedKept = Math.max(0, surfaceTotal - submittedTokens - submittedNoteBudget)
        const payback = submittedTokens > 0 ? pick.foldCostTokens / submittedTokens : Number.POSITIVE_INFINITY
        condensePick = {
          ...pick,
          spans: spansToSubmit,
          tokens: submittedTokens,
          noteTokens: submittedNoteBudget,
          keptTokens: submittedKept,
          belowFloor: belowWorkingSetFloor(surfaceTotal, submittedKept, pick.floorTokens),
          carrierCount: spansToSubmit.reduce((total, span) => total + span.carriers.length, 0),
          distilledBlocks: 0,
          paybackRequests: payback,
          gates: {
            capacity: submittedTokens >= MIN_CONDENSE_SPAN_TOKENS
              && !belowWorkingSetFloor(surfaceTotal, submittedKept, pick.floorTokens),
          },
        }
      }
    }
    // ONE content item carries the whole prefix (see condenseRangesOf: each
    // submitted range becomes its own kernel block, so per-block items would
    // multiply the barriers and duplicate this one note). The model writes ONE
    // handoff note; every submitted range carries it, because a prefix is one
    // continuous stretch of work.
    args = {
      ...args,
      content: spansToSubmit.map((span) => ({
        startSeq: span.start,
        endSeq: span.end,
        summary: note,
        topic: args.condense!.topic,
        verifiedReadings: args.condense!.verifiedReadings,
      })),
    }
  }
  // Items-level required check AFTER the envelope peel (see
  // validateContentItems for why the schema gate alone cannot do this).
  validateContentItems(args.content!)

  // Duplicate-note gate (live another live session, 2026-11-26): two adjacent condense
  // calls landed the SAME byte-identical note (932 chars) on two INTERSECTING
  // number ranges, so every later request paid that full note once per checkpoint
  // node — pure resident cost for zero new record. Refused BEFORE anything lands,
  // so the model rewrites and resends without paying a second re-prefill.
  // Condense-path only (an explicit `content` range legitimately reuses short
  // one-line summaries). Full rationale in duplicateCondenseNoteVerdict.
  if (condenseSynthesizedRange) {
    const absorbs = condensePick !== undefined && (
      condensePick.kind === 'distill'
      || condensePick.spans.some((span) => span.endBlock !== null || span.carriers.length > 0)
    )
    const verdict = duplicateCondenseNoteVerdict(
      session,
      args.content!.map((item) => ({ startSeq: Number(item.startSeq), endSeq: Number(item.endSeq), summary: item.summary, spanAbsorbsCheckpoint: absorbs })),
    )
    if (verdict !== null) return { text: verdict }
  }

  const ranges: Array<
    ResolvedSurfaceRange & {
      startSeq: number
      endSeq: number
      startRef: string
      endRef: string
      summary: string
      topic?: string
      /** B3：本段压缩时已绿的验收读数（结构化落盘）。 */
      verifiedReadings?: string[]
    }
  > = []
  // Ranges whose whole span was already shadowed by earlier compressions.
  // They land as advisory warnings, never as errors or phantom blocks.
  const alreadyCompressedNotes: string[] = []
  // Ranges rejected because they cover a CURRENT injected instruction row —
  // hard reject before the kernel apply (supersedes the F7 warn-only draft,
  // see protectedRowRejectionNote): the kernel never sees these ranges, so no
  // phantom block can exist. Computed once here: the surface is stable from
  // the orphan strip onward, the deferred compress-pair hide only touches tool
  // events, and every accepted range lands in ONE applyCompression call at the
  // end of the loop — so the set cannot go stale mid-batch.
  const rejectedNotes: string[] = []
  // Range keys (`startRef::endRef`) already claimed by an accepted range in
  // THIS call: several requested ranges can resolve to ONE balanced span
  // (single tool results expand outward to the enclosing pair), and feeding
  // the same span twice makes the kernel stamp a block per request while only
  // one durable transaction can ever land for it (issue #201).
  const seenRangeKeys = new Set<string>()
  const duplicateRangeNotes: string[] = []
  // Reversed edges (`start > end`) are a model slip, not a real span — the same
  // span with the boundaries swapped is what the caller meant (measured
  // 2026-11-01 in one live window: 8 `reversed range` rejections, every one
  // recoverable by a swap). Swapping HERE rather than inside
  // `resolveSurfaceRange` keeps that resolver's `start ≤ end` invariant intact
  // (its stale recovery and index arithmetic both assume ordered edges) and
  // lets the advisory line name the ORIGINAL edges.
  const swappedNotes: string[] = []
  const guardedSeqs = guardedSurfaceSeqsOf(session)
  // How many content ranges this call had to cut around protected rows, and how
  // many rows it cut around. Reported in the success line so the model learns the
  // shape of its own boundary instead of inferring it from a smaller block count.
  let splitRangeCount = 0
  let splitProtectedRows = 0
  // Cut blocks written by this call. Reported so the duplication cost of the
  // shared note is visible instead of silent (the note stays resident once per
  // block, and a note written for the whole span is stored N times).
  let splitBlocks = 0
  // What the shared note costs on top of a hypothetical per-piece note: the model's
  // note is stored in EVERY cut block, so N blocks pay it N times (the coverage line
  // is the only per-block part). Reported, never refused — a duplicated note is
  // cheaper than losing half a record.
  let splitExtraTokens = 0
  /**
   * Everything about one resolved span that can be decided WITHOUT touching shared
   * state: the working-set floor, the carrier rule and the kernel refs. Split out of
   * `acceptResolved` (2026-11-25) because a cut-around must PRE-FLY every piece
   * before it can write a coverage line: that line numbers the pieces that really
   * LAND ("1/2", "2/2"), which is only knowable once the whole range has been priced
   * and the duplicates have been dropped.
   */
  const planResolved = (
    resolved: ResolvedSurfaceRange,
    shadowedSpan: readonly number[],
  ): { startRef: string; endRef: string } | { refuse: string } => {
    // The working-set floor applies to a model-NAMED range too (cross-review Q1:
    // all three reviewers). The guard protects the post-fold working set, and an
    // explicit range that folds down to a sliver violates it exactly as hard as an
    // elected one — while still paying the full one-time re-prefill. Same pricing
    // implementation as the table rows (foldSpanPricing), so a row marked
    // "below floor" and the guard can never disagree.
    // `forced` covers the content range condense synthesized above, so an
    // operator-accepted thin context is never re-refused by its own guard.
    const priced = foldSpanPricing(session, resolved.start, resolved.end, foldPricingFor(env))
    if (priced.belowFloor && args.force !== true && args.condense?.force !== true) {
      return {
        refuse: '  seq ' + resolved.start + '..' + resolved.end + ' would leave ~' + priced.liveAfterTokens + ' tokens visible, under the working-set floor ~' + (foldPricingFor(env).floorTokens ?? 0) + ' — one fold costs the SAME single re-prefill, so this span is too thin to earn it back. Do NOT just widen blindly: current instruction copies, the newest skill catalog and the active user turn are refused by a DIFFERENT guard (they appear as their own rejection line when hit), so widen only across rows that stay eligible, or fold several disjoint slices in ONE call. Read consumed bytes back from files instead of folding. force:true bypasses THIS floor only — never the guarded rows. Other ranges in this batch already landed, so do not resend this one unchanged.',
      }
    }
    // An edge on an ACTIVE block's checkpoint summary node resolves to the
    // kernel block ref (bN) — the boundary that makes applyCompression distill
    // (tier 2/3) instead of folding the summary as a plain message.
    const startBlockRef = blockRefForSummarySeq(session, resolved.start)
    const endBlockRef = blockRefForSummarySeq(session, resolved.end)
    const startRef = startBlockRef ?? edgeRefForSeq(session, byRaw, resolved.start, 'start', resolved.end)
    const endRef = endBlockRef ?? edgeRefForSeq(session, byRaw, resolved.end, 'end', resolved.start)
    if (startRef === undefined || endRef === undefined) {
      throw new Error(
        `billion-context-dsh: seq ${resolved.start}..${resolved.end} has no assigned ref — `
        + 'the range must be on the current surface (run acp_status for the live seq list)',
      )
    }
    // A plain (message-ref) range must not fold a LIVE block's checkpoint
    // carrier. The kernel keeps that carrier visible — our projection marks it
    // via `summaryOfBlockId`, upstream #335 — but this host transaction replaces
    // the WHOLE span in one `surfaceOp`, so the carrier would leave the surface
    // anyway: the block's summary would vanish while the kernel still counts the
    // block as active. Block-ref boundaries (bN..bM) are the call that SHOULD
    // fold carriers (tier 2/3), so the reject fires only when NEITHER edge is a
    // block ref — the same rule the kernel applies (`resolveBoundaries` reports
    // boundary kind `block` as soon as one edge is a block ref).
    if (startBlockRef == null && endBlockRef == null) {
      const carriers = liveCheckpointCarriersInSpan(session, shadowedSpan)
      if (carriers.length > 0) {
        return { refuse: liveCarrierRejectionNote(resolved.start, resolved.end, carriers) }
      }
    }
    return { startRef, endRef }
  }

  /**
   * Point of no return: claim the span and queue it. The summary is a PARAMETER
   * (not `range.summary`) so a cut-around can prepend its coverage line — the model's
   * note was written for the WHOLE span, and a split block must say so.
   */
  const commitPlanned = (
    plan: { startRef: string; endRef: string },
    resolved: ResolvedSurfaceRange,
    origin: { startSeq: number; endSeq: number },
    range: NonNullable<CompressArgs['content']>[number],
    summary: string,
  ): void => {
    const rangeKey = `${plan.startRef}::${plan.endRef}`
    if (seenRangeKeys.has(rangeKey)) {
      // An earlier range in this call already owns this exact span: its summary
      // lands for the whole span, and a second block here could never get its
      // own durable transaction (the span leaves the surface after the first).
      duplicateRangeNotes.push(
        `  seqs ${origin.startSeq}..${origin.endSeq} resolve to the same span as an earlier range in this call (${resolved.start}..${resolved.end}) — skipped`,
      )
      return
    }
    seenRangeKeys.add(rangeKey)
    ranges.push({
      ...resolved,
      startSeq: origin.startSeq,
      endSeq: origin.endSeq,
      startRef: plan.startRef,
      endRef: plan.endRef,
      // B3：把该段声明的已绿验收读数带上（缺位=不写键）
      ...(Array.isArray(range.verifiedReadings) && range.verifiedReadings.length > 0
        ? { verifiedReadings: range.verifiedReadings.map(String) }
        : {}),
      summary,
      ...(range.topic ?? args.topic) === undefined ? {} : { topic: range.topic ?? args.topic },
    })
  }

  /**
   * One-shot path for a range that needs NO cut: plan, record the refusal, commit.
   * A range with no protected row inside goes through exactly this, which is what
   * keeps its stored summary byte-identical to the pre-split behaviour.
   */
  const acceptResolved = (
    resolved: ResolvedSurfaceRange,
    shadowedSpan: readonly number[],
    origin: { startSeq: number; endSeq: number },
    range: NonNullable<CompressArgs['content']>[number],
  ): void => {
    const plan = planResolved(resolved, shadowedSpan)
    if ('refuse' in plan) {
      rejectedNotes.push(plan.refuse)
      return
    }
    commitPlanned(plan, resolved, origin, range, range.summary)
  }
  for (const range of args.content!) {
    let startSeq = parseBoundary(range.startSeq, byRef, session)
    let endSeq = parseBoundary(range.endSeq, byRef, session)
    // Swap on POSITION, not value: after a compaction replace the surface is
    // locally non-monotonic (a checkpoint node carrying a HIGH seq sits before
    // residual nodes with LOWER seqs), so a model can legitimately ask for
    // `12..9` meaning the positionally ascending span [12, 7, 8, 9] — swapping
    // on value would invert it into a position-reversed call that the resolver
    // rejects. A genuinely misplaced call (edges swapped positionally) still
    // gets the mirrored note.
    if (indexOfSurfaceSeq(session, startSeq) > indexOfSurfaceSeq(session, endSeq)) {
      swappedNotes.push(
        `  seqs ${startSeq}..${endSeq} were reversed — swapped to ${endSeq}..${startSeq}`,
      )
      const mirrored = startSeq
      startSeq = endSeq
      endSeq = mirrored
    }
    let resolved: ResolvedSurfaceRange
    try {
      // Balance edges FIRST: the requested edges may sit mid-pair or on nodes
      // that project no bare-`${seq}` id (multi-tool-call assistants project to
      // `${seq}#${callId}` sub-ids; empty tool results project to nothing —
      // issue #155). resolveSurfaceRange shifts them to clean
      // tool-pairing-balanced cuts that ANCHOR (bare ref, sub-id, or the
      // empty-result fallback — see edgeRefForSeq), so the resolved refs exist
      // and the shadowed span matches the returned range. Edges shadowed by an
      // earlier compression (stale nudge table / old compress result) are
      // remapped to the still-live content of the span.
      resolved = resolveSurfaceRange(session, startSeq, endSeq)
    } catch (error) {
      if (error instanceof AlreadyCompressedRangeError) {
        const covering = error.coveringBlockIds
        const blockNote = covering.length === 0
          ? ''
          : ` (block ${covering[0]!.slice(0, 8)}${covering.length > 1 ? ` +${covering.length - 1} more` : ''})`
        alreadyCompressedNotes.push(
          `  seqs ${error.start}..${error.end} already compressed${blockNote} — nothing to reclaim; recall({ blockId }) to recover the originals`,
        )
        continue
      }
      throw error
    }
    // Hard reject BEFORE the kernel: a span covering a CURRENT injected policy
    // row has no legitimate outcome — the host re-injects the newest AGENTS.md
    // copy the moment it leaves the surface (compress → re-inject loop fuel,
    // issue #71), and a folded skill catalog never comes back at all (resend
    // keyed on the digest, issue #185). Stale copies pass: removing them while
    // the newest stays visible is the real cleanup and triggers no re-injection.
    // Probe the set that will ACTUALLY be shadowed (`shadowedSeqsOf`, the
    // positional slice the transaction prices) rather than a numeric interval —
    // see guardedRowsInSpan for why the interval false-positives on a locally
    // non-monotonic surface.
    const shadowedSpan = shadowedSeqsOf(session, resolved.start, resolved.end)
    const instructionHits = guardedRowsInSpan(guardedSeqs, shadowedSpan)
    // A LIVE block carrier is protected in the same sense: this host transaction
    // replaces the WHOLE span in one `surfaceOp`, so folding across one would drop
    // a summary the kernel still counts as active. It is cut around too — unless the
    // span EDGE is the carrier, where the edge already resolved to a bN and the
    // distillation path below owns the call.
    // Only a MODEL-WRITTEN range cuts around carriers. The condense election's own
    // prefix is expected to straddle them — distilling every carrier inside the
    // prefix is what a prefix fold IS — and the refs stage below already refuses the
    // plain-seq case (`startBlockRef == null && endBlockRef == null`).
    const carrierHits = condenseSynthesizedRange
      ? []
      : liveCheckpointCarriersInSpan(session, shadowedSpan)
        .filter((carrier) => carrier.seq !== resolved.start && carrier.seq !== resolved.end)
        .map((carrier) => carrier.seq)
    const protectedHits = [...new Set([...instructionHits, ...carrierHits])]
    if (protectedHits.length === 0) {
      acceptResolved(resolved, shadowedSpan, { startSeq, endSeq }, range)
      continue
    }
    // The condense election never hands us a span it did not price against the
    // guards, so a hit there is a bug in the geometry rather than a model slip:
    // keep the hard rejection (and never cut a prefix, whose whole point is to
    // fold the carriers inside it).
    if (condenseSynthesizedRange) {
      rejectedNotes.push(protectedRowRejectionNote(resolved.start, resolved.end, protectedHits, shadowedSpan, session))
      continue
    }
    // CUT AROUND the protected rows instead of refusing (2026-11-25). In the real
    // session s2115, 8 of 10 compress failures were a legal span drawn across a
    // guarded row (5 active user turns, 2 AGENTS.md copies, 1 catalog), and all 8
    // succeeded on the very NEXT step with a hand-cut range: the model can already
    // do this split, and the extra turn is the entire cost. Done here, the same
    // span lands one block per surviving side, in ONE call.
    const pieces = splitAroundProtected(shadowedSpan, protectedHits)
    // PRE-FLIGHT every piece before writing anything: the coverage line numbers the
    // pieces that really LAND (`1/2`, `2/2`), so a piece dropped by the floor, the
    // carrier rule or the duplicate rule must not be counted — and a range whose
    // pieces all drop still takes the single-range rejection path below.
    const planned: Array<{
      resolved: ResolvedSurfaceRange
      startSeq: number
      endSeq: number
      refs: { startRef: string; endRef: string }
      where: string
    }> = []
    const landedKeys = new Set(seenRangeKeys)
    for (const piece of pieces) {
      const pieceStart = piece[0]!
      const pieceEnd = piece[piece.length - 1]!
      let pieceResolved: ResolvedSurfaceRange
      try {
        // Re-balance: a cut inside a tool pair would anchor nothing.
        pieceResolved = resolveSurfaceRange(session, pieceStart, pieceEnd)
      } catch (error) {
        if (error instanceof AlreadyCompressedRangeError) continue
        throw error
      }
      const pieceShadowed = shadowedSeqsOf(session, pieceResolved.start, pieceResolved.end)
      // Balancing an edge can walk back ONTO a protected row (a cut inside a tool
      // pair extends to the pair's ends). Skipping THAT side is the honest answer:
      // the note names it, and the other side still lands.
      if (guardedRowsInSpan(guardedSeqs, pieceShadowed).length > 0) {
        rejectedNotes.push(`  seqs ${pieceStart}..${pieceEnd} skipped — balancing that cut walked back onto a protected row; fold ${pieceStart}..${pieceEnd} on its own`)
        continue
      }
      const plan = planResolved(pieceResolved, pieceShadowed)
      if ('refuse' in plan) {
        rejectedNotes.push(plan.refuse)
        continue
      }
      const key = `${plan.startRef}::${plan.endRef}`
      if (landedKeys.has(key)) continue
      landedKeys.add(key)
      planned.push({
        resolved: pieceResolved,
        startSeq: pieceStart,
        endSeq: pieceEnd,
        refs: plan,
        where: splitPiecePosition(shadowedSpan, protectedHits, pieceShadowed),
      })
    }
    if (planned.length === 0) {
      rejectedNotes.push(protectedRowRejectionNote(resolved.start, resolved.end, protectedHits, shadowedSpan, session))
    } else {
      planned.forEach((item, index) => {
        // The note the model wrote describes seqs A..B — the WHOLE requested span —
        // while this block holds one side of it. Say so, mechanically, instead of
        // letting a later reader take half a record for the whole thing.
        const coverage = splitCoverageLine(
          index + 1,
          planned.length,
          resolved.start,
          resolved.end,
          protectedHits.length,
          item.where,
        )
        commitPlanned(item.refs, item.resolved, { startSeq: item.startSeq, endSeq: item.endSeq }, range, coverage + range.summary)
        // Priced as STORED: the coverage line is per-block, the note is the model's
        // whole-span note repeated once per block.
        splitExtraTokens += defaultCountTokens(coverage)
      })
      splitExtraTokens += (planned.length - 1) * defaultCountTokens(withSummaryFramePrefix(range.summary))
      splitRangeCount += 1
      splitProtectedRows += protectedHits.length
      splitBlocks += planned.length
    }
    continue
  }

  // Nothing to do: every requested range was already compressed, rejected, or
  // a duplicate of an earlier range in the same call.
  if (ranges.length === 0) {
    const text = ['Compressed 0 block(s), ~0 tokens reclaimed.', ...swappedNotes, ...alreadyCompressedNotes, ...duplicateRangeNotes, ...rejectedNotes]
    if (alreadyCompressedNotes.length > 0) {
      text.push('  (all requested ranges were already compressed — recall a block to recover its originals)')
    } else if (duplicateRangeNotes.length > 0 && rejectedNotes.length === 0) {
      text.push('  (nothing compressed — every range resolved to a span an earlier range in this call already covers)')
    } else if (rejectedNotes.length > 0) {
      text.push('  (nothing compressed — every range covered a current guarded row, an injected policy row or the active user turn; see the rejections above)')
    }
    return { text: text.join('\n') }
  }

  const applied = env.kernel.applyCompression({
    ranges: ranges.map(({ startRef, endRef, summary, topic }) => ({ startRef, endRef, summary, topic })),
    messages: coreMessages,
    state: turn.state,
    config,
    // Deliberately NOT overriding protectedMessageIds: with the full log the
    // kernel's recent/last-user protection is computed over the same
    // non-block-covered messages as the visible feed, so default behavior is
    // preserved. Any 'Excluded N protected message(s)' warning is surfaced.
  })
  // A kernel error for ONE range must not poison the whole call: the other
  // ranges still created blocks. This matters for issue #18's "phantom range"
  // — messages absorbed into an earlier block's effectiveMessageIds (kernel
  // boundary adjustment) but still live on the surface resolve fine but make
  // the kernel throw "Range contains no compressible messages". Fail only
  // when NOTHING landed; otherwise land the successes and surface the
  // failures as advisory lines below.
  if (applied.result.errors.length > 0 && applied.result.blocksCreated === 0) {
    return { text: `compress failed: ${applied.result.errors.join('; ')}` }
  }
  env.store.set(session, applied.state)
  if (applied.result.blocksCreated > 0) {
    // Hide this compress call/result after the tool result lands, so the
    // compaction summary never sits between an assistant tool_calls block and
    // its tool response (strict providers reject that sequence).
    env.compressCallIdsToHide?.add(exec.callId)
  }

  // Match freshly created kernel blocks to the requested ranges by their
  // range key (the kernel stamps startRef/endRef onto each new block).
  const previousIds = new Set(turn.state.blocks.map((block) => block.blockId))
  const newBlocks = applied.state.blocks.filter((block) => !previousIds.has(block.blockId))
  const blockByRangeKey = new Map(newBlocks.map((block) => [`${block.startRef}::${block.endRef}`, block]))
  // Warnings carry two shapes: range-prefixed ("Skipped range (a..b) — …")
  // attributable to a specific range, and free-form ("Excluded N protected
  // message(s) …") attributable to the call as a whole.
  const warningByRangeKey = new Map<string, string[]>()
  const freeWarnings: string[] = []
  for (const warning of applied.result.warnings) {
    const match = /^Skipped range \((.+?)\.\.(.+?)\)/.exec(warning)
    if (match !== null) {
      const key = `${match[1]}::${match[2]}`
      const list = warningByRangeKey.get(key) ?? []
      list.push(warning)
      warningByRangeKey.set(key, list)
    } else {
      freeWarnings.push(warning)
    }
  }

  const lines: string[] = []
  const landedBlockIds: string[] = []
  let skippedRanges = 0
  for (const range of ranges) {
    const key = `${range.startRef}::${range.endRef}`
    const block = blockByRangeKey.get(key)
    if (block === undefined) {
      // The kernel skipped this range (already compressed / overlapped): no
      // kernel block was created, so no durable transaction is landed — the
      // ledger must never record a block the kernel does not know.
      skippedRanges += 1
      const warnings = warningByRangeKey.get(key) ?? []
      for (const warning of warnings) lines.push(`  ${warning}`)
      continue
    }
    // The edges were already balanced above; shadow exactly that span.
    const { start, end } = range
    const shadowed = shadowedSeqsOf(session, start, end)
    // Last line of defense before the durable write: if an earlier landed
    // transaction in this batch removed this span from the surface (the kernel
    // can stamp one block per requested range even when several share refs),
    // skip instead of transacting — a summary whose shadowedSeqs no longer
    // name an exact current surface span bricks the session log on next load
    // (issue #201). runCompactionTransaction re-validates and would throw;
    // skipping keeps the rest of the batch landing.
    if (!isExactSurfaceSpan(session, start, end, shadowed)) {
      skippedRanges += 1
      lines.push(`  skipped seqs ${start}..${end}: already shadowed by an earlier range in this call`)
      continue
    }
    // Price the reclaimed tokens in the HOST's token vocabulary (rule 12):
    // prefer the live meter's per-node prices, fall back to the exact mirror.
    // NEVER defaultCountTokens — that overdraws the meter on CJK (issue #54).
    const shadowedTokens = shadowedTokensViaMeter(session, shadowed, agent.ctx)
    const tier = block.tier === 2 || block.tier === 3 ? block.tier : 1
    const parentBlockIds = compactionIdsOfKernelBlocks(session, block.directBlockIds)
    // Provenance follows the LIVE route, not `agent.options`: after a mid-session
    // model switch the latter is a stale snapshot (the PREVIOUS route), so the
    // summary node would be stamped with a route the summary did not come from.
    const { provider, model } = routeFor(agent)
    const { compactionId } = runCompactionTransaction(session, {
      start,
      end,
      shadowedSeqs: shadowed,
      summary: [{ type: 'text', text: range.summary }],
      shadowedTokenCount: shadowedTokens,
      provider,
      model,
      tier,
      kernelBlockId: block.blockId,
      ...(range.topic === undefined ? {} : { topic: range.topic }),
      ...(parentBlockIds.length === 0 ? {} : { parentBlockIds }),
      // Record the kernel block's raw coverage so a restarted engine
      // rehydrates the SAME effective messages (a tier-2 block's coverage is
      // its parents' originals, not the checkpoint node).
      directMessageIds: block.directMessageIds,
      effectiveMessageIds: block.effectiveMessageIds,
      // B3：已绿验收读数随压缩块落盘（缺位=不写键）
      ...(range.verifiedReadings === undefined ? {} : { verifiedReadings: range.verifiedReadings }),
    })
    const adjusted = start !== range.startSeq || end !== range.endSeq
    // Always report the tier, even tier 1: a silently-downgraded distill
    // (boundary moved off the checkpoint seq → the kernel folds a plain
    // message) must be visible to the model immediately, or the model keeps
    // believing the distillation landed (issue #60, failure mode 2).
    const tierLabel = `, tier ${tier}`
    // B3 must be READABLE, not just durable: echo the recorded readings back in the
    // compress result, otherwise "later steps need not re-run them" is unreachable
    // (the field had no other production reader).
    const readingsLabel = range.verifiedReadings !== undefined && range.verifiedReadings.length > 0
      ? `, verified: ${range.verifiedReadings.join('; ')}`
      : ''
    const note = range.recovered === true
      ? ` (seqs ${range.startSeq}..${range.endSeq} were already shadowed — compressed the live remainder ${start}..${end})`
      : adjusted
        ? ` (adjusted from ${range.startSeq}..${range.endSeq} to balanced edges)`
        : ''
    lines.push(
      `  block ${compactionId.slice(0, 8)}: seqs ${start}..${end}, ${shadowed.length} messages shadowed${tierLabel}${readingsLabel}${note}`,
    )
    landedBlockIds.push(compactionId.slice(0, 8))
  }

  // Baseline for the nudge cost line: the request after a replace re-prefills
  // almost the whole surface once, and the delta is what that fold actually cost.
  if (applied.result.blocksCreated > 0) recordCompressCost(session, providerUsage(agent))

  // The `Compressed N block(s)` PREFIX is load-bearing — tests and the PTC caller
  // read it as "the call succeeded" — so the split detail rides AFTER it.
  // Exact singular/plural: the line is model-facing, and "1 range(s)" reads like a
  // template that was never filled in.
  const splitLine = splitRangeCount === 0
    ? ''
    : ` from ${splitRangeCount} range${splitRangeCount === 1 ? '' : 's'} (split around ${splitProtectedRows} protected line${splitProtectedRows === 1 ? '' : 's'})`
  const summaryLine = `Compressed ${applied.result.blocksCreated} block(s)${splitLine}, ~${applied.result.tokensCompressed} tokens reclaimed.`
  // Models routinely reconstruct their own history wrongly AFTER a fold: the
  // compress call is itself shadowed into the block it created, so next turn the
  // only trace left on the visible surface is this result text. Measured
  // (2026-11-25): an assistant described the call as "wrote the note file" and
  // later argued it had never compressed, while the UI showed the real call.
  // State the action first, in the imperative, so the reconstruction is anchored
  // to a sentence rather than an inference from line format.
  // The block id is known at render time — hand the model the real call, not a
  // placeholder it must translate. Second person + imperative: the reader is
  // whichever model resumes this session, and the fold happened inside the call
  // that is itself shadowed (measured 2026-11-25: an assistant described the
  // call as "wrote the note file" and later argued it had never compressed).
  const actionLine = landedBlockIds.length === 0
    ? ''
    : `You just called compress — the messages this fold covered are OFF the visible surface now. Read them back with recall({ blockId: "${landedBlockIds[0]}" }) or recall({ seq }).`
  // What the elected fold costs the NEXT request, in the currency the model
  // optimizes with (tokens, not seconds — the prefill rate is host-specific and
  // unverified here). Position fact (2026-11-15): a rewrite re-prefills EVERYTHING
  // after the rewrite point and nothing before it, which is where the fixed request
  // head (system prompt, tool registry, injected rows) lives — so that head is
  // outside this price. Do NOT justify the boundary with cache reuse: on another
  // provider a live session reported 88,775,441 uncached input tokens against 0 cache
  // read and 0 cache write (2026-11-24), so how much the host reuses is not something
  // this plugin can claim. Adding the paired request overhead here double-counted
  // that head (round-10 review B1); the overhead belongs only to the emergency
  // projection in src/index.ts.
  const condenseLine = condensePick === undefined || applied.result.blocksCreated === 0
    ? ''
        : (() => {
            const noteTok = defaultCountTokens(withSummaryFramePrefix(args.content?.[0]?.summary ?? ''))
            const netSaving = condensePick.tokens - noteTok
            const firstSpan = condensePick.spans[0]!
            const lastSpan = condensePick.spans[condensePick.spans.length - 1]!
            // The batch is priced from its EARLIEST rewrite point: everything after
            // the first span's start is re-prefilled once, and that includes every
            // later range of the same prefix fold.
            const firstIndex = indexOfSurfaceSeq(session, firstSpan.start)
            const foldCost = firstIndex < 0 ? 0 : (surfaceSuffixTokens(session)[firstIndex] ?? 0)
            // TIER 2 must never fold silently: distilling carriers replaces the originals
            // behind them with ONE summary, and those originals stop being readable at
            // message level (recall can still read the summary, not the bytes). Say that in
            // the result line, where the model reads what just happened.
            if (condensePick.kind === 'distill') {
              // M-3: the distill shape can arrive as a MULTI-span batch, so the landing
              // ref is whichever span actually ends on a carrier — the LAST span may be
              // a plain gap run after it.
              const lands = condensePick.spans
                .map((span) => span.endBlock)
                .find((block) => block !== null) ?? 'b?'
              const foldCost = firstIndex < 0 ? 0 : (surfaceSuffixTokens(session)[firstIndex] ?? 0)
              return `  distilled ${condensePick.distilledBlocks} blocks into 1 across seqs ${firstSpan.start}..${lands} (tier 2 \u2014 the originals behind those carriers are no longer recoverable at message level; recall reads the new summary, not the bytes), one re-prefill of ~${foldCost}t; the same call also folded ~${condensePick.tokens}t of gap prose that was under the ~${MIN_CONDENSE_SPAN_TOKENS}t a normal fold needs.`
            }
            const digest = condensePick.carrierCount > 0
              ? `, distilling ${condensePick.carrierCount} block(s)`
              : ''
            const paybackCopy = foldCost === 0
              ? 'pays back immediately (nothing sits after the rewrite point)'
              : `pays back in ${formatPayback(foldCost, condensePick.tokens)}`
            return `  condense folds ${condensePick.tokens}t (seqs ${firstSpan.start}..${lastSpan.end}${condensePick.spans.length > 1 ? ' in ' + condensePick.spans.length + ' ranges' : ''})${digest}, re-prefills ${foldCost}t once — ${paybackCopy} (every token AFTER the rewrite point; the fixed request head sits BEFORE it and is outside this price) — and the note (~${noteTok} tokens) stays resident after it, saving ~${netSaving} per request thereafter.\n  Batching spans into ONE call pays a single re-prefill, measured from the EARLIEST rewrite point of that batch; splitting them across calls pays one re-prefill each, so batch only what you would fold anyway.`
          })()
  // A-2（2026-11-23）：折完立刻回报「还剩多少可折」。
  // 用户实测炸会话的直接形态：第 81 turn 压掉 51996 tokens 后连续 6 个 turn 无任何
  // 压缩动作，然后撞 300K —— 模型把「折了一次」当成「折够了」。这里把剩余候选直接
  // 摆出来，并明说「一次付款一回」的节奏（每次调用各付一次 re-prefill、各留一个屏障）。
  const nextFoldLine = (() => {
    if (applied.result.blocksCreated === 0) return ''
    let next: CondensePick | CondenseMiss
    try {
      // Re-picked against the CURRENT surface: the fold just landed, so the
      // remaining geometry is what the model would face on its next call.
      next = pickCondenseSpan(session, turn, 5, foldPricingFor(env))
    } catch {
      // Advisory only — never let the follow-up hint break a landed fold.
      return ''
    }
    if ('reason' in next) {
      // Only ranges a condense could actually take: a stretch under
      // MIN_CONDENSE_SPAN_TOKENS can never carry a handoff note, so reporting it
      // every single call is pure noise (and the refusal text for it is a
      // paragraph long). Silence there is the honest answer — the surface simply
      // has nothing foldable LEFT, only residue.
      const useful = next.candidates.filter((range) => range.tokens >= MIN_CONDENSE_SPAN_TOKENS)
      if (useful.length === 0) return ''
      const widest = useful.reduce((max, range) => (range.tokens > max.tokens ? range : max), useful[0]!)
      return `  Still foldable: ${useful.length} range(s) remain — the widest is seqs ${widest.start}..${widest.end} (~${widest.tokens} tokens), held back by the working-set floor for now; the next nudge re-prices them against a shorter surface.`
    }
    const firstSpan = next.spans[0]!
    const lastSpan = next.spans[next.spans.length - 1]!
    const digest = next.carrierCount > 0 ? `, distilling ${next.carrierCount} block(s)` : ''
    // Multi-span batches are POSITION-ordered, so lastSpan.end can be numerically
    // SMALLER than firstSpan.start (the last run carries a checkpoint with a high
    // seq). Printing the raw numbers reads as a reversed/invalid range (live
    // 2026-11-26: "seqs 2441..2423" after a tier-2 batch). When the value order
    // disagrees with the position order, print each span separately and say so.
    const prefixDesc = firstSpan.start <= lastSpan.end
      ? `seqs ${firstSpan.start}..${lastSpan.end}`
      : `${next.spans.length} span(s) in position order (checkpoint seqs sit high): ${next.spans.map((s) => `${s.start}..${s.end}`).join(', ')}`
    return `  Still foldable: ${next.eligibleRangeCount} range(s) remain — the foldable prefix is ${prefixDesc} (~${next.tokens} tokens${digest}). Fold it in a LATER turn: each call pays its own re-prefill and leaves its own barrier, so stacking folds inside one turn is the expensive way.`
  })()
  const totalSkipped = skippedRanges + alreadyCompressedNotes.length + duplicateRangeNotes.length + rejectedNotes.length
  const failedLines = applied.result.errors.map((error) => `  ${error}`)
  const condenseAdvisories = condensePick === undefined || applied.result.blocksCreated === 0
    ? []
    : [
        ...condenseNoteAdvisories,
        ...(condenseNoteGaps(args.condense?.summary ?? '').length > 0
          ? ['  condense note is missing recommended field(s): ' + condenseNoteGaps(args.condense?.summary ?? '').join(', ') + ' — Pointers and Dead ends are the ones a resumed session re-derives most expensively.']
          : []),
      ]
  const warningLines = [
    ...condenseAdvisories,
    // The shared-note cost, stated instead of hidden: nothing was lost, but a note
    // written for the whole span is resident once per cut block, so the model knows
    // why a split costs more than the block sizes suggest.
    ...(splitBlocks === 0
      ? []
      : [`  ${splitBlocks} block(s) came from ${splitRangeCount} range(s) cut around ${splitProtectedRows} protected line(s): each cut block carries the model's WHOLE-SPAN note plus its own [split i/N …] coverage line, so the note is resident once per block (~${splitExtraTokens} extra tokens in total) — nothing was lost, and no bytes moved between blocks.`]),
    ...freeWarnings.map((warning) => `  ${warning}`),
    ...failedLines,
    ...alreadyCompressedNotes,
    ...duplicateRangeNotes,
    ...rejectedNotes,
    ...swappedNotes,
    ...lines,
  ]
  const footer = totalSkipped > 0
    ? `  (${totalSkipped} range(s) skipped or failed — see above)`
    : ''
  return { text: `${summaryLine}\n${[actionLine, ...warningLines, condenseLine, nextFoldLine, footer].filter((line) => line !== '').join('\n')}` }
}
