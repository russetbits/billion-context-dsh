/** Shared tool plumbing: ToolEnvironment, output wrapper, edge-ref helpers, guard/split helpers. */

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

export interface ToolEnvironment extends KernelConfigInput {
  readonly kernel: CompressionCore
  readonly store: AcpStateStore
  /** Display-only: the named preset that produced the nudge thresholds above, if any (`/acp-prune status` names it). Never read by the kernel path. */
  readonly preset?: PresetName
  /** Resolve the effective context window for an agent (optional: status falls back to modelContextLimit). */
  readonly windowFor?: (agent: Agent) => Promise<AcpWindow>
  /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
  readonly prompts?: ResolvedPrompts
  /**
   * `textOverrides` diagnostics (unknown ids / oversized values). Rendered as one
   * self-check line in `acp_status` — a composition-row config otherwise has no
   * visible surface where a typo could show up. Absent = nothing to report.
   */
  readonly textWarnings?: readonly string[]
  /** How the model can REACH these tools in the current step (src/ptc.ts). The
   *  engine swaps in the PTC variant of `prompts` together with this, so the two
   *  never disagree; a stand-in env (tests, plain compositions) defaults to
   *  `native` and keeps the direct-call copy byte-identical. */
  readonly toolCallStyle?: ResolvedCallStyle
  /**
   * Call ids of compress invocations that created a durable block. The engine
   * listens for the matching `tool/result` and hides the call/result pair from
   * the surface, preventing the compaction summary from sitting between them
   * (strict providers reject that sequence with HTTP 400).
   */
  readonly compressCallIdsToHide?: Set<string>
  /**
   * Read/write access to the runtime settings layer for `/acp-prune config`.
   * Absent surfaces (never expected — the engine always builds one) would
   * degrade the command to advice text.
   */
  readonly settingsCommand?: SettingsCommandSurface
}

export interface TextOutput {
  text: string
}

export function textOutput(): {
  schema: { type: 'object'; properties: { text: { type: 'string' } }; additionalProperties: boolean }
  render: (args: unknown, value: TextOutput) => import('@deepseek-ai/dsh-llm').ContentBlock[]
} {
  return {
    schema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      additionalProperties: false,
    },
    render: (_args, value) => [{ type: 'text', text: value.text }],
  }
}

export function requireAgent(exec: ToolRunContext): Agent {
  if (exec.agent === undefined) {
    throw new Error('billion-context-dsh: tool requires an agent execution context')
  }
  return exec.agent
}

/**
 * Resolve the effective context window for a tool or command run: probe the
 * agent's real window via `windowFor` when provided, otherwise fall back to
 * the environment's `modelContextLimit`. Shared by the compress and
 * acp_status tool handlers and the `/acp-prune` command so the resolution logic
 * lives in exactly one place (issue #63 — the tools used the 128K fallback
 * for pressure decisions even when auto-detection had found a larger window).
 */
export async function resolveEffectiveWindow(env: ToolEnvironment, agent: Agent): Promise<AcpWindow> {
  return env.windowFor === undefined
    ? { limit: env.modelContextLimit, source: 'explicit' as const }
    : await env.windowFor(agent)
}

/**
 * Pure gate helpers for the compress tool's CURRENT-instruction-row rejection.
 *
 * Decision history (issue #71 review): the first draft only WARNED when a
 * manual compress range swallowed a current injected row (F7), because the
 * compression is safe and self-healing. The owner reversed that during PR1
 * review: compressing a CURRENT row has NO legitimate outcome — the host
 * re-injects the newest AGENTS.md copy unconditionally the moment it leaves
 * the surface (presence gate, deepseek-harness
 * packages/context/agent-instructions/src/index.ts:137/:163), so the tokens
 * come straight back and the call is pure waste — and a hard reject keeps the
 * manual path consistent with the system-side GC's iron rule (PR2: never
 * clear a group's newest row). STALE copies stay compressible: removing them
 * while the newest stays visible is the actual cleanup and triggers no
 * re-injection. Skill-catalog rows joined the guard set for a DIFFERENT
 * reason (issue #185): they are not self-healing at all — the host resends
 * the catalog only when its digest changes, so a folded catalog stays gone
 * until the catalog actually changes. The range table
 * (buildCompressibleSeqRanges) never offers these rows, so the gate only fires
 * on hand-built ranges.
 *
 * `guardedRowsInSpan` is the overlap probe. It takes the POSITIONAL span the
 * transaction will actually shadow (`shadowedSeqsOf`), never a numeric
 * `start <= seq <= end` interval: the surface is locally non-monotonic after
 * earlier replacements (a checkpoint seq spliced ahead of older residual
 * nodes), so a tier-2 distill of two checkpoints can carry a CURRENT
 * instruction row numerically inside its edges while the sliced span excludes
 * it — the interval probe rejected exactly the call the nudge hands the model
 * (issue #71 review B1). Probing the slice also keeps guard and effect in
 * agreement: `shadowedSeqsOf` is what the transaction prices and
 * `assertProvenance` verifies.
 * `protectedRowRejectionNote` renders the rejection the model sees: it names
 * the offending seqs AND the compressible slices left in the span, so the model
 * can re-cut (or split into two calls) instead of retrying the same call.
 * `guardedSurfaceSeqsOf` supplies the protected set.
 */
export function guardedRowsInSpan(guarded: ReadonlySet<number>, shadowed: readonly number[]): number[] {
  const inSpan = new Set(shadowed)
  return [...guarded].filter((seq) => inSpan.has(seq)).sort((a, b) => a - b)
}

/**
 * Maximal runs of a shadowed surface slice that contain NO protected row.
 *
 * The slice is POSITIONAL (`shadowedSeqsOf`), never a numeric interval: a checkpoint
 * node carries a seq far higher than its neighbours, so `>=`/`<=` arithmetic would
 * cut the wrong side (the same reason `guardedRowsInSpan` scans a set). A protected
 * row at an EDGE simply yields one piece; a span that is nothing but protected rows
 * yields none, which is the caller's signal to keep the rejection.
 */
export function splitAroundProtected(shadowed: readonly number[], hits: readonly number[]): number[][] {
  const blocked = new Set(hits)
  const pieces: number[][] = []
  let current: number[] = []
  for (const seq of shadowed) {
    if (blocked.has(seq)) {
      if (current.length > 0) pieces.push(current)
      current = []
      continue
    }
    current.push(seq)
  }
  if (current.length > 0) pieces.push(current)
  return pieces
}

/**
 * Where one surviving piece sits relative to the cut points, in words a reader of
 * the block can act on: "BEFORE it", "AFTER them", or "between protected rows 2
 * and 3". Positions come from the POSITIONAL slice, never from seq values.
 */
export function splitPiecePosition(
  shadowedSpan: readonly number[],
  protectedSeqs: readonly number[],
  piece: readonly number[],
): string {
  const blocked = new Set(protectedSeqs)
  const hitPositions = shadowedSpan
    .map((seq, index) => (blocked.has(seq) ? index : -1))
    .filter((index) => index >= 0)
  const first = shadowedSpan.indexOf(piece[0]!)
  const last = shadowedSpan.indexOf(piece[piece.length - 1]!)
  const pronoun = hitPositions.length === 1 ? 'it' : 'them'
  if (hitPositions.length === 0 || first < hitPositions[0]!) return `BEFORE ${pronoun}`
  if (last > hitPositions[hitPositions.length - 1]!) return `AFTER ${pronoun}`
  const preceding = hitPositions.filter((index) => index < first).length
  return `between protected rows ${preceding} and ${preceding + 1}`
}

/**
 * The mechanical coverage line a CUT block carries above the model's note.
 *
 * A split stores the model's note in EVERY piece, and that note was written for the
 * WHOLE span (seqs `spanStart..spanEnd`). Without this line a later `recall({ blockId })`
 * reads a note describing content the block does not hold, and — worse — takes it for
 * the block's complete record while the rest sits in a sibling block. The line makes
 * both facts explicit, costs one short row, and is used ONLY on cut blocks: a range
 * with no protected row inside stores the note byte-identically.
 */
export function splitCoverageLine(
  index: number,
  total: number,
  spanStart: number,
  spanEnd: number,
  protectedCount: number,
  where: string,
): string {
  const rows = `${protectedCount} protected row${protectedCount === 1 ? '' : 's'}`
  return `[split ${index}/${total} of seqs ${spanStart}..${spanEnd} around ${rows} — the note below covers the WHOLE span; this block holds only the part ${where}]\n`
}

/**
 * Advisory note for a plain range that would fold a LIVE block's checkpoint
 * carrier (see `liveCheckpointCarriersInSpan`). Names the block ids the model
 * can use instead — a block-ref boundary is the call that legitimately folds
 * carriers (tier 2/3 distillation).
 */
export function liveCarrierRejectionNote(
  start: number,
  end: number,
  carriers: readonly { seq: number; kernelBlockId: string }[],
): string {
  const list = carriers.map((carrier) => `seq ${carrier.seq} (${carrier.kernelBlockId})`).join(', ')
  const first = carriers[0]!
  // Name the LAST carrier as the end edge too: a valid multi-carrier span is a
  // tier-3 condense, and suggesting `b1..b1` would silently distill only the
  // first parent.
  const last = carriers[carriers.length - 1]!
  return `  seqs ${start}..${end} rejected — checkpoint ${list} carries the visible summary of a still-active block, and a plain seq range never supersedes one. Distill it with the block ids instead (compress({ content: [{ startSeq: "${first.kernelBlockId}", endSeq: "${last.kernelBlockId}", summary }] })), or cut the span around those seqs.`
}

export function protectedRowRejectionNote(
  start: number,
  end: number,
  hits: readonly number[],
  shadowed: readonly number[],
  session?: Session,
): string {
  const preview = hits.slice(0, 4).join(', ')
  const more = hits.length > 4 ? ` +${hits.length - 4} more` : ''
  const first = shadowed.indexOf(hits[0]!)
  const last = shadowed.indexOf(hits[hits.length - 1]!)
  const before = first > 0 ? shadowed.slice(0, first) : []
  const after = last >= 0 && last < shadowed.length - 1 ? shadowed.slice(last + 1) : []
  const slices = [before, after]
    .filter((slice) => slice.length > 0)
    .map((slice) => `${slice[0]}..${slice[slice.length - 1]}`)
  const recovery = slices.length === 0
    ? 'no part of this span is compressible — every row in it is protected content that must stay live, so there is nothing to fold here; pick an OLDER span instead (acp_status lists the live ranges)'
    : `the compressible part of this span is seq ${slices.join(' and ')} — the engine cuts around those rows automatically now, so re-sending this same range folds that part with no hand cut`
  // Why the guarded rows may not be folded, per channel. Without a session the
  // hits cannot be classified, so the wording stays the instruction-only text
  // this note has always carried (pinned by tests).
  let hasInstructions = false
  let hasCatalog = false
  let hasUserTurn = false
  if (session !== undefined) {
    for (const seq of hits) {
      const event = eventAtOf(session, seq)
      if (event === undefined) continue
      if (!hasInstructions && isAgentInstructionsRow(event)) hasInstructions = true
      if (!hasCatalog && isSkillCatalogRow(event)) hasCatalog = true
      if (!hasUserTurn && isRealUserTurn(event)) hasUserTurn = true
      if (hasInstructions && hasCatalog && hasUserTurn) break
    }
  }
  const reasons: string[] = []
  if (session === undefined) {
    reasons.push('the host re-injects the newest AGENTS.md copy the moment it leaves the surface, so compressing it reclaims nothing')
  } else if (!hasInstructions && !hasCatalog && !hasUserTurn) {
    // A session that identifies none of the hits must NOT claim an AGENTS.md
    // re-injection it cannot confirm: the rows stay guarded, the mechanism
    // stays unnamed.
    reasons.push('these rows must stay visible on the surface')
  } else {
    // Per-channel WHY (issue #185): the guarded channels fail differently when
    // folded — AGENTS.md comes straight back (host presence gate); a skill
    // catalog never does (resend keyed on the catalog digest, not visibility);
    // and the active user turn would vanish from the surface entirely.
    if (hasInstructions) {
      reasons.push('the host re-injects the newest AGENTS.md copy the moment it leaves the surface, so compressing it reclaims nothing')
    }
    if (hasCatalog) {
      reasons.push('a folded skill catalog is never re-sent — its resend gate is the catalog digest, which folding cannot change')
    }
    if (hasUserTurn) {
      reasons.push('the active user message must stay live to preserve conversation intent')
    }
  }
  // A hit set that is ONLY the active user turn must not be explained with the
  // AGENTS.md re-injection story (or its stale-copy escape): neither applies to
  // a user message. `guardedSurfaceSeqsOf` may also protect the newest real user
  // turn alongside the policy rows, hence the two labels.
  const userOnly = hasUserTurn && !hasInstructions && !hasCatalog
  const rowLabel = hasUserTurn ? 'CURRENT guarded row(s)' : 'CURRENT injected instruction row(s)'
  const staleCopyTail = userOnly ? '' : ' (older/stale copies of the same channel are fine to compress)'
  return `  seqs ${start}..${end} rejected — the span covers ${hits.length} ${rowLabel} (seq ${preview}${more}); ${reasons.join('; ')} — ${recovery}${staleCopyTail}`
}

/**
 * Kernel ref for one RESOLVED range edge's surface node (issue #155).
 * Resolved edges are tool-pairing-balanced and anchorable, but NO LONGER
 * guaranteed to carry a bare-`${seq}` ref: a multi-tool-call assistant
 * projects to `${seq}#${callId}` sub-ids, and an empty tool result projects
 * to nothing. Three tiers, in order:
 *
 *  1. the bare-seq id (user turns, single-call assistants, text-bearing results);
 *  2. the node's sub-ids in projection (= content) order — FIRST for a start
 *     edge (the kernel must consume the whole node from its first sub-message),
 *     LAST for an end edge (unreachable in practice: a node with open calls
 *     makes every cut after it unbalanced, so it can never be a resolved end
 *     edge);
 *  3. the nearest message-bearing LIVE node between this edge and the opposite
 *     edge (inclusive), walked along the SURFACE — never raw-id space, where
 *     shadowed nodes keep their refs and would anchor onto already-compressed
 *     messages. Walking inward only also guarantees the returned ref names a
 *     message the transaction actually shadows; a span whose interior carries
 *     no message at all (two adjacent empty results) yields undefined and the
 *     caller raises the existing "no assigned ref" error.
 */
export function edgeRefForSeq(
  session: Session,
  byRaw: Readonly<Record<string, string>>,
  seq: number,
  role: 'start' | 'end',
  oppositeSeq: number,
): string | undefined {
  const nodes = session.surface.nodes
  let index = -1
  let oppositeIndex = -1
  for (let i = 0; i < nodes.length; i += 1) {
    if (nodes[i] === seq) index = i
    if (nodes[i] === oppositeSeq) oppositeIndex = i
  }
  if (index < 0 || oppositeIndex < 0) return undefined
  const direct = anchorRefForNode(session, byRaw, seq, role)
  if (direct !== undefined) return direct
  const step = role === 'start' ? 1 : -1
  // Walk from just past this edge toward the opposite edge, INCLUSIVE of it:
  // the opposite edge's own ref is a legitimate answer when nothing in between
  // carries one (e.g. [empty result, text result]).
  for (let i = index + step; i !== oppositeIndex + step; i += step) {
    const ref = anchorRefForNode(session, byRaw, nodes[i]!, role)
    if (ref !== undefined) return ref
  }
  return undefined
}

/** Bare-seq ref for one node, else its multi-call sub-id ref (see edgeRefForSeq). */
function anchorRefForNode(
  session: Session,
  byRaw: Readonly<Record<string, string>>,
  seq: number,
  role: 'start' | 'end',
): string | undefined {
  const direct = byRaw[String(seq)]
  if (direct !== undefined) return direct
  const event = eventAtOf(session, seq)
  if (event?.type !== 'assistant/message') return undefined
  const content = (event.data as { message?: { content?: unknown } }).message?.content
  // The projection emits one sub-message PER CALL, including calls whose id is
  // absent (keyed `${seq}#`), so the gate must mirror that trigger exactly:
  // fewer than 2 calls means this node projected to a bare seq (tier 1 above).
  const ids = toolCallsOf(content).map((call) => call.id ?? '')
  if (ids.length < 2) return undefined
  // Projection order IS content order (projectEvent maps the calls in sequence).
  const ordered = role === 'start' ? ids : [...ids].reverse()
  for (const id of ordered) {
    const ref = byRaw[`${seq}#${id}`]
    if (ref !== undefined) return ref
  }
  return undefined
}

/**
 * One RANGE a condense submits — NOT one block (redesigned 2026-11-24).
 *
 * A prefix fold distils every active block the foldable prefix carries AND folds
 * the gaps between them in the SAME call, so the elected unit is the prefix, not
 * the individual block. `start`/`end` are surface seqs of the range's edges;
 * `endBlock` is the kernel block id when the END edge sits on that block's
 * checkpoint carrier (a `bN` end makes the kernel DISTIL the block, tier 2/3),
 * and `carriers` names every active block the range distils.
 */
