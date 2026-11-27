/** compress tool argument surface: schema, seq/args parsing, envelope peel, items validation. */

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

/** Normalize a seq arg: number, "295", or "295#call_00_xxx" → 295. */
function parseSeq(value: number | string): number {
  const text = String(value).split('#')[0]!.trim()
  const seq = Number(text)
  if (!Number.isInteger(seq) || seq < 0) {
    throw new Error(`billion-context-dsh: invalid seq "${String(value)}" — use a surface seq like 295`)
  }
  return seq
}

/**
 * Match a drilldown mN ref: "m00306" / "m306" (kernel `refToIndex` semantics,
 * `m0*(\d{1,7})`), tolerating a trailing `#callId` fragment (symmetric with
 * `parseSeq`'s `#` handling). Returns the ref index, or null for non-mN input.
 *
 * The width AND the bound mirror the kernel's `REF_PATTERN` / `MAX_INDEX`
 * (0.0.101: `\d{1,7}` and `9999999`; through 0.0.63 both were 5 digits /
 * `99999`). They must be re-checked on every kernel bump (AGENTS.md §4b): a
 * session numbers refs upward and never reuses one, so once it outgrows the
 * old width the kernel keeps handing out `m100000`+ — and a host parser capped
 * at 5 digits rejects acp_status's OWN rows as "not a ref", in exactly the long
 * sessions a ref-width bump exists to rescue. `tests/tools.test.ts` round-trips
 * the kernel's `indexToRef` boundaries, so a future widening turns that test
 * red instead of degrading silently.
 */
const MN_RE = /^m0*(\d{1,7})(?:#.*)?$/i

/** Exported so the ref-width contract test can pin it without a 100K-message fixture. */
export function mnRefIndex(value: string): number | null {
  const match = MN_RE.exec(value.trim())
  if (match === null) return null
  const index = Number(match[1])
  return index >= 1 && index <= 9999999 ? index : null
}

/**
 * Resolve a compress boundary arg to a surface seq. Accepts:
 *  - a bare surface seq (number, "295", "295#call_00_x" — `parseSeq`);
 *  - a drilldown mN ref ("m00306" / "m306") — reverse-mapped via the CURRENT
 *    turn's `messageRefs.byRef` (CoreMessage.id = seq or "seq#callId" → split
 *    on "#"). Unknown mN (never assigned on the current surface) fails with
 *    guidance; a valid mN whose span was already compressed falls through to
 *    the existing recover-stale / already-compressed semantics (rule 7);
 *  - a kernel block ref ("b1") — the id `acp_status` prints per active block and
 *    `recall` already accepts — resolved to that block's checkpoint seq
 *    (`summarySeqOfKernelBlock`). Naming the checkpoint node is what makes the
 *    fold a tier 2/3 distillation: the resolver then sees the edge ON the
 *    carrier, `blockRefForSummarySeq` hands the kernel the block ref back, and
 *    `liveCheckpointCarriersInSpan`'s guard stands down (that guard exists to
 *    refuse a PLAIN seq range swallowing a live carrier). Before this, the
 *    reject note's own advice — "distill it with the block ids" — led to
 *    `invalid seq "b1"`, i.e. the tool told the model to make a call no parser
 *    accepted. A block id that is not an ACTIVE block (superseded, or never
 *    existed) fails with guidance.
 * `byRef` MUST come from `turn.state.messageRefs` (after `processTurn`), not
 * the persisted store state: acp_status's turn is never persisted, so mN refs
 * shown in a drilldown (including refs for messages that arrived since the
 * last nudge/compress) only exist on the current turn's ref map — a lookup
 * against the stored state would report a false "unknown mN" and dead-loop
 * the model between acp_status and compress.
 */
export function parseBoundary(value: number | string, byRef: Record<string, string>, session: Session): number {
  const text = String(value).trim()
  if (/^b\d+$/.test(text)) {
    const seq = summarySeqOfKernelBlock(session, text)
    if (seq === null) {
      throw new Error(
        `billion-context-dsh: block "${text}" is not an active block — run acp_status for the live block ids `
        + '(a block-id boundary distills that block, tier 2/3)',
      )
    }
    return seq
  }
  const index = mnRefIndex(text)
  if (index === null) return parseSeq(value)
  // Normalize to the kernel's padded key ("m00306") — byRef holds exact keys.
  const ref = `m${String(index).padStart(5, '0')}`
  const raw = byRef[ref]
  if (raw === undefined) {
    throw new Error(
      `billion-context-dsh: mN "${text}" not found on the current surface — re-run acp_status for fresh refs (the surface may have moved)`,
    )
  }
  const seq = Number(String(raw).split('#')[0]!)
  if (!Number.isInteger(seq) || seq < 0) {
    throw new Error(
      `billion-context-dsh: mN "${text}" maps to a non-seq id "${raw}" — re-run acp_status`,
    )
  }
  return seq
}

export interface CompressArgs {
  /** Tolerated wrapped-arguments form (model-generated double-nesting). */
  arguments?: string | { content?: CompressArgs['content'] }
  topic?: string
  content?: Array<{ startSeq: number | string; endSeq: number | string; summary: string; topic?: string; verifiedReadings?: string[] }>
  /** Engine-picked fold: the engine names the span, the model writes the handoff note. */
  force?: boolean
  condense?: {
    summary: string
    topic?: string
    verifiedReadings?: string[]
    force?: boolean
    /** 段级 note：批内每段一行模型小结（主路径）；缺位时指针块落机械描述行兜底。 */
    spans?: Array<{ startSeq: number | string; endSeq: number | string; note: string }>
  }
}

/**
 * Unwrap the tolerated wrapped-arguments forms back to the canonical shape:
 * `{ arguments: "{\"content\": [...]}" }` or `{ arguments: { content: [...] } }`
 * → `{ content: [...] }`. The direct `{ content: [...] }` form passes through
 * untouched. Returns null when no form carries content (caller raises).
 */
export function unwrapCompressArgs(args: CompressArgs): CompressArgs | null {
  if (args.content !== undefined || args.condense !== undefined) return args
  if (args.arguments === undefined) return null
  let inner: unknown = args.arguments
  if (typeof inner === 'string') {
    try {
      inner = JSON.parse(inner)
    } catch {
      return null
    }
  }
  if (typeof inner !== 'object' || inner === null || Array.isArray(inner)) return null
  const inner2 = inner as { content?: unknown; condense?: unknown }
  if (inner2.content === undefined && inner2.condense === undefined) return null
  return {
    ...args,
    content: inner2.content as CompressArgs['content'],
    condense: inner2.condense as CompressArgs['condense'],
  }
}

/**
 * Peel the tolerated wrapped-arguments envelope `{ arguments: {…} }` that some
 * model channels emit for ANY tool — the same double-nesting that birthed
 * `unwrapCompressArgs` (live-verified on acp_status: a drilldown call arrived
 * as `{"arguments":{"scope":"compressed"}}` and was silently dropped, since
 * only compress unwrapped). The envelope may be an object or a JSON string;
 * inner keys win over outer duplicates. Args without an envelope pass through
 * untouched.
 */
export function unwrapEnvelope<T extends object>(args: T): T {
  const envelope = (args as { arguments?: unknown }).arguments
  if (envelope === undefined) return args
  let inner: unknown = envelope
  if (typeof inner === 'string') {
    try {
      inner = JSON.parse(inner)
    } catch {
      return args
    }
  }
  if (typeof inner !== 'object' || inner === null || Array.isArray(inner)) return args
  return { ...args, ...(inner as object) } as T
}

/**
 * Enforce the items-level `required` contract on the EFFECTIVE content, after
 * the wrapped-arguments envelope has been peeled. The DSH schema gate only
 * sees the model's top-level arguments object — when the call arrives wrapped
 * as `{ arguments: { content: [...] } }`, the top-level `content` property is
 * absent there (it lives inside the envelope), so the gate never checks the
 * items and a missing `summary`/`startSeq`/`endSeq` sailed through to the
 * kernel, which fails late with a field-less "Summary is empty" and sent live
 * sessions into a retry loop (the same failure mode the schema gate fix for
 * the direct form closed). Running the SAME check on the unwrapped content
 * closes that window for both forms, and produces the identical
 * `invalid arguments: missing required property "content[0].summary"` surface
 * by reusing the host's `ToolArgsError` instead of a hand-rolled format.
 * An empty/whitespace-only summary counts as missing (the kernel would
 * reject it anyway — fail early with the field name instead).
 */
export function validateContentItems(content: NonNullable<CompressArgs['content']>): void {
  const violations: string[] = []
  content.forEach((item, index) => {
    const path = `content[${index}]`
    if (item.startSeq === undefined) violations.push(`missing required property "${path}.startSeq"`)
    if (item.endSeq === undefined) violations.push(`missing required property "${path}.endSeq"`)
    if (typeof item.summary !== 'string' || item.summary.trim().length === 0) {
      violations.push(`missing required property "${path}.summary"`)
    }
  })
  if (violations.length > 0) throw new ToolArgsError(violations)
}

export const compressParameters = {
  // Tolerated wrapped-arguments form: some models emit
  // `{ "arguments": "{\"content\": [...]}" }` (double-nested) or
  // `{ "arguments": { "content": [...] } }` instead of the unwrapped
  // `{ "content": [...] }`. The old DSH validator surfaced this as
  // `invalid arguments: "arguments" must be an object` and the model retried
  // forever. `arguments` is accepted as an optional JSON node so the wrapped
  // shape passes schema validation; `handleCompress` unwraps it and falls back
  // to a clear runtime error when neither form carries content. `content` is
  // intentionally NOT `required: true` — a required property would reject the
  // wrapped shape before `handleCompress` can see it. The tool description
  // still tells the model content is mandatory.
  //
  // The items fields are the opposite case: startSeq/endSeq/summary MUST be
  // `required: true`. Without that, a model call that omits `summary` (only
  // startSeq/endSeq/topic present) passed schema validation and failed late
  // inside the kernel with "Summary is empty" — and live sessions showed the
  // model retrying the identical broken call in a loop. With the fields
  // required, the same call is rejected at the schema gate with
  // `missing required property "content[0].summary"`, which tells the model
  // exactly which field to add. (`recall` is the exception: ITS fields are
  // all optional because it has two mutually exclusive forms — `query` vs
  // `blockId`/`seq` — so the exclusivity is enforced by the handler with an
  // executable refusal, not by the schema gate.)
  arguments: { type: 'json', description: DEFAULT_TOOL_PARAM_TEXTS['compress.arguments'] },
  topic: { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.topic'] },
  content: {
    type: 'array' as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['compress.content'],
    items: {
      type: 'object' as const,
      properties: {
        startSeq: {
          required: true,
          oneOf: [
            { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.startSeq'] },
            { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.startSeq.text'] },
          ],
        },
        endSeq: {
          required: true,
          oneOf: [
            { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.endSeq'] },
            { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.endSeq.text'] },
          ],
        },
        summary: { type: 'string' as const, required: true, description: DEFAULT_TOOL_PARAM_TEXTS['compress.summary'] },
        topic: { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.item.topic'] },
        // B3 (2026-09-08 governance plan): the handler and region.ts have
        // accepted verifiedReadings since the plan landed, but the declared
        // parameter schema did not list it — `additionalProperties: false`
        // then rejected every live call that carried it
        // (`invalid arguments: "content[0].verifiedReadings" is not a declared
        // property`), so the structured-loss-stopping field was unreachable
        // from the model's tool interface. Declared here; additionalProperties
        // stays false so unknown fields are still rejected.
        verifiedReadings: {
          type: 'array' as const,
          items: { type: 'string' as const },
          description: DEFAULT_TOOL_PARAM_TEXTS['compress.verifiedReadings'],
        },
      },
      additionalProperties: false,
    },
  },
  force: {
    type: 'boolean' as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['compress.force'],
  },
  condense: {
    type: 'object' as const,
    // Single source: the default lives in prompts.ts so `tool.condense` is overridable.
    description: DEFAULT_PROMPTS.tools.condense,
    properties: {
      // A-7（2026-11-23）：原来是一行罗列六个字段名，模型记不住也照不出。
      // 改成可直接照抄的填空骨架 —— 骨架比字段清单少花认知，也天然带上
      // 「文件:行号 / 精确值 / 命令」这些真正降低重新发现成本的东西。
      summary: {
        type: 'string' as const,
        required: true,
        description: 'The handoff note replacing the elected stretch (minimum 50 characters). Fill in this skeleton — replace every <…>, do NOT paste the placeholders:\n'
          + '## Task\n<verbatim or near-verbatim copy of the task this stretch served \u2014 the user\'s own wording, not a retelling>\n'
          + '## Now\n<what you are doing right now, why, and the immediate next step>; done when <concrete criterion>\n'
          + '## Plan\n- <the decomposition of that task, and what counts as complete \u2014 when it is bigger than one step>\n'
          + '## Pointers\n- <file:line / seq / blockId / path + the one fact a resumed session must not re-derive>\n'
          + '  - ALSO carry VERBATIM any user-supplied value a later step must use as-is (credentials, ids, addresses, keys): none can be reconstructed, and dropping one forces the user to repeat it.\n'
          + '## Verification\n- <what you ran, what passed, what failed + the workaround>\n'
          + '## Dead ends\n- <tried X, failed because Y \u2014 so nobody retries it>\n'
          + 'This is a BRIEF FOR THE WORK AHEAD, not a narrative of what happened: pick the facts a resumed session would otherwise re-derive and drop the running commentary.\n'
          + 'Budget: 150-400 tokens is the sweet spot; anything over 1500 is truncated to its head. Nothing here is a gate \u2014 a missing heading is named in the result and the fold LANDS anyway, so a note never needs rewriting. ## Task and ## Now are the two that matter most; Pointers and Dead ends cost a resumed session the most to re-derive, so write them even when short.',
      },
      topic: { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.topic'] },
      force: { type: 'boolean' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.force'] },
      verifiedReadings: { type: 'array' as const, items: { type: 'string' as const }, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.verifiedReadings'] },
      spans: {
        type: 'array' as const,
        items: {
          type: 'object' as const,
          properties: {
            startSeq: {
              required: true,
              oneOf: [
                { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans.startSeq'] },
                { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans.startSeq'] },
              ],
            },
            endSeq: {
              required: true,
              oneOf: [
                { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans.endSeq'] },
                { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans.endSeq'] },
              ],
            },
            note: { type: 'string' as const, required: true, description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans.note'] },
          },
          additionalProperties: false as const,
        },
        description: DEFAULT_TOOL_PARAM_TEXTS['compress.condense.spans'],
      },
    },
    additionalProperties: false,
  },
} as const
