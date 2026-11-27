/** textOverrides param patches: withParamTexts + per-tool parameter surfaces. */

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
import { compressParameters } from './compress-args.ts'
import { recallParameters } from './recall.ts'
import { statusParameters } from './status.ts'

/**
 * 换掉一个 `description`(不可变):沿路径浅拷贝,值没变就返回 `undefined` — 调用方
 * 据此跳过「无变化也新建对象」。
 */
function replaceDescriptionAt(
  node: unknown,
  path: readonly (string | number)[],
  description: string,
): unknown {
  if (path.length === 0) {
    if (node === undefined || node === null) return undefined
    if ((node as { description?: string }).description === description) return undefined
    return { ...(node as Record<string, unknown>), description }
  }
  const [head, ...rest] = path
  const child = (node as Record<string | number, unknown> | undefined)?.[head as never]
  if (child === undefined) return undefined
  const patched = replaceDescriptionAt(child, rest, description)
  if (patched === undefined) return undefined
  const clone: Record<string | number, unknown> | unknown[] = Array.isArray(node)
    ? [...node]
    : { ...(node as Record<string, unknown>) }
  ;(clone as Record<string | number, unknown>)[head as never] = patched
  return clone
}

/** `[路径, textOverrides id]` — 参数级文案的搬运清单,三条工具各一张。 */
type ParamPatch = readonly [readonly (string | number)[], string]

/**
 * 参数级文案替换。**未覆盖时返回入参对象本身**(与 `DEFAULT_TOOL_PARAM_TEXTS`
 * 引用相等即是"没覆盖"),所以默认路径零分配 — 与 `tool.condense` 的做法一致。
 * 只改 `description`;`type`/`required`/`enum`/`oneOf` 结构原样保留。
 */
function withParamTexts<T>(base: T, prompts: ResolvedPrompts, patches: readonly ParamPatch[]): T {
  if (prompts.toolParamTexts === DEFAULT_PROMPTS.toolParamTexts) return base
  let out: unknown = base
  for (const [path, id] of patches) {
    const text = prompts.toolParamTexts[id]
    if (text === undefined) continue
    const patched = replaceDescriptionAt(out, path, text)
    if (patched !== undefined) out = patched
  }
  return out as T
}

const COMPRESS_PARAM_PATCHES: readonly ParamPatch[] = [
  [['arguments'], 'compress.arguments'],
  [['topic'], 'compress.topic'],
  [['content'], 'compress.content'],
  [['content', 'items', 'properties', 'startSeq', 'oneOf', 0], 'compress.startSeq'],
  [['content', 'items', 'properties', 'startSeq', 'oneOf', 1], 'compress.startSeq.text'],
  [['content', 'items', 'properties', 'endSeq', 'oneOf', 0], 'compress.endSeq'],
  [['content', 'items', 'properties', 'endSeq', 'oneOf', 1], 'compress.endSeq.text'],
  [['content', 'items', 'properties', 'summary'], 'compress.summary'],
  [['content', 'items', 'properties', 'topic'], 'compress.item.topic'],
  [['content', 'items', 'properties', 'verifiedReadings'], 'compress.verifiedReadings'],
  [['force'], 'compress.force'],
  [['condense', 'properties', 'topic'], 'compress.condense.topic'],
  [['condense', 'properties', 'force'], 'compress.condense.force'],
  [['condense', 'properties', 'verifiedReadings'], 'compress.condense.verifiedReadings'],
]

const RECALL_PARAM_PATCHES: readonly ParamPatch[] = [
  [['query'], 'recall.query'],
  [['blockId'], 'recall.blockId'],
  [['seq'], 'recall.seq'],
  [['chars'], 'recall.chars'],
  [['maxResults'], 'recall.maxResults'],
  [['offset'], 'recall.offset'],
  [['pageLimit'], 'recall.pageLimit'],
]

const STATUS_PARAM_PATCHES: readonly ParamPatch[] = [
  [['detail'], 'acpStatus.detail'],
  [['tool'], 'acpStatus.tool'],
  [['sort'], 'acpStatus.sort'],
  [['limit'], 'acpStatus.limit'],
]

/**
 * The compress parameter sheet with the model-facing copy replaced by the
 * RESOLVED text. Returns the shared constant object untouched (same reference)
 * when nothing was overridden, so the default path allocates nothing and
 * `tests/g5-governance.test.ts` keeps reading the shipped sheet.
 *
 * Two layers ride here: the TOOL-level `condense` description (`tool.condense`)
 * and the 14 PARAMETER-level texts (`tool.param.compress.*`).
 */
export function compressParametersFor(prompts: ResolvedPrompts) {
  let out = compressParameters
  if (prompts.tools.condense !== DEFAULT_PROMPTS.tools.condense) {
    out = { ...out, condense: { ...out.condense, description: prompts.tools.condense } }
  }
  return withParamTexts(out, prompts, COMPRESS_PARAM_PATCHES)
}

/** `recall` 的 7 个参数文案(`tool.param.recall.*`)。 */
export function recallParametersFor(prompts: ResolvedPrompts) {
  return withParamTexts(recallParameters, prompts, RECALL_PARAM_PATCHES)
}

/** `acp_status` 的 4 个参数文案(`tool.param.acpStatus.*`)。 */
export function statusParametersFor(prompts: ResolvedPrompts) {
  return withParamTexts(statusParameters, prompts, STATUS_PARAM_PATCHES)
}
