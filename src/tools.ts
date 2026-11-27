/** makeTools entry point: binds the four tools to one engine (imports the split modules). */

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
import { CompressArgs, compressParameters } from './compress-args.ts'
import { handleCompress } from './condense-apply.ts'
import { RecallArgs, handleRecall } from './recall.ts'
import { StatusArgs, handleStatus, statusParameters } from './status.ts'
import { compressParametersFor, recallParametersFor, statusParametersFor } from './params.ts'
import { textOutput, type ToolEnvironment } from './shared.ts'

// M-3 (2026-11-25): the file was split by responsibility (shared/compress-args/
// condense-elect/condense-note/condense-apply/recall/status/params); this entry
// module re-exports the names external consumers (commands.ts, index.ts, tests)
// imported from './tools.ts' so their imports stay unchanged.
export type { ToolEnvironment } from './shared.ts'
export {
  resolveEffectiveWindow,
  guardedRowsInSpan, splitAroundProtected, splitPiecePosition, splitCoverageLine,
  liveCarrierRejectionNote, protectedRowRejectionNote, edgeRefForSeq,
} from './shared.ts'
export { mnRefIndex, compressParameters } from './compress-args.ts'
export type { CondensePick, CondenseSpan, CondenseMiss } from './condense-elect.ts'
export { MIN_CONDENSE_SPAN_TOKENS } from './region.ts'
export { pickCondenseSpan } from './condense-elect.ts'
export { condenseFloorRefusal, condenseNoCandidateReason, describePrefixSpans, formatPayback, paybackLabel } from './condense-note.ts'
export {
  condenseNoteGaps, missingCondenseHeadings, noteWeightRatio, noteWeightRejection,
  truncateNoteToCeiling, MAX_NOTE_WEIGHT_RATIO, TYPICAL_HANDOFF_NOTE_TOKENS,
  MAX_HANDOFF_NOTE_TOKENS, CONDENSE_BATCH_SPLIT_TOKENS, FORCED_BATCH_MIN_ADVANTAGE,
} from './condense-note.ts'
export { buildSearchDocs } from './recall.ts'
export { compressParametersFor, recallParametersFor, statusParametersFor } from './params.ts'

/**
 * Build the three ACP model tools bound to one engine.
 *
 * The four MAIN tools are `textOverrides`-addressable today: `compress`/`recall`/
 * `acp_status` through `prompts.tools.*` and the compress sheet's `condense`
 * parameter through `compressesParametersFor` above. TODO(textOverrides): the
 * remaining ~26 parameter `description` strings (compress content items, force,
 * verifiedReadings, recall's query/blockId/seq/chars/maxResults/offset/pageLimit,
 * acp_status's detail/tool/sort/limit) are still hard-coded — wiring them means
 * turning `compressParameters`/`recallParameters`/`statusParameters` into
 * prompt-derived builders, which collides with the in-flight content-path work in
 * this file; left as a deliberate follow-up rather than a half-migration.
 */
export function makeTools(env: ToolEnvironment): ToolDefinition[] {
  const prompts = env.prompts ?? DEFAULT_RESOLVED
  return [
    defineTool({
      name: 'compress',
      description: prompts.tools.compress,
      parameters: compressParametersFor(prompts),
      output: textOutput(),
      async execute(args, exec) {
        return handleCompress(env, args as CompressArgs, exec)
      },
    }),
    defineTool({
      name: 'recall',
      description: prompts.tools.recall,
      parameters: recallParametersFor(prompts),
      output: textOutput(),
      execute(args, exec) {
        return Promise.resolve(handleRecall(env, args as RecallArgs, exec))
      },
    }),
    defineTool({
      name: 'acp_status',
      description: prompts.tools.acpStatus,
      parameters: statusParametersFor(prompts),
      output: textOutput(),
      execute(args, exec) {
        return handleStatus(env, args as StatusArgs, exec)
      },
    }),
  ]
}
