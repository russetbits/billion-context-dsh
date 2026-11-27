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
import { type ToolDefinition } from '@deepseek-ai/dsh-tools';
import { type ToolEnvironment } from './shared.ts';
export type { ToolEnvironment } from './shared.ts';
export { resolveEffectiveWindow, guardedRowsInSpan, splitAroundProtected, splitPiecePosition, splitCoverageLine, liveCarrierRejectionNote, protectedRowRejectionNote, edgeRefForSeq, } from './shared.ts';
export { mnRefIndex, compressParameters } from './compress-args.ts';
export type { CondensePick, CondenseSpan, CondenseMiss } from './condense-elect.ts';
export { MIN_CONDENSE_SPAN_TOKENS } from './region.ts';
export { pickCondenseSpan } from './condense-elect.ts';
export { condenseFloorRefusal, condenseNoCandidateReason, describePrefixSpans, formatPayback, paybackLabel } from './condense-note.ts';
export { condenseNoteGaps, missingCondenseHeadings, noteWeightRatio, noteWeightRejection, truncateNoteToCeiling, MAX_NOTE_WEIGHT_RATIO, TYPICAL_HANDOFF_NOTE_TOKENS, MAX_HANDOFF_NOTE_TOKENS, CONDENSE_BATCH_SPLIT_TOKENS, FORCED_BATCH_MIN_ADVANTAGE, } from './condense-note.ts';
export { buildSearchDocs } from './recall.ts';
export { compressParametersFor, recallParametersFor, statusParametersFor } from './params.ts';
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
export declare function makeTools(env: ToolEnvironment): ToolDefinition[];
