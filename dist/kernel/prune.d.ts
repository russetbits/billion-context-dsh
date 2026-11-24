import type { CompressionState, CoreMessage } from "./types.ts";
export declare const SUMMARY_HEADER = "[Compressed conversation section]";
/**
 * The transient visible id of an active block's rendered summary message.
 * This is a VIEW-ONLY representation: it must never be persisted into
 * `effectiveMessageIds`/`directMessageIds` (the durable coverage is the
 * block's raw message ids).
 */
export declare function summaryMessageId(blockId: string): string;
export declare function isSummaryMessageId(id: string): boolean;
/**
 * Base id of a message id: everything before the first `#`. Sub-id
 * projections (`base#callId`, `base#r0`, …) are alternate renderings of the
 * same original message, so coverage is decided per original message — id
 * comparisons normalize to the base (issue #231).
 */
export declare function baseIdOf(id: string): string;
/**
 * True when `id` falls under any covered original message, regardless of
 * which projection form the block recorded and which the current view emits.
 * `coveredBases` must be the set of `baseIdOf(c)` over the covered ids,
 * built once per pass (O(1) per message instead of a scan). The exact
 * membership test is subsumed: a covered id's own base is in the set.
 */
export declare function isCovered(id: string, coveredBases: Set<string>): boolean;
/**
 * True when a message is a rendered block summary (the exact shape prune
 * emits). The id prefix alone is not sufficient — a host-authored message
 * that happens to carry a reserved id must not be treated as a rendered
 * summary (it would be silently dropped from ranges or deleted by rebuild).
 */
export declare function isRenderedSummaryMessage(message: Pick<CoreMessage, "id" | "role" | "contentType">): boolean;
export interface PruneOptions {
    injectSummaries?: boolean;
}
export declare function prune(messages: CoreMessage[], state: CompressionState, options?: PruneOptions): CoreMessage[];
