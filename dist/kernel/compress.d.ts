import { createInitialState } from "./state.ts";
import type { ApplyRetrieveResult } from "./ccr.ts";
import type { MessageContentStore } from "./content-store.ts";
import { type PipelineNode } from "./pipeline.ts";
import type { ApplyCompressionResult, CompressionBlock, CompressionState, Config, CoreMessage, ProcessTurnResult, StatusReport } from "./types.ts";
export interface Ports {
    countTokens?: (text: string) => number;
}
export interface CompressionCore {
    processTurn(input: ProcessTurnInput): ProcessTurnResult;
    /** Kernel primitive behind acp_retrieve: resolve a ref to its stored
     *  original plus the wire-safe injection pair (see ccr.ts). */
    retrieve(store: MessageContentStore, ref: string): ApplyRetrieveResult;
    applyCompression(input: ApplyCompressionInput): ApplyCompressionResult;
    defaultNodes(): PipelineNode[];
    decompress(blockId: string, state: CompressionState): CompressionBlock | undefined;
    search(query: string, state: CompressionState): CompressionBlock[];
    status(state: CompressionState, tokenCount: number, config: Config): StatusReport;
}
export interface ProcessTurnInput {
    messages: CoreMessage[];
    state: CompressionState;
    config: Config;
    tokenCount: number;
    /** Per-session CCR content store (pass result.contentStore back here).
     *  Omit for pre-CCR sessions — an empty store is used. */
    contentStore?: MessageContentStore;
}
export interface ApplyCompressionInput {
    ranges: {
        startRef: string;
        endRef: string;
        summary: string;
        topic?: string;
        compressCallId?: string;
        summaryMaxChars?: number;
    }[];
    messages: CoreMessage[];
    state: CompressionState;
    config: Config;
    protectedMessageIds?: Set<string>;
}
export declare function createCore(ports?: Ports): CompressionCore;
export { createInitialState };
