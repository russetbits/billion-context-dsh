import type { CompressionState, CoreMessage } from "./types.ts";
export interface SyncResult {
    state: CompressionState;
    deactivated: string[];
}
export declare function syncBlocks(messages: CoreMessage[], state: CompressionState): SyncResult;
