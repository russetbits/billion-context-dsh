import type { CompressionState, CoreMessage } from "./types.ts";
/** Host-declared surface facts for status reports. The kernel never resolves
 *  packs — which pack is active is host policy (billion-context #730); the
 *  kernel only renders what the host declares about the active surface. */
export interface StatusReportMeta {
    pack?: string;
    packVersion?: string;
    host?: string;
}
export interface StatusReportOptions {
    scope?: "compressed" | "uncompressed";
    view?: "ranges" | "messages";
    tool?: string;
    sort?: "size" | "time" | "tool" | "age";
    limit?: number;
    meta?: StatusReportMeta;
}
export declare function buildStatusReport(state: CompressionState, messages: CoreMessage[], countTokens: (t: string) => number, options?: StatusReportOptions): string;
export declare function buildRecap(state: CompressionState, blockId?: string): string;
