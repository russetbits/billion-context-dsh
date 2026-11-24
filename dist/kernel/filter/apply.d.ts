import type { CoreMessage } from "../types.ts";
import type { MessageFiltersConfig } from "./types.ts";
export interface ApplyResult {
    messages: CoreMessage[];
    partsFiltered: number;
    partsDropped: number;
    partsModified: number;
}
export declare function applyMessageFilters(messages: CoreMessage[], config: MessageFiltersConfig | undefined): ApplyResult;
