import type { MessageFilter } from "./types.ts";
export declare function registerMessageFilter(filter: MessageFilter): void;
export declare function getMessageFilter(name: string): MessageFilter | undefined;
export declare function listMessageFilters(): MessageFilter[];
export declare function clearMessageFilters(): void;
