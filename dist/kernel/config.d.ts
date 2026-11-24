import type { Config } from "./types.ts";
export declare function defaultConfig(modelContextLimit: number, overrides?: Partial<Config>): Config;
export declare function validateConfig(config: Config): string[];
