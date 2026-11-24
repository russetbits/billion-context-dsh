/**
 * Inlined acp-kernel (0.0.101) public surface.
 *
 * Source restored from the npm package's dist sourcemaps (`sourcesContent`) —
 * the published bundle ships no `src/`, and the upstream git clone carries no
 * 0.0.101 tag, so the sourcemap is the only faithful source. `types.ts` and
 * `filter/types.ts` are pure-type files that TypeScript erases at build time,
 * so they come from `dist/*.d.ts` instead.
 *
 * Only the closure this plugin actually imports is vendored: 45 files /
 * ~11.9K lines, out of the package's full 73 files / 19,360 lines. Everything
 * outside the closure — wire adapters (3.3K), panel, persist, cache-report,
 * parse-compress-input, image-compress, packs — is deliberately absent; we do
 * not drive any of it.
 *
 * Import specifiers inside `src/kernel/` must use `.ts` extensions: the
 * restored sources carry `.js` (they were written for a bundler) and `tsx`
 * does not rewrite `.js` → `.ts` at runtime.
 */

export { createCore } from './compress.ts'
export { COMPRESS_PHILOSOPHY, HOW_TO_COMPRESS_RULES, TIER2_DISTILL_RULES, TIER3_CONDENSE_RULES } from './compression-rules.ts'
export { defaultConfig } from './config.ts'
export { renderNudgeText } from './nudge-text.ts'
export { buildStatusReport } from './report.ts'
export { setDocCacheCap } from './search/doc-cache.ts'
export { searchBlocks } from './search/index.ts'
export { createInitialState } from './state.ts'
export { defaultCountTokens } from './tokenize.ts'

export type { CompressionCore } from './compress.ts'
// Two MessageRole definitions exist in the package: `types.ts` (includes
// "system") and `search/types.ts` (user|assistant|tool). The package's public
// surface exports the SEARCH one, and that is what our tools.ts has always
// been typed against — export the same one or the host's role type clashes.
export type { MessageRole, SearchDoc } from './search/types.ts'
export type {
  CompressionBlock,
  CompressionState,
  Config,
  ContextBreakdown,
  CoreMessage,
  NudgeDecision,
} from './types.ts'
