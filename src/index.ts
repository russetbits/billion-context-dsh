/**
 * billion-context-dsh — Active Context Pruning (ACP) for the DeepSeek Harness,
 * delivered as a `CompactionEngine` backend.
 *
 * The model decides when and what to compress (pure ACP semantics):
 *  - the `compress` tool durably shadows a surface range with the model-written
 *    summary (no second LLM summarization call — the ACP cost win);
 *  - the original events stay in the append-only session log, so `recall`
 *    (its search form and its read form) and replay always work;
 *  - refs are surface seqs carried by the injected nudge's range table (DSH
 *    has no in-memory message rewrite hook — see docs/dsh-porting-verification.md);
 *  - automatic policy never summarizes by itself: it nudges the model.
 *
 * Mount it wherever a compaction backend is expected:
 *
 * ```yaml
 * - id: compaction-billion-context
 *   name: 'billion-context-dsh'
 *   config:
 *     modelContextLimit: 128000
 * ```
 *
 * The package registers `ctx.compaction` plus the four model tools and the
 * `/acp-prune` command when the hosting composition provides `ctx.tools` /
 * `ctx.commands`.
 * @module billion-context-dsh
 */

import type { Context } from '@deepseek-ai/cordis'
import {
  CompactionEngine,
  ManualCompactionError,
  type CompactionAgentContext,
  type CompactionResult,
  type CompactionTrigger,
  type ManualCompactAgentContext,
} from '@deepseek-ai/dsh-compaction'
import { createCore, setDocCacheCap, type CompressionCore } from './kernel/index.ts'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import { CONTEXT_WINDOW_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import { DEFAULT_SESSION_CACHE_LIMIT, LruMap } from './lru.ts'
import { AcpStateStore } from './state.ts'
import { makeTools, type ToolEnvironment } from './tools.ts'
import { acpCommand } from './commands.ts'
import { healRegistrations } from './registration.ts'
import { buildNudge, EMERGENCY_NUDGE_MAX_PER_TURN, meterMediaPriceResolver, resolveTokenCount, type NudgeGate } from './nudge.ts'
import { ACP_SYSTEM_PROMPT_ORDER } from './system-prompt.ts'
import { applyTextOverrides, renderSystemPrompt, resolvePrompts, textOverrideWarningLines, type AcpPrompts, type ResolvedPrompts, type TextOverrides } from './prompts.ts'
import { DEFAULT_CONTEXT_WINDOW, probeModelWindow, projectedContextWindow, routeFor, type AcpWindow } from './window.ts'
import {
  buildCompressibleSeqRanges,
  deferCompressPairHide,
  foldSpanPricing,
  surfaceTokens,
  runCompactionTransaction,
  shadowedSeqsOf,
  stripOrphanedSurfaceToolMessages,
  pruneStaleNudgeRows,
  openToolCallIds,
  type KernelRangeView,
} from './region.ts'
import { allLogMessages, eventsToCoreMessages, extractEventText, overflowMarkerSummary, surfaceEventsOf, toolCallIdOfResultEvent } from './messages.ts'
import {
  detectPtcMode,
  looksLikeLeakedAcpCall,
  toPtcPrompts,
  type ResolvedCallStyle,
  type ToolCallStyle,
} from './ptc.ts'
import { shadowedTokensViaMeter } from './host-tokens.ts'
import { foldPricingFor, kernelConfigFor } from './config.ts'
import {
  DEFAULT_OVERFLOW_INFLATION,
  lastRequestUsage,
  pairedInflation,
  calibrateRequestPair,
  requestOverhead,
  setCalibrationAnchor,
  setCalibrationRoute,
} from './usage.ts'

/**
 * Ceiling on the smallest reclaim worth a full re-prefill on the emergency path.
 * A fold leaves a permanent un-foldable barrier, so a 1K reclaim in a large
 * window buys 1K of relief for one barrier — the worst barrier-per-byte ratio
 * possible (cross-review Q5).
 *
 * It is a CEILING, not a floor: the meaningful minimum is a fraction of the
 * window (OVERFLOW_FOLD_WINDOW_RATIO) capped by this number. An absolute floor
 * is unreachable on a small window — the e2e overflow-recovery scenario runs at
 * modelContextLimit 2000, so a hard 4096 made the emergency path refuse every
 * fold and disabled overflow recovery exactly where it matters most. Same class
 * of mistake the cross-review flagged for the degrade threshold (round-4 Q2).
 */
const MIN_OVERFLOW_FOLD_TOKENS = 4096
const OVERFLOW_FOLD_WINDOW_RATIO = 0.1
/**
 * How many disjoint spans one overflow recovery may fold inside a SINGLE retry (round-11
 * review Q5). The retry budget is separate and stays at the host default of 1: extra passes
 * cost barriers and re-prefill, never a retry. 3 is a ceiling against a pathological surface
 * where every span is barely above the minimum — the loop already stops as soon as the
 * projection fits or no eligible span remains, so this only bounds the worst case.
 */
const OVERFLOW_FOLD_PASSES = 3
import {
  ACP_SETTINGS_NAMESPACE,
  AcpPluginConfigSchema,
  AcpSettingsSchema,
  describeSettingsChange,
  filterSettingsEntry,
  liveSettingsFromRefs,
  makeSettingsCommandSurface,
  resolveAcpSettings,
  type AcpSettings,
  type AcpSettingsInput,
  type AcpSettingsService,
  type LegacySettingsService,
  type SettingsCommandSurface,
} from './settings.ts'
import { PRESETS, PRESET_NAMES, isPresetName, resolvePreset, type NudgePreset, type PresetName } from './presets.ts'

export { AcpStateStore } from './state.ts'
export { kernelConfigFor, type KernelConfigInput } from './config.ts'
export {
  PRESETS,
  PRESET_NAMES,
  isPresetName,
  resolvePreset,
  type NudgePreset,
  type PresetName,
} from './presets.ts'
export { ACP_SYSTEM_PROMPT, ACP_SYSTEM_PROMPT_ORDER } from './system-prompt.ts'
export {
  DEFAULT_PROMPTS,
  DEFAULT_RESOLVED,
  applyTextOverrides,
  ableTextIds,
  renderSystemPrompt,
  renderTemplate,
  resolvePrompts,
  textById,
  textOverrideWarningLines,
  type TextOverrideReport,
  type TextOverrides,
  type AcpPrompts,
  type NudgePrompts,
  type PromptInput,
  type PromptOverride,
  type RangeTablePrompts,
  type ResolvedPrompts,
  type ToolPrompts,
} from './prompts.ts'
export { makeTools, type ToolEnvironment } from './tools.ts'
export { acpCommand } from './commands.ts'
export { buildNudge, resolveTokenCount, EMERGENCY_NUDGE_MAX_PER_TURN, type NudgeEnvironment, type NudgeGate, type NudgeOutcome } from './nudge.ts'
export {
  DEFAULT_CONTEXT_WINDOW,
  detectContextWindow,
  projectedContextWindow,
  windowSourceLabel,
  type AcpWindow,
} from './window.ts'
export {
  AlreadyCompressedRangeError,
  rebuildBlockLedger,
  resolveSurfaceRange,
  runCompactionTransaction,
  shadowedSeqsOf,
  findOpenTurn,
  assertNoActiveCompaction,
  blockRegistry,
  blockRefForSummarySeq,
  compactionIdsOfKernelBlocks,
  summarySeqOfKernelBlock,
  expandShadowedSeqs,
  hideCompressToolPair,
  stripOrphanedSurfaceToolMessages,
  type AcpBlockLedgerEntry,
  type CompactionTransactionInput,
  type ResolvedSurfaceRange,
} from './region.ts'
export { eventsToCoreMessages, projectEvent, surfaceEventsOf, extractEventText } from './messages.ts'
export {
  ACP_SETTINGS_NAMESPACE,
  AcpPluginConfigSchema,
  AcpSettingsSchema,
  describeSettingsChange,
  filterSettingsEntry,
  isVolatileRef,
  liveSettingsFromRefs,
  makeSettingsCommandSurface,
  parseSettingValue,
  resolveAcpSettings,
  SETTINGS_KEYS,
  SETTING_DEFAULTS,
  unwrapVolatile,
  VOLATILE_WRITE,
  type AcpSettings,
  type AcpSettingsDescriptor,
  type AcpSettingsInput,
  type AcpSettingsService,
  type LegacySettingsService,
  type SettingsChangeEffect,
  type SettingsCommandSurface,
  type SettingsKey,
} from './settings.ts'

/**
 * The slice of the cordis loader this plugin touches. The base
 * `@deepseek-ai/cordis` types do not declare a `loader` field — the loader
 * package extends the context at runtime and provides the service itself
 * (`cordis-plugin-loader/lib/index.js`, `ctx.reflect.provide("loader", …)`).
 * A module augmentation would make the loader a compile-time dependency of a
 * plugin whose peer set does not include it, so the service is reached through
 * one structural assertion in `disableBasicOnce` instead.
 */
export interface LoaderEntryLike {
  options?: { id?: string }
  update(options: Record<string, unknown>): Promise<unknown> | unknown
}

export interface LoaderLike {
  entries(): Iterable<LoaderEntryLike>
}

/**
 * The isolate-realm fallback of `compaction-basic` is invisible to
 * `loader.entries()` — 实测（2026-11-27，`@deepseek-ai/cordis-plugin-loader` 1.2.0 +
 * `dsh-web-app/presets/cordis.patch.yml`）：cordis preset 把 compaction 组挂在 agent
 * preset mount 出来的独立 `EntryTree`（`dsh-agent-preset-registry` 的 `PresetTree`）里，
 * 主 loader 树的 `entries()` 永远枚举不到那棵树。但 cordis 的服务注册表是全局唯一的
 * （`ctx.reflect.store`，`cordis/lib/index.js` `provide()`）：每个 isolate realm 用一个
 * realm 私有 Symbol 作键（`compaction#compaction` 之类），impl 就存在同一个 store 里，
 * 且 `impl.ctx.fiber.entry` 直接给出对应的 loader Entry。所以 realm 内的 basic 走
 * store 扫描禁用，不走 loader。
 *
 * 判别（全部实测自 `cordis-plugin-loader/lib/index.js` `Realm.access` 与
 * `cordis/lib/index.js` `provide()`）：
 * - 服务名 `impl.name === 'compaction'`（`dsh-compaction` 的 `Service(ctx, "compaction")`）；
 * - 提供它的 entry 是 basic：`impl.ctx.fiber.entry?.options?.id === 'compaction-basic'`；
 * - 键不是根平面的 compaction symbol（根平面由 `loader.entries()` 路径负责，那里才
 *   有持久化的 patch 行；realm 的 basic 是 preset 组合的 fallback，禁掉即可）。
 */
const ISOLATE_SYMBOL = Symbol.for('cordis.isolate')

interface ReflectStoreEntry {
  name: string
  isolate?: symbol
  ctx: { fiber?: { entry?: LoaderEntryLike } }
}

export interface ReflectStoreLike {
  [key: symbol]: ReflectStoreEntry | undefined
}

export interface ReflectLike {
  store: ReflectStoreLike
}

export interface AcpConfig {
  /**
   * The context window used for pressure decisions, in tokens. When omitted,
   * `autoModelContextLimit` (default true) resolves it automatically: the live
   * host session projection (`contextPressure.contextWindow`) is preferred,
   * then the model's real window is probed via
   * `agent.ctx.llm.resolveModelInfo(provider, model)`; an explicit value
   * always wins and disables both.
   */
  readonly modelContextLimit?: number
  /** Auto-resolve the real context window: host session projection first, then the LLM runtime probe. Default true. */
  readonly autoModelContextLimit: boolean
  /** Nudge window lower bound (usage fraction; validation only — the growth-driven trigger has no percentage floor). Kernel default 0.45 — same as billion-context-pi. */
  readonly nudgeMinContextLimitPct?: number
  /**
   * Nudge window upper bound — over-limit guarantee line: above this the
   * kernel injects a nudge regardless of growth or cadence. Engine default
   * 0.70 (deliberately BELOW the kernel/billion-context-pi default 0.75 and
   * the host compaction-basic auto-compaction line 0.80, so the forced nudge
   * always fires first); an explicit value wins over this default — a
   * same-name key in `coreOverrides.nudge` wins over both (it merges last).
   */
  readonly nudgeMaxContextLimitPct?: number
  /**
   * Emergency nudge threshold (bypasses the per-turn dedup, but is capped at
   * EMERGENCY_NUDGE_MAX_PER_TURN = 3 injections per user turn — issue #108).
   * Engine default 0.85 (down from the kernel/billion-context-pi default 0.95:
   * 95% leaves the model no room to act before the API rejects, and the host's
   * 80% compaction-basic line shadows it in standard/code/cordis modes).
   */
  readonly nudgeEmergencyThresholdPct?: number
  /**
   * Named bundle for the three nudge thresholds — how eagerly the model is
   * asked to compress, in one word. One of 'preserve' | 'relaxed' | 'balanced'
   * | 'efficient' | 'aggressive' (see src/presets.ts). It fills ONLY the nudge
   * thresholds you did not set explicitly, so precedence is explicit value >
   * preset > engine default and a partial override on top of a preset still
   * wins. An unknown name fails engine construction (fail-fast). No effect on
   * any other knob (`modelContextLimit`, `autoNudge`, prompts, coreOverrides).
   * An unknown name fails construction, and so does a merged window that ends up
   * inverted (min > max, max > emergency or min > emergency — the kernel itself
   * only warns about that, see `assertNudgeThresholdOrder`).
   */
  readonly preset?: PresetName
  /**
   * Any other acp-kernel Config override (billion-context-pi's `coreOverrides`
   * escape hatch). Merge order per section: kernel defaults → the engine pct
   * knobs above → these keys land LAST, so a same-name key here wins.
   */
  readonly coreOverrides?: Partial<import('acp-kernel').Config>
  /**
   * Custom token-count function for the kernel's internal estimation.
   * Defaults to the kernel's `defaultCountTokens` (CJK: 1 char = 1 token,
   * other: 4 chars = 1 token — aligns with billion-context-pi).
   * Can be overridden for provider-specific tokenization, e.g. DeepSeek's
   * official coefficient: 1 CJK char ≈ 0.6 tokens, 1 other char ≈ 0.3 tokens.
   * Only affects the kernel's internal estimation (compressible range sizing,
   * nudge text, growth branch pending); the `projectedTokens` reading from
   * `sessionProjections` (used for nudge pressure decisions and acp_status)
   * is provider-anchored and unaffected by this function.
   */
  readonly countTokens?: (text: string) => number
  /** Register the four model tools on `ctx.tools`. Default true. */
  readonly autoTools: boolean
  /** Register the `/acp-prune` command on `ctx.commands`. Default true. */
  readonly autoCommand: boolean
  /** Inject the nudge into `agent/pre-step` when the kernel recommends it. Default true. */
  readonly autoNudge: boolean
  /**
   * How many consecutive provider-confirmed context-overflow failures one
   * agent may answer with an emergency compaction + retry before the original
   * error is preserved (mirrors compaction-basic's `maxOverflowRetries`;
   * default 1, the host's own default). The budget resets when the agent
   * makes progress (an assistant message lands) or returns to idle, so a
   * request that cannot be repaired cannot retry forever.
   */
  readonly maxOverflowRetries?: number
  /**
   * No-action degrade fold (方案 D layer 2): how many consecutive user turns a
   * session may sit at or above the emergency nudge threshold
   * (nudgeEmergencyThresholdPct, engine default 0.85) without a single
   * successful `compress` call before the engine folds the largest eligible
   * span itself in `agent/pre-step`. The counter increments once per
   * emergency nudge that actually reached the model and resets on a successful
   * compress. The degrade fires AT MOST ONCE per session (a loop guard), is
   * stamped with EMERGENCY PROVENANCE like the overflow path, and never
   * touches the overflow retry budget. Default 3; `0` disables the degrade
   * (the hard nudge copy remains). Mirrors what the host's compaction-basic
   * used to buy for free at its 80% line: a no-thought pressure reset.
   */
  readonly emergencyFoldAfterTurns?: number
  /**
   * Take over the host's own automatic compaction. A host composition that
   * ALSO loads `compaction-basic` keeps running that engine's own
   * `agent/pre-step` auto-compaction (its `auto` flag defaults to true), which
   * silently replaces the model's `compress` summaries with a host-written one
   * — measured on a live session: 121 messages / 77,280 tokens folded into a
   * 32-character summary that dropped every user constraint it covered.
   * `disableBasic` (default true) disables that loader entry at runtime, so ACP
   * owns the only compression path and no message is ever summarized by a
   * writer that does not know the note contract. Set false to keep the host's
   * native behaviour instead. The mount NEVER fails over this: an unreachable
   * loader/entry logs one warning and the plugin runs unchanged.
   */
  readonly disableBasic: boolean
  /**
   * Escape hatch: disable the runtime-settings integration entirely
   * (composition-layer ONLY — deliberately not exposed through the settings
   * layer itself: a switch that turns off its own plumbing could not be
   * reached if the plumbing broke). Default: enabled.
   */
  readonly settingsEnabled?: boolean
  /** Per-stage prompt template overrides (nudge / range table / system prompt / tool descriptions). See docs/configurable-prompts-design.md. */
  readonly prompts?: AcpPrompts
  /**
   * Flat id-addressed FULL REPLACEMENT of any model-visible text (`nudge.*`,
   * `rangeTable.*`, `tool.*`, `systemPrompt`); ids come from `ableTextIds()`.
   * Unlisted ids keep the English default. Applied AFTER `prompts` and BEFORE
   * the PTC rewrite, so an override still adapts to the active calling
   * convention. Non-string or unknown ids are ignored with one warning line —
   * a typo must never stop the plugin from starting. See src/config.ts for the
   * token-cost caveat (translation makes the copy heavier, not cheaper).
   */
  readonly textOverrides?: TextOverrides
  /**
   * How the model reaches the ACP tools. `auto` (default) detects PTC / code mode
   * from the assembled system prompt of each session every step; `native` or
   * `ptc` pin it (an operator override, or a host whose PTC copy changes).
   * Composition-layer only — deliberately NOT a settings knob, so the six-key
   * settings surface (AcpPluginConfigSchema) is untouched. See src/ptc.ts.
   */
  readonly toolCallStyle?: ToolCallStyle
}

const DEFAULT_CONFIG: AcpConfig = {
  autoModelContextLimit: true,
  autoTools: true,
  autoCommand: true,
  autoNudge: true,
  // Same default as the host's compaction-basic policy: one owned retry per
  // unrelieved overflow, then the original error is preserved.
  maxOverflowRetries: 1,
  // ACP is the only summarizer: the host's own `compaction-basic` engine would
  // otherwise fold on its own schedule with a summary contract it never read.
  disableBasic: true,
  // Nudge thresholds: engine defaults 0.70/0.85 — deliberately below the
  // kernel/billion-context-pi 0.75/0.95. 0.95 leaves no room to act before
  // the API rejects, and the host's compaction-basic line (thresholdRatio
  // 0.80) shadows it in standard/code/cordis modes; 0.70 keeps the forced
  // over-limit nudge ahead of that 80% line. Explicit values always win
  // against these defaults — `coreOverrides` merges last and beats them on
  // same-name keys.
  nudgeMaxContextLimitPct: 0.7,
  nudgeEmergencyThresholdPct: 0.85,
  // No-action degrade fold: after 3 consecutive emergency-nudged turns without
  // a successful compress the engine folds the largest eligible span itself
  // (once per session, EMERGENCY PROVENANCE stamped, no overflow retry spent).
  emergencyFoldAfterTurns: 3,
}

export function resolveAcpConfig(config: Partial<AcpConfig> = {}): AcpConfig {
  const resolved = resolvePresetThresholds({ ...DEFAULT_CONFIG, ...config }, config)
  assertNudgeThresholdOrder(resolved)
  const maxOverflowRetries = resolved.maxOverflowRetries ?? 1
  if (!Number.isInteger(maxOverflowRetries) || maxOverflowRetries < 0) {
    throw new Error(`maxOverflowRetries must be a non-negative integer (got ${maxOverflowRetries})`)
  }
  // Write the validated value BACK: an explicit `maxOverflowRetries: undefined`
  // in a Partial config survives the `{ ...DEFAULT_CONFIG, ...config }` spread
  // as `undefined`, and the request-error listener must never read that as
  // "budget 0" (recovery silently disabled while the config says default).
  return { ...resolved, maxOverflowRetries, emergencyFoldAfterTurns: normalizeEmergencyFoldAfterTurns(resolved.emergencyFoldAfterTurns) }
}

/**
 * Normalize the no-action degrade knob the same way `maxOverflowRetries` is
 * normalized: an explicit `undefined` in a Partial config survives the
 * DEFAULT_CONFIG spread and must read as "disabled" only when the operator
 * actually wrote 0 — never when the key was simply absent.
 */
function normalizeEmergencyFoldAfterTurns(value: number | undefined): number {
  if (value === undefined) return 3
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`emergencyFoldAfterTurns must be a non-negative integer (got ${value})`)
  }
  return value
}

/**
 * Apply `config.preset`, if one was given: an unknown name throws here at
 * construction, and the preset fills ONLY the thresholds the caller left unset.
 */
function resolvePresetThresholds(base: AcpConfig, config: Partial<AcpConfig>): AcpConfig {
  if (base.preset === undefined) return base
  // Fail fast on an unknown preset name (same contract as prompt-template
  // validation): a typo must break construction, never silently fall back to
  // the engine defaults.
  const preset = resolvePreset(base.preset)
  // Precedence: explicit value > preset > engine default. Read the caller's
  // EXPLICIT choices from `config`, not from `base` — base already merged
  // DEFAULT_CONFIG, so `base.X ?? preset.X` would let the engine default (e.g.
  // max 0.70) mask the preset. `config.X ?? preset.X` keeps an explicit value
  // while letting the preset fill anything the caller left unset.
  return {
    ...base,
    nudgeMinContextLimitPct: config.nudgeMinContextLimitPct ?? preset.nudgeMinContextLimitPct,
    nudgeMaxContextLimitPct: config.nudgeMaxContextLimitPct ?? preset.nudgeMaxContextLimitPct,
    nudgeEmergencyThresholdPct: config.nudgeEmergencyThresholdPct ?? preset.nudgeEmergencyThresholdPct,
  }
}

/**
 * The settings-layer view of the RAW composition row: its six scalar knobs,
 * plus — when the row names a preset — the preset values for the thresholds
 * the row leaves unset.
 *
 * Every kernel consumer reads those six knobs through `readSettingsSource()`,
 * never from `this.config` (where `resolveAcpConfig` already applied the
 * preset). Seeding the live snapshot and the seam's base layer from the raw
 * row alone would therefore drop the preset entirely: a preset-only row
 * (`{ preset: 'aggressive' }`) resolves to `{}` there, and the schema
 * defaults (0.70 / 0.85) silently win — issue #176. Filling the absent
 * thresholds with the preset's values keeps explicit > preset > default on
 * the live read path exactly as on `this.config` (only ABSENT keys are
 * filled, so an explicit value on the row still outranks its preset).
 */
function presetFilledSettingsEntry(config: Partial<AcpConfig>): AcpSettingsInput {
  const entry = filterSettingsEntry(config)
  if (config.preset === undefined) return entry
  // Safe to re-resolve: an unknown name already threw in resolveAcpConfig,
  // which runs earlier in the same constructor.
  const preset = resolvePreset(config.preset)
  return {
    ...entry,
    nudgeMinContextLimitPct: config.nudgeMinContextLimitPct ?? preset.nudgeMinContextLimitPct,
    nudgeMaxContextLimitPct: config.nudgeMaxContextLimitPct ?? preset.nudgeMaxContextLimitPct,
    nudgeEmergencyThresholdPct: config.nudgeEmergencyThresholdPct ?? preset.nudgeEmergencyThresholdPct,
  }
}

/**
 * Construction-time guard on the resolved nudge thresholds.
 *
 * The kernel tolerates an inverted window: `validateConfig` only *warns* when a
 * turn runs ("Thresholds may not fire correctly"), it never rejects the config.
 * That leaves a silent trap on this feature — combining a preset with a single
 * explicit override is the whole point of the precedence rule, and it can
 * produce e.g. `preset: 'preserve'` (min 0.55) + `nudgeMaxContextLimitPct:
 * 0.5`, where the over-limit line sits below… and an emergency line above it
 * fires first, inverting what the user asked for. We own this merge, so we
 * reject the merged result loudly instead of shipping a window that quietly
 * does something else.
 *
 * Only values that are actually set are compared: an omitted `min` falls back
 * to the kernel default (0.45) inside the kernel, and mirroring that constant
 * here would duplicate kernel state we deliberately do not track.
 */
function assertNudgeThresholdOrder(config: AcpConfig): void {
  const { nudgeMinContextLimitPct: min, nudgeMaxContextLimitPct: max, nudgeEmergencyThresholdPct: emergency } = config
  const describe = `min ${min ?? 'kernel default'} / max ${max ?? 'kernel default'} / emergency ${emergency ?? 'kernel default'}`
  if (min !== undefined && max !== undefined && min > max) {
    throw new Error(`nudge thresholds are inverted (${describe}) — nudgeMinContextLimitPct must be <= nudgeMaxContextLimitPct`)
  }
  if (max !== undefined && emergency !== undefined && max > emergency) {
    throw new Error(`nudge thresholds are inverted (${describe}) — nudgeMaxContextLimitPct must be <= nudgeEmergencyThresholdPct`)
  }
  if (min !== undefined && emergency !== undefined && min > emergency) {
    throw new Error(`nudge thresholds are inverted (${describe}) — nudgeMinContextLimitPct must be <= nudgeEmergencyThresholdPct`)
  }
}

/** One emergency context-overflow compaction, for the recovery log line. */
interface OverflowRecoveryResult {
  readonly start: number
  readonly end: number
  readonly shadowedSeqs: readonly number[]
  readonly shadowedTokenCount: number
}

/**
 * The ACP compaction backend. Subclasses the seam exactly like
 * `dsh-compaction-basic`; swaps summarization-driven compaction for
 * model-driven block compression without touching the agent loop.
 */
export class AcpCompactionEngine extends CompactionEngine {
  /**
   * Static plugin config schema — cordis reads `plugin.Config` off the raw
   * plugin value at fiber start and validates the composition row against it
   * (`resolveConfig`). Class-shaped mounts carry this (the bundle/composition
   * row path); function-shaped mounts skip validation and receive plain
   * values. On dsh-settings ≥ 0.1.7 hosts this SAME schema is what SettingsForms
   * builds its form from — the volatile fields are exactly what `/acp-prune
   * config` exposes (see AcpPluginConfigSchema for why fields are volatile and
   * default-free).
   */
  static readonly Config = AcpPluginConfigSchema
  /** The framework-agnostic ACP compression core, reused verbatim. */
  readonly kernel: CompressionCore
  /** Per-session kernel state. */
  readonly store: AcpStateStore
  /** Resolved engine configuration. */
  readonly config: AcpConfig
  /** Resolved prompt templates (validated at construction — fail-fast on template typos). */
  readonly prompts: ResolvedPrompts
  /**
   * One-line diagnostics from `textOverrides` (unknown ids, oversized values).
   * Empty when nothing was overridden. Logged once at construction and echoed by
   * `acp_status`, because a composition-row config has no other visible surface.
   */
  readonly textOverrideLines: readonly string[]
  /** The prompt set ACTUALLY applied to the current step: `prompts` in native
   *  mode, its PTC variant in code mode (src/ptc.ts). `prompts` keeps the
   *  composition value so introspection and tests see what was configured. */
  private currentPrompts: ResolvedPrompts
  /** The calling convention the currently registered surface was built for. */
  private appliedStyle: ResolvedCallStyle
  /** Disposers of the tool registrations this instance made — `tools.register()`
   *  returns the exact disposer, which is what lets a mode flip REPLACE the
   *  descriptions instead of stacking a second layer per flip. */
  private toolDisposers: Array<() => void> = []
  /** Same, for the ACP system-prompt section (`systemPrompt.section()`). */
  private systemPromptDisposer: (() => void) | undefined
  /** Per session: the newest leaked direct call already corrected, plus how many
   *  corrections were spent (the leak is silent on the host, so the nudge is the
   *  only channel — bounded to avoid a per-step nag). */
  private readonly ptcLeakNotes = new Map<string, { seq: number; count: number }>()
  /**
   * The environment wired into tools / command / nudge. Exposed so tests (and
   * introspection) can assert the forwarding actually happened: the config
   * chain user config → this.config → env → kernelConfigFor is all OPTIONAL
   * fields, so a dropped forwarding line fails typecheck silently and would
   * revive lost-config bugs with every unit test green.
   */
  readonly env: ToolEnvironment

  /** Set once the host's own auto-compaction entry has been handled (or proven absent), so the lookup runs at most once. */
  private basicCompactionHandled = false
  /**
   * Set once an ISOLATED-realm `compaction-basic` has been disabled. Kept
   * separate from `basicCompactionHandled` because the two planes resolve at
   * different moments: the loader plane may still be waiting out a late-loading
   * entry while the realm scan has already found and taken down its target.
   */
  private realmCompactionHandled = false

  private readonly lastNudgeTurn = new LruMap<string, NudgeGate>(DEFAULT_SESSION_CACHE_LIMIT)
  /** Per-session emergency-nudge injection budget for the current user turn (issue #108). */
  private readonly emergencyNudges = new Map<string, { turn: number; count: number }>()
  /** Successful compress call ids awaiting their tool/result so the pair can be hidden. */
  private readonly compressCallIdsToHide = new Set<string>()
  /** Per provider/model route the resolved window (probe failures cached too). */
  private readonly windowCache = new Map<string, AcpWindow>()
  /** Live settings snapshot thunk (composition → user settings layer); swapped when the settings provider attaches (installSection on ≤0.1.6 lines, volatile-ref re-read on the ≥0.1.7 forms line). */
  private readSettingsSource: () => AcpSettings = () => resolveAcpSettings({})
  /** Hot-apply driver: re-reads the live source and runs the change diff; called once per agent step, a no-op unless something changed. */
  private resyncSettings: () => void = () => {}
  /** The settings service, captured lazily for /acp-prune config (undefined in provider-less processes). Structurally typed — both host lines' real services satisfy it. */
  private settingsService: AcpSettingsService | undefined
  /** /acp-prune config read/write surface. */
  readonly settingsCommand: SettingsCommandSurface
  /** Per route the adapter's per-request output cap (the output reservation); null = undisclosed. */
  private readonly outputReservationCache = new Map<string, number | null>()
  /** Per-agent context-overflow recovery budget (mirrors the host's `overflowRetries`). */
  private readonly overflowRetries = new Map<Agent, number>()
  /** Per-session overflow agents, so session progress can reset the recovery budget. */
  private readonly overflowSessions = new Map<Session, Agent>()
  /**
   * 方案 D layer 3 — per-session no-action counter. +1 whenever an EMERGENCY
   * nudge (≥ nudgeEmergencyThresholdPct) is actually injected and the session
   * has not yet degraded; reset to 0 whenever a successful compress lands (the
   * tool/result listener sees the call id that handleCompress queued for
   * hiding). The counter lives for the engine's lifetime, so an idle/overnight
   * resume at high pressure counts from where it left off — the cross-day
   * "first step after idle is already over the line" scenario degrades after
   * the configured K turns, not after K fresh turns.
   */
  private readonly noActionTurns = new LruMap<string, number>(DEFAULT_SESSION_CACHE_LIMIT)
  /** Sessions that already took the one-time no-action degrade fold (loop guard). */
  private readonly degradedSessions = new LruMap<string, true>(DEFAULT_SESSION_CACHE_LIMIT)

  /**
   * Disable the host's own `compaction-basic` entry, at most once per plane.
   *
   * A composition may load BOTH engines. `compaction-basic` runs its own
   * `agent/pre-step` auto-compaction (`config.auto` defaults to true) and
   * writes a summary that never read the ACP note contract — measured on a
   * live session: 121 messages / 77,280 tokens folded into a 32-character
   * summary with every user constraint in that span gone. Disabling its loader
   * entry is a RUNTIME action: `Entry.update({ disabled: true })` writes the
   * option and then disposes that entry's fiber
   * (`cordis-plugin-loader/lib/index.js`, `Entry.update` → `fiber.dispose()`),
   * so no restart is involved.
   *
   * TWO planes, both covered (2026-11-27, live-session seq5455 evidence):
   * 1. the loader ROOT plane — the composition row the profile patch disables
   *    at rest; found through `loader.entries()` and disabled the same way;
   * 2. the ISOLATED realm — the cordis preset (dsh-web-app
   *    `presets/cordis.patch.yml`) mounts its compaction group inside the
   *    agent preset's own `EntryTree` (`dsh-agent-preset-registry` PresetTree,
   *    `isolate: { compaction: true }`), which the root loader's `entries()`
   *    can NEVER see (measured against loader 1.2.0: a preset tree is a
   *    separate `EntryTree` rooted at the agent's scope fiber, not a subtree
   *    of the main loader). Those entries are reached through the GLOBAL
   *    service store instead: every realm-private implementation registers
   *    under its own realm Symbol in `ctx.reflect.store` (cordis `provide()`
   *    keys `impl` by `ctx[Context.isolate][name]`), and
   *    `impl.ctx.fiber.entry` hands back the owning loader Entry.
   *
   * Fail-open in every direction — no loader service, no matching entry, no
   * reachable reflect store, or a throwing dispose each log ONE warning and
   * leave the plugin running. A mount failure here would cost the session its
   * only compression path, which is strictly worse than two engines competing.
   */
  private disableBasicOnce(): void {
    if (this.basicCompactionHandled || !this.config.disableBasic) return
    // Reading a cordis SERVICE that was never provided THROWS
    // (`cannot get property "loader" without inject`), so the absence of a
    // loader is not a null check — it is a caught throw. This is the whole
    // reason the lookup is wrapped: a host without the loader plugin must keep
    // running, not lose its only compression path to a failed mount.
    let loader: LoaderLike | undefined
    let loaderMissing = false
    try {
      loader = (this.ctx as unknown as { loader?: LoaderLike }).loader
    } catch {
      // Reading the service threw = the host has NO loader plugin at all, a
      // PERMANENT environment fact. Silent + final: no composition row to
      // disable, and warning every pre-step would spam forever.
      loaderMissing = true
    }
    if (!loaderMissing) {
      try {
        if (loader !== undefined && typeof loader.entries === 'function') {
          for (const entry of loader.entries()) {
            // `Entry.id` is the TREE path (`parent:child`), while `options.id` is
            // the id the composition row declared — the latter is the stable match.
            if (entry.options?.id !== 'compaction-basic') continue
            // Found it: one attempt, never retried. A failing dispose is usually
            // persistent (entry already disposed, service gone), so retrying would
            // only spam warnings; this path is best-effort by design. The realm
            // scan below still runs afterwards — the two planes are independent.
            this.basicCompactionHandled = true
            void Promise.resolve(entry.update({ disabled: true })).catch((error: unknown) => {
              const message = error instanceof Error ? error.message : String(error)
              this.ctx.logger.warn(`billion-context-dsh: could not disable compaction-basic: ${message} (will not retry)`)
            })
          }
        }
      } catch (error) {
        // `entries()` itself can throw (host internals mid-rebuild); warn once and
        // fall through to the retry next pre-step — the lookup is synchronous and
        // costs nothing, and the handled flag is deliberately NOT set here. The
        // realm scan still runs: it shares no state with the loader.
        const message = error instanceof Error ? error.message : String(error)
        this.ctx.logger.warn(`billion-context-dsh: compaction-basic lookup failed: ${message}; will retry at the next pre-step`)
        return
      }
    }
    // A host without the loader plugin is a NORMAL environment, not a fault:
    // there is no composition row to disable, so there is nothing to report.
    // But a loader WITH no compaction-basic row is ambiguous — the entry may be
    // late-loading (a dynamic install after the first pre-step), so leave the
    // flag unset and keep the cheap synchronous lookup alive on later pre-steps.
    // The realm scan is DISABLED (live-verified 2026-11-27): the preset compaction
    // group mounts compaction-basic in an ISOLATED realm as the ONLY provider of
    // the realm-local `compaction` service, and command-compact in the SAME realm
    // waits for that service (isolate:{compaction:true} never falls back to the
    // root plane where compaction-acp lives). Disposing the realm basic left
    // `ctx.get('compaction') === undefined` inside every preset realm — all four
    // presets reported "loading failed: waiting for compaction" and both the
    // /compact command and the host auto-compact died with it. The root-plane
    // disable (the loader `entries()` path above) stays: there the patch row
    // already disables basic at rest, and compaction-acp IS the root provider.
    if (loaderMissing) this.basicCompactionHandled = true
  }

  /**
   * Disable every `compaction-basic` implementation that lives in an ISOLATED
   * cordis realm (the preset-mounted compaction group), at most once.
   *
   * RETIRED (2026-11-27): this scan disposed the preset realm's compaction-basic
   * fiber, but that basic is the ONLY provider of the realm-local `compaction`
   * service and command-compact lives in the same realm — disposing it broke
   * all four presets ("waiting for compaction") plus the /compact command and
   * the host auto-compact. The method body is kept as an inert stub so the
   * call sites and tests that reference it still compile; it must NOT be
   * re-armed unless a realm-local compaction provider OTHER than basic exists
   * (see the alignment note at the root-plane call site).
   *
   * cordis keeps every service implementation in ONE global store
   * (`ctx.reflect.store`, `cordis/lib/index.js` `provide()`); a realm-private
   * registration is stored under the realm's own Symbol, so all realms are
   * enumerable from the plugin's context regardless of which `EntryTree` the
   * owning entry hangs from. The root-plane implementation is skipped — its
   * key IS the root compaction symbol, and it belongs to the loader `entries()`
   * path above (including the persisted patch row that already disables it at
   * rest). The plugin's own `compaction-acp` implementation is skipped by the
   * entry-id match: it provides `compaction` too, but from an entry whose id
   * is `compaction-acp`.
   */
  private disableRealmBasics(): void {
    if (this.realmCompactionHandled) return
    this.realmCompactionHandled = true
    // The former store-walk implementation is REMOVED, not commented out: its
    // behavior (disposing the preset realm's compaction-basic fiber) broke all
    // four presets and the /compact command (live-verified 2026-11-27). The
    // original walk is preserved in git history at the commit that introduced
    // it; see the class-doc note above for why it must not come back.
    this.ctx.logger.debug?.(
      "billion-context-dsh: isolated compaction-basic scan retired — the realm basic is the preset realm's ONLY compaction provider",
    )
  }

  constructor(ctx: Context, config: Partial<AcpConfig> = {}) {
    super(ctx)
    // Class-mounted compositions arrive with the six knobs wrapped in volatile
    // refs (the loader resolves our static Config schema before construction);
    // unwrap them so preset resolution, the threshold-order assertion and the
    // config snapshot see plain values, not opaque objects.
    this.config = resolveAcpConfig({ ...config, ...liveSettingsFromRefs(filterSettingsEntry(config)) })
    // Resolve + validate prompt templates BEFORE building env: a template typo
    // must fail engine construction, never silently leak into model context.
    // textOverrides lands between the structured `prompts` layer and the PTC
    // rewrite: `prompts` stays the template path (validated, placeholder-checked),
    // while an id override is a plain REPLACEMENT that must win over it, and must
    // still be re-rendered into the active call syntax (toPtcPrompts below).
    const textReport = applyTextOverrides(resolvePrompts(config.prompts), config.textOverrides)
    this.prompts = textReport.prompts
    this.textOverrideLines = textOverrideWarningLines(textReport)
    if (this.textOverrideLines.length > 0) {
      for (const line of this.textOverrideLines) this.ctx.logger.warn(`billion-context-dsh: ${line}`)
    }
    // Calling convention at mount: a pinned `toolCallStyle: ptc` applies right
    // away; `auto` (the default) starts native and detects per session at the
    // first pre-step — the ASSEMBLED system prompt is the only place the mode is
    // observable, because it arrives through an agent preset this engine cannot
    // read (src/ptc.ts, measured on host 0.2.2-alpha.2).
    this.appliedStyle = this.config.toolCallStyle === 'ptc' ? 'ptc' : 'native'
    this.currentPrompts = this.appliedStyle === 'ptc' ? toPtcPrompts(this.prompts) : this.prompts
    const ports = this.config.countTokens !== undefined ? { countTokens: this.config.countTokens } : {}
    this.kernel = createCore(ports)
    // The kernel's docFeatures cache (per-doc search features) defaults to an
    // 8MB SOURCE-CHAR cap — sized for multi-session server processes. A DSH
    // profile is single-user and its search corpus (ALL shadowed originals)
    // routinely exceeds 8MB, so the default re-tokenizes the corpus on every
    // recall({ query }) call (issue #133: ~18s/call on a 40MB corpus, cold and
    // warm identical). The cap cannot be tuned DOWN instead — it evicts FIFO
    // and bills source chars only, so a cap below the corpus caches nothing
    // (measured: half the corpus → 1.1× on a repeat scan). 128MB covers the
    // largest reported session (17.6M shadowed tokens ≈ 70MB text). Retained
    // feature heap is 2.1×–51× the billed chars (content-dependent, measured)
    // — accepted, since the host already holds a log of that scale; the
    // arithmetic and the upstream root cause are in AGENTS.md rule 14.
    setDocCacheCap(128 * 1024 * 1024)
    this.store = new AcpStateStore()

    // ── Runtime settings seam (M6) ──────────────────────────────────────
    // The six settings-exposed knobs resolve as: schema default → composition
    // row subset (FILTERED — a raw row also carries prompts/coreOverrides/
    // countTokens, values that must never enter the settings layer) → the
    // host's user layer (~/.dsh/settings.yaml section on the legacy line,
    // profile entry config on the forms line). `current` is the live snapshot
    // every consumer reads; `applySettings` lands an incoming change (initial
    // call included) and runs the diff handler. The integration is an
    // OPTIONAL-service consumer: with no settings provider (plain npm-install
    // compositions) nothing registers and the engine behaves exactly as
    // composed — the same values, read through the same thunk.
    // The BASE layer the seam registers is the composition row's own scalar
    // subset, taken from the RAW row — not from `this.config`, which already has
    // engine defaults merged in; using it would turn every uncomposed key into a
    // `base` override that shadows the schema default (so /acp-prune config list would
    // report `base` for keys nobody composed, and a reset would keep the value).
    // One deliberate exception (issue #176): when the row names a preset, the
    // preset values for the thresholds the row leaves unset ARE added — every
    // kernel consumer reads the six knobs through `readSettingsSource()`, never
    // `this.config`, so without them a preset-only row resolves to `{}` here
    // and the schema defaults (0.70 / 0.85) silently win. Consequence: for a
    // composed preset, /acp-prune config list attributes those threshold keys to
    // `base` and a runtime reset falls back to the preset value, not the
    // engine default. `current` is the resolved snapshot reads start from; the
    // two differ only in which keys are PRESENT, never in the values they
    // resolve to.
    // On hosts whose loader validates our static Config schema (class-mounted
    // compositions), row values arrive wrapped in volatile refs — unwrap them
    // for one-shot reads, and keep the RAW filtered entry so the forms path can
    // re-read through those refs live (form writes commit into them in place).
    const compositionEntry = presetFilledSettingsEntry(config)
    // Construction-time snapshot — what "behaves exactly as composed" means.
    // Detach falls back to THIS, never to the mutable `current`: on the
    // legacy line the provider's own dispose hook resets the source to the
    // registered entry (which resolves to exactly these values), and on the
    // forms line the service dies while the last-applied values may no longer
    // be editable through any live surface — stranding them would be worse
    // than reverting to the composition.
    const initialCurrent: AcpSettings = resolveAcpSettings(liveSettingsFromRefs(compositionEntry))
    let current: AcpSettings = initialCurrent
    this.readSettingsSource = () => current
    const engine = this
    const applySettings = (): void => {
      const next = this.readSettingsSource()
      const prev = current
      current = next
      try {
        engine.onSettingsChanged(prev, next)
      } catch (error) {
        // The watcher callback runs inside the settings commit loop; a sync
        // throw must not escape into it (the loop logs and continues, but our
        // diff handler owns its failures — warn and keep the last good).
        this.ctx.logger.warn(`billion-context-dsh: applying settings change failed: ${String(error)}`)
      }
    }
    this.resyncSettings = applySettings
    this.settingsCommand = makeSettingsCommandSurface(() => this.settingsService, () => current)
    // Detach cleanup shared by both supported paths: cordis disposes the value
    // an inject callback returns when the provider fiber unloads. Without it
    // the engine keeps a dead provider handle (/acp-prune config would still
    // report available and write into a disposed service). Reads fall back to
    // the CONSTRUCTION-TIME snapshot, not the mutable `current`: on the legacy
    // line the provider's own dispose hook resets the source to the registered
    // entry (which resolves to exactly these values), and on the forms line
    // freezing at whatever was last applied could strand values the operator
    // can no longer edit through any live surface. Order-independent either way.
    const detachSettings = (): void => {
      this.settingsService = undefined
      this.readSettingsSource = () => initialCurrent
    }
    if (this.config.settingsEnabled !== false) {
      // Two supported host shapes, detected at runtime (issue #193): the
      // ≤0.1.6 lines expose the legacy `installSection` section seam (the
      // standalone `installSettingsSection` helper this was written against
      // is gone — it is now a method on the provider); the ≥0.1.7 lines
      // expose SettingsForms, which builds its form from our static Config
      // schema and has no registration step at all. A settings-shaped
      // service speaking neither degrades with one warn (issue #173).
      ctx.inject(['settings'], (settingsCtx) => {
        // The service handle is read structurally: the seam's Context
        // augmentation differs per host line (and is absent whenever this
        // package's types stop importing dsh-settings) — feature detection
        // below only needs the runtime shape.
        const face = (settingsCtx as unknown as { settings?: unknown }).settings
        if (face === undefined || typeof face !== 'object') return undefined
        const service = face as Record<string, unknown>
        if (typeof service.installSection === 'function') {
          // LEGACY LINE (dsh-settings ≤ 0.1.6): register the composition-row
          // subset as the section's base layer; the provider pushes changes via
          // onChange and swaps the source thunk. UNWRAPPED values go across the
          // seam — volatile refs must never reach the provider's persistence
          // (they stringify to {}).
          const legacy = service as unknown as LegacySettingsService
          legacy.installSection(ctx, ACP_SETTINGS_NAMESPACE, AcpSettingsSchema, liveSettingsFromRefs(compositionEntry), {
            // The seam's source type follows the entry it registered, so `source`
            // is a partial view of the settings; re-resolve it into a
            // fully-defaulted snapshot so every reader sees the same shape the
            // composition path produced.
            setSource: (source) => {
              this.readSettingsSource = () => resolveAcpSettings(source())
            },
            onChange: applySettings,
          })
          this.settingsService = legacy
          return detachSettings
        }
        if (typeof service.describe === 'function' && typeof service.update === 'function' && typeof service.replace === 'function') {
          // FORMS LINE (dsh-settings ≥ 0.1.7): there is no registration step —
          // the profile entry IS the carrier (its id equals the bundle row's id),
          // and the volatile fields of our static Config schema are exactly what
          // its form exposes. Form writes commit into our fiber config's volatile
          // refs IN PLACE between steps and emit no event we receive, so swap the
          // thunk to re-read through the refs; agent/pre-step drives the diff.
          // The host's methods read instance state through `this` — keep the
          // original receiver (a cordis-injected face may be a proxy, and a
          // bare reference would rebind `this` to this wrapper object).
          this.settingsService = {
            describe: (service.describe as AcpSettingsService['describe']).bind(service),
            update: (service.update as AcpSettingsService['update']).bind(service),
            replace: (service.replace as AcpSettingsService['replace']).bind(service),
          }
          this.readSettingsSource = () => resolveAcpSettings(liveSettingsFromRefs(compositionEntry))
          // One-shot diagnostic: the forms line addresses settings by PROFILE
          // ENTRY ID, and we address the bundle row's id (`compaction-acp`).
          // If the operator renamed their own composition row, no descriptor
          // exists for us — without this warn, /acp-prune config would report
          // available while every write fails with an opaque entry error.
          try {
            const rows = (service.describe as AcpSettingsService['describe']).call(service)
            if (!rows.some((row) => String(row.ns) === ACP_SETTINGS_NAMESPACE)) {
              this.ctx.logger.warn(
                'billion-context-dsh: SettingsForms is present but no profile entry named "compaction-acp" is visible — '
                + 'the six knobs keep their composition values; /acp-prune config needs a composition row with that id '
                + '(the bundle install provides it)',
              )
            }
          } catch {
            // A throwing describe() is a host defect, not ours to repair — the
            // command surface surfaces it loudly on first use instead.
          }
          return detachSettings
        }
        // DEGRADE: a settings-shaped service speaking neither API (a future
        // renamed line or a foreign implementation) — one warn, no capture:
        // capturing a half-matching handle would make /acp-prune config report
        // available and fail every write with opaque errors. Values keep
        // flowing from the composition row through the untouched thunk.
        this.ctx.logger.warn(
          'billion-context-dsh: host settings service speaks neither installSection (dsh-settings <= 0.1.6) '
          + 'nor SettingsForms describe/update/replace (>= 0.1.7) — the compaction-acp settings section is '
          + 'unavailable; the six knobs keep their composition values',
        )
        return undefined
      })
    }

    const env: ToolEnvironment = {
      kernel: this.kernel,
      store: this.store,
      // The settings-exposed knobs read LIVE from the settings source, so a
      // settings.yaml edit (or /acp-prune config set) hot-applies to every
      // subsequent call — consumers never see stale numbers. (ToolEnvironment
      // fields are readonly properties; getters satisfy them.)
      get modelContextLimit() { return engine.readSettingsSource().modelContextLimit ?? DEFAULT_CONTEXT_WINDOW },
      get nudgeMinContextLimitPct() { return engine.readSettingsSource().nudgeMinContextLimitPct },
      get nudgeMaxContextLimitPct() { return engine.readSettingsSource().nudgeMaxContextLimitPct },
      get nudgeEmergencyThresholdPct() { return engine.readSettingsSource().nudgeEmergencyThresholdPct },
      coreOverrides: this.config.coreOverrides,
      // Display-only: which named preset produced the thresholds above (if any),
      // so /acp-prune status can name it. The resolved pct values above are what the
      // kernel actually reads — this field never feeds kernelConfigFor.
      preset: this.config.preset,
      windowFor: (agent) => this.windowFor(agent),
      // Prompt set and calling convention read LIVE so a mode flip applies to
      // the next call without rebuilding env (the two are swapped together in
      // `applyToolCallStyle`, never independently — a PTC nudge with native
      // descriptions would send the model back to the failing shape).
      get prompts() { return engine.currentPrompts },
      get toolCallStyle() { return engine.appliedStyle },
      get textWarnings() { return engine.textOverrideLines },
      compressCallIdsToHide: this.compressCallIdsToHide,
      settingsCommand: this.settingsCommand,
    }
    this.env = env

    // Tools and commands may not be registered yet on cold start: cordis
    // starts unrelated composition rows concurrently, so the first
    // `ctx.get('tools')` can legitimately be undefined even though the row
    // ships later in the file. HMR-style reloads always see them (already
    // present), but a fresh process races — the tools silently vanished on
    // restart. Register eagerly, then re-attempt when the service appears
    // (`internal/service`) or the app finishes booting (`ready`); guard so a
    // late callback never double-registers.
    const tools = ctx.get('tools')
    if (tools !== undefined) {
      this.registerToolSurface()
    } else {
      let done = false
      ctx.on('internal/service', (name: unknown) => {
        if (name !== 'tools' || done) return
        done = true
        this.registerToolSurface()
      })
    }
    const commands = ctx.get('commands')
    if (commands !== undefined) {
      commands.register(acpCommand(env))
    } else {
      let done = false
      const registerCommand = (): void => {
        if (done) return
        const registry = ctx.get('commands')
        if (registry === undefined) return
        done = true
        registry.register(acpCommand(env))
      }
      ctx.on('internal/service', (name: unknown) => {
        if (name === 'commands') registerCommand()
      })
    }
    // After a successful compress tool result is appended, hide its
    // call/result pair. The durable summary node was inserted mid-turn (before
    // the result), so leaving the pair visible would put a user message between
    // an assistant tool_calls block and its tool response — strict providers
    // reject that request with HTTP 400 (issue #18).
    ctx.on('session/event', (session, event) => {
      // An assistant message on a session that just overflowed means the retry
      // (or a later request) made progress: reset that agent's overflow budget
      // so a FUTURE overflow gets a fresh recovery chance (host parity).
      if (event.type === 'assistant/message') {
        const overflowAgent = this.overflowSessions.get(session)
        if (overflowAgent !== undefined) {
          this.overflowRetries.delete(overflowAgent)
          // Drop the reverse lookup as well: it exists ONLY to find the agent
          // whose budget must be reset, and both terminal paths (progress, or
          // idle after an unrelieved overflow) walk through here — keeping it
          // would retain the session and its state for the engine's lifetime.
          this.overflowSessions.delete(session)
        }
      }
      if (event.type !== 'tool/result') return
      // One shared extractor (rule 10): message-level field first, then the
      // nested tool-result block, then source.callId — one implementation for
      // call/result pairing everywhere, never a per-listener copy.
      const callId = toolCallIdOfResultEvent(event)
      if (typeof callId !== 'string' || !this.compressCallIdsToHide.has(callId)) return
      this.compressCallIdsToHide.delete(callId)
      // 方案 D layer 3: a call id queued for hiding means handleCompress landed
      // at least one durable block — the model DID act. Reset the no-action
      // counter so a working model is never degraded behind its back.
      this.noActionTurns.delete(session.id)
      // session.append is NOT reentrant: calling it synchronously inside this
      // session/event dispatch (the outer append still holds the reentry lock)
      // throws "session append cannot reenter while another append is being
      // published" on live, store-attached sessions, and the dispatcher
      // silently swallows the error — the hide would be a no-op. Defer it to a
      // microtask: microtasks drain after the append fully publishes and
      // before the agent loop resumes, so the pair is hidden before the next
      // request is built.
      deferCompressPairHide(session, callId, event.seq, (error) => {
        ctx.logger.warn(`billion-context-dsh: hide compress call/result pair failed: ${String(error)}`)
      })
    })
    ctx.on('agent/pre-step', async (payload, next) => {
      // A crash-interrupted tool leaves an orphan call/result on the surface:
      // it corrupts the pairing balance cache AND can 400 the next request
      // (strict providers reject tool messages without their call/response).
      // Clean them before EVERY step — not only when a nudge fires — so a
      // low-pressure session never hits the orphan 400 (issue #18). No call is
      // in flight at pre-step (the previous step's tools all landed), so the
      // default empty in-flight set is safe.
      // Order is load-bearing (round-8 review, all three models): clear our own
      // stale nudge rows BEFORE the orphan/broken-pair strip, so a nudge that used to
      // sit between a tool call and its result stops making that pair look broken —
      // the strip must never delete a real tool result to fix a row we authored.
      // B-1（2026-11-23）：整段 housekeeping 必须 fail-open。这里任何一处抛错
      // （宿主 registry 重建、设置热应用、校准样本异常、块账本重建）都会让整个
      // pre-step 拒绝，代价是一整个回合而不是一条提示。
      try {
        // Take the host's own auto-compaction engine down before any of our
        // bookkeeping runs: it must never fold a span of this session with a
        // summary contract it never read (`disableBasic`, default true).
        this.disableBasicOnce()
        pruneStaleNudgeRows(payload.agent.session)
        stripOrphanedSurfaceToolMessages(payload.agent.session)
        // Calibration pair for the emergency fit check (src/usage.ts): the newest billed
        // request against the surface of the step that PRODUCED it. Pairing it with THIS
        // step's surface would compare two moments — the surface already carries the
        // previous step's output, so both the ratio and the absolute overhead shrink
        // toward zero and the fit check under-quotes a fold that still overflows
        // (round-10 review S2). calibrateRequestPair carries the previous surface.
        setCalibrationRoute(payload.agent.session, `${routeFor(payload.agent).provider}/${routeFor(payload.agent).model}`)
        const billed = lastRequestUsage(payload.agent.session)
        calibrateRequestPair(payload.agent.session, billed?.tokens ?? 0, billed?.seq ?? 0, surfaceTokens(payload.agent.session))
        // Hot-apply settings BEFORE any knob is read this step (issue #193): on
        // the forms line, writes committed between steps land in our volatile
        // refs with no event we receive — this re-read is what applies them.
        engine.resyncSettings()
        // Calling-convention gate BEFORE anything that renders model-facing copy:
        // a PTC session must register `await tools.*` descriptions and nudge text,
        // never the direct-call shape the host rejects (src/ptc.ts). Cheap — a
        // marker scan over the session's system rows — and the heal below builds
        // its definitions from `env`, so it must see the applied style.
        const style = engine.resolveToolCallStyle(payload.agent.session)
        if (style !== engine.appliedStyle) engine.applyToolCallStyle(style)
        // Registration self-heal (src/registration.ts): `tools.register()` is a
        // layer effect on the REGISTRY instance, so when the host rebuilds an
        // agent realm the four ACP tools vanish while this plugin instance stays
        // mounted — and the activation `done` guard forbids a second attempt.
        // Live signature (2026-11-01): every model call answered `unknown tool`
        // for the first two steps of a resumed session. Checking here (before
        // `next()` builds the request) restores them within the SAME step, and
        // runs regardless of `autoNudge` because a nudging-off session still
        // needs the tools.
        const healed = healRegistrations(env, ctx.get('tools'), ctx.get('commands'), payload.agent)
        if (healed.tools.length > 0 || healed.command) {
          ctx.logger.warn(
            `billion-context-dsh: re-registered ${healed.tools.join(', ')}${healed.command ? ' + /acp-prune' : ''} on session ${payload.agent.session.id} — the host registry was rebuilt since activation`,
          )
        }
      } catch (error) {
        ctx.logger.warn(`billion-context-dsh: pre-step housekeeping failed (fail-open): ${String(error)}`)
      }
      if (!engine.readSettingsSource().autoNudge) return next()
      const decision = await next()
      if (decision.kind === 'reject') return decision
      // B-1：nudge 构建同样 fail-open —— 拿不到压力/窗口/区间表时退回不加 nudge，
      // 而不是让这一步直接失败。openToolCallIds 也在 try 内：它读 surface，同样的
      // 失败模式（B-1 首版漏掉这一句，由 tests/pre-step-fail-open.test.ts 抓出）。
      try {
        // Never append a nudge while a tool call of this step is still OPEN: the row would
        // land BETWEEN the call and its result, which strict providers reject and which
        // rule 8 used to resolve by deleting the real pair (round-8 review, named the
        // ship-blocking bug at the source). One step later the results have landed and
        // the same table is just as actionable — pressure never depends on this step.
        if (openToolCallIds(payload.agent.session).size > 0) return decision
        const window = await this.windowFor(payload.agent)
        const outcome = buildNudge(
          payload.agent,
          {
            ...env,
            modelContextLimit: window.limit,
            // Consumed here, not inside buildNudge: the note rides the nudge this
            // step injects, and a step without a nudge must not silently spend it.
            ptcLeakNote: engine.takePtcLeakNote(payload.agent.session),
          },
          this.lastNudgeTurn,
          this.emergencyNudges,
          () => {
            // The kernel still wants an emergency nudge but the per-turn budget
            // is spent: log WHY the model stops receiving nudges instead of
            // letting the silence look like a bug (issue #108 review).
            ctx.logger.warn(
              `billion-context-dsh: emergency nudge suppressed — per-turn budget of ${EMERGENCY_NUDGE_MAX_PER_TURN} spent (session ${payload.agent.session.id}); pressure is still above the emergency threshold`,
            )
          },
        )
        if (outcome === null) return decision
        // 方案 D layer 2/3: an EMERGENCY nudge that actually reached the model is
        // one more reminder without action. Count it; once the count reaches
        // emergencyFoldAfterTurns and this session has not degraded yet, the
        // engine folds the largest eligible span itself INSTEAD of injecting
        // another reminder — the nudge's range table is computed from the
        // pre-fold surface, so appending it after the fold would hand the model
        // stale seqs. The fold never touches the overflow retry budget.
        // Threshold note (lead 方案 D ruling): the line is nudgeEmergencyThresholdPct
        // (engine default 0.85), NOT compaction-basic's 0.80 — the emergency tier
        // already owns a per-turn budget, the kernel's emergencyOverride copy and
        // the test infrastructure, so reusing it needs no new threshold knob; the
        // 5-point gap to basic's 0.80 is the price of keeping one threshold source
        // (preset-aware, operator-tunable) instead of a second hard-coded one.
        const sessionKey = payload.agent.session.id
        if (outcome.emergency) {
          const k = this.config.emergencyFoldAfterTurns ?? 3
          const count = (this.noActionTurns.get(sessionKey) ?? 0) + 1
          this.noActionTurns.set(sessionKey, count)
          if (k > 0 && count >= k && !this.degradedSessions.has(sessionKey)) {
            this.degradedSessions.set(sessionKey, true)
            this.noActionTurns.set(sessionKey, 0)
            try {
              const folded = await this.emergencyNoActionFold(payload.agent, count)
              if (folded !== null) {
                ctx.logger.warn(
                  `billion-context-dsh: EMERGENCY no-action degrade — the model was reminded ${count} turn(s) at or above the emergency threshold without calling compress; folded procedurally (seqs ${folded.start}-${folded.end}, ~${folded.shadowedTokenCount} tokens shadowed, EMERGENCY PROVENANCE stamped, ledger row written). Semantic detail in the folded span is summarized by the marker; originals stay in the session log for recall. Overflow retry budget untouched.`,
                )
                // The folded surface is what the next request must see; skip the
                // (now stale) nudge this step and let the next step's nudge —
                // built from the folded surface — take over.
                return decision
              }
              ctx.logger.warn(
                `billion-context-dsh: no-action degrade armed but nothing eligible to fold on session ${sessionKey}; the hard nudge is delivered instead`,
              )
            } catch (error) {
              // Fail-open exactly like the rest of pre-step: the degrade is an
              // extra safety net, never a reason to reject the step.
              ctx.logger.warn(`billion-context-dsh: no-action degrade fold failed (fail-open): ${String(error)}`)
            }
          }
        }
        return { kind: 'enter', messages: [...decision.messages, outcome.message] }
      } catch (error) {
        ctx.logger.warn(`billion-context-dsh: nudge build failed (fail-open): ${String(error)}`)
        return decision
      }
    })
    // ── Context-overflow recovery (M5) ─────────────────────────────────
    // The seam rethrows a failed request unless a listener answers
    // `agent/request-error` with `{ kind: 'retry' }` (dsh-agent-loop), and
    // docs/dsh-porting-verification.md (architecture fact 2) says the ACP
    // engine should copy compaction-basic's two automatic listeners. The
    // pressure half is deliberately NOT copied — ACP stays model-driven, the
    // pre-step listener above only nudges — but the overflow half is the ONE
    // automatic action: on a provider-confirmed CONTEXT_WINDOW_EXCEEDED the
    // engine emergency-compacts the largest eligible block (a durable,
    // model-free marker summary; originals stay in the log for
    // recall) and asks the loop to retry. Without this
    // listener the error was rethrown unretried.
    ctx.on('agent/status', ({ agent, status }) => {
      if (status !== 'idle') return
      this.overflowRetries.delete(agent)
      this.overflowSessions.delete(agent.session)
    })
    ctx.on('agent/request-error', async ({ agent, failure, signal }, next) => {
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next()
      this.overflowSessions.set(agent.session, agent)
      const retries = this.overflowRetries.get(agent) ?? 0
      // `resolveAcpConfig` normalizes the value, so this fallback only guards a
      // future config shape; it MUST stay 1 (the host's own default) — reading 0
      // here would disable overflow recovery for a key that was never set to 0.
      const max = this.config.maxOverflowRetries ?? 1
      if (retries >= max) {
        this.ctx.logger.warn(
          `billion-context-dsh: context-overflow recovery budget spent (${max} retries) for session ${agent.session.id}; preserving the original request error`,
        )
        return next()
      }
      const generation = agent.session.surface.replaceGeneration
      let result: OverflowRecoveryResult | null
      try {
        result = await this.compactForOverflow(agent, signal)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        if (!signal.aborted && agent.session.surface.replaceGeneration > generation) {
          // The durable transaction landed but the follow-through threw: the
          // surface shrank, so retrying from the replacement surface is the
          // best move even though the bookkeeping failed.
          this.ctx.logger.warn(
            `billion-context-dsh: context-overflow compaction failed after durable surface progress: ${message}; retrying from the replacement surface`,
          )
          this.overflowRetries.set(agent, retries + 1)
          return { kind: 'retry' }
        }
        this.ctx.logger.warn(
          `billion-context-dsh: context-overflow compaction failed: ${message}; ${signal.aborted ? 'cancellation prevents retry' : 'preserving the original request error'}`,
        )
        return next()
      }
      if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next()
      if (result !== null) {
        this.ctx.logger.info(
          `compaction (context overflow recovery): shadowed ${result.shadowedSeqs.length} surface nodes (seqs ${result.start}-${result.end}, ~${result.shadowedTokenCount} tokens)`,
        )
      }
      this.overflowRetries.set(agent, retries + 1)
      return { kind: 'retry' }
    })
    // The load-bearing ACP guidance lives in the system prompt ONCE; nudges
    // stay short and advisory (model-driven: the model decides). The
    // systemPrompt service may not be registered yet on cold start (cordis
    // starts unrelated composition rows concurrently), so apply the same
    // retry pattern as tools and commands: eager registration, then
    // re-attempt when the service appears via `internal/service`; guard so a
    // late callback never double-registers.
    const systemPrompt = ctx.get('systemPrompt')
    if (systemPrompt !== undefined) {
      this.registerPromptSurface()
    } else {
      let done = false
      ctx.on('internal/service', (name: unknown) => {
        if (name !== 'systemPrompt' || done) return
        done = true
        this.registerPromptSurface()
      })
    }
  }

  /**
   * Register (and remember the disposer of) the four ACP tools for the CURRENT
   * prompt set. `tools.register()` returns the exact disposer, which
   * {@link applyToolCallStyle} uses to replace the surface on a mode flip.
   */
  private registerToolSurface(): void {
    const registry = this.ctx.get('tools')
    if (registry === undefined) return
    for (const tool of makeTools(this.env)) {
      const dispose = registry.register(tool)
      if (typeof dispose === 'function') this.toolDisposers.push(dispose as () => void)
    }
  }

  /** Register (and remember the disposer of) the ACP system-prompt section. */
  private registerPromptSurface(): void {
    const registry = this.ctx.get('systemPrompt')
    if (registry === undefined) return
    const dispose = registry.section({
      name: 'billion-context-dsh',
      order: ACP_SYSTEM_PROMPT_ORDER,
      text: renderSystemPrompt(this.currentPrompts),
    })
    if (typeof dispose === 'function') this.systemPromptDisposer = dispose as () => void
  }

  /**
   * Swap the model-visible calling convention (src/ptc.ts). Runs from
   * `agent/pre-step`, i.e. before the step builds its request, so the very next
   * request carries the new descriptions and section text.
   *
   * Dispose-then-register: `register()` on a name that is already registered
   * would stack a second layer contribution, so the old contributions go first.
   * A failed re-registration is not fatal — the registration self-heal
   * (`src/registration.ts`) restores whatever is missing on the next pre-step.
   */
  private applyToolCallStyle(style: ResolvedCallStyle): void {
    if (style === this.appliedStyle) return
    this.appliedStyle = style
    this.currentPrompts = style === 'ptc' ? toPtcPrompts(this.prompts) : this.prompts
    for (const dispose of this.toolDisposers) {
      try {
        dispose()
      } catch {
        // Registry already torn down by the host — nothing to release.
      }
    }
    this.toolDisposers = []
    if (this.systemPromptDisposer !== undefined) {
      try {
        this.systemPromptDisposer()
      } catch {
        // Same: a disposed realm cannot be un-registered from.
      }
      this.systemPromptDisposer = undefined
    }
    this.registerToolSurface()
    this.registerPromptSurface()
  }

  /**
   * Resolve the calling convention for this step: a pinned config always wins,
   * otherwise the assembled system prompt of the session decides (`auto`).
   */
  private resolveToolCallStyle(session: Session): ResolvedCallStyle {
    const configured = this.config.toolCallStyle
    if (configured === 'native' || configured === 'ptc') return configured
    return detectPtcMode(
      surfaceEventsOf(session)
        .filter((event) => event.type === 'system/message')
        .map(extractEventText),
    ) ? 'ptc' : 'native'
  }

  /** Per-session cap on direct-call corrections (a nag on every step would be
   *  noise; three covers a model that keeps trying the direct shape). */
  private static readonly PTC_LEAK_NOTE_MAX = 3

  /**
   * True when the model just emitted a direct ACP call as TEXT — in PTC mode the
   * host rejects the name and the markup lands in the assistant text with no call
   * and no error result, so nothing tells the model it failed. The next nudge
   * carries the correction. Scans only the recent tail (a leak is a fresh event),
   * and reports once per distinct leaked seq.
   */
  private takePtcLeakNote(session: Session): boolean {
    if (this.appliedStyle !== 'ptc') return false
    const events = surfaceEventsOf(session)
    let leakSeq = 0
    for (const event of events.slice(-40)) {
      if (event.type !== 'assistant/message') continue
      if (!looksLikeLeakedAcpCall(extractEventText(event))) continue
      const seq = event.seq
      if (typeof seq === 'number' && seq > leakSeq) leakSeq = seq
    }
    if (leakSeq === 0) return false
    const record = this.ptcLeakNotes.get(session.id) ?? { seq: 0, count: 0 }
    if (leakSeq <= record.seq || record.count >= AcpCompactionEngine.PTC_LEAK_NOTE_MAX) return false
    this.ptcLeakNotes.set(session.id, { seq: leakSeq, count: record.count + 1 })
    return true
  }

  /**
   * Resolve the effective context window for an agent. An explicitly
   * configured `modelContextLimit` always wins (no probe). Otherwise the live
   * session projection (`contextPressure.contextWindow`) is preferred when it
   * discloses one — it tracks the session's CURRENT route, so a mid-session
   * model switch repairs itself without a restart or config (see
   * projectedContextWindow). Falls back to probing the model's real window
   * via `agent.ctx.llm.resolveModelInfo` (cached per provider/model route,
   * probe failures cached too) and finally to DEFAULT_CONTEXT_WINDOW when
   * auto-detection is disabled or unavailable. On the auto-detected paths the
   * adapter's per-request output cap is then SUBTRACTED from the window
   * (applyReservation): every downstream usage computation must run against
   * the SUSTAINABLE input budget (window minus output reservation), not the
   * raw window — a 96K window with a 16K cap carries at most 80K of input,
   * so the raw denominator understates usage by cap/window (≈17% there, and
   * far worse on short-window models). An explicit limit keeps the operator's
   * exact value (they own the denominator); a failed probe keeps the raw
   * fallback.
   */
  async windowFor(agent: Agent): Promise<AcpWindow> {
    const live = this.readSettingsSource()
    if (live.modelContextLimit !== undefined) {
      return { limit: live.modelContextLimit, source: 'explicit' }
    }
    // The per-route output cap must be looked up against the session's LIVE
    // route or it lags one switch behind (a stale agent.options snapshot names
    // the PREVIOUS route) — routeFor owns that fallback chain for every caller.
    const { provider, model } = routeFor(agent)
    const key = `${provider}\0${model}`
    // Projection source first: it reflects the live route (agent.options is a
    // stale snapshot after a model switch), and it is not cached here because
    // the projection itself refreshes on every request — caching would freeze
    // the old model's window for the whole process (the false-EMERGENCY trap).
    // Only consulted when auto detection is enabled (same gate as the probe).
    if (live.autoModelContextLimit) {
      const projected = projectedContextWindow(agent)
      if (projected !== null) {
        // The window comes from the live projection; the output cap comes from
        // the (cached) model probe for the LIVE route — the projection schema
        // carries no cap, so the cap follows the live provider/model resolved
        // above (agent.options only as the pre-first-request fallback).
        const cap = await this.outputCapFor(agent, provider, model)
        return this.applyReservation({ limit: projected, source: 'projection', provider, model }, cap)
      }
    }
    const cached = this.windowCache.get(key)
    if (cached !== undefined) return cached
    let window: AcpWindow
    let cap: number | null = null
    if (!live.autoModelContextLimit) {      window = { limit: DEFAULT_CONTEXT_WINDOW, source: 'default', provider, model }
    } else {
      const probe = await probeModelWindow(agent, provider, model)
      cap = probe.outputReservation
      if (probe.contextWindow === null) {
        // Probe failures are cached below too, so the 128K fallback sticks for
        // the whole process lifetime — a gateway operator who fixes the model
        // API must restart (or set modelContextLimit) before the probe retries.
        // Warn loudly instead of failing silently: pressure numbers computed
        // against the fallback are what issue #63's false emergency nudges
        // came from (a gateway that disclosed no window read as ~55% of 128K
        // when the real window was 1M).
        this.ctx.logger.warn(
          `billion-context-dsh: context-window auto-detection failed for ${provider}/${model} — using the ${DEFAULT_CONTEXT_WINDOW} fallback (change modelContextLimit or autoModelContextLimit via /acp-prune config — or restart — to re-probe)`,
        )
        window = { limit: DEFAULT_CONTEXT_WINDOW, source: 'default', provider, model, probeFailed: true }
        cap = null // the probe failed or disclosed nothing — no cap either
      } else {
        window = { limit: probe.contextWindow, source: 'auto', provider, model }
      }
    }
    window = this.applyReservation(window, cap)
    this.windowCache.set(key, window)
    return window
  }

  /**
   * Diff handler for runtime settings changes: drop the window cache when a
   * window-related key changed (probe FAILURES are cached too — clearing is
   * what lets the next pre-step re-probe after a fix), clear the per-turn
   * nudge dedup when nudges come back on, and warn on order anomalies
   * (accepted, never rejected — rejecting a write cannot fix an externally
   * edited settings.yaml, and an invalid stored section would fail the next
   * boot loud anyway).
   */
  private onSettingsChanged(prev: AcpSettings, next: AcpSettings): void {
    const effect = describeSettingsChange(prev, next)
    for (const warning of effect.warnings) {
      this.ctx.logger.warn(`billion-context-dsh: ${warning}`)
    }
    if (effect.clearWindowCache) this.windowCache.clear()
    if (effect.clearNudgeDedup) this.lastNudgeTurn.clear()
  }

  /**
   * The adapter's per-request output cap for a route, from one
   * probeModelWindow call (a local catalog lookup — no request is sent),
   * cached per route like the window itself.
   */
  private async outputCapFor(agent: Agent, provider: string, model: string): Promise<number | null> {
    if (provider === '' || model === '') return null
    const key = `${provider}\0${model}`
    const known = this.outputReservationCache.get(key)
    if (known !== undefined) return known
    const cap = (await probeModelWindow(agent, provider, model)).outputReservation
    this.outputReservationCache.set(key, cap)
    return cap
  }

  /**
   * Subtract the output reservation from a resolved window: `limit` becomes
   * the SUSTAINABLE input budget (`rawLimit - outputReserved`) that every
   * downstream usage computation (nudge tiers, truncate, growth) measures
   * against. No-op when the cap is unknown or not smaller than the window
   * (degenerate config) — the raw-window behavior is preserved.
   */
  private applyReservation(window: AcpWindow, cap: number | null): AcpWindow {
    if (cap === null || cap >= window.limit) return window
    return { ...window, rawLimit: window.limit, outputReserved: cap, limit: window.limit - cap }  }

  /**
   * Best-effort emergency compaction for one provider-confirmed context
   * overflow: pick the largest eligible (guarded, tool-pairing-balanced)
   * surface range and land the normal durable transaction with a fixed
   * engine-written marker summary. No LLM call — the provider just rejected
   * the request for being too large, so there is no model turn available to
   * write a summary; the originals stay in the append-only log, so
   * recall still indexes and restores them, and the
   * model can re-run the compress tool over the marker later to write a real
   * summary. Returns null when nothing eligible exists (nothing to reclaim).
   */
  private async compactForOverflow(agent: Agent, signal: AbortSignal): Promise<OverflowRecoveryResult | null> {
    signal.throwIfAborted()
    const session = agent.session
    const surfaceMessages = eventsToCoreMessages(surfaceEventsOf(session))
    const tokenCount = resolveTokenCount(agent, surfaceMessages)
    const window = await this.windowFor(agent)
    const config = kernelConfigFor({ ...this.env, modelContextLimit: window.limit })
    // The ref map the kernel transaction needs (the same contract handleCompress relies on) is
    // derived INSIDE `foldPass`, per pass: pass 0 appends a `compaction/summary` plus a
    // checkpoint node, so a snapshot taken before the loop could not see the surface the next
    // pass anchors on (round-12 review Q1). Deriving it once here as well only ran
    // `processTurn` a second time on the same unchanged surface for pass 0 (round-13 review
    // Q1/Q3a); we read nothing but the ref map.
    const foldPricing = foldPricingFor(this.env)
    const minimumFold = Math.min(
      MIN_OVERFLOW_FOLD_TOKENS,
      Math.max(1, Math.floor(window.limit * OVERFLOW_FOLD_WINDOW_RATIO)),
    )
    // ONE pass folds ONE eligible span. The provider has already rejected the request and the
    // retry budget is one, so a single insufficient fold used to burn that budget and stall the
    // session (round-11 review Q5): barriers (an instruction row, a checkpoint, the active turn)
    // split the surface, so folding the LARGEST span can still leave the projection above the
    // window while folding a second, disjoint span would fit. Keep folding further disjoint
    // spans inside the SAME retry — an extra fold costs a barrier and a re-prefill, it costs no
    // retry, and the projection stays advisory at every pass.
    const foldPass = (pass: number): {
      readonly record: OverflowRecoveryResult | null
      readonly fitRisk: boolean
      readonly liveAfterTokens: number
      /** Tokens still foldable AFTER this pass (eligible rows at or above `minimumFold`). */
      readonly remainingReclaimTokens: number
      readonly inflation: number
      readonly overhead: number
    } => {
      // Each pass re-reads the log and re-derives the ref map from scratch. Pass 0 appended a
      // `compaction/summary` plus a checkpoint node to the surface, and reusing the pre-loop
      // `coreMessages` meant pass 1+ anchored and priced a surface it could not see — the new
      // nodes were absent from `byRaw`, so its span was built from a stale projection
      // (round-12 review Q1, both reviewers). `processTurn` is pure state-in/state-out, so
      // re-running it (at most OVERFLOW_FOLD_PASSES times, no LLM call) is the cheap fix.
      const messages = allLogMessages(session)
      const passTurn = this.kernel.processTurn({ messages, state: this.store.stateFor(session), config, tokenCount })
      const stateNow = passTurn.state
      this.store.set(session, stateNow)
      // One kernel range spanning the whole surface: buildCompressibleSeqRanges
      // applies the host guards (protected tail, last real user turn, CURRENT
      // instruction rows, checkpoints, the system node) and splits it into
      // balanced segments — the same geometry the nudge range table offers the
      // model, so the emergency path can only shadow what the model itself
      // could have compressed. The ref map comes from the state THIS pass was handed — after
      // a pass lands, the kernel state carries the refs of the rewritten surface.
      const byRaw = stateNow.messageRefs.byRaw
      let firstSeq: number | undefined
      let lastSeq: number | undefined
      for (const seq of session.surface.nodes) {
        if (byRaw[String(seq)] === undefined) continue
        firstSeq ??= seq
        lastSeq = seq
      }
      if (firstSeq === undefined || lastSeq === undefined) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      const view: KernelRangeView = {
        ranges: [{ startRef: byRaw[String(firstSeq)]!, endRef: byRaw[String(lastSeq)]! }],
        refs: { byRef: stateNow.messageRefs.byRef },
      }
      // `preserveRecent: 5` is region.ts's own default (the nudge range table's
      // recent-tail protection), pinned EXPLICITLY here: this path acts without
      // asking the model, so its safety margin must not silently drift with a
      // tuning change made for the advisory table.
      const ranges = buildCompressibleSeqRanges(session, view, {
        preserveRecent: 5,
        mediaPriceOf: meterMediaPriceResolver(agent, session),
        // Emergency semantics are "fold what is foldable": a stale instruction row
        // (superseded AGENTS.md injection / superseded catalog) cannot come back —
        // folding it while the newest copy stays visible reclaims real bytes and
        // never re-triggers an injection (rule 16, live-measured). Keeping it a
        // barrier here would make the emergency pool deny the only reclaim a long
        // session has left. The CURRENT copy stays a wall: it sits in
        // `protectedSurfaceSeqs` and outside `staleInstructionRowsOf` on every path.
        foldStaleInstructionRows: true,
        foldPricing,
      })
      if (ranges.length === 0) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      // Each pass folds one eligible span, largest first (the first pass is host parity),
      // priced in the same vocabulary as the nudge range table.
      //
      // The working-set floor applies here too (cross-review Q6, all three
      // reviewers): this is the only fold in the system with no model in the loop,
      // and a too-small emergency fold pays the full one-time re-prefill and then
      // re-triggers overflow a few turns later. DELIBERATE DIFFERENCE from the
      // model-facing guard: when EVERY eligible range is below the floor we still
      // fold the largest one, because refusing here means the provider rejects the
      // request outright — the session is already out of room, so a shallow working
      // set beats a hard failure. The model path refuses instead, because it can
      // choose a wider span or read from files.
      const aboveFloor = ranges.filter((range) => !range.belowFloor)
      const pool = aboveFloor.length > 0 ? aboveFloor : ranges
      const best = pool.reduce((largest, range) => (range.tokens > largest.tokens ? range : largest))
      // A fold re-prefills everything after its rewrite point (40–52 s measured at
      // an early rewrite point, ~40 s at a late one). Reclaiming a
      // trickle cannot buy a usable request, so a tiny election is worse than
      // letting the retry fail (cross-review Q1) — but the trickle test is
      // window-relative: on a 2K window the only fold available IS small, and refusing it
      // means the session dies instead.
      if (best.tokens < minimumFold) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      const priced = foldSpanPricing(session, best.start, best.end, foldPricing)
      // Fit check, calibrated rather than assumed: our estimate uses this plugin’s
      // tokenizer, not the engine's, so it is only sound as an upper bound. The
      // calibration is a PAIRED sample (billed vs the surface estimate of the same
      // moment, recorded at agent/pre-step) plus the fixed request overhead the
      // provider bills and we do not count as surface — the system prompt, the tool
      // registry, host rows. Dividing the PEAK billed request by the CURRENT surface
      // was the earlier form and it fails both ways (cross-review Q3): on a session
      // with a large durable prefix the ratio reads 3-4× and refuses forever, while
      // the overhead the fold cannot remove is missing from the live side, so a fold
      // that still overflows is accepted. The two factors are DIFFERENT kinds of
      // error and must not multiply each other (cross-review Q3): the ratio is the
      // tokenizer gap on the counted part, the overhead is a fixed absolute the fold
      // can never remove. `(liveAfter + overhead) * ratio` inflated the fixed
      // overhead by the ratio as well and refused the one fold that could have
      // saved the request.
      // The projection is ADVISORY, never a gate: this path only runs when the
      // provider has ALREADY rejected the request, so refusing here IS the recovery
      // failure — the budget is one retry, then the original error stands and the
      // session stalls. What the projection buys is a stamp instead: FIT RISK
      // rides the marker provenance, so a fold that could not fit stays visible in
      // the transcript and the block ledger rather than silently never happening.
      // Measured 2026-11-15: with the same-request calibration (S2) the paired ratio
      // reads honestly high, and refusing on it deleted the emergency fold in the
      // e2e overflow scenario outright (five assertions failed — nothing was folded).
      // A false acceptance costs one re-prefill plus one barrier, at most once per
      // turn; a false refusal costs a stalled session.
      const paired = pairedInflation(session)
      const inflation = paired ?? DEFAULT_OVERFLOW_INFLATION
      const overhead = requestOverhead(session)
      const projected = priced.liveAfterTokens * inflation + overhead
      const fitRisk = projected > window.limit
      if (fitRisk) {
        this.ctx.logger.warn(
          `billion-context-dsh: overflow fold pass ${pass} proceeds with a FIT RISK — projected `
          + `${Math.round(projected)} > window ${window.limit} (liveAfter `
          + `${priced.liveAfterTokens}, overhead ${overhead}, inflation ${inflation.toFixed(2)}, `
          + `${paired === undefined ? String(DEFAULT_OVERFLOW_INFLATION) + " fallback" : "paired sample"})`,
        )
      }
      const provenanceParts: string[] = []
      if (priced.belowFloor) {
        provenanceParts.push(`below-floor emergency fold: floor ~${foldPricing.floorTokens ?? 0}, folded ~${priced.spanTokens}, live-after ~${priced.liveAfterTokens}, window ${window.limit}, reason overflow-no-above-floor`)
      }
      if (fitRisk) {
        // Phrased as a PAST observation, not a present-tense claim (round-11 review Q9): this
        // provenance rides the marker node and stays visible on the surface for the rest of the
        // session, and a retry that then succeeded left an unfit fold reading "projected 290000
        // > window 100000" as though the CURRENT context still exceeded the window.
        provenanceParts.push(`FIT RISK when folding: projected ~${Math.round(projected)} > window ${window.limit}, live-after ~${priced.liveAfterTokens}, overhead ${overhead}, inflation ${inflation.toFixed(2)}, ${paired === undefined ? String(DEFAULT_OVERFLOW_INFLATION) + " fallback" : "paired sample"}`)
      }
      const provenance = provenanceParts.length > 0 ? provenanceParts.join("; ") : undefined
      const startRef = byRaw[String(best.start)]
      const endRef = byRaw[String(best.end)]
      if (startRef === undefined || endRef === undefined) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      // Engine-written marker: the shared constructor in src/messages.ts is what
      // makes the projection net exempt it from the model-written frame, so the
      // model-visible text and the durable/recalled text stay identical.
      const summary = overflowMarkerSummary(best.count, provenance, { startSeq: best.start, endSeq: best.end })
      const applied = this.kernel.applyCompression({
        ranges: [{ startRef, endRef, summary, topic: 'context-overflow recovery' }],
        messages,
        state: stateNow,
        config,
      })
      if (applied.result.blocksCreated === 0) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      // Commit the kernel state only once the durable transaction below can land. The old
      // ordering wrote it BEFORE the `created === undefined` / `shadowed.length === 0` early
      // returns, leaving a kernel block in the store with no `compaction/summary` on the
      // session (round-12 review Q1, both reviewers).
      const previousIds = new Set(stateNow.blocks.map((block) => block.blockId))
      const created = applied.state.blocks.find((block) => !previousIds.has(block.blockId))
      if (created === undefined) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      const shadowed = shadowedSeqsOf(session, best.start, best.end)
      if (shadowed.length === 0) return { record: null, fitRisk: false, liveAfterTokens: 0, remainingReclaimTokens: 0, inflation: DEFAULT_OVERFLOW_INFLATION, overhead: 0 }
      const shadowedTokenCount = shadowedTokensViaMeter(session, shadowed, agent.ctx)
      const { provider, model } = routeFor(agent)
      runCompactionTransaction(session, {
        start: best.start,
        end: best.end,
        shadowedSeqs: shadowed,
        summary: [{ type: 'text', text: summary }],
        shadowedTokenCount,
        provider,
        model,
        topic: 'context-overflow recovery',
        kernelBlockId: created.blockId,
        directMessageIds: created.directMessageIds,
        effectiveMessageIds: created.effectiveMessageIds,
      })
      this.store.set(session, applied.state)
      // What is still foldable after this pass: eligible rows the pool offered that are at or
      // above the minimum reclaim. The loop uses it to stop when NO combination of further
      // folds can reach the window — folding anyway spends a permanent barrier per pass for a
      // projection that is mathematically unreachable (round-13 review Q3c), the same failure
      // `overheadBlocksFit` closes for the fixed-head case.
      // Summed over EVERY eligible row, not over `pool`. `pool` is `aboveFloor` while any row
      // clears the floor; after this pass folds one of them the next pass can find `aboveFloor`
      // EMPTY and fall back to `ranges`, exposing rows this pass never saw (round-14 review Q4:
      // a 10K above-floor row plus a 24K below-floor row on a 38K surface stopped after one pass
      // because the snapshot read remainingReclaim 0 while a second fold would have fit).
      const remainingReclaim = ranges
        .filter((range) => range !== best && range.tokens >= minimumFold)
        .reduce((sum, range) => sum + range.tokens, 0)
      return {
        record: { start: best.start, end: best.end, shadowedSeqs: shadowed, shadowedTokenCount },
        fitRisk,
        liveAfterTokens: priced.liveAfterTokens,
        remainingReclaimTokens: remainingReclaim,
        inflation,
        overhead,
      }
    }
    // A fixed request head that already exceeds the window cannot be folded away: every pass
    // removes SURFACE tokens, never overhead. Continuing would burn up to OVERFLOW_FOLD_PASSES
    // permanent barriers for a projection that is mathematically unreachable (round-12 review
    // Q1, both reviewers — the unit fixture used `overhead > window` precisely to force three
    // passes, which is why the cap lives here and not inside a pass).
    const overheadBlocksFit = requestOverhead(session) >= window.limit
    let recovered: OverflowRecoveryResult | null = null
    for (let pass = 0; pass < OVERFLOW_FOLD_PASSES; pass += 1) {
      const outcome = foldPass(pass)
      if (outcome.record === null) break
      recovered ??= outcome.record
      if (!outcome.fitRisk) break
      if (overheadBlocksFit) break
      // Reachability, not just the fixed head: even folding EVERY row still eligible, the
      // un-foldable remainder (protected tail, instruction rows, the newest skill catalog, the
      // active turn, this pass's own marker) times the calibrated inflation plus the overhead
      // still beats the window — so further passes burn barriers for nothing (round-13 review
      // Q3c).
      if ((outcome.liveAfterTokens - outcome.remainingReclaimTokens) * outcome.inflation + outcome.overhead > window.limit) break
    }
    // The retried request belongs to the FOLDED surface; pairing its billed number with the
    // pre-fold anchor would read a tokenizer inflation the engine itself caused (round-11 Q4).
    if (recovered !== null) setCalibrationAnchor(session, surfaceTokens(session))
    return recovered
  }

  /**
   * 方案 D layer 2 — the no-action degrade fold. The session has been riding at
   * or above the emergency nudge threshold for `emergencyFoldAfterTurns`
   * consecutive turns with no successful `compress` call; the model has been
   * reminded (the hard rider made the nudge an action instruction) and did not
   * act, so the engine executes ONE marker fold itself — the same durable
   * transaction the overflow path runs (largest guarded eligible span,
   * engine-written marker summary, runCompactionTransaction ledger, no LLM
   * call), but NOT through `agent/request-error`: no retry is consumed and the
   * overflow budget is untouched. Runs at most once per session
   * (`degradedSessions`) so a still-ignored fold cannot loop into barrier spam.
   * The marker carries an EMERGENCY PROVENANCE naming the no-action reason and
   * the reminder count, so a later reader knows the loss of semantic detail was
   * procedural, not model-written. The calibration anchor is re-pointed at the
   * folded surface exactly like `compactForOverflow` does.
   * Returns null when nothing eligible exists (nothing to reclaim).
   */
  private async emergencyNoActionFold(agent: Agent, reminderCount: number): Promise<OverflowRecoveryResult | null> {
    const session = agent.session
    const surfaceMessages = eventsToCoreMessages(surfaceEventsOf(session))
    const tokenCount = resolveTokenCount(agent, surfaceMessages)
    const window = await this.windowFor(agent)
    const config = kernelConfigFor({ ...this.env, modelContextLimit: window.limit })
    const messages = allLogMessages(session)
    const turn = this.kernel.processTurn({ messages, state: this.store.stateFor(session), config, tokenCount })
    const stateNow = turn.state
    this.store.set(session, stateNow)
    const byRaw = stateNow.messageRefs.byRaw
    let firstSeq: number | undefined
    let lastSeq: number | undefined
    for (const seq of session.surface.nodes) {
      if (byRaw[String(seq)] === undefined) continue
      firstSeq ??= seq
      lastSeq = seq
    }
    if (firstSeq === undefined || lastSeq === undefined) return null
    const view: KernelRangeView = {
      ranges: [{ startRef: byRaw[String(firstSeq)]!, endRef: byRaw[String(lastSeq)]! }],
      refs: { byRef: stateNow.messageRefs.byRef },
    }
    // Same guard set as the overflow path, same `preserveRecent: 5` pinned
    // explicitly, same stale-instruction folding — the degrade can only shadow
    // what the model itself could have compressed.
    const ranges = buildCompressibleSeqRanges(session, view, {
      preserveRecent: 5,
      mediaPriceOf: meterMediaPriceResolver(agent, session),
      foldStaleInstructionRows: true,
      foldPricing: foldPricingFor(this.env),
    })
    if (ranges.length === 0) return null
    const aboveFloor = ranges.filter((range) => !range.belowFloor)
    const pool = aboveFloor.length > 0 ? aboveFloor : ranges
    const best = pool.reduce((largest, range) => (range.tokens > largest.tokens ? range : largest))
    const startRef = byRaw[String(best.start)]
    const endRef = byRaw[String(best.end)]
    if (startRef === undefined || endRef === undefined) return null
    const provenance = `no-action degrade: the model was reminded ${reminderCount} turn(s) at or above the emergency threshold without calling compress; this fold was written procedurally by the engine`
    const summary = overflowMarkerSummary(best.count, provenance, { startSeq: best.start, endSeq: best.end })
    const applied = this.kernel.applyCompression({
      ranges: [{ startRef, endRef, summary, topic: 'no-action emergency degrade' }],
      messages,
      state: stateNow,
      config,
    })
    if (applied.result.blocksCreated === 0) return null
    const previousIds = new Set(stateNow.blocks.map((block) => block.blockId))
    const created = applied.state.blocks.find((block) => !previousIds.has(block.blockId))
    if (created === undefined) return null
    const shadowed = shadowedSeqsOf(session, best.start, best.end)
    if (shadowed.length === 0) return null
    const shadowedTokenCount = shadowedTokensViaMeter(session, shadowed, agent.ctx)
    const { provider, model } = routeFor(agent)
    runCompactionTransaction(session, {
      start: best.start,
      end: best.end,
      shadowedSeqs: shadowed,
      summary: [{ type: 'text', text: summary }],
      shadowedTokenCount,
      provider,
      model,
      topic: 'no-action emergency degrade',
      kernelBlockId: created.blockId,
      directMessageIds: created.directMessageIds,
      effectiveMessageIds: created.effectiveMessageIds,
    })
    this.store.set(session, applied.state)
    // Same anchor discipline as compactForOverflow: the next billed request
    // belongs to the FOLDED surface and must not pair with the pre-fold one.
    setCalibrationAnchor(session, surfaceTokens(session))
    return { start: best.start, end: best.end, shadowedSeqs: shadowed, shadowedTokenCount }
  }

  /** ACP is model-driven: automatic pressure policy never summarizes by itself. */
  override async compactIfNeeded(
    _agent: CompactionAgentContext,
    _trigger: CompactionTrigger,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    signal.throwIfAborted()
    return null
  }

  /** Explicit idle-session compaction: ACP leaves the decision to the model. */
  override async compactNow(
    _agent: ManualCompactAgentContext,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    signal.throwIfAborted()
    return null
  }

  /**
   * The model-driven path lands through the `compress` tool, which runs the
   * full durable transaction directly. This seam method rejects with guidance:
   * automatic summarization is exactly what ACP replaces.
   */
  override async compactRegion(
    _start: number,
    _end: number,
    _agent: CompactionAgentContext,
    signal?: AbortSignal,
  ): Promise<CompactionResult> {
    signal?.throwIfAborted()
    throw new ManualCompactionError(
      'summary',
      'billion-context-dsh is model-driven: use the compress tool instead of automatic summarization',
    )
  }
}

export default AcpCompactionEngine
