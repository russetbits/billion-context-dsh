/**
 * M4 — ACP nudge: the kernel's compression recommendation, rendered as an
 * injected user message with a seq-based compressible-range table (D1:
 * "seq is the ref" — DSH has no in-memory message rewrite hook, so the model
 * targets ranges by surface seq rather than by <acp> tags).
 * @module billion-context-dsh/nudge
 */
import { type CompressionCore, type CompressionState, type ContextBreakdown, type CoreMessage, type NudgeDecision } from './kernel/index.ts';
import { type UserMessage } from '@deepseek-ai/dsh-llm';
import type { Agent } from '@deepseek-ai/dsh-agent';
import { AcpStateStore } from './state.ts';
import { type FoldPricing, type KernelRangeView, type MediaPriceOf, type SeqCompressibleRange } from './region.ts';
import { type ProviderUsage } from './usage.ts';
import { type KernelConfigInput } from './config.ts';
import { type ResolvedPrompts } from './prompts.ts';
import { type ResolvedCallStyle } from './ptc.ts';
export declare function stripNudgeGuidance(text: string): string;
/** Kernel inputs the nudge path shares with the compress tool. */
export interface NudgeEnvironment extends KernelConfigInput {
    readonly kernel: CompressionCore;
    readonly store: AcpStateStore;
    /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
    readonly prompts?: ResolvedPrompts;
    /** How the model can REACH these tools this step (src/ptc.ts); default `native`.
     *  `prompts` must already be the matching variant — the engine swaps both
     *  together in `applyToolCallStyle`. */
    readonly toolCallStyle?: ResolvedCallStyle;
    /** Append the one-shot direct-call correction (a leaked XML call was seen). */
    readonly ptcLeakNote?: boolean;
}
export interface NudgeOutcome {
    readonly message: UserMessage;
    readonly emergency: boolean;
}
/**
 * Resolve the best available token count for ACP pressure decisions.
 *
 * Priority chain:
 * 1. `sessionProjections.contextPressure.projectedTokens` — matches the UI's
 *    context-occupancy display (includes fixed overhead: system prompt, tool
 *    definitions, AGENTS.md, etc.). Provider-anchored; reacts to compaction.
 * 2. `tokenMeter.measure(session).surfaceTokens` — heuristic surface-only
 *    estimate (pure conversation messages, no fixed overhead). Falls back
 *    when sessionProjections is unavailable or has no provider anchor yet.
 * 3. `defaultCountTokens` character heuristic — last resort for tests and
 *    minimal hosts that lack the token-meter service.
 */
export declare function resolveTokenCount(agent: Agent, coreMessages: CoreMessage[]): number;
/**
 * A-9（2026-11-23）：批量口径汇总。
 *
 * 区间表每一行的 `fold ~N` 是「只折这一段要重读多少」，而折多段付的是同一笔
 * （rewrite point 由最早那段决定，保留量反而更少）。实测 2026-11-23 的真机
 * 区间表里，单段净收益全是负数（4426 − 46458），但把整片一起折却是 1.5 次请求
 * 回本 —— 只列单段 fold 等于把模型劝退。这里给出并集收益、共用的一次性成本与
 * 回本次数。
 *
 * `reclaimTokens` 取「不相交子集和」而不是简单相加：kernel 会给出不同粒度的
 * 重叠区间（如 881..951 与 907..941），直接相加会高估收益。按 end 升序贪心选
 * 不相交区间得到的是真实可折量的**下界**，对模型是安全的一侧。
 * `foldCostTokens` 取 union 末段的 suffix 价 —— 批量折的 re-prefill 由最后一段
 * 之后的内容决定，不是各行 fold 之和。
 */
export interface FoldPlan {
    readonly rows: number;
    readonly reclaimTokens: number;
    /** 一次性成本 = 折后**保留下来的**那部分要重读多少（不是折掉的部分）。 */
    readonly foldCostTokens: number;
    readonly breakEvenRequests: number;
}
/** 手写摘要的常驻成本（真机 condense 实测中位 ~915t），用来算保留下来的体积。 */
export declare const TYPICAL_NOTE_TOKENS = 900;
/**
 * A-9 / M-1（2026-11-23~24）：批量折叠计划。
 *
 * 判定口径是**比例**，不是「压了有没有收益」这种二元问题：
 *   保留下来的量 = suffix[start] − Σ(折掉) + note      ← 下一请求要重读的量
 *   每请求收益   = Σ(折掉) − note
 *   回本次数     = 保留量 / 每请求收益
 * 只要 Σ 足够大，保留量就足够小，回本很快；反过来，表面尾部一段小内容虽然
 * 重读成本接近 0，但也省不下多少。所以选段的标准是**回本次数最小**，而不是
 * 「折得最多」或「成本最低」。
 *
 * `suffix[start]` 用 `foldCostTokens + tokens` 还原：`foldCostTokens` 是
 * rewrite point（该段 end 之后）的价，加上该段自身的 tokens 正好是它 START
 * 之后的全部可见量。
 *
 * 候选必须来自**全部**可压段（调用方不要先 slice），否则最优下界可能落在被
 * 截掉的后半段里 —— 真机上最老 6 段恰好最碎、fold 价最高。
 */
export declare function foldPlanOf(ranges: readonly SeqCompressibleRange[], notesTokens?: number): FoldPlan | null;
/**
 * Lazy per-seq media price. The FIRST lookup triggers one meter measurement, so
 * the range walk only asks about seqs that really carry an attachment — a
 * media-free session never pays for the measurement (issue #117, issue #110).
 * Exported for the context-overflow recovery path, which ranks ranges with the
 * same vocabulary as the nudge table.
 */
export declare function meterMediaPriceResolver(agent: Agent, session: import('@deepseek-ai/dsh-session').Session): MediaPriceOf;
/**
 * Render the compressible-range table as seq refs for the model.
 *
 * The spans are the kernel's own (`compressibleRanges`, translated to surface
 * seqs) with the host guards applied on top — see buildCompressibleSeqRanges.
 * This function used to self-compute them from the surface as a labeled
 * `UPSTREAM:` workaround for kernel ref-map drift; that drift is fixed upstream
 * (acp-kernel #207) and the workaround is gone (rule 11).
 */
export declare function rangeTable(session: import('@deepseek-ai/dsh-session').Session, kernelView: KernelRangeView, prompts?: ResolvedPrompts, mediaPriceOf?: MediaPriceOf, foldPricing?: FoldPricing): string;
/**
 * Compute a SURFACE-ONLY context breakdown for display, aligned with
 * `acp_status` (kernel `buildStatusReport`/`renderOverview`).
 *
 * The kernel's own `computeContextBreakdown` (which the nudge text renders)
 * walks the message array it is fed — and `buildNudge` feeds it the FULL log
 * (`allLogMessages`, needed so T2/T3 distillation can anchor every block). So
 * a session with compressed blocks reports HISTORICAL totals there: every
 * original tool/text message already absorbed into a block is counted again,
 * e.g. `85.2K tool` for ~8.5K of live tool context. `acp_status` instead feeds
 * `buildStatusReport` the VISIBLE surface + active-block summaries, so its
 * breakdown reads the true current context. This function reproduces that
 * visible-surface reality for the nudge line so the two tools agree.
 *
 * Classification replicates kernel `computeContextBreakdown` (tool-call/
 * tool-result → tool, `system` role → system, `` code `` fence in text →
 * code, else text) EXCEPT summaries: kernel detects summaries by a
 * `[Compressed conversation section]` text prefix, which never matches a DSH
 * checkpoint node (our summary is the plain summary + `compactCheckpointSource`
 * source marker). We instead count active-block summaries directly from kernel
 * state (same source `buildStatusReport` uses), and the caller must exclude
 * checkpoint summary nodes from `messages` (they are not in any block's
 * `effectiveMessageIds` and would double-count — mirror of `/acp-prune` status's
 * `isCheckpointNode` exclusion).
 */
export declare function computeSurfaceBreakdown(state: CompressionState, messages: readonly CoreMessage[], total: number, growth: number): ContextBreakdown;
/**
 * Max emergency nudge injections within a single user turn. Bounds the
 * positive-feedback loop where an unrelieved ≥emergency-threshold pressure
 * re-injects a durable emergency nudge on every pre-step forever (issue #108).
 * Mirrors billion-context-pi commit 414acd1 (cap emergency nudge injections per
 * user turn). Normal-pressure nudges remain limited to one per turn regardless.
 */
export declare const EMERGENCY_NUDGE_MAX_PER_TURN = 3;
/**
 * Decide and build one nudge message for the agent's next pre-step. Returns
 * null when the kernel recommends no nudge or the per-turn budget is spent:
 * normal-pressure nudges fire at most once per user turn, and emergency nudges
 * are capped at {@link EMERGENCY_NUDGE_MAX_PER_TURN} per user turn so an
 * unrelieved ≥threshold pressure cannot re-inject a durable nudge on every
 * pre-step forever (issue #108). Also advances the in-memory kernel state (ref
 * assignment) so the compress tool can resolve seq → mNNNNN refs.
 *
 * `onEmergencyCapHit` (optional) fires when the kernel still wants an
 * emergency nudge but the per-turn budget is spent — the host uses it to log
 * why the model stops receiving nudges (issue #108 review).
 */
/**
 * Per-turn injection gate: WHICH turn the last normal nudge went out on, and at
 * what pressure tier.
 *
 * B-2（2026-11-23）：记 turn 而不记档位，会让「同一个 turn 内档位的升级」被自己
 * 吞掉 —— soft 档在 turn 开始时发一条，随后压力继续爬过 maxPct，真正的 hard
 * 催促却因为 `lastNudgeTurn === turn` 直接 return。存档位后只有「同 turn 且档位
 * 不升」才节流，升级仍能落地（acp-rs orchestrator.rs:532 同口径）。
 */
export interface NudgeGate {
    readonly turn: number;
    /** 0 = soft（minPct 以上）、1 = hard（maxPct 以上）、2 = emergency。 */
    readonly tier: number;
}
export declare function buildNudge(agent: Agent, env: NudgeEnvironment, lastNudgeTurn: Map<string, NudgeGate>, emergencyNudges: Map<string, {
    turn: number;
    count: number;
}>, onEmergencyCapHit?: () => void): NudgeOutcome | null;
/**
 * Render the nudge message text. DEFAULT (no `config.prompts.nudge` override)
 * calls the kernel's own `renderNudgeText` — EFFICIENCY_NOTE/EMERGENCY_HEADER,
 * context breakdown, HOW_TO_COMPRESS_RULES, tier rules, and the batch tip all
 * come from acp-kernel verbatim (the kernel-alignment principle). Only the
 * ref-ID-oriented segments are replaced with our seq-based equivalents,
 * because DSH has no `<acp>` ref tags — see docs/dsh-porting-verification.md:
 * - `rangesStr` (mNNNNN refs) → the surface-seq range table;
 * - the emergency JSON example (startId/endId) → a seq example;
 * - the tier trigger block (block ids bN) → our tier line with surface seqs.
 * When a host overrides any `prompts.nudge` slot, the template path is used so
 * `config.prompts` keeps full control (custom copy wins over kernel defaults).
 */
export declare function buildNudgeText(nudge: NudgeDecision, emergency: boolean, session: import('@deepseek-ai/dsh-session').Session, kernelView: KernelRangeView, prompts?: ResolvedPrompts, mediaPriceOf?: MediaPriceOf, style?: ResolvedCallStyle, leakNote?: boolean, usage?: ProviderUsage, foldPricing?: FoldPricing): string;
