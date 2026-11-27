/**
 * M4 — ACP nudge: the kernel's compression recommendation, rendered as an
 * injected user message with a seq-based compressible-range table (D1:
 * "seq is the ref" — DSH has no in-memory message rewrite hook, so the model
 * targets ranges by surface seq rather than by <acp> tags).
 * @module billion-context-dsh/nudge
 */

import {
  COMPRESS_PHILOSOPHY,
  HOW_TO_COMPRESS_RULES,
  TIER2_DISTILL_RULES,
  TIER3_CONDENSE_RULES,
  defaultCountTokens,
  renderNudgeText,
  type CompressionCore,
  type CompressionState,
  type ContextBreakdown,
  type CoreMessage,
  type NudgeDecision,
} from './kernel/index.ts'
import { createUserMessage, type UserMessage } from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { AcpStateStore } from './state.ts'
import { allLogMessages, eventsToCoreMessages, isCheckpointNode, surfaceEventsOf } from './messages.ts'
import {
  MIN_CONDENSE_SPAN_TOKENS,
  belowWorkingSetFloor,
  blockRegistry,
  buildCompressibleSeqRanges,
  findOpenTurn,
  summarySeqOfKernelBlock,
  surfaceSummary,
  type FoldPricing,
  type KernelRangeView,
  type MediaPriceOf,
  type SeqCompressibleRange,
} from './region.ts'
import { mediaPriceViaMeter } from './host-tokens.ts'
import { costSinceLastCompress, providerUsage, requestOverhead, type ProviderUsage } from './usage.ts'
import { sessionEventsOf } from './session-events.ts'
import { foldPricingFor, kernelConfigFor, type KernelConfigInput } from './config.ts'
import { DEFAULT_RESOLVED, renderTemplate, type ResolvedPrompts } from './prompts.ts'
import { PTC_LEAK_CORRECTION, ptcText, type ResolvedCallStyle } from './ptc.ts'

/**
 * B6 nudge 瘦身（2026-09-08 方案 §4 B6）：哲学段与压缩规则段**移出 nudge**——
 * 它们已经住在系统提示与工具描述里（各注一次），每拍复读=同一份文本反复计费
 * （本会话实测：nudge 正文 6.1 KB/次 × 3 = 18.4 KB）。nudge 只留「该压缩了 + 压缩哪些」。
 *
 * UPSTREAM: this is a labeled host-side workaround (AGENTS.md rule 11), not a
 * long-term design. The four texts below are imported from acp-kernel and are
 * removed from already-rendered nudge text, so the clean fix belongs upstream:
 * an acp-kernel option that renders the nudge without the guidance blocks.
 * Drop stripNudgeGuidance and use that option once it exists. Tracked in
 * docs/dsh-porting-verification.md.
 */
const GUIDANCE_BLOCKS = [COMPRESS_PHILOSOPHY, HOW_TO_COMPRESS_RULES, TIER2_DISTILL_RULES, TIER3_CONDENSE_RULES] as const

export function stripNudgeGuidance(text: string): string {
  let out = text
  for (const block of GUIDANCE_BLOCKS) out = out.split(block).join('')
  return out.replace(/\n{3,}/g, '\n\n').trim()
}

/** Kernel inputs the nudge path shares with the compress tool. */
export interface NudgeEnvironment extends KernelConfigInput {
  readonly kernel: CompressionCore
  readonly store: AcpStateStore
  /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
  readonly prompts?: ResolvedPrompts
  /** How the model can REACH these tools this step (src/ptc.ts); default `native`.
   *  `prompts` must already be the matching variant — the engine swaps both
   *  together in `applyToolCallStyle`. */
  readonly toolCallStyle?: ResolvedCallStyle
  /** Append the one-shot direct-call correction (a leaked XML call was seen). */
  readonly ptcLeakNote?: boolean
}

export interface NudgeOutcome {
  readonly message: UserMessage
  readonly emergency: boolean
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
export function resolveTokenCount(agent: Agent, coreMessages: CoreMessage[]): number {
  // 1. Prefer sessionProjections.contextPressure.projectedTokens (matches UI).
  const projections = agent.ctx?.get?.('sessionProjections') as
    | { snapshot?: (session: unknown) => { values?: { contextPressure?: { projectedTokens?: number } } } }
    | undefined
  const projected = projections?.snapshot?.(agent.session)?.values?.contextPressure?.projectedTokens
  if (typeof projected === 'number' && projected > 0) return projected

  // 2. Fallback to tokenMeter surfaceTokens (heuristic, no fixed overhead).
  const meter = agent.ctx?.get?.('tokenMeter') as
    | { measure?: (session: unknown) => { surfaceTokens?: number } }
    | undefined
  const surface = meter?.measure?.(agent.session)?.surfaceTokens
  if (typeof surface === 'number' && surface > 0) return surface

  // 3. Last resort: character heuristic.
  return coreMessages.reduce((sum, message) => sum + defaultCountTokens(message.text ?? ''), 0)
}

/**
 * The kernel's decision and live state in the shape the range table needs.
 *
 * The kernel owns the compressible geometry (`nudge.compressibleRanges`); the
 * ref map turns a kernel ref back into a surface seq. Both objects come from
 * the SAME turn — the map must be the one `processTurn` just returned, never a
 * rehydrated store state, or the refs point at ids this surface does not have.
 */
function kernelRangeViewOf(nudge: NudgeDecision, state: CompressionState): KernelRangeView {
  return { ranges: nudge.compressibleRanges ?? [], refs: state.messageRefs }
}

/**
 * Range-row suffix naming the image/file blocks a span carries, so the model
 * does not read "~0 tokens" as "nothing to reclaim" for a picture-heavy span
 * (issue #117). Counts, not prices: the price sits in the token column.
 */
/**
 * A-3：工具名指纹，渲染成 ` · tools: bash×12, read×5`；没有任何工具调用时返回
 * 空串，整段不进行（`{tools}` 占位符落在行尾，空串即行尾无残留）。
 */
function toolsSuffixOf(range: SeqCompressibleRange): string {
  if (range.tools.length === 0) return ''
  const list = range.tools.map((tool) => `${tool.name}\u00d7${tool.count}`).join(', ')
  return ` \u00b7 tools: ${list}`
}

function mediaSuffixOf(range: SeqCompressibleRange): string {
  if (range.images === 0 && range.files === 0 && range.spills === 0 && !range.belowFloor) return ''
  const parts: string[] = []
  if (range.images > 0) parts.push(`+${range.images} image${range.images === 1 ? '' : 's'}`)
  if (range.files > 0) parts.push(`+${range.files} file${range.files === 1 ? '' : 's'}`)
  // Spilled overflow is the opposite signal from an image: the bytes are NOT in
  // the context and are still on disk, so folding this row loses nothing the
  // model cannot re-read narrow. Named last because it argues FOR the fold.
  if (range.spills > 0) parts.push(`${range.spills} spilled`)
  // The opposite warning: this row would leave the model with too little room to
  // work in. Fold a row that still leaves the floor — or fold nothing — rather than
  // strand the working set; the widest row is the one that strands it (round-4 Q1).
  // See DEFAULT_FOLD_FLOOR_PCT in src/region.ts.
  if (range.belowFloor) parts.push('below floor')
  // ` | ` is the separator the same row already uses for the tool/text share
  // column (`[tool X% | text Y%]`) and the one AGENTS.md rule 19 documents; a
  // comma inside the bracket reads as prose.
  return ` [${parts.join(' | ')}]`
}

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
  readonly rows: number
  readonly reclaimTokens: number
  /** 一次性成本 = 折后**保留下来的**那部分要重读多少（不是折掉的部分）。 */
  readonly foldCostTokens: number
  /** `foldCostTokens / reclaimTokens` — a REQUEST count, reported for display
   *  (M-3: payback is no longer a gate; the capacity gate is the only one). */
  readonly paybackRequests: number
}

/** 手写摘要的常驻成本。2026-11-25 随 doctrine 收紧为 Appendix-A 四件套简报，
 *  从真机实测分布（s2115 n=56 / s5e5c n=33，p50 ~480t）重定为 400，
 *  与 src/tools.ts 的 TYPICAL_HANDOFF_NOTE_TOKENS 保持同一依据。 */
export const TYPICAL_NOTE_TOKENS = 400

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
export function foldPlanOf(
  ranges: readonly SeqCompressibleRange[],
  notesTokens: number = TYPICAL_NOTE_TOKENS,
  pricing?: FoldPricing,
): FoldPlan | null {
  // 低于 floor 的段 compress/condense 都会拒（`belowWorkingSetFloor`），把它们算进
  // 批量计划只会给出一份必被驳回的方案 —— 交叉评审 2026-11-24 指出新算法「取满」
  // 会系统性偏向这种段。
  const usable = ranges.filter((range) => !range.belowFloor)
  if (usable.length === 0) return null
  const ordered = [...usable].sort((a, b) => a.start - b.start)
  const floor = pricing?.floorTokens
  let best: { payback: number; reclaim: number; cost: number; set: SeqCompressibleRange[] } | null = null
  for (let index = 0; index < ordered.length; index += 1) {
    const head = ordered[index]!
    // 从 head 起按 seq 升序贪心取互不相交的段：同一 head 下多折一段只增
    // 收益、不增重读成本（它本来就在 suffix 里），所以取满是正确的。
    const set: SeqCompressibleRange[] = []
    let lastEnd = Number.NEGATIVE_INFINITY
    for (let cursor = index; cursor < ordered.length; cursor += 1) {
      const range = ordered[cursor]!
      if (range.start <= lastEnd) continue
      set.push(range)
      lastEnd = range.end
    }
    const reclaim = set.reduce((sum, range) => sum + range.tokens, 0)
    if (reclaim <= notesTokens) continue
    // H1 (cross-review 2026-11-25): price the batch EXACTLY as the tool does —
    // `foldCostTokens` is `suffix[last span end + 1]`, what the next request must
    // re-prefill ONCE. The old head-relative `suffixAtStart - reclaim + notesTokens`
    // was the NET KEPT size: on one 5-row batch it quoted 17314 while the tool billed
    // 9541, and the footer called that number "one re-prefill". Two prices for one
    // fold is the drift this change exists to remove.
    const tail = set[set.length - 1]!
    const cost = Math.max(0, tail.foldCostTokens)
    // THE SAME gate `pickCondenseSpan` applies, on the BATCH's own numbers:
    // the capacity (working-set floor) gate — M-3 (2026-11-25) removed the payback /
    // cost ceilings as gates, so the recommendation here and the tool's verdict can
    // no longer disagree on them.
    const surfaceTotal = head.liveAfterFoldTokens + head.tokens
    const kept = Math.max(0, surfaceTotal - reclaim - notesTokens)
    if (reclaim < MIN_CONDENSE_SPAN_TOKENS) continue
    if (belowWorkingSetFloor(surfaceTotal, kept, floor)) continue
    const payback = cost / reclaim
    if (best === null || payback < best.payback - 1e-9
      || (Math.abs(payback - best.payback) < 1e-9 && reclaim > best.reclaim)) {
      best = { payback, reclaim, cost, set }
    }
  }
  if (best === null) return null
  return {
    rows: best.set.length,
    reclaimTokens: best.reclaim,
    foldCostTokens: Math.round(best.cost),
    paybackRequests: best.payback,
  }
}

/**
 * Lazy per-seq media price. The FIRST lookup triggers one meter measurement, so
 * the range walk only asks about seqs that really carry an attachment — a
 * media-free session never pays for the measurement (issue #117, issue #110).
 * Exported for the context-overflow recovery path, which ranks ranges with the
 * same vocabulary as the nudge table.
 */
export function meterMediaPriceResolver(
  agent: Agent,
  session: import('@deepseek-ai/dsh-session').Session,
): MediaPriceOf {
  let prices: ReadonlyMap<number, number> | null = null
  return (seq: number) => {
    if (prices === null) prices = mediaPriceViaMeter(session, agent.ctx)
    return prices.get(seq) ?? 0
  }
}

/**
 * Render the compressible-range table as seq refs for the model.
 *
 * The spans are the kernel's own (`compressibleRanges`, translated to surface
 * seqs) with the host guards applied on top — see buildCompressibleSeqRanges.
 * This function used to self-compute them from the surface as a labeled
 * `UPSTREAM:` workaround for kernel ref-map drift; that drift is fixed upstream
 * (acp-kernel #207) and the workaround is gone (rule 11).
 */
/**
 * `1.0` — the payback period in requests (`foldCost / tokens`), one decimal. `—` when the
 * row removes nothing (a ratio against zero is meaningless, not infinite on screen).
 *
 * NOT the same function as `tools.ts:formatPayback` on purpose: that one feeds refuse/
 * recommend COPY and returns `∞`, because "it never pays back" is the fact being stated;
 * this one fills a TABLE CELL where a zero-span row simply has no ratio, so `—` is right
 * and `∞` would read as a measured value. Keep both intents; do not merge them.
 */
function paybackTextOf(foldCostTokens: number, tokens: number): string {
  if (tokens <= 0) return '—'
  const value = foldCostTokens / tokens
  if (!Number.isFinite(value)) return '—'
  return (Math.round(value * 10) / 10).toFixed(1)
}

export function rangeTable(
  session: import('@deepseek-ai/dsh-session').Session,
  kernelView: KernelRangeView,
  prompts: ResolvedPrompts = DEFAULT_RESOLVED,
  mediaPriceOf?: MediaPriceOf,
  foldPricing?: FoldPricing,
): string {
  // Duplicate spans (the kernel's multi-call sub-id edges for ONE seq range) are removed by
  // `buildCompressibleSeqRanges` itself, so the six rows the model sees are six different
  // stretches of work AND every other caller sees the same deduped list — one source, no
  // second filter to drift out of sync (round-16 review Q2).
  const allRanges = buildCompressibleSeqRanges(
    session,
    kernelView,
    {
      ...(mediaPriceOf === undefined ? {} : { mediaPriceOf }),
      // A row's price is the surface suffix after its END edge. The fixed request
      // head sits BEFORE every rewrite point, so it is outside this price; adding
      // the paired overhead here double-counted it (round-10 review B1; the older
      // overhead-inclusive price still read ~2.8x under on a real fold).
      foldPricing,
    },
  )
  // 零范围:整块省略(保留现状的提前返回与 nudge 尾部 '\n')。
  if (allRanges.length === 0) return ''
  // Cheapest rewrite point FIRST, not oldest first (cross-review 2026-11-24 G1).
  // `foldCostTokens` is everything after the rewrite point, so ascending order is
  // also the latest-first order: an earlier rewrite point re-reads more. The old
  // `slice(0, 6)` on an oldest-first array showed the six MOST EXPENSIVE rows while
  // the same payload's footer told the model to prefer the LATEST ones — the table
  // and its own advice disagreed, and the model followed the table.
  const ranges = [...allRanges]
    .sort((a, b) => a.foldCostTokens - b.foldCostTokens || a.start - b.start)
    .slice(0, 6)
  // 计划只对**表里列出的**这几段求和：rows 与 reclaim 必须是模型看得见的东西。
  // 用全量候选会算出「27 段一次折」这种表里根本没显示的承诺，指不出去就是误导。
  // prefix fold 落地后 condense 折的是**最老的前缀**：它跨几段由 guard 与 floor
  // 决定，与表里这 6 行不是同一个集合，所以 plan 仍然只按可见行计价——这行教的是
  // content 路径的批量承诺，不是 condense 的跨度。
  const plan = foldPlanOf(ranges, TYPICAL_NOTE_TOKENS, foldPricing)
  // Barriers that a fragment surface strands. `blockRegistry` is log-derived (no surface
  // write), so this stays a read-only computation.
  const liveCarriers = blockRegistry(session).filter((entry) => entry.active && entry.summarySeq !== null).length
  // The condition is "a plain batch cannot pass the gates", not "every row is tiny":
  // rows can be large and still unreachable (a long protected tail makes every rewrite
  // point too expensive), and in that case the cross-barrier fold is still the way out.
  const obstacleLine = liveCarriers >= 2 && plan === null
    ? [`obstacle: barriers ${liveCarriers}, no plain fold clears the working-set gate (each foldable stretch is under the ${MIN_CONDENSE_SPAN_TOKENS}t a note-carrying fold needs) \u2014 folding across the carriers is the only way forward: compress({ condense: { summary } }) folds them in ONE transaction (tier 2 \u2014 the originals behind those carriers become unrecoverable at message level, and the result says so).`]
    : []
  const lines = ranges.map((range) =>
    renderTemplate(prompts.rangeTable.line, {
      start: range.start,
      end: range.end,
      count: range.count,
      tokens: range.tokens,
      toolPct: range.toolPct,
      textPct: 100 - range.toolPct,
      media: mediaSuffixOf(range),
      foldCost: range.foldCostTokens,
      // Payback period in REQUESTS: the re-prefill is paid once and the saving applies
      // to every later request, so `cost / saving` is the number of requests after
      // which the fold is ahead. Printed next to the cost because on a wide surface the
      // cost column alone reads "this fold costs more than it saves" — which is exactly
      // the misreading the payback number removes (the cost is one-time, the saving is not).
      payback: paybackTextOf(range.foldCostTokens, range.tokens),
      tools: toolsSuffixOf(range),
    }),
  )
  return [
    // 前导空串元素产生 nudge 中范围表前的唯一空行(§4:parts 层不再加分隔)。
    '',
    renderTemplate(prompts.rangeTable.header, { surface: surfaceSummary(session) }),
    renderTemplate(prompts.rangeTable.title, { count: ranges.length }),
    ...lines,
    ...(plan === null
      ? []
      : [renderTemplate(prompts.rangeTable.plan, {
          rows: plan.rows,
          reclaim: plan.reclaimTokens,
          fold: plan.foldCostTokens,
          payback: plan.paybackRequests.toFixed(1),
        })]),
    // A FRAGMENTED surface is a different failure from an expensive one: every gap is
    // under the note-carrying minimum, so no plain fold can qualify no matter how long
    // the model waits. The move that still works is folding ACROSS the carriers, which
    // distils them (tier 2 — the originals behind them are gone). Advertised separately
    // from the rows because it is NOT a row and must not be compared with one.
    ...obstacleLine,
    prompts.rangeTable.footer,
  ].join('\n')
}

/**
 * The token count driving pressure decisions. Prefer `resolveTokenCount` which
 * uses `sessionProjections.contextPressure.projectedTokens` (matches the UI's
 * context-occupancy display, including fixed overhead). Falls back to
 * `tokenMeter.measure(session).surfaceTokens`, then `defaultCountTokens`
 * character heuristic for tests and minimal hosts.
 */
/**
 * The token count driving pressure decisions (and, through `processTurn`, every
 * range-row size the nudge prints). G1 口径注释（M-3）：这个数与 acp_status 的
 * `Request input sizes` 行同源（同一 tokenCount 进 kernel 状态），都是估算——
 * provider 侧 window 压力可能有出入，计费请求头才是权威。
 */
function measuredTokenCount(agent: Agent, coreMessages: CoreMessage[]): number {
  return resolveTokenCount(agent, coreMessages)
}

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
export function computeSurfaceBreakdown(
  state: CompressionState,
  messages: readonly CoreMessage[],
  total: number,
  growth: number,
): ContextBreakdown {
  let system = 0
  let tool = 0
  let code = 0
  let text = 0
  for (const message of messages) {
    const tokens = defaultCountTokens(message.text ?? '')
    if (message.contentType === 'tool-call' || message.contentType === 'tool-result') {
      tool += tokens
    } else if (message.role === 'system') {
      system += tokens
    } else if ((message.text ?? '').includes('```')) {
      code += tokens
    } else {
      text += tokens
    }
  }
  let summaries = 0
  for (const block of state.blocks) {
    if (block.active) summaries += defaultCountTokens(block.summary)
  }
  return { system, tool, summaries, code, text, total, growth }
}

/**
 * Max emergency nudge injections within a single user turn. Bounds the
 * positive-feedback loop where an unrelieved ≥emergency-threshold pressure
 * re-injects a durable emergency nudge on every pre-step forever (issue #108).
 * Mirrors billion-context-pi commit 414acd1 (cap emergency nudge injections per
 * user turn). Normal-pressure nudges remain limited to one per turn regardless.
 */
export const EMERGENCY_NUDGE_MAX_PER_TURN = 3

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
  readonly turn: number
  /** 0 = soft（minPct 以上）、1 = hard（maxPct 以上）、2 = emergency。 */
  readonly tier: number
}

export function buildNudge(
  agent: Agent,
  env: NudgeEnvironment,
  lastNudgeTurn: Map<string, NudgeGate>,
  emergencyNudges: Map<string, { turn: number; count: number }>,
  onEmergencyCapHit?: () => void,
): NudgeOutcome | null {
  const session = agent.session
  const state = env.store.stateFor(session)
  // Full log for the kernel (so block anchors survive — see handleCompress);
  // the measured token count stays a SURFACE measurement.
  const coreMessages = allLogMessages(session)
  const surfaceEvents = surfaceEventsOf(session)
  const surfaceMessages = eventsToCoreMessages(surfaceEvents)
  const tokenCount = measuredTokenCount(agent, surfaceMessages)
  const config = kernelConfigFor(env)
  const turn = env.kernel.processTurn({ messages: coreMessages, state, config, tokenCount })
  env.store.set(session, turn.state)

  const nudge = turn.nudge
  if (nudge === undefined || !nudge.shouldInject) return null
  // Media-bearing spans are invisible to every text estimator, so their price
  // comes from the host meter. Lazily: a media-free session never pays for a
  // measurement (meterMediaPriceResolver only measures on first lookup).
  const mediaPriceOf = meterMediaPriceResolver(agent, session)
  // The kernel computed `contextBreakdown` from the FULL log (allLogMessages),
  // so after any compression it reports HISTORICAL totals (a huge `tool` that
  // is really the compressed-away originals). Override it with the visible
  // surface + active-block summaries so the nudge line matches acp_status
  // (which renders from `buildStatusReport` over the surface). Checkpoint
  // summary nodes are excluded (they are not in any block's effectiveMessageIds
  // and would double-count) — the same exclusion `/acp-prune` status applies. The
  // breakdown is display-only and never drives injection, so this override is
  // safe for the decision path.
  const statusMessages = eventsToCoreMessages(
    surfaceEvents.filter((event) => isCheckpointNode(event) === false),
  )
  nudge.contextBreakdown = computeSurfaceBreakdown(turn.state, statusMessages, tokenCount, nudge.contextBreakdown?.growth ?? 0)
  const emergency = nudge.breakdown?.emergencyOverride === 1

  const turnNumber = findOpenTurn(sessionEventsOf(session)) ?? 0
  // Pressure tier for this nudge (B-2). `emergency` is the kernel's own override;
  // otherwise hard = at or above the max threshold, soft = anything else that
  // still earned an injection.
  const limitTokens = env.modelContextLimit ?? 0
  const tier = emergency
    ? 2
    : limitTokens > 0 && tokenCount / limitTokens >= (env.nudgeMaxContextLimitPct ?? 0.7) ? 1 : 0
  if (!emergency) {
    // Normal-pressure nudge: at most one PER TIER per user turn. An escalation
    // (soft → hard) still lands in the same turn; a repeat does not.
    const gate = lastNudgeTurn.get(session.id)
    if (gate !== undefined && gate.turn === turnNumber && gate.tier >= tier) return null
    lastNudgeTurn.set(session.id, { turn: turnNumber, tier })
  } else {
    // Emergency nudge: bounded per user turn. Without this cap an unrelieved
    // ≥emergencyThreshold pressure re-injects a durable nudge on EVERY pre-step
    // (each appended as a user/message event), and the nudge's own tokens push
    // usage higher → a runaway feedback loop (issue #108; mirrors pi #223/#250
    // and commit 414acd1).
    const record = emergencyNudges.get(session.id)
    if (record !== undefined && record.turn === turnNumber) {
      if (record.count >= EMERGENCY_NUDGE_MAX_PER_TURN) {
        onEmergencyCapHit?.()
        return null
      }
      record.count += 1
    } else {
      emergencyNudges.set(session.id, { turn: turnNumber, count: 1 })
    }
    // Record the tier too, so a normal nudge cannot follow an emergency one in
    // the same turn (the emergency budget already bounds its own repeats).
    lastNudgeTurn.set(session.id, { turn: turnNumber, tier })
  }

  const style: ResolvedCallStyle = env.toolCallStyle ?? 'native'
  const text = buildNudgeText(
    nudge,
    emergency,
    session,
    kernelRangeViewOf(nudge, turn.state),
    env.prompts,
    mediaPriceOf,
    style,
    // The correction rides the nudge the engine is already injecting — no extra
    // message, no extra injection surface (the leak is otherwise silent).
    style === 'ptc' && env.ptcLeakNote === true,
    providerUsage(agent),
    foldPricingFor(env),
  )
  const message = createUserMessage({
    content: [{ type: 'text', text }],
    // V4 producer kind (issue #163): DSH ≥0.1.7's V4 admission rejects the
    // legacy wrapper shape `{ kind: 'plugin', plugin: … }` outright (a wedged
    // batch fails the NEXT turn with "format v4 message requires a
    // producer-owned source kind"). `plugin:<name>` is exactly what the host's
    // own V3→V4 migration emits for unregistered plugins, and DSH 0.1.5
    // sessions accept it too (probe-verified), so no version gate is needed.
    source: { kind: 'plugin:acp-nudge' },
  })
  return { message, emergency }
}

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
export function buildNudgeText(
  nudge: NudgeDecision,
  emergency: boolean,
  session: import('@deepseek-ai/dsh-session').Session,
  kernelView: KernelRangeView,
  prompts: ResolvedPrompts = DEFAULT_RESOLVED,
  mediaPriceOf?: MediaPriceOf,
  style: ResolvedCallStyle = 'native',
  leakNote = false,
  usage?: ProviderUsage,
  foldPricing?: FoldPricing,
): string {
  // A host override of any nudge slot → template rendering (config.prompts
  // keeps its v0.1.9 contract: custom copy wins). Only the pristine default
  // reference reaches the kernel path.
  const text = prompts.nudge !== DEFAULT_RESOLVED.nudge
    ? renderNudgeFromTemplates(nudge, emergency, session, kernelView, prompts, mediaPriceOf, style, usage, foldPricing)
    : adaptKernelNudgeToSeq(renderNudgeText(nudge).text, nudge, session, kernelView, prompts, mediaPriceOf, style, usage, foldPricing)
  // A leaked direct call produced no call and no error (its markup landed as
  // assistant text), so the nudge we are already injecting is the only channel
  // that can name the failure. Gated on the mode: in native mode the direct
  // shape works, so the flag means nothing there.
  return leakNote && style === 'ptc' ? `${text}\n\n${PTC_LEAK_CORRECTION}` : text
}

/**
 * Take the kernel-rendered nudge text and replace its ref-ID-oriented segments
 * with our surface-seq equivalents.
 *
 * The B6 slim step (`stripNudgeGuidance`) runs first: it removes the
 * philosophy, HOW_TO_COMPRESS_RULES and tier-2/3 rule blocks, because they
 * already live in the system prompt and the tool descriptions — repeating them
 * in every nudge only re-billed the same ~6 KB. What stays kernel-verbatim:
 * the frame, the context breakdown, the tier line and the batch tip.
 *
 * Only the ref-ID-oriented segments are replaced with our seq-based
 * equivalents, because DSH has no `<acp>` ref tags — see
 * docs/dsh-porting-verification.md:
 */
function adaptKernelNudgeToSeq(
  text: string,
  nudge: NudgeDecision,
  session: import('@deepseek-ai/dsh-session').Session,
  kernelView: KernelRangeView,
  prompts: ResolvedPrompts,
  mediaPriceOf?: MediaPriceOf,
  style: ResolvedCallStyle = 'native',
  usage?: ProviderUsage,
  foldPricing?: FoldPricing,
): string {
  // B6：先摘掉哲学/规则段（它们住在系统提示与工具描述里），再做 seq 适配
  let out = stripNudgeGuidance(text)
  // Tier nudges: replace the kernel trigger block (block ids bN) with our tier
  // line carrying surface seqs. The kernel's TIER2/3 rule blocks were already
  // removed by stripNudgeGuidance above — those rules live in the system prompt.
  if ((nudge.tier === 2 || nudge.tier === 3) && (nudge.tierTargetBlocks?.length ?? 0) > 0) {
    out = replaceTierTrigger(out, nudge, session, prompts)
  } else if (out.includes('"startId"')) {
    // Emergency nudges: replace the ref-ID JSON example with a seq example.
    out = replaceEmergencyExample(out, style)
  }
  // Replace the ref-ID range table (mNNNNN) with the surface-seq table.
  // A zero-range table leaves the kernel's own "[No specific ranges detected]"
  // notice intact — it is a better prompt than an empty table.
  const seqTable = rangeTable(session, kernelView, prompts, mediaPriceOf, foldPricing)
  if (seqTable !== '') out = replaceRangesStr(out, seqTable)
  // M-3: the kernel 💡 tip is replaced by the engine tip (which carries the
  // two-shapes contract); a host that overrides nudge.tip keeps its own line.
  if (prompts.nudge.tip !== '') out = replaceKernelTip(out, prompts.nudge.tip)
  // PTC: the kernel copy names the direct-call shape the host rejects, so the
  // rendered text (not the kernel templates — rule 9) switches to the SDK form.
  return ptcText(out, style)
}

/**
 * The rewrite-cost line: what the last compress cost in uncached input, and the
 * reason it costs anything. Rendered only when the host reports usage AND a
 * compress has been recorded in this session, so a fresh or meter-less session
 * keeps the nudge byte-identical to before.
 */
function cacheCostLine(prompts: ResolvedPrompts, session: import('@deepseek-ai/dsh-session').Session, usage: ProviderUsage | undefined): string {
if (prompts.nudge.cacheCost === '') return ''
const cost = costSinceLastCompress(session, usage)
if (cost === null) return ''
return renderTemplate(prompts.nudge.cacheCost, { uncached: cost.delta, count: cost.count })
}

/** Replace the kernel rangesStr segment (`Compressible ranges (N, oldest first):…`) with our seq table. */
function replaceRangesStr(text: string, seqTable: string): string {
  const match = text.match(/\n\n(?:Compressible ranges \(|\[No specific ranges detected)/)
  if (!match) return text
  const start = match.index!
  const rest = text.slice(start + 2)
  const next = rest.match(/\n\n/)
  const end = next !== null ? start + 2 + next.index! : text.length
  const before = text.slice(0, start)
  const after = text.slice(end)
  // seqTable starts with '\n' (the range table's leading blank line), so
  // `before` + '\n' + seqTable yields one blank line before the table.
  return before + '\n' + seqTable + after
}

/**
 * M-3 (2026-11-25): swap the KERNEL tip for the engine tip on the kernel path.
 * The vendored kernel tip advertises a PLAIN-STRING content form our schema
 * rejects (`ONE plain string holding every range`) and carries none of the
 * two-shapes contract — the Note segment that used to correct it is gone (M-3
 * merged it into the tip), so the kernel tip is REPLACED, not appended to. The
 * engine tip (prompts.nudge.tip) carries the contract in one line.
 */
function replaceKernelTip(text: string, tip: string): string {
  const at = text.lastIndexOf('\n\n\u{1F4A1} ')
  if (at === -1 || tip === '') return text
  return text.slice(0, at) + '\n\n' + tip
}

/** Replace the kernel tier trigger segment (`[TIER n …TRIGGER]…Example: compress(…)`) with our tier line. */
function replaceTierTrigger(
  text: string,
  nudge: NudgeDecision,
  session: import('@deepseek-ai/dsh-session').Session,
  prompts: ResolvedPrompts,
): string {
  const start = text.search(/\n\n(?:\[TIER \d|\[EMERGENCY — TIER \d)/)
  if (start === -1) return text
  const rest = text.slice(start + 2)
  const next = rest.match(/\n\nHOW TO COMPRESS/)
  const end = next !== null ? start + 2 + next.index! : text.length
  const targets = nudge.tierTargetBlocks!
  const summarySeqs = targets
    .map((block) => summarySeqOfKernelBlock(session, block.blockId))
    .filter((seq): seq is number => seq != null)
    .sort((a, b) => a - b)
  const pending = nudge.tier === 2 ? nudge.breakdown?.pendingT2 : nudge.breakdown?.pendingT3
  const tokens = typeof pending === 'number' ? pending : 0
  const tierValue = nudge.tier === null ? 2 : nudge.tier
  const tierLine = renderTemplate(prompts.nudge.tier, {
    tier: tierValue,
    count: targets.length,
    prevTier: tierValue - 1,
    tokens,
    seqs: summarySeqs.join(', '),
    firstSeq: summarySeqs[0] ?? 'n/a',
    lastSeq: summarySeqs[summarySeqs.length - 1] ?? 'n/a',
  })
  return text.slice(0, start) + '\n\n' + tierLine + text.slice(end)
}

/** Replace the kernel emergency JSON example (startId/endId) with a seq example. */
function replaceEmergencyExample(text: string, style: ResolvedCallStyle = 'native'): string {
  const start = text.search(/\n\n\{ "topic":/)
  if (start === -1) return text
  const rest = text.slice(start + 2)
  const next = rest.match(/\n\nCompressible ranges |\n\n\[No specific/)
  const end = next !== null ? start + 2 + next.index! : text.length
  const example = style === 'ptc'
    ? '\n\nawait tools.compress({ content: [{ startSeq, endSeq, summary }] }) from inside `run_code` — use the seqs from the range table above.'
    : '\n\ncompress({ content: [{ startSeq, endSeq, summary }] }) — use the seqs from the range table above.'
  return text.slice(0, start) + example + text.slice(end)
}

/**
 * Template rendering path (used only when a host overrides a `prompts.nudge`
 * slot). Kept byte-compatible with the pre-refactor assembly: frame → breakdown
 * → growth → guidance → tier(+rules)/range table → tip.
 */
function renderNudgeFromTemplates(
  nudge: NudgeDecision,
  emergency: boolean,
  session: import('@deepseek-ai/dsh-session').Session,
  kernelView: KernelRangeView,
  prompts: ResolvedPrompts,
  mediaPriceOf?: MediaPriceOf,
  style: ResolvedCallStyle = 'native',
  usage?: ProviderUsage,
  foldPricing?: FoldPricing,
): string {
  // Cap the reported percentage at 100: a broken measurement (e.g. response
  // pressure folded in) must never surface as an absurd "230%" to the model.
  const pct = Math.round(Math.min(nudge.contextUsage, 1) * 100)
  const frame = renderTemplate(
    emergency ? prompts.nudge.emergency : prompts.nudge.normal,
    { pct, philosophy: COMPRESS_PHILOSOPHY },
  )
  const parts: string[] = [frame]

  // Context breakdown (kernel style, from NudgeDecision.contextBreakdown).
  if (nudge.contextBreakdown) {
    const bd = nudge.contextBreakdown
    const breakdown = renderTemplate(prompts.nudge.breakdown, {
      system: Math.round(bd.system / 1000),
      tool: Math.round(bd.tool / 1000),
      summaries: Math.round(bd.summaries / 1000),
      code: Math.round(bd.code / 1000),
      text: Math.round(bd.text / 1000),
    })
    if (breakdown !== '') parts.push('', breakdown)
    if (bd.growth > 0) {
      const growth = renderTemplate(prompts.nudge.growth, { growth: Math.round(bd.growth / 1000) })
      if (growth !== '') parts.push(growth)
    }
  }

  // HOW_TO_COMPRESS_RULES as guidance (kernel puts it in every nudge).
  if (prompts.nudge.guidance !== '') parts.push('', prompts.nudge.guidance)

  // Tier line (distillation / condensation suggestion) + tier-specific rules.
  if ((nudge.tier === 2 || nudge.tier === 3) && (nudge.tierTargetBlocks?.length ?? 0) > 0) {
    const targets = nudge.tierTargetBlocks!
    const summarySeqs = targets
      .map((block) => summarySeqOfKernelBlock(session, block.blockId))
      .filter((seq): seq is number => seq != null)
      .sort((a, b) => a - b)
    const pending = nudge.tier === 2 ? nudge.breakdown?.pendingT2 : nudge.breakdown?.pendingT3
    const tokens = typeof pending === 'number' ? pending : 0
    const tierLine = renderTemplate(prompts.nudge.tier, {
      tier: nudge.tier,
      count: targets.length,
      prevTier: nudge.tier - 1,
      tokens,
      seqs: summarySeqs.join(', '),
      firstSeq: summarySeqs[0] ?? 'n/a',
      lastSeq: summarySeqs[summarySeqs.length - 1] ?? 'n/a',
    })
    if (tierLine !== '') parts.push(tierLine)
    // Tier-specific rules from kernel (TIER2_DISTILL_RULES / TIER3_CONDENSE_RULES).
    const tierRules = nudge.tier === 2 ? TIER2_DISTILL_RULES : TIER3_CONDENSE_RULES
    parts.push('', tierRules)
  } else {
    // Range table for non-tier nudges (DSH-specific: seq-based, not ref-ID-based).
    const cacheCostText = cacheCostLine(prompts, session, usage)
    if (cacheCostText !== '') parts.push(cacheCostText)
    parts.push(rangeTable(session, kernelView, prompts, mediaPriceOf, foldPricing))
  }

  // Batch-compress tip (from kernel's nudge-text.ts style). M-3 (2026-11-25): the
  // standalone Note segment is GONE — the tip itself carries the two-shapes contract
  // (condense picks / content names, content must be an ARRAY), so the tail is ONE
  // line instead of two overlapping ones. A host that overrides only the tip now
  // owns the contract line wholesale (same hole B6 closed for guidance).
  if (prompts.nudge.tip !== '') parts.push('', prompts.nudge.tip)

  // B6：模板路径同样摘掉哲学/规则段——否则宿主只要覆盖任一 nudge 槽位（如只改 tip），
  // 整份 6 KB 指引就会重新贴回来（独立复核 2026-09-08 发现的软缺口）。
  return ptcText(stripNudgeGuidance(parts.join('\n')), style)
}
