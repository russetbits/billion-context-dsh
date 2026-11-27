export const statusParameters = {
  detail: {
    type: 'string' as const,
    enum: ['overview', 'compressed', 'visible', 'messages'] as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['acpStatus.detail'],
  },
  tool: {
    type: 'string' as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['acpStatus.tool'],
  },
  sort: {
    type: 'string' as const,
    enum: ['size', 'time', 'tool', 'age'] as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['acpStatus.sort'],
  },
  limit: {
    type: 'integer' as const,
    description: DEFAULT_TOOL_PARAM_TEXTS['acpStatus.limit'],
  },
}

import { buildStatusReport, defaultCountTokens, type CompressionCore } from './kernel/index.ts'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import { kernelConfigFor, foldPricingFor } from './config.ts'
import { resolveTokenCount } from './nudge.ts'
import { routeFor, type AcpWindow } from './window.ts'
import { attachmentsOfEvent } from './messages.ts'
import { allLogMessages, buildToolCallIndex, eventsToCoreMessages, isCheckpointNode, isRealUserTurn, surfaceEventsOf } from './messages.ts'
import { lastRequestInput, peakRequestInput, providerUsage, totalRequestInput } from './usage.ts'
import { blockRegistry, rebuildBlockLedger, MIN_CONDENSE_SPAN_TOKENS, surfaceSummary } from './region.ts'
import { sessionEventsOf } from './session-events.ts'
import { DEFAULT_TOOL_PARAM_TEXTS } from './prompts.ts'
import { callHint, type ResolvedCallStyle } from './ptc.ts'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import { ToolEnvironment, TextOutput, requireAgent, resolveEffectiveWindow } from './shared.ts'
import { condenseFloorRefusal } from './condense-note.ts'
import { unwrapEnvelope } from './compress-args.ts'

export interface StatusArgs {
  /** One-word drilldown selector — the only form exposed to the model. */
  detail?: 'overview' | 'compressed' | 'visible' | 'messages'
  /** INTERNAL: kernel `buildStatusReport` speaks scope/view; never in the schema. */
  scope?: 'compressed' | 'uncompressed'
  view?: 'ranges' | 'messages'
  tool?: string
  sort?: 'size' | 'time' | 'tool' | 'age'
  limit?: number
}

/**
 * A-5（2026-11-23）：把 `detail` 展开成 kernel 的 scope/view 组合。
 *
 * drift 的根源是「两个参数要配对」：kernel 的 `view`/`tool`/`sort` 只在
 * `scope:"uncompressed"` 下有意义，模型给错组合时 kernel 静默忽略 —— 于是它
 * 以为查了、其实拿到的是同一份 overview。一个词选一种视图，把配对从模型手里
 * 拿走：模型面只有 `detail`，`scope`/`view` 退回纯内部词汇（schema 里不声明，
 * commit 8ad126b）；未给 `detail` 的调用原样透传给 kernel（内部/脚本调用面）。
 */
function applyDetailAlias(args: StatusArgs): StatusArgs {
  const { detail, ...rest } = args
  if (detail === undefined) return args
  switch (detail) {
    case 'compressed': return { ...rest, scope: 'compressed' }
    case 'visible': return { ...rest, scope: 'uncompressed', view: rest.view ?? 'ranges' }
    case 'messages': return { ...rest, scope: 'uncompressed', view: rest.view ?? 'messages' }
    default: return rest // 'overview' — the bare report
  }
}


export async function handleStatus(env: ToolEnvironment, rawArgs: StatusArgs, exec: ToolRunContext): Promise<TextOutput> {
  // The model channel may wrap ANY tool's args under `{ arguments: {…} }`;
  // peel it or drilldown params never reach buildStatusReport (live-verified
  // `{"arguments":{"scope":"compressed"}}` silently rendered the overview).
  const args = applyDetailAlias(unwrapEnvelope<StatusArgs>(rawArgs))
  const agent = requireAgent(exec)
  const session = agent.session
  const state = env.store.stateFor(session)
  const surface = surfaceEventsOf(session)
  // One tool-call index for both projections below (P2-5): tool/result
  // toolName/toolCallId are backfilled from the assistant tool-calls.
  const toolNames = buildToolCallIndex(surface)
  const coreMessages = allLogMessages(session)
  const surfaceMessages = eventsToCoreMessages(surface, toolNames)
  const tokenCount = resolveTokenCount(agent, surfaceMessages)
  const window = await resolveEffectiveWindow(env, agent)
  const config = kernelConfigFor({ ...env, modelContextLimit: window.limit })
  // Run the same pipeline the context transform runs, so what acp_status
  // reports matches what the model actually receives. The returned turn.state
  // carries the freshly assigned refs; it is NOT persisted — acp_status is a
  // read-only view, and env.store.set would advance the nudge baseline a
  // second time in the same turn (design §6.1 P2-2).
  const turn = env.kernel.processTurn({ messages: coreMessages, state, config, tokenCount })
  // Status messages = visible surface EXCLUDING checkpoint summary nodes (P1-3).
  const statusMessages = eventsToCoreMessages(
    surface.filter((event) => isCheckpointNode(event) === false),
    toolNames,
  )
  // Upstream-aligned: the kernel renders the breakdown (percentages of the
  // VISIBLE total — no window semantics; drilldown scope/view/tool/sort/limit
  // pass through verbatim); the engine only appends the nudge decision line,
  // the DSH Surface anchor, and — in drilldown mode — the mN-vs-seq note.
  const report = buildStatusReport(turn.state, statusMessages, defaultCountTokens, args)
  const lines = [report]
  // Mirror upstream pi (`if (args.scope) return base`): a drilldown request
  // answers with the kernel report alone — the nudge decision line is an
  // overview concept. The Surface anchor stays in ALL modes: it is the model's
  // compressible-ref locator (design P2-1).
  if (args.scope === undefined) {
    const nudge = turn.nudge
    if (nudge !== undefined) {
      lines.push('', `Nudge: ${nudge.shouldInject ? 'ACTIVE' : 'idle'} — ${nudge.reason}`)
    }
    // Issue #60 P2: the model's only route to T2/T3 distillation is a LIVE
    // checkpoint seq — but acp_status (kernel buildStatusReport) is blind to
    // summary nodes (they are excluded as messages, rule 9) and shows only bN
    // refs. Append an engine-side mapping bN → checkpoint seq for ACTIVE
    // blocks (only active blocks are distillable). Appending is the
    // kernel-alignment contract: the kernel owns the report text, the engine
    // owns the wiring — this row is wiring, never a rewrite of the report.
    // `active` alone is NOT "still on the surface": it means "no parent block
    // consumed it" (region.ts blockRegistry). A plain fold can swallow a summary
    // node while leaving its block active — emergency folds still do, because
    // compactForOverflow never calls liveCheckpointCarriersInSpan. On a real
    // session that drifted this way this row listed 61 blocks, 58 of them pointing
    // at seqs that no longer exist, under a label telling the model to compress
    // them. Those seqs resolve to nothing, so the distill attempt is a guaranteed
    // failure the model cannot diagnose. Only a seq that is on the surface can be
    // named as a boundary.
    const surfaceSeqs = new Set<number>(session.surface.nodes as readonly number[])
    const checkpointRows = blockRegistry(session)
      .filter((entry) => entry.active && entry.summarySeq !== null && surfaceSeqs.has(entry.summarySeq))
      .map((entry) => `${entry.kernelBlockId} → seq ${entry.summarySeq}`)
    if (checkpointRows.length > 0) {
      lines.push('', `Checkpoint seqs (active blocks — compress a checkpoint seq to distill it): ${checkpointRows.join(', ')}`)
    }
    // Two calibers meet in this report: the pressure/nudge line is
    // provider-anchored (the session projection, where images and files carry
    // the live route's price) while the breakdown above is a text-only
    // estimate. They legitimately disagree on media-heavy sessions, and a model
    // reads that disagreement as a broken number unless it is told which is
    // which (issue #117). Emitted only when the surface actually carries media,
    // so a text-only session keeps the kernel report untouched.
    const mediaOnSurface = surface.some((event) => {
      const counts = attachmentsOfEvent(event)
      return counts.images + counts.files > 0
    })
    if (mediaOnSurface) {
      lines.push(
        '',
        'Note: the pressure line is provider-anchored (images/files priced by the live route); the breakdown above is a text-only estimate. They can differ on media-heavy sessions.',
      )
    }
  }
    // Provider-anchored billing: an engine append, never a rewrite of the kernel
    // report (rule 9). The two calibers a compression decision needs: cumulative
    // uncached vs cache-read, and peak vs total request input. A replace invalidates
    // the prefix from its rewrite point, so a compress that rewrites early re-reads
    // the whole remainder on the next request — that is the cost the model cannot
    // see otherwise.
    const usage = providerUsage(agent)
    if (usage !== undefined) {
      lines.push(
        '',
        `Provider usage (cumulative): ${usage.uncachedInputTokens} uncached input | ${usage.cacheReadTokens} cache read | ${usage.cacheWriteTokens} cache write | ${usage.outputTokens} output`,
      )
      lines.push('', `Total request input: ${totalRequestInput(usage)}`)
    }
    // Live vs peak vs room — the paper’s measurement vocabulary (arXiv 2611.05306
    // §2.4 separates total length from PEAK length, and names a "compaction
    // threshold"), made actionable here: a durable replace costs a near-full
    // re-prefill of whatever stays live, so the model cannot decide whether a
    // fold pays without knowing the live size and the remaining room. Read from
    // the log, so it works without a tokenMeter. Rule 9 used to keep window
    // semantics human-side only; measured fold pricing moved them to the model.
    const live = lastRequestInput(session)
    const peak = peakRequestInput(session)
    const contextLimit = env.windowFor !== undefined ? (await env.windowFor(agent)).limit : env.modelContextLimit
    const sizeParts: string[] = []
    if (live > 0) sizeParts.push(`live ${live}`)
    if (peak > 0) sizeParts.push(`peak ${peak}`)
    // Every fold leaves a permanent un-foldable summary barrier, so fold count is a
    // resource the model spends: more barriers means smaller eligible stretches
    // later, which is how a long session ends up refusing everything (Q5). Cheap
    // because the ledger is snapshot-cached (rule 14).
    const folds = rebuildBlockLedger(sessionEventsOf(session)).length
    if (folds > 0) {
      // A spent resource: more barriers means smaller eligible stretches later,
      // which is how a long session ends up refusing everything. No unlock number is
      // printed here — the round-4 reviews showed a fold count never measured the
      // debt it claimed (a barrier is a note, not a floor-sized hole), and whether
      // the engine may fold below the floor is read from the live rows instead.
      lines.push('', `Fold barriers: ${folds} (each splits the surface and can never be folded again) — fewer, larger folds preserve room for later folds`)
      if (folds >= 2) {
        // The fragmentation failure mode, stated where the model looks BEFORE calling: a
        // surface of many sub-minimum gaps has no plain fold left, and the only move
        // that still works is an EXPLICIT tier-2 distill — a plain condense refuses to
        // fold across carriers. Kept on its own line so it is never read as a range row.
        lines.push(`  obstacle: when every foldable gap is under the ${MIN_CONDENSE_SPAN_TOKENS}t a normal fold needs (the nudge table prints each row's size), distill the carriers explicitly instead — compress({ content: [{ startSeq: 'b1', endSeq: 'bN', summary }] }) over the carrier seqs acp_status's Checkpoint seqs row shows; the block-id boundaries are the tier-2 distill path, and the originals behind those carriers stay recall-readable (recall({ blockId }) pages them back).`)
      }
    }
    if (contextLimit > 0 && live > 0) sizeParts.push(`window ${contextLimit}`, `room ${Math.max(0, contextLimit - live)}`)
    // G1 口径标注（M-3）：live/peak/room 都是本引擎 tokenCount 的估算值；provider 侧的
    // window 压力可能与它有出入（tokenizer 差异、媒体计价），计费请求头才是权威。
    if (sizeParts.length > 0) lines.push('', `Request input sizes: ${sizeParts.join(' | ')} (estimate — provider-side window pressure may differ; the request header is authoritative)`)
    // The working-set floor is what makes a fold REFUSE, but it was only visible inside a nudge —
    // and a nudge only fires above the pressure threshold, so on a mid-pressure session the model
    // sees neither the range table nor the floor and calls `condense` blind (round-15 review Q5).
    // Printed from the SAME `foldPricingFor` the gate uses, so the number the model reads is the
    // number the gate applies. The top candidate is deliberately NOT printed: computing it runs
    // `buildCompressibleSeqRanges`, whose orphan strip APPENDS durable prune events, and this tool
    // is read-only — a status call must not change the surface it is reporting on.
    const statusPricing = foldPricingFor({ ...env, modelContextLimit: contextLimit })
    const floorTokens = statusPricing.floorTokens
    // M-3 (2026-11-25): payback is REPORTED, not gated — one definitional sentence,
    // no ceilings (they were removed from the election).
    lines.push('', `Fold payback: a fold's re-prefill is paid ONCE and its saving applies to every later request, so payback = re-prefill / saving in REQUESTS (reported for each range row; the condense election is gated by the working-set floor alone).`)
    if (floorTokens !== undefined && floorTokens > 0) {
      const close = live > 0 && live < floorTokens * 2
      // The direction matters: a fold leaves EXACTLY the room it does NOT remove, so near the
      // floor a WIDER span is the one that strands the working set — the same reasoning
      // `condenseFloorRefusal` carries (round-16 review Q1: this line told the model to widen
      // into the refusal the guard then produced).
      lines.push('', `Fold floor: a fold must leave at least ~${floorTokens} live tokens, below which compress and condense both refuse${live > 0 ? ` (live is ${live}${close ? ' — close to the floor: a fold leaves exactly the room it does NOT remove, so the SMALLER span is the one that stays above the floor; better still, stop compressing' : ''})` : ''}`)
    }
  // textOverrides self-check: a composition-row typo has no other visible
  // surface, so acp_status is where it surfaces. One line per diagnostic, and
  // nothing at all when the config is clean (no extra bytes on every request).
  if (env.textWarnings !== undefined && env.textWarnings.length > 0) {
    lines.push('', ...env.textWarnings.map((line) => `Text overrides: ${line}`))
  }
  lines.push('', `Surface: ${surfaceSummary(session)}`)
  // Drilldown rows carry kernel refs (mN, dense log-order ids) — compress
  // accepts them directly (handleCompress reverse-maps mN → live surface seq
  // via the current turn's messageRefs.byRef; issue #31). The Surface anchor
  // remains the model's compressible-seq locator for nudge-style ranges.
  if (args.scope === 'uncompressed') {
    lines.push('', 'Note: drilldown rows are kernel refs (mN) — feed them straight to compress (auto-mapped to the live surface seq); an unknown mN fails with guidance.')
  }
  return { text: lines.join('\n') }
}
