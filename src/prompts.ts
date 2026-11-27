/**
 * M4 — configurable prompt templates: the per-stage model-visible texts
 * (nudge frames, range table, system prompt, tool descriptions) rendered from
 * `config.prompts` templates with named placeholders.
 *
 * Design: docs/configurable-prompts-design.md (v4).
 * - placeholders are `{identifier}` only; literal braces like
 *   `compress({ content: [...] })` are left untouched (spaces/commas break the
 *   identifier rule);
 * - resolvePrompts merges user overrides over DEFAULT_PROMPTS per key
 *   (null/undefined → default, string → override; group-level null → whole
 *   group default for YAML hosts) and validates unknown placeholders at
 *   construction time (fail-fast, no silent typos);
 * - renderTemplate throws when a known placeholder has no value — callers
 *   must provide every value (e.g. tokens via a typeof fallback).
 * @module billion-context-dsh/prompts
 */

import { COMPRESS_PHILOSOPHY, HOW_TO_COMPRESS_RULES, TIER2_DISTILL_RULES, TIER3_CONDENSE_RULES } from './kernel/index.ts'

/** 用户可写值:字符串模板,或 null(= 用默认,等价于不写)。YAML 宿主写 null 是合法输入。 */
export type PromptInput = string | null

/** 按组生成"每键可选、可 null"的覆盖类型。 */
export type PromptOverride<T> = { [K in keyof T]?: PromptInput }

export interface NudgePrompts {
  /** 普通档首句。占位符:{pct}（`{philosophy}` 仍被校验器接受,但 B6 会把渲染出的哲学段摘掉——它只能经系统提示到达模型) */
  normal: string
  /** 紧急档首句。占位符:{pct}（`{philosophy}` 同上) */
  emergency: string
  /** 指导行（HOW_TO_COMPRESS_RULES）。无占位符 */
  guidance: string
  /** tier 蒸馏行。占位符:{tier} {count} {prevTier} {tokens} {seqs} {firstSeq} {lastSeq} */
  tier: string
  /** 上下文分解。占位符:{system} {tool} {summaries} {code} {text} */
  breakdown: string
  /** 增长行。占位符:{growth} */
  growth: string
  /** 溢出提示。无占位符 */
  tip: string
  /** 压缩后的前缀重读代价。占位符:{uncached} {count} */
  cacheCost: string
}

export interface RangeTablePrompts {
  /** 表头。占位符:{surface} */
  header: string
  /** 标题。占位符:{count}(表格行数) */
  title: string
  /** 每行。占位符:{start} {end} {count} {tokens} {toolPct} {textPct} {media} {foldCost} {payback} {tools}
   *  —— `{payback}` 是回本周期（foldCost/tokens，形如 `1.0` 或 `—`），`{tools}` 渲染成
   *  ` · tools: bash×12, read×5`（空则整段消失，见 nudge.ts） */
  line: string
  /**
   * 批量口径汇总行（A-9，2026-11-23）。占位符:{rows} {reclaim} {fold} {breakEven}。
   * 单行 fold 列只讲“这一段自己折要重读多少”，量级常远大于回收量，模型容易据此
   * 判定「压了更亏」；正确结论是相反的——一次 re-prefill 的价钱与折几段无关，所以
   * 该一次折完。本行把并集收益、共用的一次性成本与回本次数直接算给模型。
   */
  plan: string
  /** 表尾调用语法。无占位符 */
  footer: string
}

export interface ToolPrompts {
  /** 工具描述(纯文本,无占位符) */
  compress: string
  recall: string
  acpStatus: string
  /**
   * `compress` 的 `condense` 参数描述(纯文本,无占位符)。
   * 只开放这一条参数文案:`condense` 是 T2/T3 蒸馏的唯一入口,模型靠它理解
   * 「一段替换成一条 note」的语义。note 骨架(`## Task`/`## Now`/…)是校验契约,
   * 不在任何覆盖清单里(见 tools.ts 的 `MAX_HANDOFF_NOTE_TOKENS` 与骨架校验)。
   */
  condense: string
}

export interface AcpPrompts {
  readonly nudge?: PromptOverride<NudgePrompts>
  readonly rangeTable?: PromptOverride<RangeTablePrompts>
  readonly tools?: PromptOverride<ToolPrompts>
  /** 整段 system prompt 模板;`{philosophy}` 引用 kernel 的 COMPRESS_PHILOSOPHY */
  readonly systemPrompt?: PromptInput
}

/** 解析结果 —— 所有字段已填满(纯 string,无 null)、已校验。构造一次,全程复用。 */
export interface ResolvedPrompts {
  readonly nudge: NudgePrompts
  readonly rangeTable: RangeTablePrompts
  readonly tools: ToolPrompts
  /** 参数级 description 文案(键 `<tool>.<arg>`)。只由 `textOverrides` 覆盖,不在 `config.prompts` 结构化层里。 */
  readonly toolParamTexts: Readonly<Record<string, string>>
  /** 注意:这是【模板】(含 {philosophy}),不是渲染结果。渲染用 renderSystemPrompt。 */
  readonly systemPromptTemplate: string
}

/** 每槽允许的占位符名集合(构建期校验用)。 */
const NUDGE_ALLOWED: { [K in keyof NudgePrompts]: ReadonlySet<string> } = {
  normal: new Set(['pct', 'philosophy']),
  emergency: new Set(['pct', 'philosophy']),
  guidance: new Set(),
  tier: new Set(['tier', 'count', 'prevTier', 'tokens', 'seqs', 'firstSeq', 'lastSeq']),
  breakdown: new Set(['system', 'tool', 'summaries', 'code', 'text']),
  growth: new Set(['growth']),
  tip: new Set(),
  cacheCost: new Set(['uncached', 'count']),
}
const RANGE_TABLE_ALLOWED: { [K in keyof RangeTablePrompts]: ReadonlySet<string> } = {
  header: new Set(['surface']),
  title: new Set(['count']),
  line: new Set(['start', 'end', 'count', 'tokens', 'toolPct', 'textPct', 'media', 'foldCost', 'payback', 'tools']),
  plan: new Set(['rows', 'reclaim', 'fold', 'payback']),
  footer: new Set(),
}
const TOOLS_ALLOWED: { [K in keyof ToolPrompts]: ReadonlySet<string> } = {
  compress: new Set(),
  recall: new Set(),
  acpStatus: new Set(),
  condense: new Set(),
}
const SYSTEM_ALLOWED = new Set(['philosophy', 'howToCompressRules', 'tier2DistillRules', 'tier3CondenseRules'])

/** 校验单个模板:未知 `{ident}` → throw(带槽位路径)。默认模板开发期已核验,不重扫。 */
function validateTemplate(template: string, allowed: ReadonlySet<string>, path: string): string {
  const re = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g
  let match: RegExpExecArray | null
  while ((match = re.exec(template)) !== null) {
    const name = match[1]!
    if (!allowed.has(name)) {
      throw new Error(
        `${path} contains unknown placeholder {${name}} — allowed: ${[...allowed].join(', ') || '(none)'}`,
      )
    }
  }
  return template
}

/**
 * 纯替换。两个契约:
 * 1. 未知占位符不可能到达这里(构建期已校验);
 * 2. 已知占位符缺值 = 编程错误 → throw(绝不静默渲染空串)。
 */
export function renderTemplate(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_match, name: string) => {
    const value = vars[name]
    if (value === undefined) {
      throw new Error(
        `renderTemplate: missing value for placeholder {${name}} in template "${template.slice(0, 60)}…"`,
      )
    }
    return String(value)
  })
}

/**
 * 逐键合并:null / undefined → 默认;字符串 → 覆盖默认(不用 spread,
 * 否则 null 会覆盖默认,与"null = 用默认"矛盾)。组级 null/undefined →
 * 整组用默认(YAML 宿主可能写 `{ nudge: null }`,W3)。
 */
function mergeGroup<T extends Record<keyof T, string>>(
  defaults: T,
  override: PromptOverride<T> | null | undefined,
  allowed: { [K in keyof T]: ReadonlySet<string> },
  path: string,
): T {
  if (override == null) return defaults
  const out = {} as { [K in keyof T]: string }
  for (const key of Object.keys(defaults) as Array<keyof T>) {
    const value = override[key]
    out[key] = value === null || value === undefined
      ? defaults[key]
      : validateTemplate(value, allowed[key], `${path}.${String(key)}`)
  }
  return out as T
}

/**
 * 深合并 + 校验;引擎构造期调用一次,出错即抛(fail-fast)。
 * 未传入时返回 DEFAULT_RESOLVED,零校验重跑。
 */
export function resolvePrompts(input?: AcpPrompts): ResolvedPrompts {
  if (input === undefined) return DEFAULT_RESOLVED
  return {
    nudge: mergeGroup(DEFAULT_PROMPTS.nudge, input.nudge, NUDGE_ALLOWED, 'prompts.nudge'),
    rangeTable: mergeGroup(DEFAULT_PROMPTS.rangeTable, input.rangeTable, RANGE_TABLE_ALLOWED, 'prompts.rangeTable'),
    tools: mergeGroup(DEFAULT_PROMPTS.tools, input.tools, TOOLS_ALLOWED, 'prompts.tools'),
    // 参数级文案不走结构化模板层:它没有占位符,也没有「按组 null 回退」的语义,
    // 只被 textOverrides 整体替换(applyTextOverrides)。
    toolParamTexts: DEFAULT_PROMPTS.toolParamTexts,
    systemPromptTemplate:
      input.systemPrompt === null || input.systemPrompt === undefined
        ? DEFAULT_PROMPTS.systemPromptTemplate
        : validateTemplate(input.systemPrompt, SYSTEM_ALLOWED, 'prompts.systemPrompt'),
  }
}

/** 渲染 system prompt 模板(注入 kernel 压缩哲学、压缩规则、蒸馏规则)。 */
export function renderSystemPrompt(prompts: ResolvedPrompts): string {
  return renderTemplate(prompts.systemPromptTemplate, {
    philosophy: COMPRESS_PHILOSOPHY,
    howToCompressRules: HOW_TO_COMPRESS_RULES,
    tier2DistillRules: TIER2_DISTILL_RULES,
    tier3CondenseRules: TIER3_CONDENSE_RULES,
  })
}

/**
 * 每个**参数级** `description` 的默认文本 —— 工具描述讲「这个工具怎么用」,
 * 这里讲「这一格填什么」。键是 `<tool>.<arg>`(如 `compress.force`),可覆盖 id 为
 * `tool.param.<键>`;未覆盖时保持英文默认,与工具级文案同一套替换语义。
 *
 * **`compress.condense.summary` 不在此表**: 它是 note 骨架(`## Task`/`## Now`/…),
 * 是校验契约而不是文案,覆盖它会让「缺 heading 也照常落地」的约定失效。
 */
export const DEFAULT_TOOL_PARAM_TEXTS: Readonly<Record<string, string>> = {
  'compress.arguments': "Tolerated wrapped-arguments form (model-generated); unwrapped in handleCompress. Prefer passing content directly.",
  'compress.topic': "Fallback topic for entries without their own.",
  'compress.content': "One or more ranges to compress, each with startSeq/endSeq boundaries (surface seqs) and a dense summary. Required — pass it directly, not wrapped in an arguments key.",
  'compress.startSeq': "First surface seq of the range.",
  'compress.startSeq.text': "Seq as text; a trailing #callId fragment is ignored. A drilldown mN ref (\"m00306\") or a kernel block id (\"b1\", as acp_status lists) is also accepted — a block id resolves to that block's checkpoint seq, which makes the fold a tier 2/3 distillation.",
  'compress.endSeq': "Inclusive last surface seq of the range.",
  'compress.endSeq.text': "Seq as text; a trailing #callId fragment is ignored. Also accepts a drilldown mN ref or a kernel block id (\"b3\") — for a multi-block condense name the LAST block id (tier 3).",
  'compress.summary': "Complete technical summary replacing the range; keep paths, decisions, values verbatim. Minimum 50 characters.",
  'compress.item.topic': "Short label (3-5 words) for this range.",
  'compress.verifiedReadings': "The moment a verification comes back green AND a later step will depend on that result, pass the readings here (e.g. \"closedloop 414/414\", \"t4-write-epoch 6/6\"). They are stored structurally on the block and shown by acp_status and recall({blockId}), so a resumed session does not re-run the suite. Do not pass readings you have not actually run.",
  'compress.force': "Override the working-set floor guard on BOTH paths — an explicit content range whose fold leaves less than foldFloorPct of the window visible, and a condense election. Legitimate only for imminent overflow: a shallow working set is where open-weight accuracy collapses (arXiv 2611.05306 Appendix D).",
  'compress.condense.topic': "Short label (3-5 words) for the folded block.",
  'compress.condense.force': "Fold even though it leaves less than the working-set floor visible (foldFloorPct of the window). Only for imminent overflow: a shallow working set is where open-weight models lose accuracy.",
  'compress.condense.verifiedReadings': "Readings that came back green and a later step will rely on (e.g. \"closedloop 414/414\"). Stored on the block so a resumed session does not re-run them.",
  'recall.query': "SEARCH form: terms to look up across the compressed-block summaries, the shadowed originals AND the still-visible surface. Each hit names its ref (block bN, \"message seq N … still visible\", or \"in block X\") — read it back with blockId/seq. Use this OR blockId/seq, never both.",
  'recall.blockId': "READ form: dump a compressed block — the kernel block ref `bN` shown by acp_status (e.g. b1), or a compaction id / prefix from a search hit. Large blocks page: pass offset/limit and follow the continue hint. Use this OR seq, and NOT with query.",
  'recall.seq': "READ form: one message by surface seq instead of a whole block — the seqs a search reports (visible or folded) and any seq in this session log. Works after a fold too, because the log keeps the original bytes. Use this OR blockId, and NOT with query.",
  'recall.chars': "With `seq`: maximum characters to return (default 20000, capped at 100000). Long tool output is truncated with a marker, so raise it only when the answer is past the cut.",
  'recall.maxResults': "With `query`: how many hits to list (default 5).",
  'recall.offset': "With `blockId`: start position in the block's message list (default 0). Blocks are paged by size — each page stays under the host tool-result trim budget (up to 100 messages) — so follow the continue hint in the result to walk the rest.",
  'recall.pageLimit': "With `blockId`: messages per page (default 100; values above 100 are capped to 100). Pages are also bounded by a character budget, so long messages return fewer than this per call.",
  'acpStatus.detail': "What to show: \"overview\" (default — breakdown, blocks, fold floor, sizes), \"compressed\" (the block list you can recall/distill), \"visible\" (the surface grouped into foldable ranges), \"messages\" (every visible message). Pair it with tool/sort/limit.",
  'acpStatus.tool': "Filter drilldown rows to one tool name (detail:\"messages\" only).",
  'acpStatus.sort': "Row order: size (default, most tokens first), time, tool; \"age\" applies to compressed blocks.",
  'acpStatus.limit': "Cap on rows or blocks shown (default 30).",
}

/**
 * 默认模板 —— 与 v4 之前的硬编码文案逐字节一致
 * (回归锚点见 tests/prompts.test.ts 的硬编码字面量快照)。
 */
export const DEFAULT_PROMPTS: ResolvedPrompts = {
  nudge: {
    // 与 kernel nudge-text.ts EFFICIENCY_NOTE 逐字对齐——不含 "Context usage is at X%"
    // 陈述(usage 只通过 breakdown 传达);{pct} 仍可用作自定义占位符。
    // B6（2026-09-08）：正文 ≤300 B——philosophy 段移出 nudge（已住系统提示与工具描述），
    // 每拍复读同一份 6 KB 文本=重复计费。
    normal: 'Efficiency nudge: compress consumed ranges early to keep context lean — not an overflow warning. A stronger alert appears only if the context is actually full.',
    emergency: '⚠️ Context limit reached — compress now. Prioritize consumed tool outputs.',
    guidance: HOW_TO_COMPRESS_RULES,
    tier: 'Tier {tier}: {count} tier-{prevTier} block(s) distillable ({tokens} tokens) — distill them by compressing their checkpoint seq(s) [seqs {seqs}] as one range: compress({ content: [{ startSeq: {firstSeq}, endSeq: {lastSeq}, summary }] }).',
    breakdown: 'Context breakdown: {system}K system | {tool}K tool | {summaries}K summaries | {code}K code | {text}K text',
    growth: '+{growth}K since last nudge',
    tip: '💡 One call is enough: compress({ condense: { summary } }) lets the engine pick the span (folds the OLDEST consumed PREFIX, every fold barrier inside it merges into ONE block), and compress({ content: [{ startSeq, endSeq, summary }, …] }) batches explicit ranges — content must be an ARRAY (a plain-string content is rejected).',
    cacheCost: 'Since last compress: +{uncached} uncached input tokens billed ({count} compress(es) so far) — a replace re-prefills everything the prompt carries AFTER the rewrite point ONCE, so fold rarely, in bulk, and no earlier than the consumed span forces.',
  },
  rangeTable: {
    header: 'Surface: {surface}',
    title: 'Ranges ({count}, cheapest rewrite point first — condense always folds the OLDEST eligible prefix, so write your note for that stretch):',
    line: '  - seq {start}..{end} \u00b7 {count}msg ~{tokens}t tool{toolPct}%{media} \u00b7 fold ~{foldCost} \u00b7 payback {payback}{tools}',
    plan: 'Best batch here: {rows} row(s) in ONE call → ~{reclaim}t less on every later request, one re-prefill of ~{fold}t — pays back in {payback} req. The fold column prices each row ALONE; the batch pays one re-prefill for all of them.',
    footer: 'fold = re-prefilling every token AFTER the rewrite point, paid ONCE; batch rows into ONE call. DEFAULT MOVE: compress({ condense: { summary } }) — the engine folds the oldest eligible prefix and merges every barrier inside it into ONE block, so you never hand-assemble boundaries.\nUse compress({ content: [{ startSeq, endSeq, summary }] }) only for an exact span or several disjoint ones; stale seqs mean re-run acp_status. A row marked N spilled has its bytes on a FILE — folding it loses nothing; a row marked below floor strands the working set — fold a row that STILL LEAVES the floor. Only fold spans you have fully consumed: carry their open TODOs and artifact paths into the summary, or leave them live.',
  },
  tools: {
    compress: 'Replace older conversation ranges with dense summaries you write. Each message seq is a surface reference.\n'
      + 'display: condense (DEFAULT \u2014 engine elects the cheapest eligible shape, possibly a multi-span batch, you name no boundary) | content (EXPLICIT \u2014 you name the spans)\n'
      + 'condense: compress({ condense: { summary } }) folds the cheapest eligible shape of the CURRENT surface (possibly a multi-span batch across walls) in ONE replace; the note is a brief for the work AHEAD (NVIDIA App. A four-field shape, 150-400 tokens — not a narrative of what happened):\n'
      + '  ## Task (verbatim/near-verbatim copy of the task this stretch served — REQUIRED) ## Now (current work + intent + next step, done when <criterion> — REQUIRED) ## Plan (decomposition + completion criteria) ## Pointers (artifact paths/seq/blockId — keep credentials and exact values verbatim)\n'
      + '  judgment: fold what is consumed, keep what is still in use; never summarize narrative process\n'
      + 'content: compress({ content: [{ startSeq, endSeq, summary, topic?, verifiedReadings? }] }) \u2014 batch disjoint ranges in ONE call (each entry its own block; one re-prefill for the batch); boundaries = surface seqs (acp_status or nudge table), bN, or mN\n'
      + 'verifiedReadings: readings that came back green and a later step depends on (e.g. ["closedloop 414/414"]) \u2014 stored on the block, so a resumed session does not re-run the suite\n'
      + 'never: content the current step is still using\n'
      + 'moments: stage or subtask completion | strategy switch | intermediate milestone | winding up failed exploration\n'
      + 'consumed: only fold a moment whose details you have fully consumed and will not re-check\n'
      + 're-verify: before folding, ask \u2014 will I need any detail from this range again in this task? If yes, keep it live\n'
      + 'dead-end: turn dead-end exploration into a conclusion (what was tried, why it failed, the next step) \u2014 never a blow-by-blow\n'
      + 'only-record: the summary is the ONLY record a later reader has \u2014 self-contained, so a reader (or you, after a recall) can continue without the original; keep paths and values verbatim\n'
      + 'cost: a fold re-prefills every token AFTER the rewrite point ONCE \u2192 prefer the LATEST consumed stretch, batch what you would fold anyway, do not fold small spans repeatedly\n'
      + 'reject: a rejected call names exactly what to change \u2014 read it instead of guessing.',
    recall: 'Recall content the visible context no longer carries — ONE tool, two mutually exclusive forms. '
      + 'SEARCH: recall({ query: "CONTEXT_WINDOW_EXCEEDED" }) scores the compressed-block summaries, the folded originals AND the still-visible surface, and prints one ref per hit (block bN, a still-visible seq, or the block that owns a folded seq) — when a summary lacks a detail you need (exact values, error strings, decisions, verbatim code), SEARCH the compressed blocks FIRST — never guess or reconstruct from memory, then read only that block to recover the original. '
      + 'READ: recall({ blockId: "b1" }) dumps a block\'s original messages read-only (the bN ref acp_status lists, or a compaction id / prefix from a search hit; large blocks page — pass offset/pageLimit and follow the continue hint, and locate the page you need with a query first), and recall({ seq: 42 }) returns ONE event\'s original text (pass chars to read past the 20000-char cut; works after a fold because the log keeps the bytes). '
      + 'One call is one form: query + blockId/seq in the same call is refused. Examples: recall({ query: "AUTH_TOKEN_TTL" }) · recall({ blockId: "b1", offset: 0, pageLimit: 50 }).',
    acpStatus: 'Context status: overview of the current context — CONTEXT BREAKDOWN (tool/text/summaries token shares of the visible total), COMPRESSED BLOCKS ledger, and the nudge decision. No args = overview. Percentages are shares of the visible content, not the context window. Note: the block refs in COMPRESSED BLOCKS (bN, e.g. b1) work in BOTH tools — recall({ blockId: "bN" }) reads the block, and compress accepts bN as a boundary to DISTILL that block (tier 2/3); plain spans use the Surface: seq range. Drilldown: detail:"compressed" for the per-block list, detail:"visible" for the merged foldable ranges, detail:"messages" for every visible message; no detail = overview. tool narrows to one tool name (detail:"messages" only), sort reorders (size/time/tool; age for compressed), limit caps rows (default 30). Drilldown row refs are kernel ids (mN) — feed them straight to compress as startSeq/endSeq (auto-mapped to the live surface seq); bN also works as a compress boundary (it distills that block), Surface: seqs cover plain spans.',
    condense: 'Engine-picked fold: from EVERY foldable shape of the CURRENT surface (prefix, a cut inside the prefix\'s first range, a single range, a multi-span batch that folds every wall-free run in ONE rewrite) the engine elects the cheapest eligible shape (possibly a multi-span batch) whose folded span reaches the note-carrying minimum and whose remainder stays above the working-set floor (walls it may never cross: the newest injected instruction copy of every scope, the newest skill catalog, the active user turn) — and folds it in ONE durable replace — you write the handoff note, and the result names the spans it actually folded. Mutually exclusive with content. The note is a BRIEF FOR THE WORK AHEAD, not a narrative of what happened (NVIDIA App. A): "## Task" (verbatim/near-verbatim copy of the task) and "## Now" (current work + intent + next step) are required, "## Plan" (decomposition + completion criteria) and "## Pointers" (artifact paths, credentials/values verbatim) are the advisory rest — and keep it SHORT, 150-400 tokens: the note stays resident in every later request. A note without the required fields turns one fold into a re-run of the work it replaced. Price: the next request re-prefills every token AFTER the rewrite point ONCE, so the rewrite point IS the lever — a fold near the tail barely re-reads anything, one near the head re-reads almost everything. Prefer the LATEST consumed stretch, fold in bulk, and do not fold at all when nothing consumed is worth reclaiming.',
  },
  toolParamTexts: DEFAULT_TOOL_PARAM_TEXTS,
  systemPromptTemplate: `Active Context Pruning — model-driven context management

YOU decide whether and when to compress context. The nudge is an efficiency notification: when you see one, consider which ranges you have genuinely consumed and could summarise to keep working context lean.

{philosophy}

WHEN TO COMPRESS:
- A sub-agent or delegated task has returned a large result that you have already extracted the key facts from.
- Verbose command output (build/test logs, git diff, directory listings) where you have already used the information you need.
- Exploration that led nowhere.
- Repeated reads of the same file or repeated status checks once the decision is recorded.
- Resolved discussion threads where a decision has been captured in summary or in code.
- Intermediate steps of a completed multi-step task, once the final result is recorded.
- A task phase has ended — bug hunt complete, root cause found, exploration done, research sprint wrapped.

WHEN NOT TO COMPRESS:
- Content the current step is actively reading or reasoning about.
- Important user messages — preserve their exact intent, constraints, and acceptance criteria.
- Protected tool outputs — hard-excluded from compression ranges, survive intact in visible context.
- Content you will still need to cite verbatim — in review/audit/verification tasks, keep source reads un-compressed until the final report is written. If you compressed it and now need the exact detail, a recall costs a full round-trip; prefer delaying the compress.

{howToCompressRules}

CONDENSE — an engine-elected fold, for a long context with many turns still ahead:
compress({ condense: { summary } }) elects the LARGEST eligible continuous run of the
CURRENT surface and folds it in ONE replace — you do not name the span, you write the
note. Walls the election never crosses: the newest injected instruction copy of every
scope, the newest skill catalog, checkpoint carriers, the system node, the active user
turn (superseded injection copies DO fold — they cannot come back). The judgment call
is ONE line: fold what is consumed, keep what is still in use — content the current
step is reading or reasoning about stays live no matter how big it is.
The note is a BRIEF FOR THE WORK AHEAD (Appendix A four-field shape), NOT a narrative
of what happened — no stage-by-stage retelling. Keep it SHORT: 150-400 tokens; it
stays resident in every later request.
- ## Task (REQUIRED): verbatim or near-verbatim copy of the task this stretch served
- ## Now (REQUIRED): current work + intent + the immediate next step, with its done-when
- ## Plan: the decomposition and each step's completion criterion (when is it DONE)
- ## Pointers: artifact paths / seqs / blockIds, credentials and exact values verbatim
A note without the required headings turns one fold into a re-run of the work it replaced.
Price: the next request re-prefills every token AFTER the
rewrite point ONCE, so WHERE the rewrite lands is the lever — a fold near the tail
barely re-reads anything, one near the head re-reads almost everything, and three
small folds pay that re-prefill three times. Prefer the LATEST consumed stretch, fold
in bulk, and not at all when nothing consumed is worth reclaiming.
Compression tools (refs are SURFACE SEQS, not ids):
- compress: replace one or more seq ranges, each with your own dense summary. Single range: compress({ content: [{ startSeq, endSeq, summary }] }). Batch multiple unrelated segments in one call (each entry becomes its own block): compress({ content: [{ startSeq: 1, endSeq: 5, summary: '...' }, { startSeq: 12, endSeq: 18, summary: '...' }] }). Keep ranges disjoint — overlapping entries in one batch are skipped. Edges are auto-balanced to tool-call/result boundaries; a trailing #callId fragment in a seq is ignored. Seq refs must be on the current surface: seqs from older nudges or earlier compresses go stale as the surface moves, so a stale span is auto-remapped to its still-live remainder (the result reports the adjusted span), a fully compressed span is reported as already compressed, and invented/other-session seqs fail with guidance. The block refs (bN, e.g. b1) in acp_status COMPRESSED BLOCKS are compress boundaries too — naming one folds the checkpoint node of that block, which is what makes the fold a tier 2/3 distillation; recall accepts the same ref.
- recall: the ONE way back into the log, two mutually exclusive forms. SEARCH — recall({ query }) scores the compressed-block summaries, the folded originals and the still-visible surface, and prints the ref per hit (block bN, a still-visible seq, or the owning block of a folded seq); SEARCH the compressed blocks FIRST — never guess or reconstruct from memory (exact values, error strings, decisions, verbatim code are what a summary drops) — then read only that block. READ — recall({ blockId }) dumps a block's originals read-only (bN ref from acp_status, or a compaction id; large blocks page by offset/pageLimit, follow the continue hint), recall({ seq }) returns ONE event's text (chars caps the length). One call is one form: query + blockId/seq is refused.
- acp_status: current context usage and the live compressible-range list. Run it right before compressing — the only seqs that never go stale are the ones you just read. Drilldown (detail/tool/sort/limit) lists per-message or per-block sizes; drilldown rows are kernel ids (mN) — compress accepts them directly (auto-mapped to the live surface seq).

Tiered compression: each compressed block appears on the surface as one summary node. Compressing that node again DISTILLS the block (tier 2): the parent summary folds into your new summary and the original messages are freed. Distilling a tier-2 block yields tier 3. Distill when a summary itself is consumed — recall on the tier-2 block recovers the full originals.

{tier2DistillRules}

{tier3CondenseRules}

When you write a summary, it becomes the ONLY record of that range: keep file paths, signatures, exact values, decisions, and error strings verbatim so a later reader (or you, after a recall) can continue without the original. A mid-task compression must also hand off STATE, not just facts: where you are in the task and what remains (open TODOs), which approaches you already ruled out (dead ends), and which intermediate artifacts are already done and where they live — so the next step continues instead of re-running work you already did. Never reuse historical seqs — the surface moves as messages land and compress; verify with acp_status.`,
}

/** 模块级默认缓存:默认参/兜底直接引用,避免每次调用重跑校验。 */
export const DEFAULT_RESOLVED: ResolvedPrompts = DEFAULT_PROMPTS

// ─────────────────────────────────────────────────────────────────────────────
// textOverrides — flat, id-addressed FULL REPLACEMENT of any model-visible text
// ─────────────────────────────────────────────────────────────────────────────

/**
 * `textOverrides` 的键是点号路径 id(`<group>.<key>`),值是**完整替换文本**。
 * 未列出的 id 保持英文默认。REPLACE, never append: 覆盖值直接取代原文案,
 * 所以 token 成本大致持平,配置不会悄悄变贵(见 src/config.ts 的注释)。
 */
export type TextOverrides = Record<string, string>

/**
 * 覆盖值的长度上限(相对该 id 默认值的倍数)。超过只 warning,不拒绝:
 * 最常见的事故是把英文翻成中文,而中文并不更省 token(见 src/config.ts)。
 */
export const MAX_TEXT_OVERRIDE_RATIO = 1.5

/**
 * Token 估算,与 `defaultCountTokens` 同口径:CJK 1 字/token,其余 4 字/token。
 * 只用于长度告警,不参与任何校验或计价。
 */
export function estimateTextTokens(text: string): number {
  let cjk = 0
  for (const ch of text) if (/\p{Script=Han}|\p{Script=Hiragana}|\p{Script=Katakana}|\p{Script=Hangul}/u.test(ch)) cjk += 1
  return cjk + Math.ceil((text.length - cjk) / 4)
}

/**
 * 每个可覆盖文案的 id(`nudge.*` / `rangeTable.*` / `tool.*` / `systemPrompt`)。
 * 由 DEFAULT_PROMPTS 的键推导,所以新增一个槽位就自动出现在这里 ——
 * `tests/text-overrides.test.ts` 用它逐 id 验证「能覆盖且生效」,防止 id 表与实际漂移。
 */
export function ableTextIds(): string[] {
  return [
    ...Object.keys(DEFAULT_PROMPTS.nudge).map((key) => `nudge.${key}`),
    ...Object.keys(DEFAULT_PROMPTS.rangeTable).map((key) => `rangeTable.${key}`),
    ...Object.keys(DEFAULT_PROMPTS.tools).map((key) => `tool.${key}`),
    ...Object.keys(DEFAULT_PROMPTS.toolParamTexts).map((key) => `tool.param.${key}`),
    'systemPrompt',
  ]
}

/** 按 id 读一组已解析文案里的文本;未知 id 返回 undefined。 */
export function textById(prompts: ResolvedPrompts, id: string): string | undefined {
  if (id === 'systemPrompt') return prompts.systemPromptTemplate
  if (id.startsWith('tool.param.')) return prompts.toolParamTexts[id.slice('tool.param.'.length)]
  const dot = id.indexOf('.')
  if (dot < 0) return undefined
  const group = id.slice(0, dot)
  const key = id.slice(dot + 1)
  const bag = group === 'nudge' ? prompts.nudge : group === 'rangeTable' ? prompts.rangeTable : group === 'tool' ? prompts.tools : undefined
  return bag === undefined ? undefined : (bag as unknown as Record<string, string>)[key]
}

/** 一条覆盖告警(拼接成一行给启动日志 / acp_status 自检)。 */
export interface TextOverrideDiagnostic {
  readonly id: string
  readonly tokens: number
  readonly defaultTokens: number
}

export interface TextOverrideReport {
  /** 覆盖后的文案(不可变;未覆盖的字段与入参同一个引用)。 */
  readonly prompts: ResolvedPrompts
  /** 配置里有、但不在 `ableTextIds()` 里的 id — 忽略,不报错。 */
  readonly unknown: string[]
  /** 比默认值长出 MAX_TEXT_OVERRIDE_RATIO 倍的覆盖 — 照常生效,只告警。 */
  readonly oversized: TextOverrideDiagnostic[]
}

/**
 * 应用扁平覆盖。**只改呈现,不参与任何校验**:note 骨架(`## Task`/`## Now`)、
 * `missingCondenseHeadings`、`MAX_HANDOFF_NOTE_TOKENS` 都读固定的默认值或各自的
 * 常量,和这里无关。占位符(`{pct}` 等)由后续 `renderTemplate` 照常替换 ——
 * 覆盖值缺占位符是合法的(它就是普通文本),多出未知占位符会在这里被
 * `validateTemplate` 拒绝吗?**不会**: 扁平覆盖是纯文本替换,校验留给
 * `config.prompts` 那条结构化路径。这是有意的取舍:拼错一个 `{pctt}` 只会在
 * 模型面前原样出现,不会让插件起不来。
 */
export function applyTextOverrides(
  resolved: ResolvedPrompts,
  overrides?: TextOverrides | null,
): TextOverrideReport {
  if (overrides == null) return { prompts: resolved, unknown: [], oversized: [] }
  const known = new Set(ableTextIds())
  const unknown: string[] = []
  const oversized: TextOverrideDiagnostic[] = []
  const nudge = { ...resolved.nudge } as Record<string, string>
  const rangeTable = { ...resolved.rangeTable } as Record<string, string>
  const tools = { ...resolved.tools } as Record<string, string>
  let toolParamTexts = resolved.toolParamTexts
  let systemPromptTemplate = resolved.systemPromptTemplate
  for (const [id, value] of Object.entries(overrides)) {
    if (!known.has(id) || typeof value !== 'string') {
      unknown.push(id)
      continue
    }
    const dot = id.indexOf('.')
    if (id === 'systemPrompt') systemPromptTemplate = value
    else if (id.startsWith('tool.param.')) {
      toolParamTexts = { ...toolParamTexts, [id.slice('tool.param.'.length)]: value }
    }
    else if (id.startsWith('nudge.')) nudge[id.slice(dot + 1)] = value
    else if (id.startsWith('rangeTable.')) rangeTable[id.slice(dot + 1)] = value
    else tools[id.slice(dot + 1)] = value
    const defaultTokens = estimateTextTokens(textById(DEFAULT_PROMPTS, id) ?? '')
    const tokens = estimateTextTokens(value)
    if (defaultTokens > 0 && tokens > defaultTokens * MAX_TEXT_OVERRIDE_RATIO) {
      oversized.push({ id, tokens, defaultTokens })
    }
  }
  return {
    prompts: {
      nudge,
      rangeTable,
      tools,
      toolParamTexts,
      systemPromptTemplate,
    } as unknown as ResolvedPrompts,
    unknown,
    oversized,
  }
}

/** 诊断行(启动日志与 acp_status 共用一份文案,不各写一套)。 */
export function textOverrideWarningLines(report: TextOverrideReport): string[] {
  const lines: string[] = []
  if (report.unknown.length > 0) {
    lines.push(`textOverrides: ignored ${report.unknown.length} unknown id(s): ${report.unknown.join(', ')} — run ableTextIds() for the accepted ids`)
  }
  for (const item of report.oversized) {
    lines.push(
      `textOverrides: ${item.id} is ~${item.tokens} tokens vs ~${item.defaultTokens} default (${(item.tokens / item.defaultTokens).toFixed(2)}x) — overrides REPLACE the text, so this makes every request heavier`,
    )
  }
  return lines
}
