/**
 * M4 — configurable prompts: template rendering, per-key merge + validation,
 * and byte-identical default snapshots (the anti-regression anchor for the
 * template migration — literals below were captured from the PRE-change
 * implementation, so they are independent of the new rendering code).
 * Design: docs/configurable-prompts-design.md (v4).
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import { createCore, type CompressionCore, type NudgeDecision } from 'acp-kernel'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { AcpStateStore } from '../src/state.ts'
import { buildNudge, buildNudgeText, rangeTable, type NudgeEnvironment } from '../src/nudge.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import {
  DEFAULT_PROMPTS,
  renderSystemPrompt,
  renderTemplate,
  resolvePrompts,
} from '../src/prompts.ts'
import { buildTextSession, wholeSurfaceRangeView } from './helpers.ts'
import AcpCompactionEngine, { AcpCompactionEngine as Named } from '../src/index.ts'

function fakeAgent(session: import('@deepseek-ai/dsh-session').Session): Agent {
  return {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx: new Context(),
  } as unknown as Agent
}

function makeEnv(limit: number): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: limit,
  }
}

function fakeDecision(pct: number, emergency: boolean): NudgeDecision {
  return {
    shouldInject: true,
    reason: 'probe',
    compressibleRanges: [],
    tierTargetBlocks: [],
    contextUsage: pct / 100,
    tier: null,
    breakdown: {
      usage: pct / 100,
      growth: 0,
      growthReference: 0,
      effectiveThreshold: 0,
      nudgeGrowthTokens: 50000,
      growthFloor: 20000,
      hasPendingNudge: 0,
      overLimit: emergency ? 1 : 0,
      emergencyOverride: emergency ? 1 : 0,
      pendingT1: 0,
      pendingT2: 0,
      pendingT3: 0,
    },
  } as never
}

/** 快照前段:system prompt 去掉"Nothing forces you"后的首段。 */
const SYSTEM_PROMPT_HEAD = `Active Context Pruning — model-driven context management

YOU decide whether and when to compress context. The nudge is an efficiency notification: when you see one, consider which ranges you have genuinely consumed and could summarise to keep working context lean.`

test('M4/prompts 1: default system prompt contains key sections in correct order', () => {
  const rendered = renderSystemPrompt(resolvePrompts())
  // Head
  assert.ok(rendered.startsWith(SYSTEM_PROMPT_HEAD), 'system prompt starts with the new efficiency-notification framing')
  // Philosophy
  assert.ok(rendered.includes('Compression Philosophy:\n- All compression serves the primary task'), 'philosophy section present')
  // WHEN TO / WHEN NOT
  assert.ok(rendered.includes('WHEN TO COMPRESS:'), 'WHEN TO COMPRESS section present')
  assert.ok(rendered.includes('WHEN NOT TO COMPRESS:'), 'WHEN NOT TO COMPRESS section present')
  assert.ok(rendered.includes('A task phase has ended'), 'WHEN TO bullet present')
  assert.ok(rendered.includes('Protected tool outputs'), 'WHEN NOT bullet present')
  assert.ok(rendered.includes('Content you will still need to cite verbatim'), 'WHEN NOT cite-verbatim bullet present (issue #55)')
  // HOW_TO_COMPRESS_RULES
  assert.ok(rendered.includes('HOW TO COMPRESS'), 'HOW_TO_COMPRESS_RULES section present')
  assert.ok(rendered.includes('KEEP VERBATIM'), 'KEEP VERBATIM section present')
  assert.ok(rendered.includes('Write dense, scannable bullets'), 'formatting guidance present')
  // Tool descriptions (DSH-specific: seq-based)
  assert.ok(rendered.includes('Compression tools (refs are SURFACE SEQS, not ids):'), 'tool section present')
  assert.ok(rendered.includes('compress: replace one or more seq ranges'), 'compress tool description present')
  assert.ok(rendered.includes('Tiered compression:'), 'tiered compression section present')
  // Tier rules
  assert.ok(rendered.includes('TIER 2 COMPRESSION — DISTILLATION'), 'TIER2_DISTILL_RULES present')
  assert.ok(rendered.includes('TIER 3 COMPRESSION — ULTRA-CONDENSATION'), 'TIER3_CONDENSE_RULES present')
  assert.ok(rendered.includes('When you write a summary, it becomes the ONLY record'), 'summary writing rules present')
  // No "suggestion, not an order" language
  assert.ok(!rendered.includes('suggestion, not an order'), 'no "suggestion, not an order" language')
  assert.ok(!rendered.includes('Nothing forces you'), 'no "Nothing forces you" language')
})

test('M4/prompts 2: default normal nudge — efficiency note + tip, guidance moved OUT (B6)', () => {
  const session = buildTextSession(4)
  const text = buildNudgeText(fakeDecision(7, false), false, session, wholeSurfaceRangeView(session))
  // Frame: kernel EFFICIENCY_NOTE verbatim (no "Context usage is at X%" statement)
  assert.ok(text.startsWith('This is an efficiency nudge to compress early and keep context lean — not an overflow warning.'), 'frame starts with efficiency note')
  assert.ok(!text.includes('Context usage is at'), 'no "Context usage is at" usage statement in the nudge')
  // B6（2026-09-08 方案 §4 B6）：哲学段与压缩规则段移出 nudge（已住系统提示/工具描述），
  // 每拍复读同一份 6 KB 文本=重复计费。nudge 只留「该压缩了 + 压缩哪些」。
  assert.ok(!text.includes('Compression Philosophy:'), '哲学段不在 nudge（B6）')
  assert.ok(!text.includes('HOW TO COMPRESS'), '压缩规则段不在 nudge（B6）')
  assert.ok(!text.includes('KEEP VERBATIM'), 'KEEP VERBATIM 不在 nudge（B6）')
  // No "suggestion, not a requirement"
  assert.ok(!text.includes('suggestion, not a requirement'), 'no "suggestion, not a requirement"')
  // Tip at end (M-3: the ENGINE tip replaces the kernel one — the kernel copy
  // advertised a plain-string content form this schema rejects, and the merged
  // tip carries the two-shapes contract in one line).
  assert.ok(text.endsWith('💡 One call is enough: compress({ condense: { summary } }) lets the engine pick the span (folds the OLDEST consumed PREFIX, every fold barrier inside it merges into ONE block), and compress({ content: [{ startSeq, endSeq, summary }, …] }) batches explicit ranges — content must be an ARRAY (a plain-string content is rejected).'), 'engine tip at end')
})

test('M4/prompts 2b: default nudge renders through the kernel renderNudgeText path (seq table replaces ref table)', () => {
  // With a compressible range present, the DEFAULT nudge must come from the
  // kernel's renderNudgeText (EFFICIENCY_NOTE / breakdown / HOW_TO_COMPRESS_RULES /
  // tip verbatim) with ONLY the ref-ID rangesStr replaced by our seq table.
  const session = buildTextSession(12)
  const text = buildNudgeText(fakeDecision(50, false), false, session, wholeSurfaceRangeView(session))
  // Kernel frame + tip verbatim；B6 起哲学/规则段被摘掉（改住系统提示）
  assert.ok(text.startsWith('This is an efficiency nudge to compress early and keep context lean — not an overflow warning.'), 'kernel EFFICIENCY_NOTE frame')
  assert.ok(!text.includes('HOW TO COMPRESS'), 'B6：规则段不在 nudge')
  assert.ok(text.endsWith('💡 One call is enough: compress({ condense: { summary } }) lets the engine pick the span (folds the OLDEST consumed PREFIX, every fold barrier inside it merges into ONE block), and compress({ content: [{ startSeq, endSeq, summary }, …] }) batches explicit ranges — content must be an ARRAY (a plain-string content is rejected).'), 'engine batch tip at end')
  // Ref-ID rangesStr replaced by the surface-seq table. The title KEEPS the
  // kernel header format ("N, oldest first") with the seq dialect appended —
  // the kernel mN range rows are what must be gone.
  assert.ok(!text.includes('m00150'), 'kernel ref range sample gone')
  assert.ok(text.includes('Ranges ('), 'seq-dialect title injected')
  assert.ok(text.includes('seq 1..7'), 'surface-seq range table injected')
  assert.ok(text.includes('Surface: 12 nodes, seqs 1..12'), 'surface summary present')
  // No usage statement; the only mNNNNN tokens are inside kernel's own
  // HOW_TO_COMPRESS_RULES example text (m00420 key anchors), not a leak.
  assert.ok(!text.includes('Context usage is at'), 'no usage statement')
})

test('M4/prompts 2c (M-3): the tip carries the two-shapes contract — the standalone Note segment is gone', () => {
  // M-3 merged the Note and 💡 segments (they repeated the same contract): the
  // tip now carries both shapes AND the array requirement in ONE line, so the
  // tail is one line instead of two overlapping ones. Pinned against REAL
  // rendered output on both prompt paths; the tip stays the closing line.
  const session = buildTextSession(12)
  const kernelPath = buildNudgeText(fakeDecision(50, false), false, session, wholeSurfaceRangeView(session))
  assert.ok(kernelPath.includes('lets the engine pick the span'), 'the condense shape is spelled out')
  assert.ok(kernelPath.includes('content must be an ARRAY'), 'the array requirement survives in the tip')
  assert.ok(kernelPath.includes('batches explicit ranges'), 'the content shape is still spelled out')
  assert.ok(!kernelPath.includes('two shapes work'), 'the standalone Note segment is gone (M-3)')
  assert.ok(kernelPath.includes('💡'), 'a tip still closes the nudge')

  const templatePath = buildNudgeText(
    fakeDecision(50, false), false, session, wholeSurfaceRangeView(session),
    resolvePrompts({ nudge: { tip: '自定义尾注' } }),
  )
  assert.ok(templatePath.includes('自定义尾注'), 'host tip replaces the contract line it owns')
  assert.ok(templatePath.endsWith('自定义尾注'), 'host tip still closes the nudge')
})

test('M4/prompts 3: default emergency nudge — ⚠️ frame + HOW_TO_COMPRESS_RULES + seq example', () => {
  const session = buildTextSession(4)
  const text = buildNudgeText(fakeDecision(96, true), true, session, wholeSurfaceRangeView(session))
  // Frame: emergency style（B6 起哲学/规则段摘出，只留动作指令）
  assert.ok(text.startsWith('⚠️ Context limit reached — compress now. Prioritize consumed tool outputs.'), 'emergency frame starts with compress now')
  assert.ok(!text.includes('Compression Philosophy:'), 'B6：哲学段不在 emergency nudge')
  // No "choice and timing are yours"
  assert.ok(!text.includes('choice and timing are yours'), 'no "choice and timing are yours"')
  assert.ok(!text.includes('HOW TO COMPRESS'), 'B6：规则段不在 emergency nudge')
  assert.ok(!text.includes('KEEP VERBATIM'), 'B6：KEEP VERBATIM 不在 emergency nudge')
  // Ref-ID example replaced with the seq example (kernel emergency has no 💡 tip)
  assert.ok(!text.includes('startId'), 'no ref-ID startId in the emergency example')
  assert.ok(text.includes('compress({ content: [{ startSeq, endSeq, summary }] })'), 'seq example replaces the ref-ID example')
  assert.ok(!text.includes('💡 Compress all ranges'), 'kernel emergency nudge has no batch tip (kernel parity)')
})

test('M4/prompts 4: range table snapshot (with ranges) and zero-range early return', () => {
  const empty = buildTextSession(4)
  assert.equal(rangeTable(empty, wholeSurfaceRangeView(empty)), '')
  const paged = buildTextSession(12)
  assert.equal(
    rangeTable(paged, wholeSurfaceRangeView(paged)),
    `
Surface: 12 nodes, seqs 1..12
Ranges (1, cheapest rewrite point first — condense always folds the OLDEST eligible prefix, so write your note for that stretch):
  - seq 1..7 · 7msg ~7227t tool0% · fold ~5164 · payback 0.7
Best batch here: 1 row(s) in ONE call → ~7227t less on every later request, one re-prefill of ~5164t — pays back in 0.7 req. The fold column prices each row ALONE; the batch pays one re-prefill for all of them.
fold = re-prefilling every token AFTER the rewrite point, paid ONCE; batch rows into ONE call. DEFAULT MOVE: compress({ condense: { summary } }) — the engine folds the oldest eligible prefix and merges every barrier inside it into ONE block, so you never hand-assemble boundaries.
Use compress({ content: [{ startSeq, endSeq, summary }] }) only for an exact span or several disjoint ones; stale seqs mean re-run acp_status. A row marked N spilled has its bytes on a FILE — folding it loses nothing; a row marked below floor strands the working set — fold a row that STILL LEAVES the floor. Only fold spans you have fully consumed: carry their open TODOs and artifact paths into the summary, or leave them live.`
  )
})

/**
 * 语义载体断言：或为自然语言原话，或为该语义点的专属结构化键行（`key: value`）。
 * 用户 2026-11-25 定：测试保护的是**语义**，不是措辞 —— 精简/结构化改写不该
 * 迫使测试跟着改，但语义消失必须红。每个键只对应一个语义点，避免退化成永真。
 */
function carriesSemantic(text: string, naturalPhrase: string, structuredKey: string): boolean {
  if (text.includes(naturalPhrase)) return true
  return new RegExp(`(^|\\n)\\s*${structuredKey}\\s*:`, 'i').test(text)
}

test('M4/prompts 5: tool descriptions render the defaults byte-identical', () => {
  const descriptions = Object.fromEntries(makeTools(makeEnv(128000)).map((t) => [t.name, t.description]))
  assert.equal(
    descriptions['compress'],
    'Replace older conversation ranges with dense summaries you write. Each message seq is a surface reference.\ndisplay: condense (DEFAULT — engine elects the cheapest eligible shape, possibly a multi-span batch, you name no boundary) | content (EXPLICIT — you name the spans)\ncondense: compress({ condense: { summary } }) folds the cheapest eligible shape of the CURRENT surface (possibly a multi-span batch across walls) in ONE replace; the note is a brief for the work AHEAD (NVIDIA App. A four-field shape, 150-400 tokens — not a narrative of what happened):\n  ## Task (verbatim/near-verbatim copy of the task this stretch served — REQUIRED) ## Now (current work + intent + next step, done when <criterion> — REQUIRED) ## Plan (decomposition + completion criteria) ## Pointers (artifact paths/seq/blockId — keep credentials and exact values verbatim)\n  judgment: fold what is consumed, keep what is still in use; never summarize narrative process\ncontent: compress({ content: [{ startSeq, endSeq, summary, topic?, verifiedReadings? }] }) — batch disjoint ranges in ONE call (each entry its own block; one re-prefill for the batch); boundaries = surface seqs (acp_status or nudge table), bN, or mN\nverifiedReadings: readings that came back green and a later step depends on (e.g. ["closedloop 414/414"]) — stored on the block, so a resumed session does not re-run the suite\nnever: content the current step is still using\nmoments: stage or subtask completion | strategy switch | intermediate milestone | winding up failed exploration\nconsumed: only fold a moment whose details you have fully consumed and will not re-check\nre-verify: before folding, ask — will I need any detail from this range again in this task? If yes, keep it live\ndead-end: turn dead-end exploration into a conclusion (what was tried, why it failed, the next step) — never a blow-by-blow\nonly-record: the summary is the ONLY record a later reader has — self-contained, so a reader (or you, after a recall) can continue without the original; keep paths and values verbatim\ncost: a fold re-prefills every token AFTER the rewrite point ONCE → prefer the LATEST consumed stretch, batch what you would fold anyway, do not fold small spans repeatedly\nreject: a rejected call names exactly what to change — read it instead of guessing.',
  )
  assert.equal(descriptions['recall'], 'Recall content the visible context no longer carries — ONE tool, two mutually exclusive forms. SEARCH: recall({ query: "CONTEXT_WINDOW_EXCEEDED" }) scores the compressed-block summaries, the folded originals AND the still-visible surface, and prints one ref per hit (block bN, a still-visible seq, or the block that owns a folded seq) — when a summary lacks a detail you need (exact values, error strings, decisions, verbatim code), SEARCH the compressed blocks FIRST — never guess or reconstruct from memory, then read only that block to recover the original. READ: recall({ blockId: "b1" }) dumps a block\'s original messages read-only (the bN ref acp_status lists, or a compaction id / prefix from a search hit; large blocks page — pass offset/pageLimit and follow the continue hint, and locate the page you need with a query first), and recall({ seq: 42 }) returns ONE event\'s original text (pass chars to read past the 20000-char cut; works after a fold because the log keeps the bytes). One call is one form: query + blockId/seq in the same call is refused. Examples: recall({ query: "AUTH_TOKEN_TTL" }) · recall({ blockId: "b1", offset: 0, pageLimit: 50 }).')
  assert.equal(descriptions['acp_status'], 'Context status: overview of the current context — CONTEXT BREAKDOWN (tool/text/summaries token shares of the visible total), COMPRESSED BLOCKS ledger, and the nudge decision. No args = overview. Percentages are shares of the visible content, not the context window. Note: the block refs in COMPRESSED BLOCKS (bN, e.g. b1) work in BOTH tools — recall({ blockId: "bN" }) reads the block, and compress accepts bN as a boundary to DISTILL that block (tier 2/3); plain spans use the Surface: seq range. Drilldown: detail:"compressed" for the per-block list, detail:"visible" for the merged foldable ranges, detail:"messages" for every visible message; no detail = overview. tool narrows to one tool name (detail:"messages" only), sort reorders (size/time/tool; age for compressed), limit caps rows (default 30). Drilldown row refs are kernel ids (mN) — feed them straight to compress as startSeq/endSeq (auto-mapped to the live surface seq); bN also works as a compress boundary (it distills that block), Surface: seqs cover plain spans.')
})

test('M4/prompts 5b: default compress description ships the #43 semantic-timing and actionable-summary guidance', () => {
  // issue #43 (AgentFold/CAT paper insights): the DEFAULT compress tool
  // description must carry (a) semantic compression moments and (b) the
  // actionable-summary rule (dead-end → conclusion, summary is the ONLY
  // record). These were proven out locally via config.prompts overrides and
  // are now promoted into the shipped default — byte-identical snapshot is
  // test #5, this test pins the two semantic carriers explicitly.
  const descriptions = Object.fromEntries(makeTools(makeEnv(128000)).map((t) => [t.name, t.description]))
  const compress = descriptions['compress']
  assert.ok(
    carriesSemantic(compress, 'Good compression moments: stage or subtask completion whose details you have fully consumed and will not re-check, strategy switches, intermediate milestones, and wrapping up failed exploration', 'moments'),
    'semantic compression moments present (issue #43)',
  )
  assert.ok(
    carriesSemantic(compress, 'turn dead-end exploration into a conclusion (what was tried, why it failed, the next step) — not a blow-by-blow', 'dead-end'),
    'actionable dead-end→conclusion rule present (issue #43)',
  )
  assert.ok(
    carriesSemantic(compress, 'keep the summary the ONLY record: self-contained, so a later reader (or you, after a recall) can continue without the original', 'only-record'),
    'summary-as-only-record rule present (issue #43)',
  )
})

test('M4/prompts 5c: search-first carriers ship in the default recall description AND the system prompt quick-reference (issue #44)', () => {
  // issue #44: when a summary lacks details, the model must SEARCH the
  // compressed blocks first — never guess or reconstruct from memory.
  // Two independent entry points carry the rule: (a) the default
  // recall TOOL description and (b) the system-prompt
  // quick-reference line. Pinning both as semantic carriers (not
  // byte-identical — that is test #5) guards against either drifting back
  // to the old passive wording independently.
  const descriptions = Object.fromEntries(makeTools(makeEnv(128000)).map((t) => [t.name, t.description]))
  const recallDesc = descriptions['recall']
  assert.ok(
    recallDesc.includes('SEARCH the compressed blocks FIRST — never guess or reconstruct from memory'),
    'search-first rule present in the default recall description (issue #44)',
  )
  assert.ok(
    recallDesc.includes('then read only that block to recover the original'),
    'search-locates-then-reads step present (issue #44)',
  )
  const rendered = renderSystemPrompt(resolvePrompts())
  assert.ok(
    rendered.includes('SEARCH the compressed blocks FIRST — never guess or reconstruct from memory'),
    'search-first rule present in the system prompt quick-reference (issue #44)',
  )
  assert.ok(
    rendered.includes('exact values, error strings, decisions, verbatim code'),
    'quick-reference carrier list matches the tool description (issue #44)',
  )
})

test('M4/prompts 5d: anti-over-compression carriers ship in the default compress description AND the system prompt (issue #55)', () => {
  // issue #55: in review/audit/verification tasks the model compressed source
  // reads and then decompressed them back (compress -> decompress ->
  // re-compress loop). Three carriers guide it away: (a) the WHEN NOT TO
  // COMPRESS cite-verbatim bullet, (b) the "fully consumed" qualifier on the
  // subtask completion trigger, (c) the re-verify self-check at the end of
  // the compress description. Proven out locally via config.prompts
  // overrides (live-verified in cordis.patch.yml) before promotion to the
  // defaults.
  const descriptions = Object.fromEntries(makeTools(makeEnv(128000)).map((t) => [t.name, t.description]))
  const compress = descriptions['compress']
  assert.ok(
    carriesSemantic(compress, 'stage or subtask completion whose details you have fully consumed and will not re-check', 'consumed'),
    'consumed-qualifier on subtask completion present (issue #55)',
  )
  assert.ok(
    carriesSemantic(compress, 'Before compressing, ask: will I need to re-verify any detail from this range in this task? If yes, keep it live.', 're-verify'),
    're-verify self-check present (issue #55)',
  )
  const rendered = renderSystemPrompt(resolvePrompts())
  assert.ok(
    rendered.includes('Content you will still need to cite verbatim — in review/audit/verification tasks, keep source reads un-compressed until the final report is written.'),
    'cite-verbatim WHEN NOT bullet present in system prompt (issue #55)',
  )
  assert.ok(
    rendered.includes('a recall costs a full round-trip; prefer delaying the compress'),
    'round-trip cost warning present in system prompt (issue #55)',
  )
})

test('M4/prompts 5e: liveness-aware compression carriers ship in the range-table footer AND the system prompt (issue #149)', () => {
  // issue #149: compressing an in-progress span dropped live task state
  // (where you are / open TODOs, ruled-out paths, ready artifacts + their
  // locations), so the model re-ran consumed work after the summary replaced
  // it. Two engine-owned carriers now guard it — the slim nudge no longer
  // carries the full rule set (B6), so the point-of-action cue lives right
  // under the ranges (footer) and the load-bearing one-time guidance (system
  // prompt) names the handoff explicitly.
  const rendered = renderSystemPrompt(resolvePrompts())
  assert.ok(
    rendered.includes('A mid-task compression must also hand off STATE, not just facts'),
    'task-state handoff rule present in the system prompt (issue #149)',
  )
  assert.ok(
    rendered.includes('which intermediate artifacts are already done and where they live'),
    'ready-artifact-location carrier present in the system prompt (issue #149)',
  )
  const paged = buildTextSession(12)
  const table = rangeTable(paged, wholeSurfaceRangeView(paged))
  assert.ok(
    table.includes('Only fold spans you have fully consumed'),
    'action-zone liveness reminder present in the range-table footer (issue #149)',
  )
  assert.ok(
    table.includes('open TODOs and artifact paths'),
    'footer names ruled-out paths + artifact locations (issue #149)',
  )
})

test('M4/P0-3: Appendix-A note doctrine ships in the compress description, condense description AND the system prompt', () => {
  // paper-alignment §4 P0-3 (2026-11-25): the controlled experiment (§8.2) attributes
  // -62% re-retrieval to doctrine alone, and NVIDIA Appendix A defines what a good
  // note IS — a brief for the work ahead (Task verbatim / Now / Plan+done-when /
  // Pointers with verbatim credentials), never a narrative of what happened, and
  // SHORT (the paper's median is 185t; our TYPICAL_HANDOFF_NOTE_TOKENS is 400).
  // Three engine-owned carriers must all say it: the compress tool description
  // (point of call), the condense param description, and the system prompt CONDENSE
  // section (one-time guidance). Kernel copy (HOW_TO_COMPRESS_RULES, rule 9) is
  // upstream — this pins only OUR layers.
  const descriptions = Object.fromEntries(makeTools(makeEnv(128000)).map((t) => [t.name, t.description]))
  const compress = descriptions['compress']
  assert.ok(
    compress.includes('brief for the work AHEAD') && compress.includes('not a narrative of what happened'),
    'compress description frames the note as a forward brief (Appendix A)',
  )
  assert.ok(compress.includes('## Task') && compress.includes('REQUIRED'), 'Task named REQUIRED')
  assert.ok(compress.includes('## Plan') && compress.includes('## Pointers'), 'Plan and Pointers named')
  assert.ok(compress.includes('150-400 tokens'), 'the short-note budget is stated')
  assert.ok(compress.includes('fold what is consumed, keep what is still in use'), 'the consumed-vs-live judgment line')
  const compressTool = makeTools(makeEnv(128000)).find((t) => t.name === 'compress')
  const condenseCopy = String((compressTool!.parameters as { properties: { condense?: { description?: string } } }).properties.condense?.description ?? '')
  assert.ok(condenseCopy.includes('BRIEF FOR THE WORK AHEAD'), 'the condense parameter description ships the Appendix-A frame')
  assert.ok(condenseCopy.includes('## Task') && condenseCopy.includes('## Pointers'), 'condense description carries the same four-field shape')
  assert.ok(condenseCopy.includes('150-400'), 'condense description states the short budget')
  const rendered = renderSystemPrompt(resolvePrompts())
  assert.ok(rendered.includes('fold what is consumed, keep what is still in use'), 'the judgment line lives in the system prompt')
  assert.ok(rendered.includes('NOT a narrative'), 'the anti-narrative rule lives in the system prompt')
  assert.ok(rendered.includes('150-400 tokens'), 'the short-note budget lives in the system prompt')
  assert.ok(rendered.includes('verbatim or near-verbatim copy of the task'), 'Task is near-verbatim in the system prompt')
  assert.ok(rendered.includes('completion criterion (when is it DONE)'), 'the Plan completion criterion lives in the system prompt')
  assert.ok(rendered.includes('credentials and exact values verbatim'), 'Pointers carry verbatim credentials')
})

test('M4/prompts 6: partial overrides merge per key; key/group null falls back to default', () => {
  const merged = resolvePrompts({ nudge: { normal: '自定义 {pct}' } })
  assert.equal(merged.nudge.normal, '自定义 {pct}')
  assert.equal(merged.nudge.emergency, DEFAULT_PROMPTS.nudge.emergency)
  const keyNull = resolvePrompts({ nudge: { guidance: null } })
  assert.equal(keyNull.nudge.guidance, DEFAULT_PROMPTS.nudge.guidance)
  const groupNull = resolvePrompts({ nudge: null } as never)
  assert.equal(groupNull.nudge, DEFAULT_PROMPTS.nudge)
})

test('M4/prompts 7: nudge normal template substitutes {pct}', () => {
  const prompts = resolvePrompts({ nudge: { normal: '上下文使用率 {pct}%' } })
  const session = buildTextSession(4)
  const text = buildNudgeText(fakeDecision(7, false), false, session, wholeSurfaceRangeView(session), prompts)
  assert.ok(text.startsWith('上下文使用率 7%'))
})

test('M4/prompts 8: unknown placeholder throws with the slot path', () => {
  assert.throws(
    () => resolvePrompts({ nudge: { normal: '…{pctt}…' } }),
    /prompts\.nudge\.normal contains unknown placeholder \{pctt\}/,
  )
  assert.throws(
    () => resolvePrompts({ rangeTable: { line: '  - {start}..{wrong}' } }),
    /prompts\.rangeTable\.line contains unknown placeholder \{wrong\}/,
  )
  assert.throws(
    () => resolvePrompts({ systemPrompt: 'x {philosophyy} y' }),
    /prompts\.systemPrompt contains unknown placeholder \{philosophyy\}/,
  )
})

test('M4/prompts 9: renderTemplate throws on a missing value for a known placeholder', () => {
  assert.throws(
    () => renderTemplate('{tokens} tokens', {}),
    /missing value for placeholder \{tokens\}/,
  )
})

test('M4/prompts 10: empty guidance removes the line cleanly (frame + newline + table + tip)', () => {
  const session = buildTextSession(12)
  const prompts = resolvePrompts({
    nudge: {
      guidance: '',
      breakdown: '',
      growth: '',
      tip: '',
    },
  })
  // B6：默认 frame 不再内嵌哲学段（哲学住系统提示）
  const frame = 'Efficiency nudge: compress consumed ranges early to keep context lean — not an overflow warning. A stronger alert appears only if the context is actually full.'
  const text = buildNudgeText(fakeDecision(7, false), false, session, wholeSurfaceRangeView(session), prompts)
  assert.equal(text, `${frame}\n${rangeTable(session, wholeSurfaceRangeView(session))}`)
  assert.ok(!text.includes('HOW TO COMPRESS'))
  assert.ok(!text.includes('Compress all ranges'))
})

test('M4/prompts 11: tier line renders (0 tokens) when pending is missing (B2 fallback)', () => {
  const decision: NudgeDecision = {
    shouldInject: true,
    reason: 'tier-2 distillation recommended',
    compressibleRanges: [],
    tierTargetBlocks: [{ blockId: 'b1', tier: 1, effectiveMessageIds: ['m1'], compressedTokens: 4750, summary: 's', active: true, directMessageIds: [], directBlockIds: [], createdAt: 1, survivedCount: 0, generation: 'young' }],
    contextUsage: 0.9,
    tier: 2,
    // pendingT2 deliberately absent at runtime: NudgeBreakdown requires it
    // statically (acp-kernel types.d.ts:183-185), so cast per test convention.
    breakdown: {
      usage: 0.9,
      growth: 0,
      growthReference: 0,
      effectiveThreshold: 0,
      nudgeGrowthTokens: 50000,
      growthFloor: 20000,
      hasPendingNudge: 0,
      overLimit: 1,
      emergencyOverride: 0,
      pendingT1: 0,
      pendingT3: 0,
    } as never,
  }
  const session = buildTextSession(12)
  const text = buildNudgeText(decision, false, session, wholeSurfaceRangeView(session))
  assert.match(text, /Tier 2: 1 tier-1 block\(s\) distillable \(0 tokens\)/)
  assert.doesNotMatch(text, /\( tokens\)/)
})

test('M4/prompts 12: buildNudge forwards env.prompts into the injected message (B1)', () => {
  const env: NudgeEnvironment = {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    // ~77.5% usage for 12 messages: over-limit (>= 0.70) but below emergency
    // (0.85) → the NORMAL frame renders.
    modelContextLimit: 16000,
    prompts: resolvePrompts({ nudge: { normal: 'CUSTOM normal {pct}' } }),
  }
  const outcome = buildNudge(fakeAgent(buildTextSession(12)), env, new Map<string, number>(), new Map())
  assert.ok(outcome !== null, 'over-limit nudge fires')
  assert.equal(outcome!.emergency, false)
  const text = outcome!.message.content.map((block) => (block as { text?: string }).text ?? '').join('')
  assert.ok(text.startsWith('CUSTOM normal '), 'the custom normal frame reached the injected message')
})

test('M4/prompts 13: Chinese override smoke (i18n scenario)', () => {
  const prompts = resolvePrompts({
    nudge: {
      normal: '上下文使用率 {pct}%。这是效率提示，不是溢出警告。',
      emergency: '⚠️ 上下文已达上限，立即压缩。',
      guidance: '压缩规则：保持路径、签名、错误原文。',
      tier: '第 {tier} 层:{count} 个第 {prevTier} 层块可蒸馏({tokens} tokens)。',
      breakdown: '上下文分解：{system}K 系统 | {tool}K 工具 | {summaries}K 摘要 | {code}K 代码 | {text}K 文本',
      growth: '+{growth}K 自上次提示',
      tip: '💡 一次调用压缩多个范围。',
    },
    rangeTable: {
      header: '表面:{surface}',
      title: '可压缩范围(仅供参考):',
      line: '  - seq {start}..{end} — {count} 条消息,约 {tokens} tokens',
      footer: '用 compress 压缩;批量条目各成一个块。',
    },
    systemPrompt: '主动上下文剪枝 —— 模型驱动。\n{philosophy}\n{howToCompressRules}\n压缩工具:compress/recall/acp_status。',
  })
  const session = buildTextSession(12)
  const text = buildNudgeText(fakeDecision(7, false), false, session, wholeSurfaceRangeView(session), prompts)
  assert.ok(text.includes('上下文使用率 7%'))
  assert.ok(text.includes('表面:12 nodes, seqs 1..12'))
  assert.ok(text.includes('可压缩范围(仅供参考):'))
  assert.ok(text.includes('💡 一次调用压缩多个范围'))
  const emerg = buildNudgeText(fakeDecision(96, true), true, session, wholeSurfaceRangeView(session), prompts)
  assert.ok(emerg.includes('上下文已达上限'))
  const sys = renderSystemPrompt(prompts)
  assert.ok(sys.includes('主动上下文剪枝'))
  assert.ok(sys.includes('Compression Philosophy:'))
  assert.ok(sys.includes('compress/recall/acp_status'))
})

test('M4/prompts 14: engine-level — config.prompts reaches system prompt section, tool descriptions, and the pre-step nudge', async () => {
  const ctx = new Context()
  const sections: Array<{ name: string; order: number; text: string }> = []
  const tools: Array<{ name: string; description: string }> = []
  ctx.provide('systemPrompt' as never, { section: (opts: never) => sections.push(opts as never) } as never)
  ctx.provide('tools' as never, { register: (tool: never) => tools.push(tool as never) } as never)
  ctx.plugin(AcpCompactionEngine as never, {
    modelContextLimit: 16000,
    prompts: {
      nudge: { normal: '引擎级自定义 {pct}' },
      systemPrompt: '引擎级系统提示\n{philosophy}',
      tools: { compress: '引擎级压缩工具描述' },
    },
  } as never)
  await new Promise((resolve) => setTimeout(resolve, 20))
  const engine = ctx.compaction as Named
  assert.ok(engine instanceof Named)
  assert.equal(engine.prompts.nudge.normal, '引擎级自定义 {pct}', 'engine resolved the custom template')

  // System prompt section: rendered once with the custom template + kernel philosophy.
  assert.equal(sections.length, 1, 'ACP section registered exactly once')
  assert.ok(sections[0]!.text.includes('引擎级系统提示'))
  assert.ok(sections[0]!.text.includes('Compression Philosophy:'), 'kernel philosophy embedded via {philosophy}')

  // Tool descriptions: registered from env.prompts (proves env carried this.prompts).
  const compressTool = tools.find((t) => t.name === 'compress')
  assert.equal(compressTool!.description, '引擎级压缩工具描述')

  // pre-step: the custom normal frame reaches the injected nudge message.
  const decision = await (ctx.waterfall(
    'agent/pre-step' as never,
    { agent: fakeAgent(buildTextSession(12)) } as never,
    async () => ({ kind: 'enter', messages: [] }) as never,
  ) as never)
  const decisionObj = decision as { kind: string; messages: Array<{ content: Array<{ type: string; text?: string }> }> }
  assert.equal(decisionObj.kind, 'enter')
  assert.equal(decisionObj.messages.length, 1, 'the nudge message was appended to the decision')
  const text = decisionObj.messages[0]!.content.map((block) => block.text ?? '').join('')
  assert.ok(text.startsWith('引擎级自定义 '), 'the custom nudge frame reached the injected message')
})

test('M4/prompts 15: engine construction fails fast on a template typo', () => {
  assert.throws(
    () => new Named(new Context(), { prompts: { nudge: { normal: '{bad}' } } }),
    /prompts\.nudge\.normal contains unknown placeholder \{bad\}/,
  )
  // Group-level null from YAML-ish input is tolerated (whole group falls back).
  const engine = new Named(new Context(), { prompts: { nudge: null } } as never)
  assert.equal(engine.prompts.nudge.normal, DEFAULT_PROMPTS.nudge.normal)
})

/**
 * Round-12 review Q9.6: the whitelist for `rangeTable.line` listed the retired `rewriteCost`
 * slot while the default template renders `{foldCost}` — a user copying the default string into
 * a custom `prompts.rangeTable.line` (or writing `{foldCost}` themselves) threw at startup.
 */
test('M4/prompts: a custom rangeTable.line may use the {foldCost} slot the default renders', () => {
  const resolved = resolvePrompts({
    rangeTable: { line: '  - seq {start}..{end} \u00b7 {count}msg ~{tokens}t tool{toolPct}%{media} \u00b7 fold ~{foldCost}' },
  })
  assert.match(resolved.rangeTable.line, /fold ~\{foldCost\}/)
})
