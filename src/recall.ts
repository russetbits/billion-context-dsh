import { searchBlocks, defaultCountTokens, type MessageRole, type SearchDoc } from './kernel/index.ts'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import { ToolArgsError, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import { unwrapEnvelope } from './compress-args.ts'
import { ToolEnvironment, TextOutput, requireAgent } from './shared.ts'
import { DEFAULT_TOOL_PARAM_TEXTS, type ResolvedPrompts } from './prompts.ts'
import { callHint, type ResolvedCallStyle } from './ptc.ts'
import { sliceDecompressPage, DEFAULT_DECOMPRESS_PAGE, DEFAULT_DECOMPRESS_PAGE_CHARS, blockIdOfKernelRef, rebuildBlockLedger, expandShadowedSeqs, summarySeqOfKernelBlock, blockRefForSummarySeq, compactionIdsOfKernelBlocks } from './region.ts'
import { eventAtOf, sessionEventsOf } from './session-events.ts'
import { classifySurfaceEvent, extractEventText, isCheckpointNode, isSkillCatalogRow, toolCallsOf } from './messages.ts'
import { condenseNoCandidateReason } from './condense-note.ts'

/**
 * `recall` merges the two ways back into the append-only log into ONE tool:
 * the SEARCH form (`query`) and the READ form (`blockId` / `seq`). They are
 * mutually exclusive — a search returns refs, a read returns bytes — and the
 * handler refuses a call that carries both (see `handleRecall`).
 */
export const recallParameters = {
  query: { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.query'] },
  blockId: { type: 'string' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.blockId'] },
  seq: { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.seq'] },
  chars: { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.chars'] },
  maxResults: { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.maxResults'] },
  offset: { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.offset'] },
  pageLimit: { type: 'integer' as const, description: DEFAULT_TOOL_PARAM_TEXTS['recall.pageLimit'] },
} as const

export interface RecallArgs {
  query?: string
  blockId?: string
  seq?: number
  chars?: number
  maxResults?: number
  offset?: number
  pageLimit?: number
}

/**
 * C-2（2026-11-23）：按 seq 读一条消息的原文。
 *
 * C-1 让 recall 的检索形态会命中**仍可见**的消息并报出它的 seq，但模型拿到这个
 * seq 后无路可走 —— 按块读取只认 blockId，可见消息不属于任何块。这条路径把
 * 「检索命中 → 读到内容」接上，同时也能读任何历史 seq（日志保留原文），所以在
 * 折叠之后同样有效。
 */
const SEQ_READ_DEFAULT_CHARS = 20_000
const SEQ_READ_MAX_CHARS = 100_000

function handleSeqRead(session: Session, rawSeq: number, rawChars: number | undefined): TextOutput {
  if (!Number.isInteger(rawSeq) || rawSeq <= 0) return { text: `recall: seq must be a positive integer (got ${String(rawSeq)})` }
  const event = eventAtOf(session, rawSeq)
  if (event === undefined) {
    return { text: `recall: seq ${rawSeq} is not in this session log (see acp_status for the Surface anchor, or recall({ query }) for seqs)` }
  }
  const text = extractEventText(event)
  if (text.length === 0) return { text: `recall: seq ${rawSeq} (${event.type}) carries no text` }
  const cap = Math.min(SEQ_READ_MAX_CHARS, Math.max(1_000, rawChars ?? SEQ_READ_DEFAULT_CHARS))
  // `surface.nodes` is branded (SessionSeq); a plain-number probe needs the cast.
  const visible = (session.surface.nodes as readonly number[]).includes(rawSeq) ? 'visible' : 'folded (original kept in the log)'
  const body = text.length > cap
    ? `${text.slice(0, cap)}\n\n…[truncated at ${cap} of ${text.length} chars — pass chars to read further]`
    : text
  return { text: `[seq ${rawSeq} · ${event.type} · ${visible}]\n\n${body}` }
}

/** Resolve a block arg to its durable compaction id: exact `bN` kernel ref
 *  first (acp_status shows `bN`), then the compaction-id prefix match that
 *  `recall` and /acp-prune have always used. The `bN` branch is exact
 *  (`/^b\d+$/` with `$`), so a UUID that happens to start with `b1` cannot be
 *  shadowed — full UUIDs and 8-char prefixes never match the anchored regex. */
function resolveBlockId(session: Session, arg: string): string | null {
  const byKernelRef = blockIdOfKernelRef(session, arg)
  if (byKernelRef !== null) return byKernelRef
  // A canonical `bN` form is NEVER read as a compaction-id prefix. `blockIdOfKernelRef` already
  // refused it (no active block carries that ref), and compaction ids are UUIDs whose leading
  // chars are hex — `b01` is itself a valid UUID prefix, so falling through to the prefix scan
  // resolved a malformed ref to a real block with probability 1/16^len (round-14 review Q6: the
  // one-off `M3` flake, unreproducible by construction).
  if (/^b\d+$/.test(arg)) return null
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  const byPrefix = ledger.find((entry) => entry.blockId.startsWith(arg))
  return byPrefix?.blockId ?? null
}

/** READ form, `blockId` branch: page a block's original messages. */
function handleBlockRead(session: Session, rawBlockId: string, offset: number | undefined, limit: number | undefined): TextOutput {
  if (rawBlockId === '') {
    return { text: 'recall: pass blockId (bN or compaction id, from acp_status or a search hit) or seq (one message)' }
  }
  const blockId = resolveBlockId(session, rawBlockId)
  if (blockId === null) {
    return { text: `recall: block "${rawBlockId}" not found (see acp_status for the block list)` }
  }
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  const block = ledger.find((entry) => entry.blockId === blockId)
  if (block === undefined) {
    return { text: `recall: block "${rawBlockId}" not found (see acp_status for the block list)` }
  }
  // Tier-2/3 blocks shadow parent checkpoint nodes: expand to the originals.
  // Page by BOTH message count and rendered chars so a normal page stays under
  // the host's tool-result pruner threshold (DEFAULT_DECOMPRESS_PAGE_CHARS);
  // renderLen prices each message exactly as it will appear on this page.
  const expanded = expandShadowedSeqs(session, block.blockId)
  const page = sliceDecompressPage(
    expanded,
    offset ?? 0,
    limit ?? DEFAULT_DECOMPRESS_PAGE,
    DEFAULT_DECOMPRESS_PAGE_CHARS,
    (seq) => {
      const event = eventAtOf(session, seq)
      const text = event === undefined ? '' : extractEventText(event)
      return text.length === 0 ? 0 : `[seq ${seq}] ${text}`.length
    },
  )
  if (page.total === 0 || page.seqs.length === 0) {
    const where = page.total === 0 ? '' : ` has ${page.total} messages; offset ${page.offset} is past the end — use an offset below ${page.total}, or omit it`
    return { text: page.total === 0 ? `Block ${block.blockId} — ${block.summary}\n\n(no recoverable content)` : `recall: block ${block.blockId}${where}` }
  }
  const parts: string[] = []
  for (const seq of page.seqs) {
    const event = eventAtOf(session, seq)
    const text = event === undefined ? '' : extractEventText(event)
    if (text.length > 0) parts.push(`[seq ${seq}] ${text}`)
  }
  const tierNote = block.tier > 1 ? ` (tier ${block.tier}, distills ${block.parentBlockIds.length} block(s))` : ''
  // Marker + continue hint LEAD the payload, not trail it: if a page is ever
  // oversized and the host drops its middle, the "this page is partial" line
  // survives up top rather than being the exact line that gets trimmed away.
  const lines: string[] = []
  lines.push(`[messages ${page.offset + 1}..${page.offset + page.seqs.length} of ${page.total}]`)
  if (!page.exhausted) lines.push(`More available — continue with recall({ blockId: "${block.blockId}", offset: ${page.offset + page.seqs.length} })`)
  return {
    text: `Block ${block.blockId} — ${block.summary}${tierNote}\n\n${lines.join('\n')}\n\n${parts.join('\n\n') || '(no text content on this page)'}`,
  }
}

/** The "no form given" refusal: name BOTH shapes with an executable example so
 *  the next call is copy-paste, not a guess. */
function recallNoFormNote(): string {
  return 'recall: pass exactly one form — `query` (search the log and the blocks: recall({ query: "error string" })) '
    + 'or `blockId`/`seq` (read the original bytes: recall({ blockId: "b1" }) / recall({ seq: 42 })). Nothing was done.'
}

/** Both forms in one call: refuse explicitly and say which field to drop, so a
 *  retry is mechanical instead of a re-read of the description. */
function recallExclusiveNote(given: string): string {
  return `recall: \`query\` (search) and ${given} (read the original) are mutually exclusive — one call does one of them. `
    + `Drop \`query\` to read the original, or drop ${given} to run the search. Nothing was done.`
}

/**
 * The ONE model-facing way back into the append-only log.
 *
 * Two mutually exclusive forms, dispatched here:
 *  - SEARCH — `query` → hybrid retrieval over block summaries, shadowed
 *    originals and the still-visible surface (`handleSearch`);
 *  - READ   — `seq` (one event, `chars`-capped) or `blockId` (a paged block
 *    dump) → original bytes from the log, read-only.
 * A call carrying both forms, or neither, is refused with an executable note:
 * the old split (two tools, one schema each) made an ambiguous call impossible,
 * a merged schema does not, so the refusal has to teach the retry.
 */
export function handleRecall(env: ToolEnvironment, rawArgs: RecallArgs, exec: ToolRunContext): TextOutput {
  const args = unwrapEnvelope<RecallArgs>(rawArgs)
  const session = requireAgent(exec).session
  const hasQuery = typeof args.query === 'string'
  const hasBlock = typeof args.blockId === 'string' && args.blockId !== ''
  const hasSeq = args.seq !== undefined
  // Three refusals, not two: `blockId` + `seq` is ALSO ambiguous (both name a
  // READ target, and before this check `seq` silently won while the schema text
  // said "use this OR blockId") — cross-review 2026-11-24.
  if (hasBlock && hasSeq) return { text: recallExclusiveNote('`blockId`/`seq`') }
  if (hasQuery && (hasBlock || hasSeq)) {
    return { text: recallExclusiveNote(hasBlock ? '`blockId`' : '`seq`') }
  }
  if (hasQuery) return handleSearch(env, { query: args.query ?? '', limit: args.maxResults }, exec)
  if (hasSeq) return handleSeqRead(session, args.seq!, args.chars)
  if (hasBlock) return handleBlockRead(session, args.blockId!, args.offset, args.pageLimit)
  return { text: recallNoFormNote() }
}

interface SearchArgs {
  query: string
  limit?: number
}

/** Event type → kernel message role (drives hybrid role weighting). */
function roleOfEvent(event: SessionEvent): MessageRole | null {
  switch (event.type) {
    case 'user/message': return 'user'
    case 'assistant/message': return 'assistant'
    case 'tool/result': return 'tool'
    default: return null
  }
}

// The search corpus (block summaries + all shadowed originals) is a pure
// function of the append-only log: rebuild it once per log snapshot and reuse
// across searches until the next append (issue #133 — the per-call full
// rebuild re-extracted and re-counted every shadowed message on every search;
// the snapshot array is stable until the next append, see sessionEventsOf).
const searchDocsCache = new WeakMap<readonly SessionEvent[], SearchDoc[]>()

/**
 * Build the unified SearchDoc[] from the log: one block doc per ledger entry
 * (ref = compactionId, so `recall({ blockId })` closes the loop) plus one
 * message doc per shadowed ORIGINAL (expanded through distilled parents; each
 * seq is claimed by the earliest/innermost block that covered it, mirroring
 * pi's owner map — recall on that block recovers the original).
 * Cached per log snapshot (see searchDocsCache). Exported for the issue #133
 * regression tests (not part of the public API — index.ts re-exports only).
 */
export function buildSearchDocs(session: Session): SearchDoc[] {
  const events = sessionEventsOf(session)
  const cached = searchDocsCache.get(events)
  if (cached !== undefined) return cached
  const ledger = rebuildBlockLedger(events)
  const docs: SearchDoc[] = []
  const claimed = new Set<number>()
  for (const block of ledger) {
    docs.push({
      kind: 'block',
      ref: block.blockId,
      text: block.summary,
      title: block.summary.slice(0, 60) || block.blockId,
      blockId: block.blockId,
      tier: block.tier,
      tokens: defaultCountTokens(block.summary),
    })
    for (const seq of expandShadowedSeqs(session, block.blockId)) {
      if (claimed.has(seq)) continue
      claimed.add(seq)
      const event = eventAtOf(session, seq)
      if (event === undefined) continue
      const role = roleOfEvent(event)
      const text = extractEventText(event)
      if (role === null || text.length === 0) continue
      docs.push({
        kind: 'message',
        ref: `seq ${seq}`,
        text,
        title: `${role}: ${text.slice(0, 60)}`,
        role,
        blockId: block.blockId,
        tier: block.tier,
        tokens: defaultCountTokens(text),
      })
    }
  }
  // C-1（2026-11-23）：把**仍然可见**的表面节点也纳入语料。
  //
  // 原来只收「块摘要 + 被折叠的原文」，所以模型问「刚才那个报错在哪」时，只要
  // 那段还没被压，search 反而一无所获 —— 它得靠翻可见消息找，而可见面正是长
  // 会话里最翻不动的部分。acp-rs 的全日志面是唯一实测更优的能力（同一会话
  // 934 事件/88.7 万字符，2.234ms vs 1.837ms）。
  //
  // 排除三类噪音，它们占位却不含可检索的会话内容：宿主元数据行（nudge 回显、
  // compress 配对桩、prune 墓碑 —— `classifySurfaceEvent` 的 'metadata'）、
  // checkpoint 摘要节点（摘要本身已是 block doc）与 skill 目录行（每步复读的
  // 同一份清单）。
  for (const seq of session.surface.nodes) {
    if (claimed.has(seq)) continue
    const event = eventAtOf(session, seq)
    if (event === undefined) continue
    if (classifySurfaceEvent(event) === 'metadata') continue
    if (isCheckpointNode(event) || isSkillCatalogRow(event)) continue
    const role = roleOfEvent(event)
    const text = extractEventText(event)
    if (role === null || text.length === 0) continue
    docs.push({
      kind: 'message',
      ref: `seq ${seq}`,
      text,
      title: `${role}: ${text.slice(0, 60)}`,
      role,
      tokens: defaultCountTokens(text),
    })
  }
  searchDocsCache.set(events, docs)
  return docs
}

function handleSearch(env: ToolEnvironment, rawArgs: SearchArgs, exec: ToolRunContext): TextOutput {
  const args = unwrapEnvelope<SearchArgs>(rawArgs)
  const session = requireAgent(exec).session
  if (args.query.trim() === '') return { text: 'recall: empty query (no matches)' }
  const docs = buildSearchDocs(session)
  // Trust the kernel: hybrid (0.7×BM25 stemmed + 0.3×fuzzy n-gram) is the
  // algorithm contract — no engine-side gate or threshold re-implements
  // search policy. Scores are surfaced so the model can judge a weak hit
  // (fuzzy-only tops out near 0.3).
  const results = searchBlocks(docs, args.query, { limit: args.limit ?? 5, previewLength: 160 })
  if (results.length === 0) return { text: `recall: no matches for "${args.query}"` }
  const lines = results.map((r) => {
    const kind = r.kind === 'block'
      ? `block ${r.ref}`
      : r.blockId === undefined
        // C-1：命中仍在表面上的消息 —— 不必读原文，直接读那一 seq。
        ? `message ${r.ref} (${r.role ?? '?'}, still visible — no read needed, recall({ seq: ${r.ref.slice(4)} }))`
        : `message ${r.ref} (${r.role ?? '?'}, in block ${r.blockId})`
    return `  - ${kind} (score ${r.score.toFixed(2)}): ${r.preview}`
  })
  const style = env.toolCallStyle ?? 'native'
  return {
    text: `Matches for "${args.query}":\n${lines.join('\n')}\n\nRead a hit with: ${callHint(style, 'recall', '{ blockId }')} or ${callHint(style, 'recall', '{ seq }')}`,
  }
}
