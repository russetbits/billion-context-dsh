// Load a DSH session log into a replayable in-memory Session.
//
// Three steps make a `.v4.jsonl.zstd` file replayable, and all three are load-bearing:
//
//   1. decode it as CONCATENATED zstd frames (scripts/lib/zstd-frames.mjs) — a
//      one-shot decode yields only the header frame;
//   2. drop the leading `type:'session'` row — `Session.create` wants event rows
//      only, and the header carries keys its envelope validator rejects
//      (`version`/`id`/`createdAt`) plus no `seq`;
//   3. flatten `sourceEventSeqs` range entries with the host's own
//      `decodeSeqRanges` — the durable encoding allows `number | [start, end]`,
//      but the seed validator checks each entry with `isEventSeq` (numbers only)
//      and never decodes. A real 15349-event log carries 55 such entries across
//      32 checkpoint events; without this step `Session.create` throws
//      `session event "user/message" sourceEventSeqs must densely contain
//      non-negative safe integers` at the first one.
//
// A session file that is still being written drifts under the reader (measured:
// nodes 143 → 146 → 147 across three reads of one live file). Reading is therefore
// BOUNDED by two stats — before and after — and a change fails loudly instead of
// silently returning a mixed snapshot. Pass a `--copy-to` destination (the CLI does
// it for you) when replaying anything a live process may still be appending to.

import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import { Session, decodeSeqRanges } from '@deepseek-ai/dsh-session'
import { decodeSessionLog } from './zstd-frames.mjs'

/** The session header row type; it is storage metadata, never an event. */
export const SESSION_HEADER_TYPE = 'session'

const driftError = (filePath) =>
  new Error(`snapshot drifted while reading (mtime/size changed): ${filePath} — copy the file first (--copy-to <dir>)`)

const statIdentity = (filePath) => {
  const stat = statSync(filePath)
  return { mtimeMs: stat.mtimeMs, bytes: stat.size }
}

/**
 * Flatten `sourceEventSeqs` range entries. Returns the normalized events and how
 * many range entries were expanded (0 when the log already stored flat seqs).
 */
const normalizeSeqRanges = (events) => {
  let rangeEntries = 0
  const normalized = events.map((event) => {
    if (!Array.isArray(event.sourceEventSeqs)) return event
    rangeEntries += event.sourceEventSeqs.filter((entry) => Array.isArray(entry)).length
    return { ...event, sourceEventSeqs: decodeSeqRanges(event.sourceEventSeqs) }
  })
  return { events: normalized, rangeEntries }
}

/**
 * Read one session log and rebuild its live surface.
 *
 * @param {string} filePath `.jsonl.zstd` (multi-frame) or plain `.jsonl` log
 * @returns {{
 *   filePath: string, events: object[], surface: Session, nodes: number[],
 *   frames: number, rows: number, headerRows: number, bytes: number,
 *   mtimeMs: number, sha1: string, rangeEntries: number, seqGaps: number
 * }}
 * @throws when the log will not decode, its seqs are not contiguous from 0, the
 *   seed is rejected, or the file changed between the bounding stats.
 */
export const loadSessionSnapshot = (filePath) => {
  const before = statIdentity(filePath)
  const bytes = readFileSync(filePath)
  const sha1 = createHash('sha1').update(bytes).digest('hex')

  let decoded
  try {
    decoded = decodeSessionLog(bytes)
  } catch (error) {
    // A half-written frame and a mid-read append look alike at the byte level; the
    // second stat is what tells them apart, so a drift report wins over "corrupt".
    const failedAt = statIdentity(filePath)
    if (failedAt.mtimeMs !== before.mtimeMs || failedAt.bytes !== before.bytes) throw driftError(filePath)
    throw error
  }

  const rows = decoded.text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line))
  const headerRows = rows[0]?.type === SESSION_HEADER_TYPE ? 1 : 0
  const rawEvents = headerRows === 1 ? rows.slice(1) : rows

  const { events, rangeEntries } = normalizeSeqRanges(rawEvents)

  const gaps = []
  events.forEach((event, index) => {
    if (event.seq !== index) gaps.push({ index, got: event.seq })
  })
  if (gaps.length > 0) {
    const detail = gaps.slice(0, 5).map((gap) => `index ${gap.index} has seq ${gap.got}`).join('; ')
    throw new Error(`session log seqs are not contiguous from 0 (${gaps.length} gap(s)): ${detail}`)
  }

  const session = Session.create(`replay-${sha1.slice(0, 12)}`, events)

  const after = statIdentity(filePath)
  if (after.mtimeMs !== before.mtimeMs || after.bytes !== before.bytes) throw driftError(filePath)

  return {
    filePath,
    events,
    surface: session,
    nodes: session.surface.nodes,
    frames: decoded.frames,
    rows: rows.length,
    headerRows,
    bytes: bytes.length,
    mtimeMs: before.mtimeMs,
    sha1,
    rangeEntries,
    seqGaps: 0,
  }
}
