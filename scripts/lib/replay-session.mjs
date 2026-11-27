#!/usr/bin/env node
// Replay a DSH session log statically: report the rebuilt surface and its carriers.
//
// Usage:
//   node --import tsx scripts/lib/replay-session.mjs <file> --count
//   node --import tsx scripts/lib/replay-session.mjs <file> --carriers
//   node --import tsx scripts/lib/replay-session.mjs <file> --count --json
//   node --import tsx scripts/lib/replay-session.mjs <file> --carriers --copy-to .tmp/snapshots
//
// `--count` needs only the session package; `--carriers` also loads the plugin's own
// `blockRegistry` (src/region.ts), which is why `--import tsx` is the supported line.
//
// Carrier definitions are NOT interchangeable, and the wrong one produced a 5x
// over-count during the probe that found this:
//   * surface nodes carrying `surfaceOp: { op: 'replace' }` — NOT carriers. On a real
//     session that set held 5 nodes and only 1 was a block summary; the other 4 were a
//     system-prompt replacement, two developer injections and a superseded nudge.
//   * `blockRegistry(session).filter(active)` — the ledger view: blocks no later block
//     distilled. Real session: 72 of 84.
//   * active block whose `summarySeq` is still on the live surface — the carrier you
//     can actually SEE. Real session: 1, because 71 active blocks had their summary
//     node folded away by later plain ranges (65 of them provably inside a later
//     replace span).
// The default `--carriers` verdict is the third one.

import { copyFileSync, mkdirSync } from 'node:fs'
import { basename, join, resolve } from 'node:path'
import { loadSessionSnapshot } from './session-snapshot.mjs'

const RANGE_HEAD = (seqs) => {
  const list = seqs.filter((seq) => typeof seq === 'number')
  if (list.length === 0) return 'empty'
  return `seqs ${Math.min(...list)}..${Math.max(...list)}`
}

const usage = () => {
  console.error('usage: node --import tsx scripts/lib/replay-session.mjs <file> [--count] [--carriers] [--json] [--copy-to <dir>]')
  process.exit(2)
}

const argv = process.argv.slice(2)
const copyToIndex = argv.indexOf('--copy-to')
const copyTo = copyToIndex === -1 ? null : argv[copyToIndex + 1]
// Drop the flag AND its value before splitting flags from positionals, or the
// destination directory counts as the session file.
const rest = copyToIndex === -1 ? argv : argv.filter((_, index) => index !== copyToIndex && index !== copyToIndex + 1)
const flags = new Set(rest.filter((arg) => arg.startsWith('--')))
const positional = rest.filter((arg) => !arg.startsWith('--'))
if (positional.length !== 1 || flags.size === 0) usage()
if (copyToIndex !== -1 && (copyTo === undefined || copyTo.startsWith('--'))) usage()
const file = positional[0]

// A live session file drifts under the reader; copy first when the caller asks, and
// loadSessionSnapshot still bounds every read with two stats either way.
let target = resolve(file)
let copiedTo = null
if (copyTo !== null) {
  mkdirSync(copyTo, { recursive: true })
  copiedTo = join(resolve(copyTo), basename(file))
  copyFileSync(target, copiedTo)
  target = copiedTo
}

let snapshot
try {
  snapshot = loadSessionSnapshot(target)
} catch (error) {
  console.error(`FAILED: ${error && error.message ? error.message : String(error)}`)
  process.exit(1)
}

const base = {
  file: resolve(file),
  readFrom: target,
  copiedTo,
  sha1: snapshot.sha1,
  bytes: snapshot.bytes,
  mtimeMs: snapshot.mtimeMs,
  mtime: new Date(snapshot.mtimeMs).toISOString(),
  frames: snapshot.frames,
  rows: snapshot.rows,
  headerRows: snapshot.headerRows,
  events: snapshot.events.length,
  nodes: snapshot.nodes.length,
  nodeSpan: RANGE_HEAD(snapshot.nodes),
  rangeEntriesFlattened: snapshot.rangeEntries,
  seqGaps: snapshot.seqGaps,
}

const wantsCarriers = flags.has('--carriers')

const carrierReport = async () => {
  let blockRegistry
  try {
    ;({ blockRegistry } = await import('../../src/region.ts'))
  } catch (error) {
    throw new Error(`--carriers needs the plugin module (src/region.ts); re-run with: node --import tsx <script> (${error && error.message ? error.message : String(error)})`)
  }
  const registry = blockRegistry(snapshot.surface)
  const active = registry.filter((entry) => entry.active)
  const onSurface = new Set(snapshot.nodes)
  const bySeq = new Map(snapshot.events.map((event) => [event.seq, event]))
  const replaceNodes = snapshot.nodes.filter((seq) => bySeq.get(seq)?.surfaceOp?.op === 'replace')
  const summaries = new Set(active.map((entry) => entry.summarySeq).filter((seq) => seq !== null))
  // Active blocks whose summary node is still visible: the carrier you can SEE.
  const carriers = active.filter((entry) => entry.summarySeq !== null && onSurface.has(entry.summarySeq))
  return {
    carriers: carriers.length,
    carrierRefs: carriers.map((entry) => `${entry.kernelBlockId}@${entry.summarySeq}`),
    activeBlocks: active.length,
    totalBlocks: registry.length,
    activeOffSurface: active.length - carriers.length,
    replaceOpNodes: replaceNodes.length,
    replaceOpNodesThatAreBlockSummaries: replaceNodes.filter((seq) => summaries.has(seq)).length,
    tiers: registry.reduce((acc, entry) => ({ ...acc, [entry.tier]: (acc[entry.tier] ?? 0) + 1 }), {}),
  }
}

if (wantsCarriers) {
  try {
    base.carriers = await carrierReport()
  } catch (error) {
    console.error(`FAILED: ${error && error.message ? error.message : String(error)}`)
    process.exit(1)
  }
}

if (flags.has('--json')) {
  console.log(JSON.stringify(base, null, 2))
  process.exit(0)
}

console.log(`file: ${base.file}`)
if (base.copiedTo !== null) console.log(`copied to: ${base.copiedTo} (read from the copy)`)
console.log(`sha1: ${base.sha1}`)
console.log(`bytes: ${base.bytes}  mtime: ${base.mtime}`)
console.log(`frames: ${base.frames}  rows: ${base.rows} (${base.headerRows} header + ${base.events} events)`)
console.log(`events: ${base.events}  seq gaps: ${base.seqGaps}  range entries flattened: ${base.rangeEntriesFlattened}`)
console.log(`nodes: ${base.nodes}  (${base.nodeSpan})`)

if (wantsCarriers) {
  const c = base.carriers
  console.log('')
  console.log(`carriers (blockRegistry active AND summary node on surface): ${c.carriers}`)
  console.log(`blockRegistry active: ${c.activeBlocks}   total: ${c.totalBlocks}   tiers: ${JSON.stringify(c.tiers)}`)
  console.log(`active blocks whose summary left the surface: ${c.activeOffSurface}`)
  console.log(`NOT the carrier count — surface nodes carrying surfaceOp 'replace': ${c.replaceOpNodes} (${c.replaceOpNodesThatAreBlockSummaries} of them are block summaries)`)
  if (c.carrierRefs.length > 0) console.log(`carrier refs: ${c.carrierRefs.join(', ')}`)
}
