// Concatenated-frame Zstandard decoding for DSH session logs.
//
// A `session.v4.jsonl.zstd` file is NOT one zstd frame. The persistence backend
// appends ONE frame per event batch (`ZSTD_e_flush`), so a live session carries
// thousands of concatenated frames (measured 2026-11-24: a 9.37 MB session = 6278
// frames / 15232 lines).
//
// Both `zlib.zstdDecompressSync` and the zstd STREAM API decode only the FIRST
// frame and report nothing about the rest — no error, no warning. Decoding a whole
// file that way makes every real session read as a 1-row "empty shell" (the
// 2026-10-14 incident: 302 sessions misjudged as empty and deleted; `docs/02`
// §约束 2 now forbids judging a session empty from a single-frame decode).
//
// So: split on the frame magic, decode each frame on its own, join the text. A
// frame that fails to decode THROWS — skipping it would be the same bug wearing a
// different hat (silent data loss behind a green check).
//
// Only STANDARD frames are split (`28 B5 2F FD`); skippable frames (`0x184D2A5x`) are
// not handled. This host never writes one (`zstd -l` reports `Skips 0`), and if that
// ever changes the skippable bytes land inside the previous frame's slice and trip
// `corrupt session log` — a deliberate fail-closed, not a defect.
//
// Two callers share this one implementation on purpose: the e2e persistence net and
// the static-replay tooling over real session files. Two copies would eventually
// drift, and the wrong copy is invisible until it deletes something.

import { zstdDecompressSync } from 'node:zlib'

const FRAME_MAGIC = [0x28, 0xb5, 0x2f, 0xfd]

const isFrameMagicAt = (bytes, at) =>
  bytes[at] === FRAME_MAGIC[0] &&
  bytes[at + 1] === FRAME_MAGIC[1] &&
  bytes[at + 2] === FRAME_MAGIC[2] &&
  bytes[at + 3] === FRAME_MAGIC[3]

/**
 * Decode a DSH session log buffer.
 *
 * @param {Buffer} bytes raw file contents
 * @returns {{ text: string, frames: number }} `frames` is 0 for a plaintext
 *   `.jsonl` log (no frame magic at offset 0); a compressed log with no decodable
 *   frame is an error, never `frames: 0`.
 */
export const decodeSessionLog = (bytes) => {
  // An uncompressed `.jsonl` log: nothing to split, the file IS the text.
  if (bytes.length < 4 || !isFrameMagicAt(bytes, 0)) return { text: bytes.toString('utf8'), frames: 0 }

  // Frame boundaries are found by the magic itself: zstd frames are concatenated
  // back to back, and Node exposes no length-of-frame API. This is the same stride
  // the workspace's production reader (tools/bin/read-v4-session.mjs) uses.
  const starts = []
  for (let i = 0; i + 3 < bytes.length; i += 1) if (isFrameMagicAt(bytes, i)) starts.push(i)

  const parts = starts.map((start, index) => {
    const end = index + 1 < starts.length ? starts[index + 1] : bytes.length
    try {
      return zstdDecompressSync(bytes.subarray(start, end)).toString('utf8')
    } catch (error) {
      const detail = error && error.message ? error.message : String(error)
      throw new Error(`corrupt session log: frame ${index} at byte ${start} failed (${detail})`)
    }
  })

  // Frames normally end with their own newline; add one only when the tail lacks it
  // so the last JSONL row is never glued to the previous one.
  const text = parts.join('\n')
  return { text: text.endsWith('\n') ? text : `${text}\n`, frames: starts.length }
}
