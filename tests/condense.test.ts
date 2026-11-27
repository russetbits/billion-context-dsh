/**
 * condense — the engine elects the stretch, the model writes the handoff note.
 *
 * Pinned contracts:
 * 1. The elected span is the OLDEST consumed prefix. The fold price is the prompt
 *    tail AFTER the rewrite point ONLY — the fixed request head sits before every
 *    rewrite point and is therefore outside that price (round-10 review B1).
 * 2. The CURRENT injected instruction copy is a wall; a SUPERSEDED copy folds with
 *    the run (it cannot come back, and it is most of a long session residue).
 * 3. The note must carry ## Task and ## Now, or nothing lands.
 * 4. condense and content are mutually exclusive.
 * 5. The result prices the fold: one-time re-prefill plus the resident note. The
 *    price no longer borrows the paired request sample (B1), so the copy carries no
 *    LOWER-bound hedge and no request-overhead claim in either state.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { allLogMessages } from '../src/messages.ts'
import { surfaceSuffixTokens } from '../src/region.ts'
import { recordRequestPair } from '../src/usage.ts'
import { MAX_NOTE_WEIGHT_RATIO, MIN_CONDENSE_SPAN_TOKENS, makeTools, missingCondenseHeadings, pickCondenseSpan, type ToolEnvironment } from '../src/tools.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

/** Two stacked longText copies — one message alone clears the kernel 5000-char floor. */
const bigText = (label: string, seed: number): string => longText(label, seed) + longText(label + 'b', seed + 100)

const NOTE = '## Task\nBuild the token rotation in src/auth/login.ts and get the auth suite green.\n'
  + '## Goal\nShip the token rotation.\n'
  + '## Next\nRun node --import tsx --test tests/auth.test.ts; done at 12/12 pass.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    // This file tests span election, not the working-set guard; the fixtures are
    // small on purpose, so the 15% floor would refuse every fold (see
    // tests/fold-floor.test.ts for the guard itself).
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session, callId: string): never {
  const agent = {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx: { tokenMeter: undefined },
  }
  return {
    callId,
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent,
  } as never
}

function compressTool(env: ToolEnvironment) {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  return definition
}

/** An AGENTS.md injection row in the host shape, always the same scope. */
function appendInstruction(session: Session, digest: string, scope = './\u0000AGENTS.md', path = 'AGENTS.md'): void {
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Instructions v' + digest + ': ' + longText('instr' + digest, 77) }],
    source: {
      kind: 'agent-instructions',
      form: 'instructions',
      baseline: true,
      changes: [{ action: 'set', scope, path, digest }],
    },
  }), { surfaceOp: 'append' })
}

/**
 * A-2 fixture: TWO current instruction scopes, so the surface carries two walls and
 * a fold of the first stretch leaves a second eligible stretch behind.
 */
function twoWallFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  // Big first stretch (net saving positive: 8 big nodes against a 4-node suffix).
  for (let index = 1; index <= 8; index += 1) {
    if (index % 2 === 1) appendUser(session, bigText('u' + index, index))
    else appendAssistant(session, bigText('a' + index, index))
  }
  appendInstruction(session, 'w1')
  // Middle stretch: still eligible AFTER the first fold.
  for (let index = 9; index <= 12; index += 1) {
    if (index % 2 === 1) appendUser(session, bigText('u' + index, index))
    else appendAssistant(session, bigText('a' + index, index))
  }
  appendInstruction(session, 'w2', './\u0000docs/other.md', 'docs/other.md')
  // Small tail: keeps the protected window cheap so the first fold pays back.
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'small tail 1' }], source: { kind: 'user' } }), { surfaceOp: 'append' })
  session.append('assistant/message', { turn: 1, step: 1, stream: [], message: { role: 'assistant', content: [{ type: 'text', text: 'small tail 2' }], provider: 'test-provider', model: 'test-model' } } as never, { surfaceOp: 'append' })
  return session
}

/** Work nodes with two injections of the SAME scope in the middle. */
function workFixture(id: string): { session: Session; staleSeq: number; newestSeq: number } {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, bigText('u1', 1))
  appendAssistant(session, bigText('a2', 2))
  appendUser(session, bigText('u3', 3))
  appendAssistant(session, bigText('a4', 4))
  appendInstruction(session, 'v1')
  const staleSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  appendUser(session, bigText('u5', 5))
  appendAssistant(session, bigText('a6', 6))
  appendInstruction(session, 'v2')
  const newestSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  appendUser(session, bigText('u7', 7))
  appendAssistant(session, bigText('a8', 8))
  appendUser(session, bigText('u9', 9))
  appendAssistant(session, bigText('a10', 10))
  appendUser(session, bigText('u11', 11))
  appendAssistant(session, bigText('a12', 12))
  return { session, staleSeq, newestSeq }
}

function turnOf(env: ToolEnvironment, session: Session) {
  return env.kernel.processTurn({
    messages: allLogMessages(session),
    state: env.store.stateFor(session),
    config: kernelConfigFor({ modelContextLimit: 128000 }),
    tokenCount: 300000,
  })
}

test('condense elects the largest eligible run and prices it as one re-prefill', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-largest')
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c1'))
  const text = (result as { text: string }).text
  // M-1 (2026-11-25): the election may return a multi-span batch (walls stay),
  // landing several blocks in ONE call at ONE rewrite point — the pricing copy
  // below is what this test really pins.
  assert.match(text, /Compressed \d+ block/, 'the elected shape lands')
  assert.match(text, /condense folds \d+t/, 'the result names what the fold removes')
  assert.match(text, /re-prefills \d+t once/, 'the result prices the one-time re-prefill')
  assert.match(text, /pays back (in [\d.]+|immediately)/, 'the result states the payback period')
  assert.match(text, /stays resident after it/, 'the result names the durable note cost')
})

test('A-2: a landed fold reports the candidates that REMAIN', async () => {
  const env = makeEnv()
  const session = twoWallFixture('condense-remaining')
  // F2 (cross-review 2026-11-25): the OLD assertion was `A || B`, which is always
  // true the moment either copy appears and would pass even when BOTH lines went
  // missing in one shape. Branch on the elected shape instead: a single span must
  // name what remains, a multi-span batch that folded everything must name itself.
  const prePick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok(!('reason' in prePick), 'the fixture is foldable: ' + JSON.stringify(prePick))
  const spanCount = (prePick as { spans: ReadonlyArray<unknown> }).spans.length
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c1'))
  const text = (result as { text: string }).text
  assert.match(text, /Compressed \d+ block/, 'the fold lands')
  if (spanCount === 1) {
    assert.match(text, /Still foldable: \d+ range\(s\) remain/, 'a single-span fold names what remains: ' + text)
    assert.doesNotMatch(text, /in \d+ ranges/, 'and does not announce a batch: ' + text)
  } else {
    assert.match(text, /in \d+ ranges/, 'a multi-span batch names its shape: ' + text)
    assert.doesNotMatch(text, /Still foldable: \d+ range\(s\) remain/, 'and does not claim leftovers it folded: ' + text)
  }
})

test('A-2: nothing left to fold adds no noise line', async () => {
  const env = makeEnv()
  // A dedicated small fixture (since 2026-11-25): MIN_CONDENSE_SPAN_TOKENS dropped to
  // 1143, so workFixture's residual bigText row (~2065t) now IS a legal fold and a
  // "Still foldable" line about it is the honest answer, not noise. This fixture folds
  // the whole big stretch and leaves only a guarded instruction wall plus a protected
  // short tail — nothing a condense could take — so silence is the expected output.
  const session = Session.create('condense-no-remaining')
  appendTurn(session, 1)
  appendUser(session, bigText('u1', 1))
  appendAssistant(session, bigText('a2', 2))
  appendUser(session, bigText('u3', 3))
  appendAssistant(session, bigText('a4', 4))
  appendInstruction(session, 'v1')
  appendUser(session, 'short tail 1')
  appendAssistant(session, 'short tail 2')
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c1'))
  const text = (result as { text: string }).text
  assert.match(text, /Compressed 1 block/, 'the fold lands')
  assert.doesNotMatch(text, /Still foldable/, 'a fully folded surface must not carry an empty follow-up hint')
})

test('condense folds the superseded injection copy and never the current one', async () => {
  const env = makeEnv()
  const { session, staleSeq, newestSeq } = workFixture('condense-walls')
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok(!('reason' in pick), 'a prefix is eligible on this fixture: ' + JSON.stringify(pick))
  const spans = (pick as { spans: ReadonlyArray<{ start: number; end: number }> }).spans
  const folds = (seq: number): boolean => spans.some((span) => span.start <= seq && seq <= span.end)
  assert.ok(folds(staleSeq), 'the stale copy seq ' + staleSeq + ' folds inside the elected prefix ' + JSON.stringify(spans))
  assert.ok(!folds(newestSeq), 'the current copy seq ' + newestSeq + ' is a wall, not inside ' + JSON.stringify(spans))
})

test('condense without both headings lands anyway and names the gap', async () => {
  // A missing heading used to refuse the fold. That cost the model a full note
  // rewrite to buy a heading — and output tokens are priced several times above
  // cached input, so the retry loop was the most expensive failure this tool had.
  // The note still beats the raw stretch it replaces, so it lands with a warning.
  const env = makeEnv()
  const { session } = workFixture('condense-headings')
  const result = await compressTool(env).execute({ condense: { summary: 'Explored the auth module; the token rotation lives in src/auth/login.ts and the suite is green.' } } as never, execStub(session, 'c2'))
  const text = (result as { text: string }).text
  assert.match(text, /condense note is missing/, 'the gap is named')
  assert.match(text, /## Task/, 'Task is listed')
  assert.match(text, /## Now/, 'Now is listed')
  assert.match(text, /the fold landed anyway/, 'and it says the fold still happened')
  const wrote = session.snapshotEvents().filter((event) => event.type === 'compaction/summary')
  // M-1: the election may be a multi-span batch — several compaction/summary rows
  // in ONE call is still "the fold LANDS, no rewrite loop".
  assert.ok(wrote.length >= 1, 'the fold LANDS — no rewrite loop (got ' + wrote.length + ' block(s))')
})

test('condense and content are mutually exclusive', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-exclusive')
  const result = await compressTool(env).execute({
    condense: { summary: NOTE },
    content: [{ startSeq: 1, endSeq: 2, summary: NOTE }],
  } as never, execStub(session, 'c3'))
  assert.match((result as { text: string }).text, /mutually exclusive/, 'the engine refuses the ambiguous call')
})

test('a surface with nothing contiguous enough reports a miss with its reason', async () => {
  const env = makeEnv()
  const session = Session.create('condense-miss')
  appendTurn(session, 1)
  appendUser(session, 'short question')
  appendAssistant(session, 'short answer')
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok('reason' in pick, 'nothing is eligible: ' + JSON.stringify(pick))
  assert.match((pick as { reason: string }).reason, /minimum range/, 'the reason names the character floor')
})

test('the note gate accepts the Chinese headings too', () => {
  assert.deepEqual(missingCondenseHeadings('## \u4efb\u52a1\n\u628a\u767b\u5f55\u6001\u6ce8\u5165\u505a\u5b8c\n## \u76ee\u6807\n\u505a\u767b\u5f55\u6001\u6ce8\u5165\n## \u4e0b\u4e00\u6b65\n\u8dd1\u6d4b\u8bd5'), [])
  const missing = missingCondenseHeadings('just a paragraph')
  assert.equal(missing.length, 2, 'both headings are reported missing')
  assert.match(missing[0]!, /## Task/, 'first missing heading is Task')
  assert.match(missing[1]!, /## Now/, 'second missing heading is Now')
})

test('the fold price is the rewrite-point suffix alone, with no overhead hedge', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-no-sample')
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c-no-sample'))
  const text = (result as { text: string }).text
  assert.match(text, /condense folds \d+t/, 'the fold lands')
  assert.match(text, /every token AFTER the rewrite point/, 'the copy names the real price basis')
  assert.doesNotMatch(text, /fixed request overhead/, 'the head is not part of the fold price (B1)')
  assert.doesNotMatch(text, /LOWER bound/, 'no paired sample is needed, so no hedge is printed')
})

test('a paired request sample changes nothing in the fold price copy', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-with-sample')
  // Same call with a paired sample recorded: the fold price never reads it, so the
  // copy must stay identical in shape (the sample only feeds the emergency check).
  recordRequestPair(session, 254_000, 250_000)
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c-measured'))
  const text = (result as { text: string }).text
  assert.match(text, /condense folds \d+t/, 'the fold lands')
  assert.doesNotMatch(text, /LOWER bound|fixed request overhead/, 'a sample neither adds a claim nor a hedge')
})


test('a slow-payback shape elects after M-3 — payback is REPORTED, never gated', async () => {
  // M-3 (2026-11-25): the re-prefill price is no longer a GATE. The shape the old
  // rule refused (re-prefill > what the row removes) now elects, and the ratio stays
  // on the pick for the model to judge — the batch pricing covers the "white fold"
  // risk instead of a ceiling.
  const env = makeEnv()
  const { session } = workFixture('condense-cost-gated')
  // A small eligible row sitting in front of a big resident stretch: the re-prefill
  // exceeds what the row removes, which is exactly the shape the old rule refused.
  appendUser(session, longText('a large resident summary in front of every older row', 9).repeat(120))
  const turn = turnOf(env, session)
  const pick = pickCondenseSpan(session, turn, 0) as { tokens: number; foldCostTokens: number; paybackRequests: number }
  assert.ok(!('reason' in pick), 'the shape elects now: ' + JSON.stringify(pick))
  assert.ok(pick.tokens > 0, 'the elected shape carries volume')
  assert.ok(pick.paybackRequests > 1, 'and its payback really is above one request — still reported: ' + pick.paybackRequests)
})


/**
 * Round-13 review Q5: condense elected a 750-token run, the model wrote a note for it, and the
 * write-time weight cap refused it — a turn and a re-prefill burned on a span that can never
 * carry a handoff. The election must refuse such a span up front and say what to do instead.
 */
test('a span too small to carry a handoff note is never elected', () => {
  const env = makeEnv()
  const session = Session.create('condense-span-floor')
  appendTurn(session, 1)
  // The candidate run: ~3400 chars ≈ 850 tokens — under MIN_CONDENSE_SPAN_TOKENS
  // (1143 since 2026-11-25: TYPICAL_HANDOFF_NOTE_TOKENS re-measured at 400 after the
  // doctrine moved to the Appendix-A four-field briefing; the 2026-11-13 fixture used
  // ~1450 tokens against the old 2572 floor). The assistant reply keeps the run
  // balanced; what follows is the LAST real user turn (host-protected) plus a stub too
  // small for the character minimum, so this stub run IS the largest eligible run —
  // the shape the live session produced at 750 tokens.
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Authentication uses JWT access tokens with 15 minute expiry. '.repeat(57) }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  appendAssistant(session, 'Noted. '.repeat(40))
  appendUser(session, 'The live turn that follows. '.repeat(140))
  appendAssistant(session, 'Acknowledged. '.repeat(20))
  const pick = pickCondenseSpan(session, turnOf(env, session), 0)
  assert.ok('reason' in pick, 'the small run must be refused at election time: ' + JSON.stringify(pick))
  assert.match((pick as { reason: string }).reason, /capacity gate/, 'the refusal names the gate that failed')
  assert.match((pick as { reason: string }).reason, new RegExp('under the ~' + MIN_CONDENSE_SPAN_TOKENS + '-token minimum'), 'and gives the size it needed: ' + (pick as { reason: string }).reason)
  assert.match((pick as { reason: string }).reason, /NO fallback/, 'the election says it has no fallback shape')
})
