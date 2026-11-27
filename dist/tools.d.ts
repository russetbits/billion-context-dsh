/**
 * M3 — the three model tools: compress / recall / acp_status, registered
 * through `ctx.tools` (defineTool).
 *
 * compress is the heart of ACP: the model writes the summary and the tool
 * lands it as a durable surface replacement (no second LLM summarization
 * call). recall is the ONE way back into the log — either a SEARCH over the
 * block summaries, the shadowed originals and the still-visible surface
 * (`query`), or a READ of original bytes (`blockId` / `seq`); the two forms
 * are mutually exclusive, so the model face carries one schema instead of two.
 * acp_status reports the block ledger and pressure.
 * @module billion-context-dsh/tools
 */
import { type ToolDefinition } from '@deepseek-ai/dsh-tools';
import { type CompressionCore, type SearchDoc } from './kernel/index.ts';
import type { Agent } from '@deepseek-ai/dsh-agent';
import type { Session } from '@deepseek-ai/dsh-session';
import type { AcpStateStore } from './state.ts';
import { type KernelConfigInput } from './config.ts';
import { type AcpWindow } from './window.ts';
import { type SeqCompressibleRange, type FoldPricing } from './region.ts';
import { type ResolvedPrompts } from './prompts.ts';
import { type ResolvedCallStyle } from './ptc.ts';
import type { SettingsCommandSurface } from './settings.ts';
import type { PresetName } from './presets.ts';
export interface ToolEnvironment extends KernelConfigInput {
    readonly kernel: CompressionCore;
    readonly store: AcpStateStore;
    /** Display-only: the named preset that produced the nudge thresholds above, if any (`/acp-prune status` names it). Never read by the kernel path. */
    readonly preset?: PresetName;
    /** Resolve the effective context window for an agent (optional: status falls back to modelContextLimit). */
    readonly windowFor?: (agent: Agent) => Promise<AcpWindow>;
    /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
    readonly prompts?: ResolvedPrompts;
    /**
     * `textOverrides` diagnostics (unknown ids / oversized values). Rendered as one
     * self-check line in `acp_status` — a composition-row config otherwise has no
     * visible surface where a typo could show up. Absent = nothing to report.
     */
    readonly textWarnings?: readonly string[];
    /** How the model can REACH these tools in the current step (src/ptc.ts). The
     *  engine swaps in the PTC variant of `prompts` together with this, so the two
     *  never disagree; a stand-in env (tests, plain compositions) defaults to
     *  `native` and keeps the direct-call copy byte-identical. */
    readonly toolCallStyle?: ResolvedCallStyle;
    /**
     * Call ids of compress invocations that created a durable block. The engine
     * listens for the matching `tool/result` and hides the call/result pair from
     * the surface, preventing the compaction summary from sitting between them
     * (strict providers reject that sequence with HTTP 400).
     */
    readonly compressCallIdsToHide?: Set<string>;
    /**
     * Read/write access to the runtime settings layer for `/acp-prune config`.
     * Absent surfaces (never expected — the engine always builds one) would
     * degrade the command to advice text.
     */
    readonly settingsCommand?: SettingsCommandSurface;
}
/**
 * Resolve the effective context window for a tool or command run: probe the
 * agent's real window via `windowFor` when provided, otherwise fall back to
 * the environment's `modelContextLimit`. Shared by the compress and
 * acp_status tool handlers and the `/acp-prune` command so the resolution logic
 * lives in exactly one place (issue #63 — the tools used the 128K fallback
 * for pressure decisions even when auto-detection had found a larger window).
 */
export declare function resolveEffectiveWindow(env: ToolEnvironment, agent: Agent): Promise<AcpWindow>;
export declare const compressParameters: {
    readonly arguments: {
        readonly type: 'json';
        readonly description: string | undefined;
    };
    readonly topic: {
        readonly type: 'string';
        readonly description: string | undefined;
    };
    readonly content: {
        readonly type: 'array';
        readonly description: string | undefined;
        readonly items: {
            readonly type: 'object';
            readonly properties: {
                readonly startSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: string | undefined;
                    }, {
                        readonly type: 'string';
                        readonly description: string | undefined;
                    }];
                };
                readonly endSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: string | undefined;
                    }, {
                        readonly type: 'string';
                        readonly description: string | undefined;
                    }];
                };
                readonly summary: {
                    readonly type: 'string';
                    readonly required: true;
                    readonly description: string | undefined;
                };
                readonly topic: {
                    readonly type: 'string';
                    readonly description: string | undefined;
                };
                readonly verifiedReadings: {
                    readonly type: 'array';
                    readonly items: {
                        readonly type: 'string';
                    };
                    readonly description: string | undefined;
                };
            };
            readonly additionalProperties: false;
        };
    };
    readonly force: {
        readonly type: 'boolean';
        readonly description: string | undefined;
    };
    readonly condense: {
        readonly type: 'object';
        readonly description: string;
        readonly properties: {
            readonly summary: {
                readonly type: 'string';
                readonly required: true;
                readonly description: string;
            };
            readonly topic: {
                readonly type: 'string';
                readonly description: string | undefined;
            };
            readonly force: {
                readonly type: 'boolean';
                readonly description: string | undefined;
            };
            readonly verifiedReadings: {
                readonly type: 'array';
                readonly items: {
                    readonly type: 'string';
                };
                readonly description: string | undefined;
            };
        };
        readonly additionalProperties: false;
    };
};
/** Exported so the ref-width contract test can pin it without a 100K-message fixture. */
export declare function mnRefIndex(value: string): number | null;
/**
 * Pure gate helpers for the compress tool's CURRENT-instruction-row rejection.
 *
 * Decision history (issue #71 review): the first draft only WARNED when a
 * manual compress range swallowed a current injected row (F7), because the
 * compression is safe and self-healing. The owner reversed that during PR1
 * review: compressing a CURRENT row has NO legitimate outcome — the host
 * re-injects the newest AGENTS.md copy unconditionally the moment it leaves
 * the surface (presence gate, deepseek-harness
 * packages/context/agent-instructions/src/index.ts:137/:163), so the tokens
 * come straight back and the call is pure waste — and a hard reject keeps the
 * manual path consistent with the system-side GC's iron rule (PR2: never
 * clear a group's newest row). STALE copies stay compressible: removing them
 * while the newest stays visible is the actual cleanup and triggers no
 * re-injection. Skill-catalog rows joined the guard set for a DIFFERENT
 * reason (issue #185): they are not self-healing at all — the host resends
 * the catalog only when its digest changes, so a folded catalog stays gone
 * until the catalog actually changes. The range table
 * (buildCompressibleSeqRanges) never offers these rows, so the gate only fires
 * on hand-built ranges.
 *
 * `guardedRowsInSpan` is the overlap probe. It takes the POSITIONAL span the
 * transaction will actually shadow (`shadowedSeqsOf`), never a numeric
 * `start <= seq <= end` interval: the surface is locally non-monotonic after
 * earlier replacements (a checkpoint seq spliced ahead of older residual
 * nodes), so a tier-2 distill of two checkpoints can carry a CURRENT
 * instruction row numerically inside its edges while the sliced span excludes
 * it — the interval probe rejected exactly the call the nudge hands the model
 * (issue #71 review B1). Probing the slice also keeps guard and effect in
 * agreement: `shadowedSeqsOf` is what the transaction prices and
 * `assertProvenance` verifies.
 * `protectedRowRejectionNote` renders the rejection the model sees: it names
 * the offending seqs AND the compressible slices left in the span, so the model
 * can re-cut (or split into two calls) instead of retrying the same call.
 * `guardedSurfaceSeqsOf` supplies the protected set.
 */
export declare function guardedRowsInSpan(guarded: ReadonlySet<number>, shadowed: readonly number[]): number[];
/**
 * Maximal runs of a shadowed surface slice that contain NO protected row.
 *
 * The slice is POSITIONAL (`shadowedSeqsOf`), never a numeric interval: a checkpoint
 * node carries a seq far higher than its neighbours, so `>=`/`<=` arithmetic would
 * cut the wrong side (the same reason `guardedRowsInSpan` scans a set). A protected
 * row at an EDGE simply yields one piece; a span that is nothing but protected rows
 * yields none, which is the caller's signal to keep the rejection.
 */
export declare function splitAroundProtected(shadowed: readonly number[], hits: readonly number[]): number[][];
/**
 * Where one surviving piece sits relative to the cut points, in words a reader of
 * the block can act on: "BEFORE it", "AFTER them", or "between protected rows 2
 * and 3". Positions come from the POSITIONAL slice, never from seq values.
 */
export declare function splitPiecePosition(shadowedSpan: readonly number[], protectedSeqs: readonly number[], piece: readonly number[]): string;
/**
 * The mechanical coverage line a CUT block carries above the model's note.
 *
 * A split stores the model's note in EVERY piece, and that note was written for the
 * WHOLE span (seqs `spanStart..spanEnd`). Without this line a later `recall({ blockId })`
 * reads a note describing content the block does not hold, and — worse — takes it for
 * the block's complete record while the rest sits in a sibling block. The line makes
 * both facts explicit, costs one short row, and is used ONLY on cut blocks: a range
 * with no protected row inside stores the note byte-identically.
 */
export declare function splitCoverageLine(index: number, total: number, spanStart: number, spanEnd: number, protectedCount: number, where: string): string;
/**
 * Advisory note for a plain range that would fold a LIVE block's checkpoint
 * carrier (see `liveCheckpointCarriersInSpan`). Names the block ids the model
 * can use instead — a block-ref boundary is the call that legitimately folds
 * carriers (tier 2/3 distillation).
 */
export declare function liveCarrierRejectionNote(start: number, end: number, carriers: readonly {
    seq: number;
    kernelBlockId: string;
}[]): string;
export declare function protectedRowRejectionNote(start: number, end: number, hits: readonly number[], shadowed: readonly number[], session?: Session): string;
/**
 * Kernel ref for one RESOLVED range edge's surface node (issue #155).
 * Resolved edges are tool-pairing-balanced and anchorable, but NO LONGER
 * guaranteed to carry a bare-`${seq}` ref: a multi-tool-call assistant
 * projects to `${seq}#${callId}` sub-ids, and an empty tool result projects
 * to nothing. Three tiers, in order:
 *
 *  1. the bare-seq id (user turns, single-call assistants, text-bearing results);
 *  2. the node's sub-ids in projection (= content) order — FIRST for a start
 *     edge (the kernel must consume the whole node from its first sub-message),
 *     LAST for an end edge (unreachable in practice: a node with open calls
 *     makes every cut after it unbalanced, so it can never be a resolved end
 *     edge);
 *  3. the nearest message-bearing LIVE node between this edge and the opposite
 *     edge (inclusive), walked along the SURFACE — never raw-id space, where
 *     shadowed nodes keep their refs and would anchor onto already-compressed
 *     messages. Walking inward only also guarantees the returned ref names a
 *     message the transaction actually shadows; a span whose interior carries
 *     no message at all (two adjacent empty results) yields undefined and the
 *     caller raises the existing "no assigned ref" error.
 */
export declare function edgeRefForSeq(session: Session, byRaw: Readonly<Record<string, string>>, seq: number, role: 'start' | 'end', oppositeSeq: number): string | undefined;
/**
 * One RANGE a condense submits — NOT one block (redesigned 2026-11-24).
 *
 * A prefix fold distils every active block the foldable prefix carries AND folds
 * the gaps between them in the SAME call, so the elected unit is the prefix, not
 * the individual block. `start`/`end` are surface seqs of the range's edges;
 * `endBlock` is the kernel block id when the END edge sits on that block's
 * checkpoint carrier (a `bN` end makes the kernel DISTIL the block, tier 2/3),
 * and `carriers` names every active block the range distils.
 */
export interface CondenseSpan {
    readonly start: number;
    readonly end: number;
    readonly endBlock: string | null;
    readonly carriers: ReadonlyArray<string>;
    readonly tokens: number;
}
/** The foldable PREFIX a condense submits, priced as ONE rewrite. */
export interface CondensePick {
    /**
     * The ranges to submit, oldest-first. ONE range carries the whole prefix; a
     * SECOND, gap-only range appears when the prefix ends on a gap that sits after
     * the last carrier (`condenseRangesOf` explains why the two cannot merge).
     */
    readonly spans: ReadonlyArray<CondenseSpan>;
    readonly tokens: number;
    /** Note tokens that stay resident after the fold — one note per submitted range. */
    readonly noteTokens: number;
    readonly keptTokens: number;
    readonly floorTokens: number;
    readonly belowFloor: boolean;
    /**
     * Whether ONE eligible row folded ALONE still leaves the floor. A refusal that
     * tells the model to fold bigger when nothing bigger is eligible is a loop
     * generator: every wider span is guarded (cross-review Q3).
     */
    readonly aboveFloorAvailable: boolean;
    /**
     * How many surface rows the guards refuse (current instruction copies, the
     * newest skill catalog, the active user turn). A session that is wide but
     * walled is NOT stuck — telling it to fold wider sends it into the guard
     * (cross-review Q4).
     */
    readonly guardedRowCount: number;
    /** How many eligible ranges the geometry produced at all (0 = nothing clears the kernel minimum). */
    readonly eligibleRangeCount: number;
    /** Active blocks this fold distils (summed over `spans`). */
    readonly carrierCount: number;
}
export interface CondenseMiss {
    readonly reason: string;
    readonly candidates: SeqCompressibleRange[];
}
/**
 * The foldable PREFIX of the current surface, for condense.
 *
 * Geometry is the range table's own (buildCompressibleSeqRanges, rule 3), so a
 * condense can never shadow a row the durable surface forbids; the timeline adds
 * the ACTIVE block carriers, which that function deliberately excludes (it
 * flushes on every checkpoint). The election walks the merged timeline
 * oldest-first and stops at the first wall — a guarded row (current instruction
 * copy, newest skill catalog, active user turn) or the working-set floor. It
 * never SKIPS a wall: a prefix with a hole in it could not be submitted as one
 * range, which is the whole point of the design.
 *
 * How long the prefix turns out to be is decided by the guards, not by the block
 * count: on a real 87-node session (4 carriers on the surface, 41K tokens) the
 * oldest foldable prefix crossed exactly ONE carrier, because the last real user
 * turn sat near the front and every later carrier was inside the protected window.
 * A 50-block synthetic surface (tests/prefix-fold.test.ts P-8) crosses all 50. So
 * the useful question is never "how many blocks exist" but "how many of them are on
 * the OLD side of the guards" — `carrierCount` is what the pick reports.
 *
 * What the fold costs is REPORTED (`CondensePick.tokens` against the note), never
 * an election gate: the engine cannot predict how many requests a session has
 * left, and a cost model it cannot validate only removes folds a real session
 * needs. The floor check is `belowWorkingSetFloor` (arXiv 2611.05306 Appendix D),
 * the note-size check happens at write time (`noteWeightRatio`).
 */
export declare function pickCondenseSpan(session: Session, turn: ReturnType<CompressionCore['processTurn']>, preserveRecent: number, foldPricing?: FoldPricing): CondensePick | CondenseMiss;
/**
 * Refusal copy for a below-floor condense. The THREE branches ARE the point
 * (cross-review Q4): two-branch copy conflated "nothing wider is above the floor"
 * with "nothing is eligible at all", so a session whose reclaimable area is merely
 * FRAGMENTED by the guards was told to stop compressing — and a session with
 * nothing to fold was told to widen into a guard. Both are loops. Exported so
 * every branch is testable without building three fixtures.
 */
export declare function condenseFloorRefusal(pick: CondensePick, floor: number): string;
export declare function missingCondenseHeadings(note: string): string[];
export declare function condenseNoteGaps(note: string): string[];
/**
 * Does the note actually COMPRESS? Three cross-reviews killed the earlier
 * post-fold duplication warning: it fires AFTER the fold lands, so obeying it
 * means rewriting the note and paying a second full re-prefill — costlier to act
 * on than to ignore — and lexical shingles flag exactly the verbatim identifiers,
 * paths and error strings a good note SHOULD carry.
 *
 * The replacement is deterministic, pre-fold and language-independent: compare
 * the note’s weight against the stretch it replaces. A note weighing a third of
 * the folded span is not a briefing, it is a second copy that stays resident in
 * every later request — the fold never earns its one-time re-prefill back.
 */
export declare const MAX_NOTE_WEIGHT_RATIO = 0.35;
/**
 * What a handoff note actually weighs on this host, measured 2026-11-15: a real model-written
 * `condense` note stored ~915 tokens while missing FOUR of the six Appendix-G fields. That is the
 * number the span floor has to clear, not a guess: a floor derived from a note the model will
 * never write lets a span into the election the weight gate then refuses (round-15 review Q7.1:
 * at `MAX_NOTE_WEIGHT_RATIO` a 2000-token span admits only ~700 stored tokens, so every span
 * between 2000 and ~2614 tokens was elected, paid for by a turn of writing, and then refused).
 */
export declare const TYPICAL_HANDOFF_NOTE_TOKENS = 900;
/**
 * Hard ceiling on a stored handoff note, independent of the weight ratio.
 *
 * The ratio alone does not bound a note: at `MAX_NOTE_WEIGHT_RATIO` a 84K stretch
 * admits 29K tokens, so it never fires on the folds that matter. Measured on a
 * real 13-block session (2026-11-24) the notes averaged ~1.78K and peaked at 5.2K
 * against a 900-token assumption — the stored area was underestimated by ~4K per
 * block, and 23.1K of a 161K surface was handoff prose. The paper's control
 * harness reports a 185-token median with p97.5 <= 2K, so a note past this
 * ceiling is a narrative, not the "brief for the work ahead" the prompt asks for.
 *
 * Set above the 900-token typical so a good note is never clipped, and low enough
 * to catch the 3-5K outliers that made the stored area grow.
 */
export declare const MAX_HANDOFF_NOTE_TOKENS = 1500;
/**
 * Cut a note down to a token ceiling, keeping whole LINES from the top.
 *
 * A note written to the skeleton puts its load-bearing headings first (Task,
 * Now, then Pointers), so dropping the tail drops narrative — which is exactly
 * what arXiv 2611.05306 Appendix A says to leave out. Truncating costs nothing;
 * refusing costs the model a full rewrite, and output tokens are priced well
 * above cached input.
 */
export declare function truncateNoteToCeiling(note: string, maxTokens: number): {
    note: string;
    droppedTokens: number;
};
/**
 * Smallest stretch condense will elect: the smallest span that can carry a note this host's
 * models actually write inside the weight cap — `TYPICAL_HANDOFF_NOTE_TOKENS / MAX_NOTE_WEIGHT_RATIO`,
 * so a copy or ratio change follows it instead of drifting. The kernel minimum (`minCompressRange`)
 * is a CHARACTER threshold (5000 chars ≈ 1250 tokens), which filters nothing; the real floor is
 * the note's weight (round-13 review Q5: a 750-token election, then the weight gate).
 */
export declare const MIN_CONDENSE_SPAN_TOKENS: number;
/**
 * Note weight, priced as the note is STORED: the summary frame prefix the checkpoint node
 * carries is part of what stays resident, so the ratio must count it or a note just under 35%
 * raw would exceed 35% once stored (round-13 review Q1). `storedTokens` lets a caller that
 * already counted the stored form pass it in instead of counting twice.
 */
export declare function noteWeightRatio(note: string, spanTokens: number, storedTokens?: number): number;
export declare function noteWeightRejection(note: string, spanTokens: number, storedTokens?: number): string;
/**
 * Build the unified SearchDoc[] from the log: one block doc per ledger entry
 * (ref = compactionId, so `recall({ blockId })` closes the loop) plus one
 * message doc per shadowed ORIGINAL (expanded through distilled parents; each
 * seq is claimed by the earliest/innermost block that covered it, mirroring
 * pi's owner map — recall on that block recovers the original).
 * Cached per log snapshot (see searchDocsCache). Exported for the issue #133
 * regression tests (not part of the public API — index.ts re-exports only).
 */
export declare function buildSearchDocs(session: Session): SearchDoc[];
/**
 * The compress parameter sheet with the model-facing copy replaced by the
 * RESOLVED text. Returns the shared constant object untouched (same reference)
 * when nothing was overridden, so the default path allocates nothing and
 * `tests/g5-governance.test.ts` keeps reading the shipped sheet.
 *
 * Two layers ride here: the TOOL-level `condense` description (`tool.condense`)
 * and the 14 PARAMETER-level texts (`tool.param.compress.*`).
 */
export declare function compressParametersFor(prompts: ResolvedPrompts): {
    readonly arguments: {
        readonly type: 'json';
        readonly description: string | undefined;
    };
    readonly topic: {
        readonly type: 'string';
        readonly description: string | undefined;
    };
    readonly content: {
        readonly type: 'array';
        readonly description: string | undefined;
        readonly items: {
            readonly type: 'object';
            readonly properties: {
                readonly startSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: string | undefined;
                    }, {
                        readonly type: 'string';
                        readonly description: string | undefined;
                    }];
                };
                readonly endSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: string | undefined;
                    }, {
                        readonly type: 'string';
                        readonly description: string | undefined;
                    }];
                };
                readonly summary: {
                    readonly type: 'string';
                    readonly required: true;
                    readonly description: string | undefined;
                };
                readonly topic: {
                    readonly type: 'string';
                    readonly description: string | undefined;
                };
                readonly verifiedReadings: {
                    readonly type: 'array';
                    readonly items: {
                        readonly type: 'string';
                    };
                    readonly description: string | undefined;
                };
            };
            readonly additionalProperties: false;
        };
    };
    readonly force: {
        readonly type: 'boolean';
        readonly description: string | undefined;
    };
    readonly condense: {
        readonly type: 'object';
        readonly description: string;
        readonly properties: {
            readonly summary: {
                readonly type: 'string';
                readonly required: true;
                readonly description: string;
            };
            readonly topic: {
                readonly type: 'string';
                readonly description: string | undefined;
            };
            readonly force: {
                readonly type: 'boolean';
                readonly description: string | undefined;
            };
            readonly verifiedReadings: {
                readonly type: 'array';
                readonly items: {
                    readonly type: 'string';
                };
                readonly description: string | undefined;
            };
        };
        readonly additionalProperties: false;
    };
};
/** `recall` 的 7 个参数文案(`tool.param.recall.*`)。 */
export declare function recallParametersFor(prompts: ResolvedPrompts): {
    readonly query: {
        readonly type: 'string';
        readonly description: string | undefined;
    };
    readonly blockId: {
        readonly type: 'string';
        readonly description: string | undefined;
    };
    readonly seq: {
        readonly type: 'integer';
        readonly description: string | undefined;
    };
    readonly chars: {
        readonly type: 'integer';
        readonly description: string | undefined;
    };
    readonly maxResults: {
        readonly type: 'integer';
        readonly description: string | undefined;
    };
    readonly offset: {
        readonly type: 'integer';
        readonly description: string | undefined;
    };
    readonly pageLimit: {
        readonly type: 'integer';
        readonly description: string | undefined;
    };
};
/** `acp_status` 的 4 个参数文案(`tool.param.acpStatus.*`)。 */
export declare function statusParametersFor(prompts: ResolvedPrompts): {
    detail: {
        type: 'string';
        enum: readonly ['overview', 'compressed', 'visible', 'messages'];
        description: string | undefined;
    };
    tool: {
        type: 'string';
        description: string | undefined;
    };
    sort: {
        type: 'string';
        enum: readonly ['size', 'time', 'tool', 'age'];
        description: string | undefined;
    };
    limit: {
        type: 'integer';
        description: string | undefined;
    };
};
/**
 * Build the three ACP model tools bound to one engine.
 *
 * The four MAIN tools are `textOverrides`-addressable today: `compress`/`recall`/
 * `acp_status` through `prompts.tools.*` and the compress sheet's `condense`
 * parameter through `compressesParametersFor` above. TODO(textOverrides): the
 * remaining ~26 parameter `description` strings (compress content items, force,
 * verifiedReadings, recall's query/blockId/seq/chars/maxResults/offset/pageLimit,
 * acp_status's detail/tool/sort/limit) are still hard-coded — wiring them means
 * turning `compressParameters`/`recallParameters`/`statusParameters` into
 * prompt-derived builders, which collides with the in-flight content-path work in
 * this file; left as a deliberate follow-up rather than a half-migration.
 */
export declare function makeTools(env: ToolEnvironment): ToolDefinition[];
