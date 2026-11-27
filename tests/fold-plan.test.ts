/**
 * A-9（2026-11-23）：区间表的「批量口径」汇总行。
 *
 * 背景（真机实测）：切换后第一份 nudge 的 6 个可压区间 fold 成本几乎相同
 * （39112～46478），而单段收益只有 20～11500t —— 每行单项净收益都是负数。
 * 按「每行 net」呈现会把模型全部劝退，正确结论却相反：一次 re-prefill 的价钱
 * 与折几段无关，该一次折完。`foldPlanOf` 给出并集收益、共用一次性成本、回本次数。
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { foldPlanOf } from '../src/nudge.ts'
import type { SeqCompressibleRange } from '../src/region.ts'

/** This file pins the SUM arithmetic; the three gates are tested in
 *  condense-election.test.ts (E-17..E-19), so the pricing here disables the two
 *  numeric ones and leaves the shared minimum/floor in force. */
const OPEN = { maxPaybackRequests: null, maxFoldCostTokens: null } as const

function range(partial: Partial<SeqCompressibleRange> & Pick<SeqCompressibleRange, 'start' | 'end' | 'tokens' | 'foldCostTokens'>): SeqCompressibleRange {
  return {
    count: 1,
    toolPct: 0,
    images: 0,
    files: 0,
    liveAfterFoldTokens: 100000,
    belowFloor: false,
    spills: 0,
    ...partial,
  }
}

test('A-9: 重叠区间按不相交子集求和（下界），不重复计收益', () => {
  // 真机形状：881..951 与 907..941 重叠，956..1034 与 965..1105 重叠。
  const plan = foldPlanOf([
    range({ start: 881, end: 951, tokens: 4426, foldCostTokens: 46458 }),
    range({ start: 907, end: 941, tokens: 1268, foldCostTokens: 45200 }),
    range({ start: 956, end: 1034, tokens: 11500, foldCostTokens: 39112 }),
    range({ start: 965, end: 1105, tokens: 7109, foldCostTokens: 39961 }),
    range({ start: 1032, end: 1034, tokens: 20, foldCostTokens: 39112 }),
  ], 900, OPEN)
  assert.ok(plan !== null)
  // 最大不相交子集：881..951(4426) + 956..1034(11500) = 15926
  // （贪心按 end 会让 907..941 挤掉 881..951，只报 12768 —— 少 21%）
  assert.equal(plan.reclaimTokens, 15926)
  assert.equal(plan.rows, 2)
  // H1 (cross-review 2026-11-25): cost = the LAST member's foldCostTokens (the tool's
  // `suffix[last span end + 1]`), i.e. the tail row 956..1034's 39112 — not the
  // head-relative net kept size (35858).
  assert.equal(plan.foldCostTokens, 39112)
  assert.ok(Math.abs(plan.paybackRequests - 39112 / 15926) < 1e-9)
})

test('A-9: 单个区间也给回本建议（比例判定对单段同样成立）', () => {
  const plan = foldPlanOf([range({ start: 10, end: 20, tokens: 5000, foldCostTokens: 4000 })], 900, OPEN)
  assert.ok(plan !== null)
  assert.equal(plan.rows, 1)
  // H1 (cross-review 2026-11-25): the single row's own foldCostTokens, 4000.
  assert.equal(plan.foldCostTokens, 4000)
  assert.ok(Math.abs(plan.paybackRequests - 4000 / 5000) < 1e-9)
  assert.equal(plan.reclaimTokens, 5000)
  assert.equal(foldPlanOf([]), null)
})

test('A-9: 回本次数至少为 1（fold 成本 0 的尾部折叠）', () => {
  const plan = foldPlanOf([
    range({ start: 10, end: 20, tokens: 5000, foldCostTokens: 0 }),
    range({ start: 30, end: 40, tokens: 4000, foldCostTokens: 0 }),
  ], 900, OPEN)
  assert.ok(plan !== null)
  assert.equal(plan.foldCostTokens, 0)
  assert.equal(plan.paybackRequests, 0)
  assert.equal(plan.reclaimTokens, 9000)
})

test('A-9: 收益为 0 的区间集合不产生汇总（避免除零与无意义行）', () => {
  const plan = foldPlanOf([
    range({ start: 10, end: 20, tokens: 0, foldCostTokens: 5000 }),
    range({ start: 30, end: 40, tokens: 0, foldCostTokens: 4000 }),
  ], 900, OPEN)
  assert.equal(plan, null)
})

test('评审 2026-11-24：belowFloor 的段不进计划（compress 会拒，计划不能推荐废案）', () => {
  const plan = foldPlanOf([
    range({ start: 10, end: 20, tokens: 5000, foldCostTokens: 4000, belowFloor: true }),
    range({ start: 30, end: 40, tokens: 5000, foldCostTokens: 300 }),
  ], 900, OPEN)
  assert.ok(plan !== null)
  assert.equal(plan.rows, 1, 'the below-floor run was dropped')
  assert.equal(plan.reclaimTokens, 5000)
  // H1 (cross-review 2026-11-25): the batch cost is the LAST row's foldCostTokens —
  // the tool's own `suffix[last span end + 1]` — not the head-relative net kept size.
  assert.equal(plan.foldCostTokens, 300)
  assert.ok(Math.abs(plan.paybackRequests - 300 / 5000) < 1e-9)
  // 全都是 below-floor 时没有可推荐的东西
  assert.equal(foldPlanOf([range({ start: 1, end: 2, tokens: 9000, foldCostTokens: 0, belowFloor: true })], 900, OPEN), null)
})
