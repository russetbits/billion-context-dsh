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
/** 用户可写值:字符串模板,或 null(= 用默认,等价于不写)。YAML 宿主写 null 是合法输入。 */
export type PromptInput = string | null;
/** 按组生成"每键可选、可 null"的覆盖类型。 */
export type PromptOverride<T> = {
    [K in keyof T]?: PromptInput;
};
export interface NudgePrompts {
    /** 普通档首句。占位符:{pct}（`{philosophy}` 仍被校验器接受,但 B6 会把渲染出的哲学段摘掉——它只能经系统提示到达模型) */
    normal: string;
    /** 紧急档首句。占位符:{pct}（`{philosophy}` 同上) */
    emergency: string;
    /** 指导行（HOW_TO_COMPRESS_RULES）。无占位符 */
    guidance: string;
    /** tier 蒸馏行。占位符:{tier} {count} {prevTier} {tokens} {seqs} {firstSeq} {lastSeq} */
    tier: string;
    /** 上下文分解。占位符:{system} {tool} {summaries} {code} {text} */
    breakdown: string;
    /** 增长行。占位符:{growth} */
    growth: string;
    /** 溢出提示。无占位符 */
    tip: string;
    /** 压缩后的前缀重读代价。占位符:{uncached} {count} */
    cacheCost: string;
}
export interface RangeTablePrompts {
    /** 表头。占位符:{surface} */
    header: string;
    /** 标题。占位符:{count}(表格行数) */
    title: string;
    /** 每行。占位符:{start} {end} {count} {tokens} {toolPct} {textPct} {media} {foldCost} {tools}
     *  —— `{tools}` 渲染成 ` · tools: bash×12, read×5`（空则整段消失，见 nudge.ts） */
    line: string;
    /**
     * 批量口径汇总行（A-9，2026-11-23）。占位符:{rows} {reclaim} {fold} {breakEven}。
     * 单行 fold 列只讲“这一段自己折要重读多少”，量级常远大于回收量，模型容易据此
     * 判定「压了更亏」；正确结论是相反的——一次 re-prefill 的价钱与折几段无关，所以
     * 该一次折完。本行把并集收益、共用的一次性成本与回本次数直接算给模型。
     */
    plan: string;
    /** 表尾调用语法。无占位符 */
    footer: string;
}
export interface ToolPrompts {
    /** 工具描述(纯文本,无占位符) */
    compress: string;
    recall: string;
    acpStatus: string;
    /**
     * `compress` 的 `condense` 参数描述(纯文本,无占位符)。
     * 只开放这一条参数文案:`condense` 是 T2/T3 蒸馏的唯一入口,模型靠它理解
     * 「一段替换成一条 note」的语义。note 骨架(`## Task`/`## Now`/…)是校验契约,
     * 不在任何覆盖清单里(见 tools.ts 的 `MAX_HANDOFF_NOTE_TOKENS` 与骨架校验)。
     */
    condense: string;
}
export interface AcpPrompts {
    readonly nudge?: PromptOverride<NudgePrompts>;
    readonly rangeTable?: PromptOverride<RangeTablePrompts>;
    readonly tools?: PromptOverride<ToolPrompts>;
    /** 整段 system prompt 模板;`{philosophy}` 引用 kernel 的 COMPRESS_PHILOSOPHY */
    readonly systemPrompt?: PromptInput;
}
/** 解析结果 —— 所有字段已填满(纯 string,无 null)、已校验。构造一次,全程复用。 */
export interface ResolvedPrompts {
    readonly nudge: NudgePrompts;
    readonly rangeTable: RangeTablePrompts;
    readonly tools: ToolPrompts;
    /** 参数级 description 文案(键 `<tool>.<arg>`)。只由 `textOverrides` 覆盖,不在 `config.prompts` 结构化层里。 */
    readonly toolParamTexts: Readonly<Record<string, string>>;
    /** 注意:这是【模板】(含 {philosophy}),不是渲染结果。渲染用 renderSystemPrompt。 */
    readonly systemPromptTemplate: string;
}
/**
 * 纯替换。两个契约:
 * 1. 未知占位符不可能到达这里(构建期已校验);
 * 2. 已知占位符缺值 = 编程错误 → throw(绝不静默渲染空串)。
 */
export declare function renderTemplate(template: string, vars: Record<string, string | number>): string;
/**
 * 深合并 + 校验;引擎构造期调用一次,出错即抛(fail-fast)。
 * 未传入时返回 DEFAULT_RESOLVED,零校验重跑。
 */
export declare function resolvePrompts(input?: AcpPrompts): ResolvedPrompts;
/** 渲染 system prompt 模板(注入 kernel 压缩哲学、压缩规则、蒸馏规则)。 */
export declare function renderSystemPrompt(prompts: ResolvedPrompts): string;
/**
 * 每个**参数级** `description` 的默认文本 —— 工具描述讲「这个工具怎么用」,
 * 这里讲「这一格填什么」。键是 `<tool>.<arg>`(如 `compress.force`),可覆盖 id 为
 * `tool.param.<键>`;未覆盖时保持英文默认,与工具级文案同一套替换语义。
 *
 * **`compress.condense.summary` 不在此表**: 它是 note 骨架(`## Task`/`## Now`/…),
 * 是校验契约而不是文案,覆盖它会让「缺 heading 也照常落地」的约定失效。
 */
export declare const DEFAULT_TOOL_PARAM_TEXTS: Readonly<Record<string, string>>;
/**
 * 默认模板 —— 与 v4 之前的硬编码文案逐字节一致
 * (回归锚点见 tests/prompts.test.ts 的硬编码字面量快照)。
 */
export declare const DEFAULT_PROMPTS: ResolvedPrompts;
/** 模块级默认缓存:默认参/兜底直接引用,避免每次调用重跑校验。 */
export declare const DEFAULT_RESOLVED: ResolvedPrompts;
/**
 * `textOverrides` 的键是点号路径 id(`<group>.<key>`),值是**完整替换文本**。
 * 未列出的 id 保持英文默认。REPLACE, never append: 覆盖值直接取代原文案,
 * 所以 token 成本大致持平,配置不会悄悄变贵(见 src/config.ts 的注释)。
 */
export type TextOverrides = Record<string, string>;
/**
 * 覆盖值的长度上限(相对该 id 默认值的倍数)。超过只 warning,不拒绝:
 * 最常见的事故是把英文翻成中文,而中文并不更省 token(见 src/config.ts)。
 */
export declare const MAX_TEXT_OVERRIDE_RATIO = 1.5;
/**
 * Token 估算,与 `defaultCountTokens` 同口径:CJK 1 字/token,其余 4 字/token。
 * 只用于长度告警,不参与任何校验或计价。
 */
export declare function estimateTextTokens(text: string): number;
/**
 * 每个可覆盖文案的 id(`nudge.*` / `rangeTable.*` / `tool.*` / `systemPrompt`)。
 * 由 DEFAULT_PROMPTS 的键推导,所以新增一个槽位就自动出现在这里 ——
 * `tests/text-overrides.test.ts` 用它逐 id 验证「能覆盖且生效」,防止 id 表与实际漂移。
 */
export declare function ableTextIds(): string[];
/** 按 id 读一组已解析文案里的文本;未知 id 返回 undefined。 */
export declare function textById(prompts: ResolvedPrompts, id: string): string | undefined;
/** 一条覆盖告警(拼接成一行给启动日志 / acp_status 自检)。 */
export interface TextOverrideDiagnostic {
    readonly id: string;
    readonly tokens: number;
    readonly defaultTokens: number;
}
export interface TextOverrideReport {
    /** 覆盖后的文案(不可变;未覆盖的字段与入参同一个引用)。 */
    readonly prompts: ResolvedPrompts;
    /** 配置里有、但不在 `ableTextIds()` 里的 id — 忽略,不报错。 */
    readonly unknown: string[];
    /** 比默认值长出 MAX_TEXT_OVERRIDE_RATIO 倍的覆盖 — 照常生效,只告警。 */
    readonly oversized: TextOverrideDiagnostic[];
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
export declare function applyTextOverrides(resolved: ResolvedPrompts, overrides?: TextOverrides | null): TextOverrideReport;
/** 诊断行(启动日志与 acp_status 共用一份文案,不各写一套)。 */
export declare function textOverrideWarningLines(report: TextOverrideReport): string[];
