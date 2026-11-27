/**
 * M4 — the `/acp-prune` slash command: a human-friendly window into the same
 * machinery the model tools expose (status, one-shot compress, the block dump `recall` pages,
 * runtime settings read/write).
 * @module billion-context-dsh/commands
 */
import type { CommandDefinition } from '@deepseek-ai/dsh-commands';
import { type ToolEnvironment } from './shared.ts';
/** Register the /acp-prune command (idempotent per engine). */
export declare function acpCommand(env: ToolEnvironment): CommandDefinition;
