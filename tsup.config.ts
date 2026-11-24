import { defineConfig } from 'tsup'

// Bundle the adapter into one self-contained dist/index.js: zero runtime deps.
//
// The kernel used to be npm's `acp-kernel@0.0.101` kept inline via an explicit
// `noExternal` (tsup externalizes `dependencies` by default, and the v0.2.4
// drift was a published package that still imported it at runtime). Since
// 2026-11-24 the kernel's closure lives in `src/kernel/` as our own source, so
// there is nothing left to externalize or inline — the vendored files are part
// of the entry's module graph like any other src file.
//
// The @deepseek-ai/* seam packages stay external — the hosting DSH deployment
// provides them.
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  target: 'node20',
  dts: false,
  sourcemap: true,
  clean: true,
  external: [/^@deepseek-ai\//],
})
