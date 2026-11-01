# Changelog

Changes to what this repository publishes. Version numbers start with 0.1.0, the first release; until then
everything is listed under Unreleased.

## Unreleased

### Changed in this snapshot

- LFM3 runs faster by default, in prompt processing and in generation. When the GGUF header names the `lfm3`
  architecture, the engine turns on a set of faster code paths before it loads the weights:
  - `Q8_0` and `Q4_0` weights are reordered at load, into the same number of bytes, so that each block's quants are
    aligned and the scales sit apart (`js/repack.js`);
  - generation uses new matrix-vector kernels, and projections that read the same input share one dispatch
    (`js/kernels.gemv.wgsl.js`);
  - prompt processing unpacks the weights inside the matrix multiply instead of into a scratch buffer
    (`js/kernels.gemm2.wgsl.js`);
  - attention runs as one fused kernel, in prompt processing and in generation (`js/kernels.flash.wgsl.js`);
  - several small steps around the projections run in fewer dispatches (`js/kernels.fusion.wgsl.js`).

  Each path is checked against the previous one, and LFM3 with all of them on is checked against llama.cpp, in
  `Q8_0` and `Q4_0`. Other architectures keep the previous paths, because they have not been checked with the new
  ones.
- The runtime has 33 modules (20 before).

### Added in this snapshot

- The `Q4_0` weight type.
- `globalThis.__ELFFUSS_RAPIDO = false`, set before `load()`, keeps the previous code paths for LFM3. The individual
  switches (`__ELFFUSS_REPACK`, `__ELFFUSS_GEMV2`, `__ELFFUSS_GEMM2`, `__ELFFUSS_GEMM2_COLA`, `__ELFFUSS_FLASH`,
  `__ELFFUSS_FUSION`) can also be set on `globalThis` before `load()`; a value the page sets, `true` or `false`, is
  kept.
- Two more switches that no model turns on: `__ELFFUSS_PLAN` reuses bind groups, uniforms and readback buffers from
  token to token, and `__ELFFUSS_TOPK_GPU` picks the sampling candidates on the device and reads back only those.

### Added in the previous snapshot

- The `lfm3` architecture (LFM3 1.7B): gated short convolutions, sliding-window and global attention with their own
  rotary bases and `rope_freqs`, the per-head output gate and the norms after attention and after the feed-forward.
  The tokenizer reads the `lfm2` pre-tokenizer, and the prompt starts with the model's start token.
- LFM3 1.7B is the default model of the registry.
- From 64 prompt tokens on, prompt processing multiplies `Q8_0` weights in half precision: each multiply unpacks the
  weight into a scratch buffer and runs a matrix multiply without shared memory (`js/kernels.drift.wgsl.js`). Each
  registry entry can also set its own prompt batch.
- Generation submits its WebGPU work layer by layer instead of once per token.

### Fixed in the previous snapshot

- The sampler sorted the whole vocabulary on every token; it now keeps the top candidates in a heap.
- The reply was decoded again in full on every token; it is now decoded as it streams.

### Fixed

- Sampling with a temperature above zero could return the least likely candidate about half of the time, because
  the random draw fell outside its range; consecutive seeds also gave correlated draws. The draw is now uniform.
- `js/tokenizer.js` is plain text: three literal NUL characters are written as `'\u0000'`, which is the same string.

### Changed

- The sampler follows llama.cpp's order: penalties, top-k, top-p, min-p, then temperature. Top-p and min-p are now
  computed before the temperature is applied.
- The seed advances with every generated token.
- When the app sets no sampling, the engine uses the one the registry declares for the model, if any; otherwise it
  decodes greedily with its default repeat penalty (1.1 over the last 64 tokens). `setSampling()` merges into that.
- The published tree is the import closure of `js/provider.js` (19 modules) and the files around it. The README
  and `package.json` were rewritten for this repository; `package.json` declares `Apache-2.0 AND MIT`, as
  `js/quant-tables.js` does, and lists `assets/`.

### Added

- `setSampling()` accepts `minP`, `presence` and `repeatPenalty`.
- Models that reason get an open reasoning block and a reply budget of their own; `setSampling({ razonar })` forces
  reasoning on (`true`) or off (`false`), and `null` leaves it to the registry.
- `NOTICE`, and the complete MIT licence of the ggml authors at the top of `js/quant-tables.js`.

### Removed

- The two articles in `docs/`, which were in Spanish. They will come back rewritten in English.
