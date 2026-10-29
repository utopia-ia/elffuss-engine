# Changelog

Changes to what this repository publishes. Version numbers start with 0.1.0, the first release; until then
everything is listed under Unreleased.

## Unreleased

### Added in this snapshot

- The `lfm3` architecture (LFM3 1.7B): gated short convolutions, sliding-window and global attention with their own
  rotary bases and `rope_freqs`, the per-head output gate and the norms after attention and after the feed-forward.
  The tokenizer reads the `lfm2` pre-tokenizer, and the prompt starts with the model's start token.
- LFM3 1.7B is the default model of the registry.
- From 64 prompt tokens on, prompt processing multiplies `Q8_0` weights in half precision: each multiply unpacks the
  weight into a scratch buffer and runs a matrix multiply without shared memory (`js/kernels.drift.wgsl.js`). Each
  registry entry can also set its own prompt batch.
- Generation submits its WebGPU work layer by layer instead of once per token.

### Fixed in this snapshot

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
