<div align="center">

<img src="assets/logo.svg" alt="" width="72">

# Elffuss Engine

**A modular WebGPU runtime for GGUF language models: a GGUF reader, a tokenizer and a sampler in plain ES modules, behind a small chat API. No dependencies, no build step.**

[![License](https://img.shields.io/badge/license-Apache--2.0%20AND%20MIT-c8a06a?style=flat-square)](#licence)
[![Dependencies](https://img.shields.io/badge/dependencies-none-5b6169?style=flat-square)](#layout)
[![Runs on](https://img.shields.io/badge/runs%20on-WebGPU-8272ff?style=flat-square)](#what-it-runs)

[Elffuss](https://elffuss.com)

</div>

> **Provisional.** This is version 0.0.0, a snapshot of the engine published before its first release. The API below
> works and will change: read the [limits of this version](#limits-of-this-version) and the
> [changelog](CHANGELOG.md) before you depend on it.

**In this README:** [quick start](#quick-start) · [what it runs](#what-it-runs) · [limits of this version](#limits-of-this-version) · [layout](#layout) · [how it is checked](#how-it-is-checked) · [licence](#licence)

## Quick start

Inference runs in the page, on WebGPU, and the weights are read from Hugging Face with HTTP range requests. The page
needs a secure context (`https://`, or `http://localhost` and `http://127.0.0.1`) and a WebGPU adapter that offers the
`shader-f16` feature and a storage-buffer binding limit large enough for the model's largest tensor (see the
[limits of this version](#limits-of-this-version)). Serve this repository with any static file server and open a page
like this one from it.

```html
<!doctype html>
<meta charset="utf-8">
<pre id="out"></pre>
<script type="module">
  import * as engine from './js/provider.js';

  console.log(engine.models());          // [{ id, label }, …]
  engine.configure('qwen35-0.8b');       // the small model, read from Hugging Face
  await engine.load((status) => console.log(status));

  engine.setSampling({ temperature: 0.7, seed: 1, maxTokens: 120 });
  const out = document.getElementById('out');
  const reply = await engine.chat(
    [{ role: 'user', content: 'Write one short sentence about the sea.' }],
    'You are a concise assistant.',
    (piece) => { out.textContent += piece; },
  );
  console.log(reply);

  await engine.unload();
</script>
```

This exact page is the one we ran, in headless Chrome, against this tree served from `http://127.0.0.1`.

- `models()` lists the registry: an `id` and a `label` per model.
- `configure(id)` selects a model. It returns `true` when the selection changes, and `false` otherwise, also for an
  `id` that does not exist.
- `load(onProgress)` downloads and prepares the selected model; `onProgress` receives status text.
- `setSampling({ temperature, seed, p, k, minP, presence, repeatPenalty, maxTokens, razonar })` merges into the
  current sampling settings, where `p` is top-p and `k` is top-k; other keys, `topP` and `topK` among them, are
  ignored in this version. For a model that reasons, `razonar: true` or `false` forces reasoning on or off, and
  `null` leaves it to the registry. `setSampling(null)` goes back to the model's defaults.
- `chat(history, system, onToken, options)` runs one turn. `history` is an array of `{ role, content }`, and any
  role other than `assistant` is read as `user`; `system` is the system prompt; `onToken` receives each piece of
  text as it is generated; the promise resolves to the reply.
- To stop a turn, pass `{ signal }` as `options` (a bare `AbortSignal` or an `AbortController` works too), or call
  `abort()`, which stops the turn in progress and the ones waiting behind it. A stopped turn rejects with an error
  whose `name` is `AbortError`; its `partial` holds the text emitted until then, and `porDescarga` is `true` when
  `unload()` or a `configure()` to another model stopped it.
- `unload()` releases the model and the WebGPU device.

## What it runs

- The model of the quick start: Qwen3.5 0.8B in `Q8_0`, from
  [`ggml-org/Qwen3.5-0.8B-GGUF`](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) on Hugging Face, Apache-2.0 as its
  GGUF header declares. Its architecture is `qwen35`, a hybrid of state-space and attention layers.
- The default model of the registry: LFM3 1.7B in `Q8_0`, the GGUF that Liquid AI publishes in
  [`LiquidAI/LFM3-1.7B-GGUF`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), under the
  [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LICENSE). Its architecture is
  `lfm3`: gated short convolutions with sliding-window and global attention. The registry reads it from the host of
  the Elffuss apps, which is why the quick start selects the Qwen model instead.
- `models()` also lists two larger models that are hosted for the Elffuss apps. They, and the copy of LFM3 the
  registry points at, are not meant to be loaded from other sites.
- Weight types the engine reads: `F32`, `F16`, `BF16`, `Q8_0`, `Q2_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ2_XXS`,
  and two third-party ternary formats, `PQ2_0` and `PTQ1_0`.

Other GGUF files may fail to load, or load and answer worse without an error: the engine has three architecture
profiles (`qwen35`, `llama` and `lfm3`), reads only the `gpt2` tokenizer family, and always writes a ChatML prompt
instead of running the chat template of the file. It reads the template only to adjust that prompt: whether the model
reasons, how its turns are closed and, for `lfm3`, the start token.

## Limits of this version

- **WebGPU features.** The attention cache is stored in half precision, so the engine needs the `shader-f16` feature:
  without it, `load()` finishes and the first `chat()` fails when it compiles its shaders. It also refuses a model
  that has a tensor larger than the adapter's `maxStorageBufferBindingSize`, a limit it asks the adapter for.
- **No cache of its own.** Inside the Elffuss sites the host app keeps the model in the browser. Anywhere else, every
  `load()` reads the model from the network again. Even so, `load()` first checks the browser's storage estimate as if
  it were going to keep the model, and refuses with a message about disk space when the estimate is too small.
- **Host files.** `load()` and `chat()` look for optional files of the host app, described under Bundlers below.
  Outside the Elffuss sites they answer 404, which the browser console shows, and the engine carries on without them
  (for the missing store it also logs a warning).
- **History.** Outside the Elffuss apps the engine keeps the last 12 messages of `history`, then fits them to the
  context.
- **One model per page.** The module holds a single engine; selecting another model unloads the current one.
- **Spanish messages.** Progress and error texts are in Spanish, and the console gets a few lines per turn.
- **Reasoning is streamed too.** With a model that reasons, `onToken` also receives the reasoning; the resolved reply
  does not include it.
- **Bundlers.** The engine looks for three optional files of the host app (`../runtime/model-store.js`,
  `../runtime/model-broker.js` and `../context.js`) with literal relative imports, and carries on without them. A
  bundler may try to resolve those paths when it builds; serve the modules as they are.

## Layout

```
js/         the runtime: 20 ES modules, served as they are written
            provider.js    the chat API of the quick start
            registro.js    the model registry; it imports nothing, so a page can list models without the engine
            gguf.js        the GGUF reader, which reads by byte ranges
            tokenizer.js   the tokenizer
            sample.js      the sampler
            model.js       a model session on WebGPU
            device.js      the WebGPU device
            the rest       the shaders (kernels.*.wgsl.js) and what runs them (ops.js, state.js, hadamard.js,
                           quant-tables.js)
assets/     the logo
```

This repository receives curated snapshots of the engine. Development happens elsewhere, so the history here moves in
snapshots.

## How it is checked

We check it internally against llama.cpp: the GGUF reader against llama.cpp's reader, the tokenizer token by token
against `llama-tokenize`, the weight types against llama.cpp's decoder (the two ternary formats, which llama.cpp
does not have, against their published definition), and forward passes, layer by layer, against llama.cpp's
activations for the model formats that llama.cpp reads. The ternary model is checked piece by piece and end to end.
Those tests and their reference data are not part of this repository.

## Licence

Apache-2.0, see [LICENSE](LICENSE). `js/quant-tables.js` carries grid tables copied from llama.cpp under the MIT licence
of the ggml authors, reproduced in full at the top of that file; see [NOTICE](NOTICE).
