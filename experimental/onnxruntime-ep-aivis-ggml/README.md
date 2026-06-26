# onnxruntime-ep-aivis-ggml

Minimal ONNX Runtime Plugin Execution Provider package for Aivis
Style-Bert-VITS2 GGML inference.

This package is kept outside the normal AivisSpeech Engine runtime path. The
engine only registers the shared library and prepends
`AivisGgmlExecutionProvider` when `--onnx_provider ggml` is explicitly selected.

## Python Helpers

The Python package exposes discovery helpers:

- `get_library_path() -> str`
- `get_ep_name() -> str`
- `get_ep_names() -> list[str]`
- `get_default_provider_options() -> dict[str, str]`

The engine also imports `onnxruntime_ep_aivis_ggml.cache.prepare_ggml_cache`
to convert supported AIVMX/ONNX synthesis models into a GGUF cache before the
first ONNX session is opened.

## Native Provider

The native shared library exports the ONNX Runtime Plugin EP symbols:

- `CreateEpFactories`
- `ReleaseEpFactory`

Provider name:

```text
AivisGgmlExecutionProvider
```

Important provider options:

- `backend`: `vulkan`, `metal`, or `cpu`
- `device`: backend-local device id
- `precision`: `accurate` or `fast`
- `gguf_path`: prepared Style-Bert-VITS2 synthesis GGUF
- `jp_bert_gguf_path`: prepared Style-Bert-VITS2 JP-BERT GGUF
- `tts_cpp_library_path`: TTS.cpp shared library exposing
  `tts_style_bert_vits2_*`
- `eager_load_model`: `0` or `1`
- `claim_synthesis_graph`: `0` or `1`
- `claim_jp_bert_graph`: `0` or `1`
- `n_threads`: TTS.cpp runtime thread count, where `0` keeps the runtime default

Graph claim is opt-in. Without `claim_synthesis_graph=1` or
`claim_jp_bert_graph=1`, the provider registers but leaves execution on the
fallback ONNX providers.

## Current Scope

This branch keeps the integration deliberately narrow:

- Aivis starts unchanged unless `--onnx_provider ggml` is selected.
- `--onnx_provider ggml` registers `AivisGgmlExecutionProvider` in strict mode.
- AIVMX/ONNX synthesis weights are converted to GGUF through the local cache
  helper before the ONNX session is created.
- JP-BERT uses the published prebuilt GGUF bundle while the existing tokenizer,
  Japanese frontend, and `word2ph` expansion stay in Aivis/Style-Bert-VITS2.
- The native EP claims only the known full synthesis and JP-BERT ONNX graphs.

Native sidecar/backend experiments, benchmark artifacts, EPContext packaging,
and generic ONNX-to-GGML compilation are intentionally outside this branch.
