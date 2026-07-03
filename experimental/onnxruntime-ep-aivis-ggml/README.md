# onnxruntime-ep-aivis-ggml

Aivis Style-Bert-VITS2 の GGML 推論を ONNX Runtime Plugin Execution Provider
として呼び出すための最小パッケージです。

このパッケージは通常の AivisSpeech Engine 実行経路から分離されています。
Engine は `--onnx_provider ggml` が明示された場合だけ共有ライブラリを登録し、
`AivisGgmlExecutionProvider` をフォールバック Provider より前に追加します。

## Python helper

Python package は次の helper を公開します。

- `get_library_path() -> str`
- `get_ep_name() -> str`
- `get_ep_names() -> list[str]`
- `get_default_provider_options() -> dict[str, str]`

Engine は `onnxruntime_ep_aivis_ggml.cache.prepare_ggml_cache` も import します。
これは、対応している AIVMX/ONNX synthesis model を、最初の ONNX session を
開く前に GGUF キャッシュへ変換するためです。

## Native Provider

native 共有ライブラリは ONNX Runtime Plugin EP の symbol を export します。

- `CreateEpFactories`
- `ReleaseEpFactory`

Provider name:

```text
AivisGgmlExecutionProvider
```

主な provider option:

- `backend`: `vulkan`, `metal`, `cpu`
- `device`: バックエンド内の device id
- `precision`: `accurate`, `fast`
- `gguf_path`: 変換済み Style-Bert-VITS2 synthesis GGUF
- `jp_bert_gguf_path`: 変換済み Style-Bert-VITS2 JP-BERT GGUF
- `tts_cpp_library_path`: `tts_style_bert_vits2_*` を公開する TTS.cpp shared library
- `eager_load_model`: `0`, `1`
- `claim_synthesis_graph`: `0`, `1`
- `claim_jp_bert_graph`: `0`, `1`
- `n_threads`: TTS.cpp runtime の thread 数。`0` は runtime 既定値を使う

graph claim は opt-in です。`claim_synthesis_graph=1` または
`claim_jp_bert_graph=1` を指定しない限り、Provider は登録されますが、
実行はフォールバック ONNX Provider に残ります。

## 現在のスコープ

このブランチでは統合範囲を意図的に狭くしています。

- `--onnx_provider ggml` を選ばない限り、Aivis の起動と推論経路は変えない。
- `--onnx_provider ggml` では `AivisGgmlExecutionProvider` を strict mode で登録する。
- AIVMX/ONNX synthesis weights は ONNX session 作成前に local GGUF キャッシュへ変換する。
- synthesis GGUF の既定は FP32 とし、レビュー基準を保守的にする。
- JP-BERT は既定の F16 `linear` GGUF を使う。既存 tokenizer、日本語 frontend、
  `word2ph` 展開は Aivis / Style-Bert-VITS2 側に残す。
- native EP は既知の full synthesis graph と JP-BERT graph だけを claim する。

native sidecar/backend の追加実験、ベンチマーク生成物、EPContext packaging、
汎用 ONNX-to-GGML compile はこのブランチの対象外です。
