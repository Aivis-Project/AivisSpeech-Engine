# ONNX GGML Plugin EP Benchmark

This PR review benchmark compares only the ONNX paths affected by the minimal
GGML Plugin EP integration:

- ONNX CPU: existing `StyleBertVITS2TTSEngine` ONNX path with `CPUExecutionProvider`
- ONNX CUDA: existing ONNX path with `CUDAExecutionProvider`
- ONNX GGML Vulkan: existing ONNX path with `AivisGgmlExecutionProvider`
  claiming the synthesis and JP-BERT ONNX graphs

RTF is `elapsed_seconds / output_duration_seconds`; lower is better. Audio
encoding is intentionally excluded from measured runs.

## Scope

- Measurement date: 2026-06-25, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- Model: AIVMX/ONNX Style-Bert-VITS2 model
- Style: `888753760`
- GGML model path: AIVMX/ONNX is converted to synthesis GGUF by the Plugin EP
  cache path; JP-BERT uses `kevinzhow/style-bert-vits2-gguf`
  `frontend/style-bert-vits2-jp-bert.gguf`
- GGML provider options: `backend=vulkan`, `precision=accurate`,
  `claim_synthesis_graph=1`, `claim_jp_bert_graph=1`, `eager_load_model=1`

| label | text | chars |
| --- | --- | ---: |
| short | `テストです。` | 6 |
| medium | `今日はいい天気ですね。` | 11 |
| long | `これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。` | 41 |

## Device Parameters

| component | value |
| --- | --- |
| OS | Ubuntu 26.04 LTS, kernel `7.0.0-22-generic` |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores / 16 threads |
| ONNX Runtime | `onnxruntime-gpu 1.26.0`; providers include `CUDAExecutionProvider`, `CPUExecutionProvider` |
| CUDA GPU | NVIDIA GeForce RTX 3060, driver `595.71.05`, VRAM `12288 MiB` |
| Vulkan GPU | NVIDIA GeForce RTX 3060, Vulkan API `1.4.329`, UMA `0`, fp16 `0`, bf16 `1`, warp size `32`, shared memory `49152`, int dot `1` |
| GGML Vulkan device pin | `GGML_VK_VISIBLE_DEVICES=1` |

## RTF Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | ONNX GGML Plugin EP Vulkan RTF |
| --- | ---: | ---: | ---: |
| short | `0.354` | `0.267` | `0.145` |
| medium | `0.271` | `0.184` | `0.101` |
| long | `0.200` | `0.065` | `0.062` |
| overall mean | `0.275` | `0.172` | `0.102` |

Provider evidence from the GGML Plugin EP run:

```json
{
  "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
  "bert_active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
  "provider_options": {
    "backend": "vulkan",
    "claim_jp_bert_graph": "1",
    "claim_synthesis_graph": "1",
    "eager_load_model": "1",
    "precision": "accurate"
  }
}
```

Interpretation:

- The Plugin EP path keeps the normal ONNX frontend and replaces only the
  supported synthesis and JP-BERT ONNX graphs with TTS.cpp GGML execution.
- On the RTX 3060 run, ONNX GGML Plugin EP Vulkan is faster than ONNX CPU for
  all three text lengths and slightly faster than ONNX CUDA overall.
- Long text is effectively tied between ONNX CUDA and ONNX GGML Plugin EP
  Vulkan in this run.

## Reproduction Command

The benchmark script added for this PR installs the provided AIVMX into a
temporary `Models` directory, clears the process-global JP-BERT ONNX cache
before each backend, validates the actual ONNX provider after model load, and
then measures only `synthesize_wave()`.

Set local paths before running:

```bash
export AIVMX_PATH="<path-to-model.aivmx>"
export STYLE_ID="888753760"
export CUDA12_NVIDIA_LIBS="<colon-separated CUDA 12/cuDNN library dirs>"
export AIVIS_GGML_ONNX_EP_LIBRARY_PATH="<path-to-libaivis_ggml_onnx_ep.so>"
export TTS_CPP_NATIVE_LIBRARY_PATH="<path-to-libtts.so>"
export BENCHMARK_OUTPUT_JSON="<path-to-output-json>"
```

Run ONNX CPU, ONNX CUDA, and ONNX GGML Plugin EP Vulkan in one process:

```bash
LD_LIBRARY_PATH="${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=1 \
uv run python tools/benchmark_onnx_ggml_provider.py \
  --aivmx_path "$AIVMX_PATH" \
  --style_id "$STYLE_ID" \
  --backend onnx-cpu \
  --backend onnx-cuda \
  --backend onnx-ggml-vulkan \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_LIBRARY_PATH" \
  --ggml_vulkan_precision accurate \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_OUTPUT_JSON"
```

Strict provider checks:

- `onnx-cpu` must select `CPUExecutionProvider`
- `onnx-cuda` must select `CUDAExecutionProvider`
- `onnx-ggml-vulkan` must select `AivisGgmlExecutionProvider`

If ONNX Runtime silently falls back to CPU, the script fails instead of
recording a misleading GPU result.
