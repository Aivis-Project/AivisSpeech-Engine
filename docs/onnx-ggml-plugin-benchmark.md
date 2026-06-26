# ONNX GGML Plugin EP Benchmark

This PR review benchmark compares only the ONNX paths affected by the minimal
GGML Plugin EP integration:

- ONNX CPU: existing `StyleBertVITS2TTSEngine` ONNX path with `CPUExecutionProvider`
- ONNX DirectML: existing ONNX path with `DmlExecutionProvider`
- ONNX CUDA: existing ONNX path with `CUDAExecutionProvider`
- ONNX GGML Vulkan: existing ONNX path with `AivisGgmlExecutionProvider`
  claiming the synthesis and JP-BERT ONNX graphs

RTF is `elapsed_seconds / output_duration_seconds`; lower is better. Audio
encoding is intentionally excluded from measured runs.

## Scope

- Measurement date: 2026-06-25, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- AudioQuery: `tempoDynamicsScale=0.0`
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

## Audio Preview

These AAC files are representative outputs for qualitative review. They are not
included in the RTF timing window.

| text length | ONNX CPU | ONNX CUDA | ONNX GGML Plugin EP Vulkan |
| --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cpu_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-cuda_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/onnx-ggml-vulkan_long.m4a) |

## Windows Intel Arc B580 Local Run (2026-06-27)

This local Windows run adds DirectML to the same CPU/GGML comparison. Raw
results are stored in
[windows-arc-b580-directml-ggml-cpu.json](res/onnx-ggml-plugin-benchmark/windows-arc-b580-directml-ggml-cpu.json).

### Scope

- Measurement date: 2026-06-27, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- AudioQuery: `tempoDynamicsScale=1.0`, matching the Engine `/audio_query`
  default used by the app
- OS: Microsoft Windows 11 Home `10.0.26200`, 64-bit
- CPU: AMD Ryzen 5 5600, 6 cores / 12 threads
- GPU: Intel(R) Arc(TM) B580 Graphics, driver `32.0.101.8826`
- ONNX Runtime: `onnxruntime-directml 1.24.4`; providers include
  `DmlExecutionProvider`, `CPUExecutionProvider`
- Model: AIVMX/ONNX `コハク` model, version `1.1.0`
- Style: `1878365376` (`ノーマル`)
- GGML provider options: `backend=vulkan`, `precision=accurate`,
  strict Plugin EP provider validation

| label | text | chars |
| --- | --- | ---: |
| short | `テストです。` | 6 |
| medium | `今日はいい天気です。` | 10 |
| long | `これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。` | 41 |

### RTF Results

| text length | ONNX CPU RTF | ONNX DirectML RTF | ONNX GGML Plugin EP Vulkan RTF |
| --- | ---: | ---: | ---: |
| short | `0.423` | `2.212` | `0.178` |
| medium | `0.341` | `0.233` | `0.172` |
| long | `0.278` | `0.483` | `0.145` |
| overall mean | `0.347` | `0.976` | `0.165` |

Provider evidence from the run:

```json
{
  "onnx-cpu": {
    "active_providers": ["CPUExecutionProvider"]
  },
  "onnx-directml": {
    "active_providers": ["DmlExecutionProvider", "CPUExecutionProvider"]
  },
  "onnx-ggml-vulkan": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"]
  }
}
```

Interpretation:

- DirectML is active in this run and is not silently falling back to CPU.
- On this Intel Arc B580 machine with ONNX Runtime `1.24.4`, DirectML is not
  consistently faster on the app-default `tempoDynamicsScale=1.0` path. It is
  faster than CPU for the medium sample, but slower for the short and long
  samples in this run.
- GGML Plugin EP Vulkan remains faster than CPU for all three text lengths.
- Short text measurements are especially sensitive to fixed ONNX Runtime and
  provider overhead. This table still excludes the first warmup synthesis per
  text; the app's first synthesis for a new sentence can be slower than these
  warm-run numbers when DirectML has not compiled that input shape yet.

### Audio Preview

These WAV files are representative outputs for qualitative review. They are not
included in the RTF timing window.

| text length | ONNX CPU | ONNX DirectML | ONNX GGML Plugin EP Vulkan |
| --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_short.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_short.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_short.wav) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_medium.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_medium.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_medium.wav) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-cpu_long.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-directml_long.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580/onnx-ggml-vulkan_long.wav) |

### Windows Reproduction Command

```powershell
$env:PATH = "C:\path\to\tts.cpp\bin;$env:PATH"

uv run python tools\benchmark_onnx_ggml_provider.py `
  --aivmx_path "$env:APPDATA\AivisSpeech-Engine-Dev\Models\22e8ed77-94fe-4ef2-871f-a86f94e9a579.aivmx" `
  --style_id 1878365376 `
  --backend onnx-cpu `
  --backend onnx-directml `
  --backend onnx-ggml-vulkan `
  --ggml_native_library_path "C:\path\to\tts.dll" `
  --onnx_ep_library_path "C:\path\to\aivis_ggml_onnx_ep.dll" `
  --ggml_vulkan_precision accurate `
  --warmup_runs 1 `
  --runs 3 `
  --tempo_dynamics_scale 1.0 `
  --output_json "docs\res\onnx-ggml-plugin-benchmark\windows-arc-b580-directml-ggml-cpu.json" `
  --audio_output_dir "docs\res\onnx-ggml-plugin-benchmark\audio\windows-arc-b580"
```

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
  --tempo_dynamics_scale 0.0 \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_OUTPUT_JSON"
```

Strict provider checks:

- `onnx-cpu` must select `CPUExecutionProvider`
- `onnx-directml` must select `DmlExecutionProvider`
- `onnx-cuda` must select `CUDAExecutionProvider`
- `onnx-ggml-vulkan` must select `AivisGgmlExecutionProvider`

If ONNX Runtime silently falls back to CPU, the script fails instead of
recording a misleading GPU result.
