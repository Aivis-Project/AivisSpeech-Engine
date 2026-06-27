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

## Linux RTX 3060 Local Run (2026-06-27)

Raw results are stored in
[linux-rtx3060-cuda-ggml-cpu.json](res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json).

### Scope

- Measurement date: 2026-06-27, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- AudioQuery: `tempoDynamicsScale=1.0`, matching the Engine `/audio_query`
  default used by the app
- Engine: `30ead73c88a7`
- TTS.cpp: `0c6678415023`
- Model: AIVMX/ONNX `コハク` model, version `1.1.0`
- Style: `1878365376` (`ノーマル`)
- GGML model path: AIVMX/ONNX is converted to synthesis GGUF by the Plugin EP
  cache path; JP-BERT uses `kevinzhow/style-bert-vits2-gguf`
  `frontend/style-bert-vits2-jp-bert.gguf`
- GGML provider options: `backend=vulkan`, `precision=accurate`,
  `claim_synthesis_graph=1`, `claim_jp_bert_graph=1`, `eager_load_model=1`

| label | text | chars |
| --- | --- | ---: |
| short | `テストです。` | 6 |
| medium | `今日はいい天気です。` | 10 |
| long | `これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。` | 41 |

### Device Parameters

| component | value |
| --- | --- |
| OS | Ubuntu 26.04 LTS, kernel `7.0.0-22-generic` |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores / 16 threads |
| ONNX Runtime | `onnxruntime-gpu 1.26.0`; providers include `CUDAExecutionProvider`, `CPUExecutionProvider` |
| CUDA GPU | NVIDIA GeForce RTX 3060, driver `595.71.05`, VRAM `12288 MiB` |
| Vulkan GPU | NVIDIA GeForce RTX 3060, Vulkan API `1.4.329`, UMA `0`, fp16 `0`, bf16 `1`, warp size `32`, shared memory `49152`, int dot `1` |
| GGML Vulkan device pin | `GGML_VK_VISIBLE_DEVICES=1` |

### RTF Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | ONNX GGML Plugin EP Vulkan RTF |
| --- | ---: | ---: | ---: |
| short | `0.296` | `1.390` | `0.176` |
| medium | `0.259` | `1.003` | `0.169` |
| long | `0.225` | `0.237` | `0.152` |
| overall mean | `0.260` | `0.877` | `0.166` |

Provider evidence from the run:

```json
{
  "onnx-cpu": {
    "active_providers": ["CPUExecutionProvider"]
  },
  "onnx-cuda": {
    "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "onnx-ggml-vulkan": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"]
  }
}
```

Interpretation:

- The Plugin EP path keeps the normal ONNX frontend and replaces only the
  supported synthesis and JP-BERT ONNX graphs with TTS.cpp GGML execution.
- ONNX CUDA is active and not silently falling back to CPU. This run required
  CUDA 12 runtime libraries to be present in `LD_LIBRARY_PATH`; without them,
  the benchmark fails instead of recording a CPU fallback as a CUDA result.
- On this RTX 3060 run, ONNX CUDA is slower than ONNX CPU for short and medium
  text and slightly slower for long text. The result reflects this model/input
  shape on `onnxruntime-gpu 1.26.0`, not a provider fallback.
- GGML Plugin EP Vulkan is faster than both ONNX CPU and ONNX CUDA for all
  three text lengths in this run.

### Audio Preview

These AAC files are representative outputs for qualitative review. They are not
included in the RTF timing window.

| text length | ONNX CPU | ONNX CUDA | ONNX GGML Plugin EP Vulkan |
| --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_long.m4a) |

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
- TTS.cpp: `7b83c9c1408ae01712d612b5ac35f63b76861e0a`; ggml submodule
  `a78c352bb70b312daa7ef1361485fbb94392713e`
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
| short | `0.425` | `2.402` | `0.105` |
| medium | `0.373` | `1.390` | `0.098` |
| long | `0.284` | `0.207` | `0.056` |
| overall mean | `0.361` | `1.333` | `0.087` |

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
  faster than CPU for the long sample, but slower for the short and medium
  samples in this run.
- GGML Plugin EP Vulkan is faster than both ONNX CPU and ONNX DirectML for all
  three text lengths after the TTS.cpp Style-Bert conv1d fallback fix.
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
  --text "テストです。" `
  --text "今日はいい天気です。" `
  --text "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。" `
  --ggml_native_library_path "C:\path\to\tts.dll" `
  --onnx_ep_library_path "C:\path\to\aivis_ggml_onnx_ep.dll" `
  --ggml_vulkan_precision accurate `
  --warmup_runs 1 `
  --runs 3 `
  --tempo_dynamics_scale 1.0 `
  --output_json "docs\res\onnx-ggml-plugin-benchmark\windows-arc-b580-directml-ggml-cpu.json" `
  --audio_output_dir "docs\res\onnx-ggml-plugin-benchmark\audio\windows-arc-b580"
```

## Linux Reproduction Command

The benchmark script installs the provided AIVMX into a temporary `Models`
directory, clears the process-global JP-BERT ONNX cache before each backend,
validates the actual ONNX provider after model load, and then measures only
`synthesize_wave()`.

Set local paths before running:

```bash
export AIVMX_PATH="<path-to-model.aivmx>"
export STYLE_ID="1878365376"
export CUDA12_NVIDIA_LIBS="<colon-separated CUDA 12/cuDNN library dirs>"
export AIVIS_GGML_ONNX_EP_LIBRARY_PATH="<path-to-libaivis_ggml_onnx_ep.so>"
export TTS_CPP_NATIVE_LIBRARY_PATH="<path-to-libtts.so>"
export BENCHMARK_OUTPUT_JSON="docs/res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json"
export BENCHMARK_AUDIO_WAV_DIR="<path-to-temporary-wav-output-dir>"
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
  --text "テストです。" \
  --text "今日はいい天気です。" \
  --text "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_LIBRARY_PATH" \
  --ggml_vulkan_precision accurate \
  --tempo_dynamics_scale 1.0 \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_OUTPUT_JSON" \
  --audio_output_dir "$BENCHMARK_AUDIO_WAV_DIR"
```

Strict provider checks:

- `onnx-cpu` must select `CPUExecutionProvider`
- `onnx-directml` must select `DmlExecutionProvider`
- `onnx-cuda` must select `CUDAExecutionProvider`
- `onnx-ggml-vulkan` must select `AivisGgmlExecutionProvider`

If ONNX Runtime silently falls back to CPU, the script fails instead of
recording a misleading GPU result.
