# ONNX GGML Plugin EP Benchmark

This PR review benchmark compares only the ONNX paths affected by the minimal
GGML Plugin EP integration:

- ONNX CPU: existing `StyleBertVITS2TTSEngine` ONNX path with `CPUExecutionProvider`
- ONNX DirectML: existing ONNX path with `DmlExecutionProvider`
- ONNX CUDA: existing ONNX path with `CUDAExecutionProvider`
- ONNX GGML Vulkan: existing ONNX path with `AivisGgmlExecutionProvider`
  claiming the synthesis and JP-BERT ONNX graphs; the Linux run compares
  JP-BERT F16 `linear` versus all-FP32, crossed with FP16 versus FP32 synthesis
  voice GGUF caches

RTF is `elapsed_seconds / output_duration_seconds`; lower is better. Audio
encoding is intentionally excluded from measured runs.

## Linux RTX 3060 Local Run (2026-06-28)

Raw results are stored in
[linux-rtx3060-cuda-ggml-cpu.json](res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json).

### Scope

- Measurement date: 2026-06-28, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- AudioQuery: `tempoDynamicsScale=1.0`, matching the Engine `/audio_query`
  default used by the app
- Engine: `feat/onnx-ggml-minimal-upstream` with ONNX Runtime `1.26.0`
  compatibility, FP16 GGUF cache defaults, an explicit FP32 voice-GGUF
  benchmark selector, and an explicit JP-BERT FP32 GGUF benchmark selector
- TTS.cpp: `a053e7270261`; ggml submodule `a78c352bb70b`
- CUDA provider option: `cudnn_conv_algo_search=HEURISTIC`
- Model: AIVMX/ONNX `コハク` model, version `1.1.0`
- Style: `1878365376` (`ノーマル`)
- GGML model path: AIVMX/ONNX is converted to synthesis GGUF by the Plugin EP
  cache path using either the F16 `no-embed-norm-no-ups` recipe or the all-F32
  voice recipe. JP-BERT is tested as both the default
  `kevinzhow/style-bert-vits2-gguf`
  `frontend/style-bert-vits2-jp-bert.gguf` F16 `linear` artifact and the
  all-FP32 GGUF baseline documented in
  [JP-BERT GGUF Quantization Notes](jp-bert-gguf-quantization.md).
- GGML provider options: `backend=vulkan`, `precision=fast`,
  `device=0`, `claim_synthesis_graph=1`, `claim_jp_bert_graph=1`,
  `eager_load_model=1`

| label | text | chars |
| --- | --- | ---: |
| short | `テストです。` | 6 |
| medium | `今日はいい天気ですね。` | 11 |
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

| text length | ONNX CPU RTF | ONNX CUDA RTF | JP-BERT FP16 + voices FP16 | JP-BERT FP16 + voices FP32 | JP-BERT FP32 + voices FP16 | JP-BERT FP32 + voices FP32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | `0.278` | `0.034` | `0.120` | `0.121` | `0.121` | `0.122` |
| medium | `0.231` | `0.027` | `0.089` | `0.092` | `0.091` | `0.091` |
| long | `0.220` | `0.020` | `0.062` | `0.062` | `0.061` | `0.062` |
| overall mean | `0.243` | `0.027` | `0.090` | `0.092` | `0.091` | `0.092` |

Provider evidence from the run:

```json
{
  "onnx-cpu": {
    "active_providers": ["CPUExecutionProvider"]
  },
  "onnx-cuda": {
    "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "onnx-ggml-vulkan-jpbert-fp16-voices-fp16": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1",
    "ggml_jp_bert_precision": "fp16-linear"
  },
  "onnx-ggml-vulkan-jpbert-fp16-voices-fp32": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f32-v1",
    "ggml_jp_bert_precision": "fp16-linear"
  },
  "onnx-ggml-vulkan-jpbert-fp32-voices-fp16": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1",
    "ggml_jp_bert_precision": "fp32",
    "ggml_jp_bert_gguf_path": "<local-gguf-dir>/jp-bert-8207b37b84342787.gguf"
  },
  "onnx-ggml-vulkan-jpbert-fp32-voices-fp32": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f32-v1",
    "ggml_jp_bert_precision": "fp32",
    "ggml_jp_bert_gguf_path": "<local-gguf-dir>/jp-bert-8207b37b84342787.gguf"
  }
}
```

Interpretation:

- The Plugin EP path keeps the normal ONNX frontend and replaces only the
  supported synthesis and JP-BERT ONNX graphs with TTS.cpp GGML execution.
- ONNX CUDA is active and not silently falling back to CPU. This run required
  CUDA 12 runtime libraries to be present in `LD_LIBRARY_PATH`; without them,
  the benchmark fails instead of recording a CPU fallback as a CUDA result.
- ONNX CUDA uses `cudnn_conv_algo_search=HEURISTIC`. The previous `DEFAULT`
  setting triggered a slow CUDA convolution path for the app-default
  `tempoDynamicsScale=1.0` SDP run on this RTX 3060, raising short and medium
  RTF above `1.0` even though CUDA was active.
- GGML Plugin EP Vulkan uses `precision=fast` in this run, which opts into the
  TTS.cpp Vulkan fast conv1d lowering while keeping ggml-vulkan F16/coopmat
  disabled. The four GGML columns differ only in JP-BERT GGUF storage and
  synthesis voice GGUF storage. `precision=accurate` remains the conservative
  direct-F32-conv mode and is not the performance number shown in this table.
- With the CUDA convolution search fix and CUDA 12 libraries available, ONNX
  CUDA is the fastest path on this RTX 3060 run. GGML Plugin EP Vulkan is still
  faster than ONNX CPU for all three text lengths and does not require NVIDIA
  CUDA runtime libraries.
- JP-BERT FP32 does not improve RTF on this RTX 3060 run. The default JP-BERT
  F16 `linear` artifact is therefore still the better practical choice because
  it keeps the same performance class while cutting the JP-BERT GGUF from
  `1,314,386,784` bytes to `710,407,072` bytes.
- FP16 voices are slightly faster than FP32 voices in this run, but the speedup
  is small because the remaining ceiling is decoder execution rather than model
  file size. The larger practical win is memory and disk footprint.

### GGUF Precision Matrix

The synthesis voice GGUF files were generated from the same AIVMX/ONNX `コハク`
model in the same benchmark run. The JP-BERT FP32 GGUF is a local benchmark
artifact derived from the all-F32 baseline; the default production artifact
remains the HF F16 `linear` GGUF.

| JP-BERT GGUF | voice GGUF | JP-BERT size / tensors | voice size / tensors | short samples | medium samples | long samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 `linear` | FP16 voices | `710,407,072` bytes / `250 F32 + 144 F16` | `129,812,864` bytes / `574 F32 + 326 F16` | `56,962` | `90,261` | `345,994` |
| FP16 `linear` | FP32 voices | `710,407,072` bytes / `250 F32 + 144 F16` | `248,034,656` bytes / `900 F32` | `56,960` | `90,261` | `345,994` |
| FP32 | FP16 voices | `1,314,386,784` bytes / `394 F32` | `129,812,864` bytes / `574 F32 + 326 F16` | `56,960` | `90,261` | `345,994` |
| FP32 | FP32 voices | `1,314,386,784` bytes / `394 F32` | `248,034,656` bytes / `900 F32` | `56,962` | `90,261` | `345,482` |

The FP16 voice cache is about `47.7%` smaller than the FP32 voice cache, and the
JP-BERT F16 `linear` cache is about `45.9%` smaller than the JP-BERT FP32
baseline. Under the app-default `tempoDynamicsScale=1.0` settings used here,
the adopted JP-BERT F16 `linear` + FP16 voices path differs from ONNX CPU only
by `+2` samples on the short text and matches medium/long sample counts.

### Precision Path Validation

This validation fixes `tempoDynamicsScale=0.0`, `noise_scale=0.0`, and
`noise_scale_w=0.0` so ONNX CPU and GGML output length can be compared without
sampling noise. It uses the same Linux RTX 3060 environment and the same three
texts as the benchmark table above.

| GGML path | Vulkan math | short RTF | medium RTF | long RTF | sample-count delta vs ONNX CPU | decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `precision=accurate` | direct F32 conv, F16/coopmat disabled | `0.168` | `0.166` | `0.148` | `0 / 0 / 0` | Too slow for the performance target |
| `precision=fast` | fast conv lowering, F16/coopmat disabled | `0.116` | `0.092` | `0.061` | `0 / 0 / 0` | Adopted path |
| experimental true-F16 fast | fast conv lowering, F16/coopmat enabled | `0.078` | `0.060` | `0.042` | `-2 / +528 / +723` | Rejected: changes duration |

Audio PCM deltas for the adopted `precision=fast` path against ONNX CPU were:
short `rmse=0.00088`, medium `rmse=0.00448`, and long `rmse=0.00332`, with
identical output sample counts for all three texts. This points to the conv
lowering as the correct performance lever; enabling ggml-vulkan F16/coopmat is
not safe for this model because it changes duration.

### GGML Vulkan Profile Run (2026-06-28)

This profile keeps the performance-oriented mixed-precision synthesis GGUF
cache: F16 for Style-Bert-VITS2 weights except embeddings, norms, decoder
upsample weights, biases, and style vectors. The generated synthesis GGUF for
this run was `129,812,864` bytes with `574 F32` tensors and `326 F16` tensors.
The decoder upsample exception is intentional: allowing those tensors to become
F16 moved a `CONV_TRANSPOSE_1D` decoder node to CPU and regressed RTF.

Strict backend validation passed with `TTS_BACKEND_STRICT=1`, confirming the
short-sentence decoder graph stayed on `Vulkan0` instead of falling back to CPU.

Run settings:

- Measurement date: 2026-06-28, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=1`
- Backend: `onnx-ggml-vulkan`, `precision=fast`
- Device pin: `GGML_VK_VISIBLE_DEVICES=1`
- TTS settings: `tempoDynamicsScale=1.0`, `noise_scale=0.0`,
  `noise_scale_w=0.0`
- Profile env: `STYLE_BERT_VITS2_DEBUG_TIMINGS=1`,
  `STYLE_BERT_VITS2_PROFILE_DECODER_NODES=1`

RTF results:

| text length | RTF | output samples |
| --- | ---: | ---: |
| short | `0.134` | `56,962` |
| medium | `0.099` | `90,261` |
| long | `0.064` | `345,994` |

Measured-run phase timings:

| text length | text encoder | duration predictor | SDP condition | SDP reverse | latent | decoder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | `4.539 ms` | `1.232 ms` | `1.068 ms` | `2.307 ms` | `51.503 ms` | `68.486 ms` |
| medium | `4.454 ms` | `1.155 ms` | `1.218 ms` | `2.428 ms` | `44.964 ms` | `94.897 ms` |
| long | `10.334 ms` | `1.533 ms` | `2.050 ms` | `4.189 ms` | `98.597 ms` | `309.778 ms` |

Decoder hot operators from the measured runs:

| text length | top decoder operators |
| --- | --- |
| short | `MUL_MAT 18.746 ms`, `IM2COL 16.202 ms`, `CONV_TRANSPOSE_1D 10.173 ms`, `ADD 10.004 ms`, `RESHAPE 7.076 ms` |
| medium | `MUL_MAT 29.390 ms`, `IM2COL 25.576 ms`, `CONV_TRANSPOSE_1D 16.568 ms`, `ADD 8.843 ms`, `RESHAPE 6.932 ms` |
| long | `IM2COL 93.853 ms`, `MUL_MAT 90.215 ms`, `CONV_TRANSPOSE_1D 75.883 ms`, `ADD 24.709 ms`, `LEAKY_RELU 13.740 ms` |

Conclusion: the remaining performance ceiling is synthesis decoder execution,
especially decoder `IM2COL`, `MUL_MAT`, and `CONV_TRANSPOSE_1D`. JP-BERT,
duration predictor, and SDP condition/reverse are not the dominant runtime
costs on this RTX 3060 Vulkan path. Full synthesis FP16 is not the next lever
unless ggml-vulkan can keep those decoder kernels on Vulkan and preserve output
duration parity.

### Audio Preview

These AAC files are representative outputs for qualitative review. They are not
included in the RTF timing window.

Baseline ONNX outputs:

| text length | ONNX CPU | ONNX CUDA |
| --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a) |

GGML Vulkan precision matrix:

| text length | JP-BERT FP16 + voices FP16 | JP-BERT FP16 + voices FP32 | JP-BERT FP32 + voices FP16 | JP-BERT FP32 + voices FP32 |
| --- | --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp16_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp16-voices-fp32_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp16_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan-jpbert-fp32-voices-fp32_long.m4a) |

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
| medium | `今日はいい天気ですね。` | 11 |
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
  --text "今日はいい天気ですね。" `
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
export TTS_CPP_NATIVE_LIBRARY_DIRS="<colon-separated dirs containing libtts.so and ggml libs>"
export JP_BERT_FP32_GGUF_PATH="<path-to-jp-bert-fp32.gguf>"
export BENCHMARK_OUTPUT_JSON="docs/res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json"
export BENCHMARK_AUDIO_WAV_DIR="<path-to-temporary-wav-output-dir>"
```

Run ONNX CPU, ONNX CUDA, and the ONNX GGML Plugin EP Vulkan JP-BERT/voice
precision matrix in one process:

```bash
LD_LIBRARY_PATH="${TTS_CPP_NATIVE_LIBRARY_DIRS}:${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=1 \
uv run python tools/benchmark_onnx_ggml_provider.py \
  --aivmx_path "$AIVMX_PATH" \
  --style_id "$STYLE_ID" \
  --backend onnx-cpu \
  --backend onnx-cuda \
  --backend onnx-ggml-vulkan-jpbert-fp16-voices-fp16 \
  --backend onnx-ggml-vulkan-jpbert-fp16-voices-fp32 \
  --backend onnx-ggml-vulkan-jpbert-fp32-voices-fp16 \
  --backend onnx-ggml-vulkan-jpbert-fp32-voices-fp32 \
  --text "テストです。" \
  --text "今日はいい天気ですね。" \
  --text "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_LIBRARY_PATH" \
  --ggml_jp_bert_fp32_gguf_path "$JP_BERT_FP32_GGUF_PATH" \
  --ggml_vulkan_device 0 \
  --ggml_vulkan_precision fast \
  --tempo_dynamics_scale 1.0 \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_OUTPUT_JSON" \
  --audio_output_dir "$BENCHMARK_AUDIO_WAV_DIR"
```

Convert the representative WAV files to AAC/M4A for the Markdown audio preview:

```bash
for wav in "$BENCHMARK_AUDIO_WAV_DIR"/*.wav; do
  base="$(basename "$wav" .wav)"
  ffmpeg -y -hide_banner -loglevel error \
    -i "$wav" \
    -c:a aac \
    -b:a 128k \
    -movflags +faststart \
    "docs/res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/${base}.m4a"
done
```

Strict provider checks:

- `onnx-cpu` must select `CPUExecutionProvider`
- `onnx-directml` must select `DmlExecutionProvider`
- `onnx-cuda` must select `CUDAExecutionProvider`
- `onnx-ggml-vulkan-jpbert-fp16-voices-fp16` must select
  `AivisGgmlExecutionProvider`
- `onnx-ggml-vulkan-jpbert-fp16-voices-fp32` must select
  `AivisGgmlExecutionProvider`
- `onnx-ggml-vulkan-jpbert-fp32-voices-fp16` must select
  `AivisGgmlExecutionProvider`
- `onnx-ggml-vulkan-jpbert-fp32-voices-fp32` must select
  `AivisGgmlExecutionProvider`

If ONNX Runtime silently falls back to CPU, the script fails instead of
recording a misleading GPU result.
