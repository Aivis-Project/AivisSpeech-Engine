# ONNX GGML Plugin EP Benchmark

This PR review benchmark primarily compares the ONNX paths affected by the
minimal GGML Plugin EP integration:

- ONNX CPU: existing `StyleBertVITS2TTSEngine` ONNX path with `CPUExecutionProvider`
- ONNX DirectML: existing ONNX path with `DmlExecutionProvider`
- ONNX CUDA: existing ONNX path with `CUDAExecutionProvider`
- ONNX GGML Vulkan: existing ONNX path with `AivisGgmlExecutionProvider`
  claiming the synthesis and JP-BERT ONNX graphs. The Linux run now uses the
  production default GGML path: JP-BERT F16 `linear`, FP16 synthesis voice GGUF,
  `precision=fast`, and `vulkan_math_mode=coopmat`.
- ONNX GGML Metal: existing ONNX path with `AivisGgmlExecutionProvider`
  claiming the synthesis and JP-BERT ONNX graphs on macOS. The current Metal
  validation uses the same memory-saving assets as the production profile:
  JP-BERT F16 `linear` plus the FP16 synthesis voice GGUF.

RTF is `elapsed_seconds / output_duration_seconds`; lower is better. Audio
encoding is intentionally excluded from measured runs.

Benchmark rule: warmup synthesis must use texts that are different from the
short/medium/long measured texts. This avoids warming text-specific frontend,
symbol, graph, or runtime caches with the exact sample later used for timing.

## Deployment Default and Memory Profile

The deployable GGML path should keep the same memory-saving assets used by this
benchmark: JP-BERT F16 `linear` and FP16 synthesis voice GGUF caches. Do not
ship F32 voice or JP-BERT GGUF files for the production/default profile unless a
specific parity investigation requires them. The expected deployment knobs are:

- `backend=vulkan`
- `precision=fast`
- `vulkan_math_mode=coopmat`
- `claim_synthesis_graph=1`
- `claim_jp_bert_graph=1`
- synthesis voice GGUF generated with the F16 `no-embed-norm-no-ups` recipe
- JP-BERT GGUF from the F16 `linear` artifact

This keeps resident model memory lower while preserving the duration-safe path
validated below. Runtime Vulkan F16 remains disabled in the default profile;
only the GGUF storage precision and cooperative matrix kernels are part of the
deployment default.

Deployment smoke validation on 2026-07-01 used the local `myoland.tts` tailnet
debug stack with the same class of TTS.cpp Style-Bert-VITS2 GGUF assets:

- host: Ubuntu 26.04, AMD Radeon 780M selected with
  `MESA_VK_DEVICE_SELECT=1002:1900!`
- phoneme source: `sbv2-aivis`
- TTS.cpp decoder: `Vulkan:0`
- TTS.cpp JP-BERT: `Vulkan:0`
- sidecar path: `style_bert_vits2_ggml`, `ggml_path=symbol-front`
- output: mono PCM WAV, 44.1 kHz, 8.998 s, 793,644 bytes
- smoke RTF: `0.156`

This smoke proves the deployable SBV2 -> Style-Bert GGML -> TTS.cpp Vulkan
chain can generate audio with the FP16 GGUF deployment profile. It is not a
replacement for the benchmark tables below; it only records that the runtime
stack is wired correctly before documenting the deployment default.

## Linux RTX 3060 + AMD 780M Local Run (2026-06-29)

Raw results are stored in
[linux-rtx3060-cuda-ggml-cpu.json](res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json)
and
[linux-780m-ggml.json](res/onnx-ggml-plugin-benchmark/linux-780m-ggml.json).

### Scope

- Measurement date: 2026-06-29, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- Warmup uses separate non-measured texts; measured short/medium/long samples
  are never used for warmup.
- AudioQuery: `tempoDynamicsScale=1.0`, matching the Engine `/audio_query`
  default used by the app
- Style-Bert-VITS2 noise settings: benchmark arguments leave
  `noise_scale` and `noise_scale_w` unset, so synthesis uses the model defaults
  (`noise=0.6`, `noise_w=0.8`). This is intentional for audio preview; forcing
  `noise_w=0.0` was isolated as the source of the metallic/electric artifact in
  the previous documentation audio.
- Engine: `6a43469` on `feat/onnx-ggml-minimal-upstream`, with ONNX Runtime
  `1.26.0` compatibility and FP16 GGUF cache defaults
- TTS.cpp: `94792ed`; ggml submodule `a78c352bb70b`
- CUDA provider option: `cudnn_conv_algo_search=HEURISTIC`
- Model: AIVMX/ONNX `まお` model, version `1.2.0`
- Style: `888753760` (`ノーマル`)
- GGML model path: AIVMX/ONNX is converted to synthesis GGUF by the Plugin EP
  cache path using the F16 `no-embed-norm-no-ups` recipe. JP-BERT uses the
  default `kevinzhow/style-bert-vits2-gguf`
  `frontend/style-bert-vits2-jp-bert.gguf` F16 `linear` artifact.
- GGML provider options: `backend=vulkan`, `precision=fast`,
  `vulkan_math_mode=coopmat`,
  `device=1` for RTX 3060 or `device=0` for AMD 780M,
  `claim_synthesis_graph=1`, `claim_jp_bert_graph=1`,
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
| Vulkan dGPU | NVIDIA GeForce RTX 3060, Vulkan API `1.4.329`, UMA `0`, fp16 `0`, bf16 `1`, warp size `32`, shared memory `49152`, int dot `1`, matrix cores `NV_coopmat2` |
| Vulkan iGPU | AMD Radeon 780M Graphics (RADV PHOENIX), UMA `1`, fp16 `0`, bf16 `0`, warp size `64`, shared memory `65536`, int dot `1`, matrix cores `KHR_coopmat` |
| GGML Vulkan device pins | RTX 3060: `--ggml_vulkan_device 1`; AMD 780M: `--ggml_vulkan_device 0` |

### RTF Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | GGML Vulkan RTX 3060 RTF | GGML Vulkan AMD 780M RTF |
| --- | ---: | ---: | ---: | ---: |
| short | `0.339` | `0.186` | `0.089` | `0.163` |
| medium | `0.237` | `0.118` | `0.063` | `0.135` |
| long | `0.210` | `0.034` | `0.045` | `0.108` |
| overall mean | `0.262` | `0.113` | `0.066` | `0.135` |

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
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1",
    "ggml_jp_bert_precision": "fp16-linear"
  }
}
```

Interpretation:

- The Plugin EP path keeps the normal ONNX frontend and replaces only the
  supported synthesis and JP-BERT ONNX graphs with TTS.cpp GGML execution.
- This Linux refresh uses natural Style-Bert-VITS2 stochastic defaults for the
  saved audio previews. The current JSON records `noise_scale=null`,
  `noise_scale_w=null`, and `truth_comparison_enabled=false`; deterministic PCM
  comparison should be run separately with fixed noise parameters.
- ONNX CUDA is active and not silently falling back to CPU. This run required
  CUDA 12 runtime libraries to be present in `LD_LIBRARY_PATH`; without them,
  the benchmark fails instead of recording a CPU fallback as a CUDA result.
- ONNX CUDA uses `cudnn_conv_algo_search=HEURISTIC`. The previous `DEFAULT`
  setting triggered a slow CUDA convolution path for the app-default
  `tempoDynamicsScale=1.0` SDP run on this RTX 3060, raising short and medium
  RTF above `1.0` even though CUDA was active.
- GGML Plugin EP Vulkan uses `precision=fast` and `vulkan_math_mode=coopmat`.
  The Vulkan probe reported `matrix cores: NV_coopmat2` for the RTX 3060 and
  `matrix cores: KHR_coopmat` for the AMD 780M. Runtime F16 remains disabled in
  this mode; only cooperative matrix kernels are enabled.
- The standalone Android TTS.cpp benchmark has a different knob surface:
  `STYLE_BERT_VITS2_VULKAN_PRECISION=fast` enables fast conv lowering but does
  not disable ggml-vulkan runtime F16 by itself. JP-BERT has a separate
  `STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION` switch. To match this Linux
  duration-safe intent on Android, run both components in `fast` mode with
  `GGML_VK_DISABLE_F16=1` and leave coopmat enabled when the device supports
  matrix cores. On the tested Adreno 840 physical device, NDK `28.2.13676358`'s
  bundled `glslc` was too old and compiled ggml-vulkan without integer-dot or
  cooperative-matrix shader support (`int dot: 0`, `matrix cores: none`).
  Rebuilding with Homebrew `shaderc 2026.2` enabled `int dot: 1` and detected
  `matrix cores: KHR_coopmat`, but ggml then disabled cooperative matrix because
  Adreno exposes F16-accumulate modes instead of the F16-input/F32-accumulate
  mode required by the current duration-safe ggml path.
- The Android physical-device all-GGUF run on 2026-07-02 used the same
  deployment asset classes: JP-BERT F16 `linear` plus the FP16 synthesis voice
  GGUF. It still used host-generated `input_ids`/phone/tone/language/style
  bundle data, but JP-BERT feature extraction and synthesis both ran on the
  device. With `STYLE_BERT_VITS2_VULKAN_PRECISION=fast`,
  `STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION=fast`, and
  `GGML_VK_DISABLE_F16=1`, Vulkan matched CPU sample counts for all three texts
  and measured RTF `1.498 / 0.825 / 0.560` for short/medium/long after the
  host `glslc` fix. The older NDK-`glslc` build measured
  `2.554 / 1.398 / 1.000` with the same runtime env. Bare Android `fast` with
  runtime F16 enabled did not produce the first measured sample after more than
  90 seconds and is not a valid comparison path. Forcing the local experimental
  F16-only coopmat path first failed during JP-BERT pipeline creation at
  `matmul_f16_f32_f16acc_aligned_l`. After constraining that experiment to pure
  F16/F16 unaligned small-tile coopmat, it ran but failed consistency: JP-BERT
  BERT RMSE was `1.32054 / 1.33430 / 1.22560`, and output samples changed to
  `24576 / 51200 / 234496`. A voice-only run using bundled BERT features
  produced the same wrong sample counts, so Adreno F16-accumulate coopmat is not
  duration-safe for this model.
- Pure F32 cooperative matrix is present in the Adreno 840 Vulkan properties
  (`A=B=C=Result=float32`, `M=64`, `N=64/32/16`, `K=8`), but current ggml-vulkan
  does not generate a true `fp16=false + COOPMAT` KHR matmul shader family. Its
  KHR `cm1` matmul shaders are generated from the `fp16=true` path, while the
  true F32 `_fp32` shaders are non-coopmat. Testing pure F32 acceleration would
  therefore require new shader generation, C++ pipeline wiring, and FP32 GGUF
  assets, including FP32 JP-BERT for a full Android device-side parity run.
- With the CUDA convolution search fix and CUDA 12 libraries available, ONNX
  CUDA is active and still the fastest path on the long sample. GGML Plugin EP
  Vulkan is faster than ONNX CPU for all three text lengths and faster than ONNX
  CUDA on the short and medium samples on the RTX 3060 run, without requiring
  NVIDIA CUDA runtime libraries. The AMD 780M iGPU path is still faster than
  ONNX CPU on all three text lengths, but slower than the RTX 3060 dGPU path.
- The benchmark no longer lists the older JP-BERT/voice precision experiments.
  The practical default is the smaller JP-BERT F16 `linear` artifact plus FP16
  synthesis voice cache.
- Saved audio preview files are AAC transcodes of the representative WAV output
  from this same run and are not included in the RTF timing window.

### Audio Quality Fix

The previous documentation audio forced `noise_scale=0.0` and
`noise_scale_w=0.0` while keeping `tempoDynamicsScale=1.0`. That setting is not
the app's natural synthesis path and was isolated as the cause of the audible
metallic/electric artifact, especially in the long sample. The current Linux
preview audio therefore leaves both noise arguments unset and records that state
in the JSON as `noise_scale=null` and `noise_scale_w=null`.

For deterministic provider parity checks, rerun the benchmark with fixed noise
parameters and treat those WAV files as validation artifacts only, not as the
qualitative preview audio.

### Precision Path Validation (Historical)

This validation fixes `tempoDynamicsScale=0.0`, `noise_scale=0.0`, and
`noise_scale_w=0.0` so ONNX CPU and GGML output length can be compared without
sampling noise. It uses the same Linux RTX 3060 environment and the same three
texts as the benchmark table above, but predates the current `まお` refresh and
is retained only as precision-path decision history.

| GGML path | Vulkan math | short RTF | medium RTF | long RTF | sample-count delta vs ONNX CPU | decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `precision=accurate`, `vulkan_math_mode=f32` | direct F32 conv, F16/coopmat disabled | `0.168` | `0.166` | `0.148` | `0 / 0 / 0` | Too slow for the performance target |
| `precision=fast`, `vulkan_math_mode=f32` | fast conv lowering, F16/coopmat disabled | `0.116` | `0.092` | `0.061` | `0 / 0 / 0` | Historical adopted path |
| `precision=fast`, `vulkan_math_mode=fp16-coopmat` | fast conv lowering, F16/coopmat enabled | `0.078` | `0.060` | `0.042` | `-2 / +528 / +723` | Rejected: changes duration |

Audio PCM deltas for the adopted `precision=fast` path against ONNX CPU were:
short `rmse=0.00088`, medium `rmse=0.00448`, and long `rmse=0.00332`, with
identical output sample counts for all three texts. This points to the conv
lowering as the correct performance lever; enabling ggml-vulkan runtime F16
with coopmat is not safe for this model because it changes duration. Current
Linux Lunar Lake testing enables coopmat without Vulkan runtime F16 by default
(`vulkan_math_mode=coopmat`).

### GGML Vulkan Profile Run (Historical, 2026-06-28)

This profile keeps the performance-oriented mixed-precision synthesis GGUF
cache: F16 for Style-Bert-VITS2 weights except embeddings, norms, decoder
upsample weights, biases, and style vectors. The generated synthesis GGUF for
this earlier `コハク` run was `129,812,864` bytes with `574 F32` tensors and
`326 F16` tensors. The decoder upsample exception is intentional: allowing
those tensors to become F16 moved a `CONV_TRANSPOSE_1D` decoder node to CPU and
regressed RTF.

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

| text length | ONNX CPU | ONNX CUDA | GGML Vulkan RTX 3060 | GGML Vulkan AMD 780M |
| --- | --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cpu_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-cuda_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/onnx-ggml-vulkan_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/linux-780m/onnx-ggml-vulkan_long.m4a) |

## macOS Apple M1 Pro Metal Local Run (2026-07-02)

Raw results are stored in
[macos-m1pro-metal-fp16-appdefault.json](res/onnx-ggml-plugin-benchmark/macos-m1pro-metal-fp16-appdefault.json)
and
[macos-m1pro-metal-fp16-consistency.json](res/onnx-ggml-plugin-benchmark/macos-m1pro-metal-fp16-consistency.json).
The previous F32 synthesis-weight comparison remains available as
[macos-m1pro-metal-f32-appdefault.json](res/onnx-ggml-plugin-benchmark/macos-m1pro-metal-f32-appdefault.json)
and
[macos-m1pro-metal-f32-consistency.json](res/onnx-ggml-plugin-benchmark/macos-m1pro-metal-f32-consistency.json).
The latest TTS.cpp direct decoder kernel iteration result is stored separately
in
[macos-m1pro-metal-direct-decoder-20260702.json](res/onnx-ggml-plugin-benchmark/macos-m1pro-metal-direct-decoder-20260702.json).

### Scope

- Measurement date: 2026-07-02, Asia/Shanghai
- Profile: `warmup_runs=1`, `runs=3` for the app-default RTF/audio-preview
  run; `warmup_runs=1`, `runs=1` for the deterministic consistency run
- Warmup uses separate non-measured texts; measured short/medium/long samples
  are never used for warmup.
- AudioQuery for app-default RTF/audio preview: `tempoDynamicsScale=1.0`, with
  `noise_scale` and `noise_scale_w` unset, matching the natural app path used
  by the Linux and Windows preview runs.
- AudioQuery for consistency: `tempoDynamicsScale=0.0`, `noise_scale=0.0`, and
  `noise_scale_w=0.0`; these WAV files are validation artifacts, not the
  qualitative preview audio.
- OS: macOS `27.0` beta build `26A5368g`, arm64
- GPU: Apple M1 Pro, 14 GPU cores, Metal 4
- ONNX Runtime: `onnxruntime 1.26.0`; built-in providers include
  `CoreMLExecutionProvider` and `CPUExecutionProvider`
- Engine: `06affa3d` on `feat/onnx-ggml-minimal-upstream`
- TTS.cpp: latest `main` at `7562414`, with local ggml changes for FP16
  `GGML_OP_KOKORO_CONV_1D` weights, `GGML_METAL=ON`, and shared
  `libtts.dylib` rebuilt before the benchmark
- Model: AIVMX/ONNX `まお` model, version `1.2.0`
- Style: `888753760` (`ノーマル`)
- GGML provider options: `backend=metal`, `precision=fast`,
  `vulkan_math_mode=coopmat`, `claim_synthesis_graph=1`,
  `claim_jp_bert_graph=1`, `eager_load_model=1`
- GGML assets: JP-BERT uses the local F16 `linear` GGUF artifact. Synthesis
  uses the production/default FP16 converter recipe
  `tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1`.

The Metal runtime log reported `simdgroup matrix mul. = true`,
`use fusion = true`, `use concurrency = true`, and `use graph optimize = true`
on Apple M1 Pro. It also reported `has tensor = false`; the newer Metal tensor
API path is disabled for this pre-M5/pre-A19 device.

The FP16 synthesis voice path required a joint TTS.cpp/ggml refactor because
the fused decoder conv1d extension previously accepted only F32 weights. The
local change allows F16 weights at `ggml_kokoro_conv_1d_ex`, CPU fallback, Metal
`supports_op`, Metal pipeline selection, Metal stride math, and the MSL kernel
entry point. The smoke run compiled and used `kernel_kokoro_conv_1d_f16`.

### RTF Results

| text length | ONNX CPU RTF | ONNX GGML Metal RTF |
| --- | ---: | ---: |
| short | `0.294` | `0.250` |
| medium | `0.260` | `0.198` |
| long | `0.252` | `0.173` |
| overall mean | `0.269` | `0.207` |

Provider evidence from the app-default run:

```json
{
  "onnx-cpu": {
    "active_providers": ["CPUExecutionProvider"]
  },
  "onnx-ggml-metal": {
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1",
    "ggml_jp_bert_precision": "fp16-linear"
  }
}
```

### Consistency Validation

The consistency run fixes `tempoDynamicsScale=0.0`, `noise_scale=0.0`, and
`noise_scale_w=0.0` to compare deterministic PCM output against ONNX CPU.

| text length | sample-count delta vs ONNX CPU | RMSE | max abs diff | correlation | Metal RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | `0` | `0.00485` | `0.07620` | `0.99941` | `0.239` |
| medium | `0` | `0.00567` | `0.09808` | `0.99945` | `0.194` |
| long | `0` | `0.00722` | `0.27725` | `0.99864` | `0.172` |

Interpretation:

- Metal is active through the Plugin EP and is not silently using ONNX CPU for
  the claimed JP-BERT and synthesis graphs.
- With FP16 synthesis GGUF and JP-BERT F16 `linear`, Metal is roughly `1.2x` to
  `1.5x` faster than ONNX CPU on this M1 Pro app-default run.
- Deterministic consistency preserves output duration exactly for all three
  measured texts. The remaining PCM deltas are expected backend arithmetic
  differences, not duration drift.
- The previous F32 synthesis-weight JSON records a faster Metal run
  (`0.185 / 0.147 / 0.128` RTF for short/medium/long), but that result is not
  reproducible with the current rebuilt `libtts.dylib`: a same-command F32
  rerun measured `0.250 / 0.200 / 0.173`. Current F32 and FP16 voice paths are
  therefore performance-equivalent on this M1 Pro.
- A follow-up profile found the current Metal bottleneck in
  `GGML_OP_KOKORO_CONV_1D`: decoder node profiling reported roughly
  `670ms / 73` `KOKORO_CONV_1D` nodes out of a `916ms` decoder pass for the
  long text. Disabling the fused conv1d epilogue reduced raw
  `KOKORO_CONV_1D` time but added enough separate `ADD` and `LEAKY_RELU` nodes
  to regress end-to-end RTF, so the default full fusion remains the best tested
  Metal mode.
- The F16 Metal kernel now avoids an unnecessary `half -> float -> half`
  round trip when loading FP16 conv weights. A long-text A/B after the fix
  measured FP16 at `0.172` RTF and F32 at `0.174` RTF under the same rebuilt
  binary; this fixes the local FP16-specific regression, while broader Metal
  speedups still require `KOKORO_CONV_1D` kernel tuning.
- A later direct decoder kernel iteration added a parallel epilogue for the
  Metal fused conv1d kernel and lowered the fused-kernel default minimum output
  length from `8192` to `4096`. This is documented in the direct decoder
  section below because it bypasses ONNX Runtime and is not directly comparable
  to the Engine Plugin EP RTF table above.

### TTS.cpp Direct Decoder Kernel Benchmark

This benchmark is for Metal kernel iteration only. It bypasses ONNX Runtime,
does not run the Engine AudioQuery path, directly loads the rebuilt
`libtts.dylib`, constructs deterministic decoder inputs, and times
`style_bert_vits2_runner::decode()`.

Reproduction guardrail: use this direct decoder path when validating TTS.cpp or
ggml Metal kernel changes. Use `tools/benchmark_onnx_ggml_provider.py` only for
formal Engine/Plugin EP measurements, and first verify that the macOS ONNX
Runtime package has `libonnxruntime_providers_shared.dylib`. The attempted
Plugin EP smoke on 2026-07-02 stopped before inference because the active
`uv` environment had `onnxruntime 1.23.0`, providers
`CoreMLExecutionProvider`, `AzureExecutionProvider`, and
`CPUExecutionProvider`, and no `libonnxruntime_providers_shared.dylib`. That is
an ONNX Runtime packaging/registration blocker, not a Metal backend timing
result.

Configuration:

- Measurement date: 2026-07-02, Asia/Shanghai
- Device: Apple M1 Pro; Metal reported `simdgroup matrix mul. = true` and
  `has tensor = false`
- Engine: `12d508d7`; TTS.cpp: `7daeaea`; ggml submodule: `023ace57`, with
  local Metal conv1d changes
- Build: rebuilt latest local `libtts.dylib` via
  `cmake --build /Users/kevinzhow/Github/TTS.cpp/build-metal-shared --target tts kokoro-conv-1d-bench`
- Runtime: `TTS_BACKEND=metal`, `TTS_BACKEND_STRICT=1`,
  `STYLE_BERT_VITS2_METAL_FUSED_CONV1D=full`
- Kernel tuning: default `metal_tiled_conv1d_min_output=4096`; parallel
  epilogue enabled by default unless
  `GGML_METAL_KOKORO_CONV_1D_EPILOGUE=0|off|serial`
- Profile: `warmup=2`, `iters=9`, sample rate `44100`

Latest direct decoder result:

| precision | frames | output samples | median ms | min ms | max ms | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | `23` | `11776` | `40.313` | `36.950` | `44.079` | `0.150970` |
| FP16 | `64` | `32768` | `74.411` | `71.976` | `79.871` | `0.100144` |
| FP16 | `128` | `65536` | `129.557` | `128.076` | `140.692` | `0.087181` |
| FP16 | `294` | `150528` | `311.122` | `301.688` | `328.069` | `0.091149` |
| FP32 | `23` | `11776` | `37.830` | `37.355` | `38.565` | `0.141669` |
| FP32 | `64` | `32768` | `67.648` | `66.000` | `75.269` | `0.091042` |
| FP32 | `128` | `65536` | `129.267` | `128.047` | `130.629` | `0.086986` |
| FP32 | `294` | `150528` | `288.504` | `283.797` | `292.409` | `0.084523` |

Validation:

- Parallel epilogue vs serial epilogue produced bit-identical decoder output
  for FP16 frames `23` and `294` (`max_abs=0`, `rms=0`).
- Default `4096` threshold vs old `8192` threshold produced bit-identical FP16
  decoder output for frames `23`, `64`, `128`, and `294`.
- The standalone `kokoro-conv-1d-bench` passed with the same accepted numerical
  range as before; the worst printed difference was `max_abs=0.00045099854`,
  `rms=0.00013016167`.

Interpretation:

- The `4096` threshold improves short/medium synthetic decoder shapes because
  it fuses the `5888` and `11776` output-length stages while still avoiding
  very small `2352` output stages. The old `8192` threshold skipped too much
  short-shape work; very high thresholds such as `32768` split more long-shape
  graph nodes and regressed direct timing.
- FP16 and FP32 remain close on this direct decoder harness. FP32 is slightly
  faster in the latest run, so this result should be treated as a Metal conv1d
  scheduling target rather than proof of an end-to-end FP16 win.

### Audio Preview

These AAC files are representative app-default outputs for qualitative review.
They are not included in the RTF timing window.

| text length | ONNX CPU | ONNX GGML Metal |
| --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_short.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_short.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_short.m4a) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_medium.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_medium.m4a) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-cpu_long.m4a) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_long.m4a"></audio><br>[AAC](res/onnx-ggml-plugin-benchmark/audio/macos-m1pro-metal-fp16/onnx-ggml-metal_long.m4a) |

## Windows Intel Arc B580 Local Run (2026-06-29)

Raw results are stored in
[windows-arc-b580-fp16-matrix.json](res/onnx-ggml-plugin-benchmark/windows-arc-b580-fp16-matrix.json).

### Scope

- Measurement date: 2026-06-29, Asia/Tokyo
- Profile: `warmup_runs=1`, `runs=3`
- Warmup uses separate non-measured texts; measured short/medium/long samples
  are never used for warmup.
- AudioQuery: `tempoDynamicsScale=1.0`, matching the Engine `/audio_query`
  default used by the app
- Style-Bert-VITS2 noise settings: benchmark arguments leave
  `noise_scale` and `noise_scale_w` unset, so synthesis uses the model defaults
  (`noise=0.6`, `noise_w=0.8`) for qualitative audio preview.
- OS: Microsoft Windows 11 Home `10.0.26200`, 64-bit
- CPU: AMD Ryzen 5 5600, 6 cores / 12 threads
- GPU: Intel(R) Arc(TM) B580 Graphics, driver `32.0.101.8826`
- ONNX Runtime: `onnxruntime-directml 1.24.4`; providers include
  `DmlExecutionProvider`, `CPUExecutionProvider`
- Engine: `d7f9098d95b0` on `feat/onnx-ggml-minimal-upstream`
- TTS.cpp: `94792ed25996`; ggml submodule `a78c352bb70b`
- Model: AIVMX/ONNX Mao model, version `1.2.0`
- Style: `888753760`
- GGML provider options: `backend=vulkan`, `precision=fast`,
  `vulkan_math_mode=coopmat`, `claim_synthesis_graph=1`,
  `claim_jp_bert_graph=1`, `eager_load_model=1`
- GGML Vulkan device pin: `GGML_VK_VISIBLE_DEVICES=1`; the Vulkan probe saw
  one device, `Intel(R) Arc(TM) B580 Graphics`, with `fp16: 0` and
  `matrix cores: KHR_coopmat`

### RTF Results

| text length | ONNX CPU RTF | ONNX DirectML RTF | ONNX GGML Vulkan RTF |
| --- | ---: | ---: | ---: |
| short | `0.415` | `0.551` | `0.209` |
| medium | `0.347` | `0.862` | `0.164` |
| long | `0.243` | `0.220` | `0.038` |
| overall mean | `0.335` | `0.544` | `0.137` |

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
    "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
    "ggml_synthesis_converter_version": "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1",
    "ggml_jp_bert_precision": "fp16-linear"
  }
}
```

Interpretation:

- This Windows refresh follows the current benchmark rule: warmup texts differ
  from measured texts. Short and medium RTF therefore include more shape/cache
  first-use cost than the older same-text warm-run table.
- The Intel Arc B580 GGML Vulkan long sample is `0.038` RTF on the default
  production GGML path.
- The smaller JP-BERT FP16 `linear` plus FP16 voice cache remains the benchmark
  default.
- ONNX DirectML is active, but this run shows it is still very shape-sensitive
  for the Style-Bert-VITS2 app-default path. It is faster than CPU on the long
  sample, slower on short and medium, and slower than GGML Vulkan on all three
  measured text lengths.
- The current Windows audio preview uses natural Style-Bert-VITS2 stochastic
  defaults. The JSON records `noise_scale=null`, `noise_scale_w=null`, and
  `truth_comparison_enabled=false`; run deterministic validation separately
  when PCM parity is the goal.

### Audio Preview

These WAV files are representative outputs for qualitative review. They are not
included in the RTF timing window.

| text length | ONNX CPU | ONNX DirectML | ONNX GGML Vulkan |
| --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_short.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_short.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_short.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_short.wav) |
| medium | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_medium.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_medium.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_medium.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_medium.wav) |
| long | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-cpu_long.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-directml_long.wav) | <audio controls preload="none" src="res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_long.wav"></audio><br>[WAV](res/onnx-ggml-plugin-benchmark/audio/windows-arc-b580-fp16/onnx-ggml-vulkan_long.wav) |

## Historical Windows Intel Arc B580 Local Run (2026-06-27)

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
export STYLE_ID="888753760"
export CUDA12_NVIDIA_LIBS="<colon-separated CUDA 12/cuDNN library dirs>"
export AIVIS_GGML_ONNX_EP_LIBRARY_PATH="<path-to-libaivis_ggml_onnx_ep.so>"
export TTS_CPP_NATIVE_LIBRARY_PATH="<path-to-libtts.so>"
export TTS_CPP_NATIVE_LIBRARY_DIRS="<colon-separated dirs containing libtts.so and ggml libs>"
export BENCHMARK_RTX3060_OUTPUT_JSON="docs/res/onnx-ggml-plugin-benchmark/linux-rtx3060-cuda-ggml-cpu.json"
export BENCHMARK_RTX3060_AUDIO_WAV_DIR="<path-to-temporary-rtx3060-wav-output-dir>"
export BENCHMARK_780M_OUTPUT_JSON="docs/res/onnx-ggml-plugin-benchmark/linux-780m-ggml.json"
export BENCHMARK_780M_AUDIO_WAV_DIR="<path-to-temporary-780m-wav-output-dir>"
```

Run ONNX CPU, ONNX CUDA, and the ONNX GGML Plugin EP Vulkan default path in one
process. Leave `noise_scale` and `noise_scale_w` unset for the qualitative
audio-preview run; use deterministic noise overrides only for separate provider
parity validation.

```bash
LD_LIBRARY_PATH="${TTS_CPP_NATIVE_LIBRARY_DIRS}:${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
uv run python tools/benchmark_onnx_ggml_provider.py \
  --aivmx_path "$AIVMX_PATH" \
  --style_id "$STYLE_ID" \
  --backend onnx-cpu \
  --backend onnx-cuda \
  --backend onnx-ggml-vulkan \
  --text "テストです。" \
  --text "今日はいい天気ですね。" \
  --text "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。" \
  --warmup_text "測定用ではない短い文です。" \
  --warmup_text "ウォームアップのために別の文章を読み上げます。" \
  --warmup_text "測定対象とは異なる長めのウォームアップ文章です。バックエンドの初回処理だけを先に済ませます。" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_LIBRARY_PATH" \
  --ggml_vulkan_device 1 \
  --ggml_vulkan_precision fast \
  --ggml_vulkan_math_mode coopmat \
  --tempo_dynamics_scale 1.0 \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_RTX3060_OUTPUT_JSON" \
  --audio_output_dir "$BENCHMARK_RTX3060_AUDIO_WAV_DIR" \
  --skip_truth_comparison
```

Then rerun the GGML Plugin EP Vulkan path on the AMD 780M iGPU:

```bash
LD_LIBRARY_PATH="${TTS_CPP_NATIVE_LIBRARY_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
uv run python tools/benchmark_onnx_ggml_provider.py \
  --aivmx_path "$AIVMX_PATH" \
  --style_id "$STYLE_ID" \
  --backend onnx-ggml-vulkan \
  --text "テストです。" \
  --text "今日はいい天気ですね。" \
  --text "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。" \
  --warmup_text "測定用ではない短い文です。" \
  --warmup_text "ウォームアップのために別の文章を読み上げます。" \
  --warmup_text "測定対象とは異なる長めのウォームアップ文章です。バックエンドの初回処理だけを先に済ませます。" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_LIBRARY_PATH" \
  --ggml_vulkan_device 0 \
  --ggml_vulkan_precision fast \
  --ggml_vulkan_math_mode coopmat \
  --tempo_dynamics_scale 1.0 \
  --warmup_runs 1 \
  --runs 3 \
  --output_json "$BENCHMARK_780M_OUTPUT_JSON" \
  --audio_output_dir "$BENCHMARK_780M_AUDIO_WAV_DIR" \
  --skip_truth_comparison
```

Convert the representative WAV files to AAC/M4A for the Markdown audio preview:

```bash
for wav in "$BENCHMARK_RTX3060_AUDIO_WAV_DIR"/*.wav; do
  base="$(basename "$wav" .wav)"
  ffmpeg -y -hide_banner -loglevel error \
    -i "$wav" \
    -c:a aac \
    -b:a 128k \
    -movflags +faststart \
    "docs/res/onnx-ggml-plugin-benchmark/audio/linux-rtx3060/${base}.m4a"
done

for wav in "$BENCHMARK_780M_AUDIO_WAV_DIR"/*.wav; do
  base="$(basename "$wav" .wav)"
  ffmpeg -y -hide_banner -loglevel error \
    -i "$wav" \
    -c:a aac \
    -b:a 128k \
    -movflags +faststart \
    "docs/res/onnx-ggml-plugin-benchmark/audio/linux-780m/${base}.m4a"
done
```

Strict provider checks:

- `onnx-cpu` must select `CPUExecutionProvider`
- `onnx-directml` must select `DmlExecutionProvider`
- `onnx-cuda` must select `CUDAExecutionProvider`
- `onnx-ggml-vulkan` must select `AivisGgmlExecutionProvider`

If ONNX Runtime silently falls back to CPU, the script fails instead of
recording a misleading GPU result.
