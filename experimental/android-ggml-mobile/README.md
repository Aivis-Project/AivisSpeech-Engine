# Android GGML Mobile Benchmark

This experiment runs Style-Bert-VITS2 GGUF workloads on Android through
TTS.cpp. The current benchmark supports two bundle modes:

- synthesis-only: phone IDs, tone IDs, language IDs, JP-BERT features, style
  vector, and synthesis scalar parameters are precomputed on the host.
- device-side JP-BERT + synthesis: phone IDs, tone IDs, language IDs, JP-BERT
  `input_ids`, `word2ph`, style vector, and scalar parameters are packed into a
  bundle, then Android runs JP-BERT GGUF feature extraction and synthesis GGUF.

Raw text normalization, OpenJTalk/G2P, and tokenizer execution are still outside
the Android process. Do not describe the second mode as full raw text-to-wave on
device until those frontend pieces are ported too.

## Current Status

Branch: `feat/mobile-support`

### Physical Android Device Result (2026-07-02)

Measured on a local Android physical device:

| Item | Value |
| --- | --- |
| Device | PLP110 |
| Android | 16 / API 36 |
| ABI | arm64-v8a |
| Vulkan device | Adreno (TM) 840, Qualcomm Technologies Inc. Adreno Vulkan Driver |
| ggml-vulkan feature signal, NDK `glslc` | UMA `1`, fp16 `1`, bf16 `0`, warp size `64`, shared memory `32768`, int dot `0`, matrix cores `none` |
| ggml-vulkan feature signal, `shaderc 2026.2` `glslc` | UMA `1`, fp16 `0`, bf16 `0`, warp size `64`, shared memory `32768`, int dot `1`, matrix cores `KHR_coopmat`, then `No suitable matrix core mode found` |
| TTS.cpp | `46099d9`; ggml submodule `a78c352b` |
| Android build toolchain | NDK `28.2.13676358` + Khronos Vulkan-Headers `v1.3.275` + host `shaderc 2026.2` `glslc` |
| Voice | `a59cb814-0083-4369-8542-f51a29e72af7` / style `888753760` |
| Device-side production voice GGUF | `a59cb814-0083-4369-8542-f51a29e72af7-1_2_0-98004407f97f5608.gguf`, 129,814,912 bytes, F16 `no-embed-norm-no-ups` |
| Device-side JP-BERT GGUF | `jp-bert-968bba0e105e1d10.gguf`, 710,407,072 bytes, F16 `linear` |
| Earlier synthesis-only voice GGUF | `mao-full-sdp.gguf`, 239 MB |

The physical-device result uses the deterministic consistency bundle, not the
audio-preview bundle:

- `tempoDynamicsScale=1.0`
- `noise_scale=0.0`
- `noise_scale_w=0.0`
- `warmup_runs=1`
- `runs=1`
- warmup texts differ from measured texts

The important Android finding is that mobile `precision=fast` by itself is not
the same as the Linux Plugin EP default. Bare `fast` leaves ggml-vulkan runtime
F16 enabled on this Adreno device (`fp16: 1`), while the Linux default
`vulkan_math_mode=coopmat` keeps runtime F16 disabled and enables cooperative
matrix kernels only when the GPU supports them. With NDK `28.2.13676358`'s
bundled `glslc`, ggml-vulkan compiles without `GL_KHR_cooperative_matrix` and
`GL_EXT_integer_dot_product` shader support, so the runtime reports `int dot: 0`
and `matrix cores: none`.

Use a newer host shader compiler for the Android build. `shaderc 2026.2`
successfully compiles ggml's `GL_KHR_cooperative_matrix` and
`GL_EXT_integer_dot_product` feature tests while the Android target still uses
the NDK Vulkan C headers and `libvulkan.so`. On Adreno 840, this enables
`int dot: 1`. KHR cooperative matrix is also detected, but ggml disables it
after probing the device shapes because Adreno exposes F16 accumulate, not the
F16-input/F32-accumulate mode that ggml currently requires for its duration-safe
path.

The adopted Android match for the Linux duration-safe intent is therefore:

```bash
STYLE_BERT_VITS2_VULKAN_PRECISION=fast
STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION=fast
GGML_VK_DISABLE_F16=1
```

Do not set `GGML_VK_DISABLE_COOPMAT` for this comparison. On this device it has
no useful effect because ggml probes KHR cooperative matrix and disables it for
lack of a suitable F32-accumulate mode, but leaving it unset preserves the same
intent as Linux `vulkan_math_mode=coopmat`.

Device-side JP-BERT + synthesis results against the CPU baseline use the
production FP16 deployment assets. The RTF window includes both Android
JP-BERT GGUF feature extraction and Android synthesis GGUF execution:

| Text | CPU RTF | Vulkan RTF | Vulkan JP-BERT / synthesis seconds | sample delta | PCM RMSE / corr | BERT RMSE vs host ONNX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | `2.737` | `1.498` | `0.768 / 0.728` | `0` | `0.00135 / 0.99970` | `0.00177` |
| medium | `1.686` | `0.825` | `0.503 / 0.915` | `0` | `0.00208 / 0.99953` | `0.00120` |
| long | `1.867` | `0.560` | `0.750 / 3.447` | `0` | `0.00343 / 0.99867` | `0.00084` |

The earlier NDK-`glslc` all-device run used the same runtime env but compiled
without integer-dot and cooperative-matrix shader support. It matched sample
counts but was slower: RTF `2.554 / 1.398 / 1.000`.

The Adreno 840 cooperative matrix probe reports:

- `VK_KHR_cooperative_matrix=yes`, `feature.cooperativeMatrix=1`,
  `supportedStages=compute|all`.
- F16 modes are `M=64`, `N=64/32/16`, `K=16`, with
  `A=B=C=Result=float16`.
- F32 modes are `M=64`, `N=64/32/16`, `K=8`, with
  `A=B=C=Result=float32`.
- INT8 modes are `M=64`, `N=64/32/16`, `K=32`, with
  `sint8/sint8 -> sint32` and `uint8/uint8 -> uint32`.
- There is no `float16 x float16 -> float32` cooperative matrix mode and no
  mixed `float16 x float32` mode. Forcing the local experimental
  `GGML_VK_ALLOW_F16_COOPMAT_ONLY=1` path first failed during JP-BERT pipeline
  creation at `matmul_f16_f32_f16acc_aligned_l` with
  `vk::Device::createComputePipeline: ErrorUnknown`. After constraining that
  experiment to pure F16/F16 unaligned small-tile coopmat, it ran but failed
  consistency: device-side JP-BERT BERT RMSE was `1.32054 / 1.33430 / 1.22560`
  for short/medium/long, and output samples changed to
  `24576 / 51200 / 234496`. A voice-only run using bundled BERT features
  produced the same wrong sample counts, so Adreno F16-accumulate coopmat is not
  a duration-safe path for this model.
- Pure F32 cooperative matrix is a separate hardware mode on Adreno 840, but it
  is not a drop-in switch for the current ggml-vulkan build. The generated KHR
  `cm1` matmul shader family is produced from the `fp16=true` shader generator
  path; the true `fp16=false` F32 shader family is emitted only for non-coopmat
  `_fp32` kernels. Enabling pure F32 coopmat would require adding a new
  `fp16=false + COOPMAT` shader family, wiring C++ pipeline creation to those
  arrays, and testing FP32 GGUF assets. It would also require an FP32 JP-BERT GGUF
  for full device-side comparison; the current production assets are JP-BERT F16
  `linear` plus FP16 synthesis voice GGUF.

Bare device-side `fast` without `GGML_VK_DISABLE_F16=1` reported `fp16: 1` and
did not produce the first measured sample after more than 90 seconds, so it is
not a valid all-device consistency path.

The earlier synthesis-only physical-device consistency results below used
host-precomputed JP-BERT features and a 239 MB full-sdp voice GGUF. They are
kept as exploratory driver evidence, not as the production FP16 deployment
profile:

| Backend / mode | Runtime Vulkan features | short delta / RTF / RMSE | medium delta / RTF / RMSE | long delta / RTF / RMSE | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| ggml Vulkan `fast` | fp16 `1`, matrix cores `none` | `-512` / `6.420` / `0.05118` | `+1024` / `7.847` / `1.00069` | `+512` / `8.404` / `0.07811` | invalid |
| ggml Vulkan `accurate` | fp16 `0`, coopmat disabled | `0` / `3.814` / `0.00069` | `0` / `0.728` / `0.00125` | `0` / `0.614` / silent output | invalid |
| ggml Vulkan `accurate`, safe env | fp16 `0`, async/graph-opt/multi-add disabled | `0` / `1.283` / `0.00069` | `0` / `0.844` / `0.00125` | `0` / `0.937` / `0.00132` | valid fallback |
| ggml Vulkan `fast`, `GGML_VK_DISABLE_F16=1` | fp16 `0`, matrix cores `none` | `0` / `0.940` / `0.00075` | `0` / `0.516` / `0.00166` | `0` / `0.478` / `0.00146` | adopted for Android Adreno |

In that synthesis-only run, the fast-no-F16 Android Adreno path had identical
CPU/GPU output sample counts for all three measured texts and PCM correlation
above `0.99969`.

Measured on Android Emulator x86_64 with host GPU rendering:

| Item | Value |
| --- | --- |
| Android | 16 |
| ABI | x86_64 |
| Android Emulator | 36.6.11.0 |
| Emulator GPU | `-gpu host` |
| Host GPU exposed to guest | AMD Radeon 780M Graphics, RADV |
| SurfaceFlinger | Google AMD OpenGL ES Translator |
| Guest Vulkan | AMD Radeon 780M Graphics (RADV PHOENIX), Vulkan 1.3 |
| Guest Vulkan feature signal | `shaderFloat16=1`, `storageBuffer16BitAccess=1`, subgroup size 64 |
| TTS.cpp backend | ggml CPU and ggml Vulkan |
| Voice | `a59cb814-0083-4369-8542-f51a29e72af7` / style `888753760` |
| Voice GGUFs tested | FP16 129 MB, FP32 248 MB |

CPU synthesis is valid but too slow for realtime:

| Backend | Voice | Text | Output samples | Output duration | Elapsed | RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ggml CPU | FP16 | short | 44,032 | 0.998 s | 4.221 s | 4.227 |
| ggml CPU | FP16 | medium | 76,288 | 1.730 s | 7.400 s | 4.278 |
| ggml CPU | FP16 | long | 326,656 | 7.407 s | 32.391 s | 4.373 |

Vulkan currently runs but is not correct on Android Emulator for this model. The
raw RTF looks faster, but generated audio length and peak values do not match
the CPU baseline, so these numbers must not be treated as usable performance.

| Backend | Voice | Mode | Text | Output samples | CPU baseline samples | Raw RTF | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ggml Vulkan | FP16 | fast | short | 3,584 | 44,032 | 3.524 | invalid |
| ggml Vulkan | FP16 | fast | medium | 19,968 | 76,288 | 0.462 | invalid |
| ggml Vulkan | FP16 | fast | long | 84,480 | 326,656 | 0.170 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | short | 18,432 | 44,032 | 0.357 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | medium | 512 | 76,288 | 8.375 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | long | 512 | 326,656 | 8.012 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | short | 512 | 44,032 | 16.762 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | medium | 512 | 76,288 | 16.038 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | long | 57,344 | 326,656 | 0.703 | invalid |

Updating Android Emulator from 36.5.11 to 36.6.11 did not fix correctness. It
did improve capability detection after the local ggml-vulkan core-feature patch:
the benchmark now prints `fp16: 1` for fast mode, but output lengths are still
wrong. The current failure is therefore not explained by the older emulator
version alone.

The likely class of issue is graphics/Vulkan compatibility between Android
Emulator host-GPU translation, the host RADV driver, and ggml-vulkan graph
execution. Android's own emulator documentation warns that unsupported or
problematic graphics acceleration can cause crashes or incorrect output and
recommends software rendering modes when host graphics has compatibility issues.
The ggml/llama.cpp community also has active reports of Android Vulkan poor
performance and Vulkan-only wrong-output regressions on specific driver/model
combinations. For this project, Android Emulator Vulkan should stay a diagnostic
target only until CPU/GPU sample-count and audio parity match.

Android 37.0 was also tested with the 16 KB page-size x86_64 image:

| Item | Value |
| --- | --- |
| Android | 17 / API 37 |
| System image | `system-images;android-37.0;google_apis_ps16k;x86_64` |
| Page size | 16 KB |
| Android Emulator | 36.6.11.0 |
| Host GPU exposed to guest | Goldfish GFXStream over AMD Radeon 780M Graphics, RADV |
| Binary LOAD alignment | 16 KB (`0x4000`) |

CPU remained valid on Android 37:

| Backend | Voice | Text | Output samples | Output duration | Elapsed | RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ggml CPU | FP16 | short | 44,032 | 0.998 s | 4.598 s | 4.605 |
| ggml CPU | FP16 | medium | 76,288 | 1.730 s | 7.644 s | 4.419 |
| ggml CPU | FP16 | long | 326,656 | 7.407 s | 32.949 s | 4.448 |

Android 37 did not fix Vulkan correctness:

| Backend | Voice | Mode | Text | Output samples | CPU baseline samples | Raw RTF | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ggml Vulkan | FP16 | fast | short | 512 | 44,032 | 13.588 | invalid |
| ggml Vulkan | FP16 | fast | medium | 31,744 | 76,288 | 0.330 | invalid |
| ggml Vulkan | FP16 | fast | long | 512 | 326,656 | 10.995 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | short | 512 | 44,032 | 10.589 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | medium | 512 | 76,288 | 10.556 | invalid |
| ggml Vulkan | FP16 | fast, different warmup enabled | long | 512 | 326,656 | 11.026 | invalid |
| ggml Vulkan | FP16 | fast, integer dot disabled | short | 512 | 44,032 | 14.264 | invalid |
| ggml Vulkan | FP16 | fast, integer dot disabled | medium | 1,536 | 76,288 | 5.101 | invalid |
| ggml Vulkan | FP16 | fast, integer dot disabled | long | 512 | 326,656 | 12.483 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | short | 27,648 | 44,032 | 0.816 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | medium | 8,192 | 76,288 | 0.969 | invalid |
| ggml Vulkan | FP32 | accurate, async/graph-opt/multi-add/f16/coopmat disabled | long | 512 | 326,656 | 10.679 | invalid |

On Android 37, `adb shell cmd gpu vkjson` is not safe on this host/AVD
combination. It caused the emulator process to abort with
`Unhandled Vulkan structure type [1000584001]`. Use the emulator log,
SurfaceFlinger, and ggml-vulkan's own device log for GPU verification instead.

Two Android-specific compatibility fixes are applied only inside this benchmark
build:

- `--n-threads 0` is normalized to an automatic hardware thread count before
  calling TTS.cpp, because current TTS.cpp crashes if `n_threads=0` reaches
  `ggml_threadpool_new`.
- ggml-vulkan is locally patched at CMake configure time so Android Vulkan core
  1.1/1.2 feature exposure does not require requesting unavailable extension
  names during `vkCreateDevice`.

## Build

From the repository root:

```bash
export ANDROID_NDK_ROOT=/path/to/android-sdk/ndk/28.2.13676358
export TTS_CPP_ROOT=/path/to/TTS.cpp
export ANDROID_ABI=arm64-v8a

# macOS/Homebrew host:
brew install shaderc spirv-headers
export VULKAN_GLSLC_EXECUTABLE="$(brew --prefix shaderc)/bin/glslc"
export SPIRV_HEADERS_DIR="$(brew --prefix spirv-headers)/share/cmake/SPIRV-Headers"

# On Linux, use a recent LunarG Vulkan SDK or distro shaderc package:
# export VULKAN_GLSLC_EXECUTABLE=/path/to/recent/glslc
# export SPIRV_HEADERS_DIR=/usr/share/cmake/SPIRV-Headers

mkdir -p build/_deps
git clone --depth 1 --branch v1.3.275 \
  https://github.com/KhronosGroup/Vulkan-Headers.git \
  build/_deps/Vulkan-Headers-1.3.275

cmake -S experimental/android-ggml-mobile \
  -B "build/android-ggml-mobile-${ANDROID_ABI}" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM=android-30 \
  -DTTS_CPP_ROOT="$TTS_CPP_ROOT" \
  -DVulkan_GLSLC_EXECUTABLE="$VULKAN_GLSLC_EXECUTABLE" \
  -DSPIRV-Headers_DIR="$SPIRV_HEADERS_DIR" \
  -DVULKAN_HPP_INCLUDE_DIR="$PWD/build/_deps/Vulkan-Headers-1.3.275/include" \
  -DSPIRV_HEADERS_INCLUDE_DIR="$ANDROID_NDK_ROOT/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include"

cmake --build "build/android-ggml-mobile-${ANDROID_ABI}" \
  --target aivis_ggml_mobile_bench aivis_vulkan_coopmat_probe \
  -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
```

`VULKAN_HPP_INCLUDE_DIR` must match the NDK Vulkan C headers. NDK
`28.2.13676358` exposes `VK_HEADER_VERSION 275`, so use Khronos
Vulkan-Headers `v1.3.275`. Do not mix NDK 28 with a newer host header such as
Homebrew Vulkan-Headers `1.4.350`, because `vulkan.hpp` will assert a header
version mismatch.

The host `glslc` is separate from the target Vulkan headers. NDK
`28.2.13676358`'s bundled `shader-tools` `glslc` is too old for this benchmark:
it rejects ggml's `GL_KHR_cooperative_matrix` and `GL_EXT_integer_dot_product`
feature tests. A newer host `glslc`, verified with Homebrew `shaderc 2026.2`,
keeps the target headers at Vulkan `1.3.275` while enabling ggml-vulkan's
integer-dot and cooperative-matrix shader variants.

Run the probe target on device when changing phones or drivers:

```bash
adb push build/android-ggml-mobile-${ANDROID_ABI}/aivis_vulkan_coopmat_probe \
  /data/local/tmp/aivis-ggml-mobile-real/aivis_vulkan_coopmat_probe
adb shell 'chmod 755 /data/local/tmp/aivis-ggml-mobile-real/aivis_vulkan_coopmat_probe && \
  /data/local/tmp/aivis-ggml-mobile-real/aivis_vulkan_coopmat_probe'
```

For an x86_64 emulator build, set `ANDROID_ABI=x86_64` and use a separate build
directory.

## Prepare Inputs

The bundle contains warmup texts that differ from the measured short, medium,
and long texts. Keep qualitative audio preview and deterministic consistency
validation as separate bundles.

Audio preview bundle, using the model's natural stochastic defaults:

```bash
uv run python tools/export_mobile_ggml_bundle.py \
  --aivmx_path "$AIVIS_AIVMX_PATH" \
  --style_id 888753760 \
  --output build/android-ggml-mobile-bundle/mao-default.aivis_mobile_bundle
```

Consistency bundle, using fixed noise overrides. The current bundle format is
v2 and includes JP-BERT `input_ids` plus `word2ph` so Android can run JP-BERT
GGUF locally:

```bash
uv run python tools/export_mobile_ggml_bundle.py \
  --aivmx_path "$AIVIS_AIVMX_PATH" \
  --style_id 888753760 \
  --tempo_dynamics_scale 1.0 \
  --noise_scale 0 \
  --noise_scale_w 0 \
  --output build/android-ggml-mobile-bundle/mao-consistency-v2.aivis_mobile_bundle
```

The consistency bundle is for sample-count and PCM parity only. Do not use it
for qualitative audio preview, because forcing `noise_w=0.0` is known to change
the perceived audio quality.

## Run On Physical Device

Push benchmark assets:

```bash
adb shell 'mkdir -p /data/local/tmp/aivis-ggml-mobile-real'
adb push build/android-ggml-mobile-arm64-v8a/aivis_ggml_mobile_bench \
  /data/local/tmp/aivis-ggml-mobile-real/aivis_ggml_mobile_bench
adb push build/android-ggml-mobile-bundle/mao-consistency-v2.aivis_mobile_bundle \
  /data/local/tmp/aivis-ggml-mobile-real/mao-consistency-v2.aivis_mobile_bundle
adb push "$VOICE_GGUF_PATH" /data/local/tmp/aivis-ggml-mobile-real/mao-voice-fp16.gguf
adb push "$JP_BERT_GGUF_PATH" /data/local/tmp/aivis-ggml-mobile-real/jp-bert-fp16-linear.gguf
adb shell 'chmod 755 /data/local/tmp/aivis-ggml-mobile-real/aivis_ggml_mobile_bench'
```

CPU consistency baseline, including device-side JP-BERT:

```bash
adb shell 'cd /data/local/tmp/aivis-ggml-mobile-real && \
  TTS_BACKEND=cpu \
  ./aivis_ggml_mobile_bench \
    --model mao-voice-fp16.gguf \
    --jp-bert-model jp-bert-fp16-linear.gguf \
    --bundle mao-consistency-v2.aivis_mobile_bundle \
    --backend cpu \
    --cpu-only \
    --runs 1 \
    --warmup-runs 1 \
    --output-json android-ggml-mobile-full-device-fp16-cpu.json \
    --audio-dir android-ggml-mobile-full-device-fp16-cpu'
```

Android Adreno adopted Vulkan path, aligned with the Linux duration-safe intent
and including device-side JP-BERT:

```bash
adb shell 'cd /data/local/tmp/aivis-ggml-mobile-real && \
  TTS_BACKEND=vulkan \
  TTS_BACKEND_STRICT=1 \
  STYLE_BERT_VITS2_VULKAN_PRECISION=fast \
  STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION=fast \
  GGML_VK_DISABLE_F16=1 \
  ./aivis_ggml_mobile_bench \
    --model mao-voice-fp16.gguf \
    --jp-bert-model jp-bert-fp16-linear.gguf \
    --bundle mao-consistency-v2.aivis_mobile_bundle \
    --backend vulkan \
    --precision fast \
    --runs 1 \
    --warmup-runs 1 \
    --output-json android-ggml-mobile-full-device-fp16-vulkan-fast-no-f16-jpfast.json \
    --audio-dir android-ggml-mobile-full-device-fp16-vulkan-fast-no-f16-jpfast'
```

If fast-no-F16 regresses on another Android driver, use the conservative
fallback:

```bash
adb shell 'cd /data/local/tmp/aivis-ggml-mobile-real && \
  TTS_BACKEND=vulkan \
  TTS_BACKEND_STRICT=1 \
  STYLE_BERT_VITS2_VULKAN_PRECISION=accurate \
  STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION=accurate \
  GGML_VK_DISABLE_ASYNC=1 \
  GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 \
  GGML_VK_DISABLE_MULTI_ADD=1 \
  ./aivis_ggml_mobile_bench \
    --model mao-voice-fp16.gguf \
    --jp-bert-model jp-bert-fp16-linear.gguf \
    --bundle mao-consistency-v2.aivis_mobile_bundle \
    --backend vulkan \
    --precision accurate \
    --runs 1 \
    --warmup-runs 1 \
    --output-json android-ggml-mobile-consistency-vulkan-accurate-safe.json \
    --audio-dir android-ggml-mobile-consistency-vulkan-accurate-safe'
```

Compare sample counts and PCM deltas against CPU before using RTF. A Vulkan run
with different output length, saturated peaks, or silent output is invalid even
when the raw RTF looks good.

## Run On Emulator

Start the emulator with host GPU rendering rather than `-no-window`:

```bash
export DISPLAY=:0
export XAUTHORITY="$(ls -t /run/user/1000/.mutter-Xwaylandauth.* | head -1)"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

emulator -avd miraa_api36_a \
  -no-audio \
  -gpu host \
  -qt-hide-window \
  -no-boot-anim \
  -no-snapshot
```

Push benchmark assets:

```bash
adb push build/android-ggml-mobile-x86_64/aivis_ggml_mobile_bench \
  /data/local/tmp/aivis_ggml_mobile_bench
adb push build/android-ggml-mobile-bundle/mao-default.aivis_mobile_bundle \
  /data/local/tmp/mao-default.aivis_mobile_bundle
adb push "$VOICE_GGUF_PATH" /data/local/tmp/mao-voice.gguf
adb shell chmod 755 /data/local/tmp/aivis_ggml_mobile_bench
```

Valid CPU baseline:

```bash
adb shell 'cd /data/local/tmp && \
  TTS_BACKEND=cpu \
  ./aivis_ggml_mobile_bench \
    --model mao-voice.gguf \
    --bundle mao-default.aivis_mobile_bundle \
    --backend cpu \
    --cpu-only \
    --runs 1 \
    --warmup-runs 0 \
    --output-json android-ggml-mobile-cpu.json \
    --audio-dir android-ggml-mobile-audio-cpu'
```

Current Vulkan repro command:

```bash
adb shell 'cd /data/local/tmp && \
  TTS_BACKEND=vulkan \
  TTS_BACKEND_STRICT=1 \
  STYLE_BERT_VITS2_VULKAN_PRECISION=fast \
  ./aivis_ggml_mobile_bench \
    --model mao-voice.gguf \
    --bundle mao-default.aivis_mobile_bundle \
    --backend vulkan \
    --precision fast \
    --runs 1 \
    --warmup-runs 0 \
    --output-json android-ggml-mobile-vulkan.json \
    --audio-dir android-ggml-mobile-audio-vulkan'
```

The Vulkan output must be compared against CPU output duration/sample counts
before using RTF. On this emulator, the Vulkan path is currently faster but not
correct.

## References

- Android Emulator hardware acceleration:
  https://developer.android.com/studio/run/emulator-acceleration
- Android Emulator troubleshooting for graphics/Vulkan issues:
  https://developer.android.com/studio/run/emulator-troubleshooting
- Android Vulkan implementation requirements:
  https://source.android.com/docs/core/graphics/implement-vulkan
- ggml/llama.cpp RADV Vulkan performance notes:
  https://github.com/ggml-org/llama.cpp/discussions/23295
- ggml/llama.cpp Android Vulkan performance discussion:
  https://github.com/ggml-org/llama.cpp/discussions/9464
- ggml/llama.cpp Vulkan wrong-output regression example:
  https://github.com/ggml-org/llama.cpp/issues/17013
