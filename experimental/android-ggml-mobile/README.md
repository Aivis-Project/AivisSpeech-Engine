# Android GGML Mobile Benchmark

This experiment keeps the Aivis frontend outside the device and runs only the
Style-Bert-VITS2 synthesis GGUF on Android through TTS.cpp. It matches the
current ONNX GGML EP boundary: phone IDs, tone IDs, language IDs, JP-BERT
features, style vector, and synthesis scalar parameters are precomputed on the
host and packed into a small binary bundle.

The goal is to measure the mobile GGUF synthesis path without including Python,
OpenJTalk, tokenizer, or JP-BERT ONNX latency.

## Current Status

Branch: `feat/mobile-support`

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

mkdir -p build/_deps
git clone --depth 1 --branch v1.3.275 \
  https://github.com/KhronosGroup/Vulkan-Headers.git \
  build/_deps/Vulkan-Headers-1.3.275

cmake -S experimental/android-ggml-mobile \
  -B build/android-ggml-mobile-x86_64 \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=x86_64 \
  -DANDROID_PLATFORM=android-30 \
  -DTTS_CPP_ROOT="$TTS_CPP_ROOT" \
  -DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc \
  -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DVULKAN_HPP_INCLUDE_DIR="$PWD/build/_deps/Vulkan-Headers-1.3.275/include" \
  -DSPIRV_HEADERS_INCLUDE_DIR=/usr/include

cmake --build build/android-ggml-mobile-x86_64 \
  --target aivis_ggml_mobile_bench \
  -j"$(nproc)"
```

## Prepare Inputs

The bundle contains warmup texts that differ from the measured short, medium,
and long texts.

```bash
uv run python tools/export_mobile_ggml_bundle.py \
  --aivmx_path "$AIVIS_AIVMX_PATH" \
  --style_id 888753760 \
  --output build/android-ggml-mobile-bundle/mao-default.aivis_mobile_bundle
```

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
