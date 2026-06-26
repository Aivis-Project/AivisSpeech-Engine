# ONNX GGML Linux 构建说明

这份文档只覆盖当前已验证的 Linux x64 路径：构建 AivisSpeech Engine，使 App 可以通过 ONNX Runtime Plugin EP 调用 TTS.cpp / ggml Vulkan 后端。Windows 和 macOS 不是本阶段目标。

## 产物结构

最终给 App 使用的是 PyInstaller 打包后的 Engine 目录：

```text
dist/run/
  run
  lib/
    libtts.so
    libggml*.so*
  onnxruntime_ep_aivis_ggml/
    lib/
      libaivis_ggml_onnx_ep.so
```

App 在 Linux 下启动 GGML ONNX 后端时使用这些相对路径：

```text
--onnx_provider ggml
--ggml_tts_server_backend vulkan
--ggml_native_library_path lib/libtts.so
--onnx_ep_library_path onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so
```

这里的路径必须相对 Engine 根目录可解析。不要依赖 `LD_LIBRARY_PATH` 作为成功标准，打包后的 `.so` 需要通过 `$ORIGIN` rpath 在 `dist/run/lib` 内自洽解析。

## 依赖来源

| 依赖 | 来源 | 用途 |
| --- | --- | --- |
| AivisSpeech Engine | 当前仓库 | Engine 主程序、ONNX Provider 选择、AIVM/AIVMX 到 GGUF 缓存转换、PyInstaller 打包 |
| ONNX GGML Plugin EP | `experimental/onnxruntime-ep-aivis-ggml` | 注册 `AivisGgmlExecutionProvider`，把已支持的 JP-BERT / synthesis ONNX 图转交给 TTS.cpp |
| TTS.cpp | `https://github.com/clawd20130/TTS.cpp.git`，当前 pin `0c6678415023c44d52dcf322827c33d36a352cb2` | 提供 `libtts.so`、ggml runtime、Vulkan 后端和 Style-Bert-VITS2 C API |
| ONNX Runtime headers | `https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz` | 只用于编译 Plugin EP；运行时仍使用 Engine Python 环境中的 `onnxruntime` |
| Vulkan SDK | LunarG `1.3.296.0` | 提供较新的 `glslc` 和 Vulkan CMake 查找路径，避免系统 SDK 太旧导致 ggml Vulkan 编译失败 |
| `libvulkan-dev` | Linux 发行版包管理器 | Vulkan loader / headers 基础依赖 |
| `patchelf` | Linux 发行版包管理器 | 把打包进 `dist/run/lib` 的 `.so` rpath 改成 `$ORIGIN` |
| `xz-utils` | Linux 发行版包管理器 | 解压 LunarG Vulkan SDK |
| `uv` + Python build group | 本仓库 `pyproject.toml` | 安装 Engine build 依赖并执行 PyInstaller |

CI 中这些版本写在 [.github/workflows/build-engine.yml](../.github/workflows/build-engine.yml)，本地复现时也应以那里为准。

## 系统依赖

Ubuntu 22.04/24.04 上至少需要：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  libvulkan-dev \
  patchelf \
  xz-utils
```

如果本机 `glslc` 不存在或版本太旧，使用和 CI 一致的 LunarG SDK：

```bash
VULKAN_SDK_VERSION=1.3.296.0
VULKAN_SDK_ARCHIVE="vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
mkdir -p download build/vulkan-sdk
curl -sSL \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/${VULKAN_SDK_ARCHIVE}" \
  -o "download/${VULKAN_SDK_ARCHIVE}"
tar -xf "download/${VULKAN_SDK_ARCHIVE}" -C build/vulkan-sdk
export VULKAN_SDK="$(find "$PWD/build/vulkan-sdk" -path '*/bin/glslc' -type f | head -n 1 | xargs dirname | xargs dirname)"
export CMAKE_PREFIX_PATH="${VULKAN_SDK}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PATH="${VULKAN_SDK}/bin:${PATH}"
```

## 本地构建

下面的命令使用占位路径，不要把本机绝对路径写入文档或提交记录：

```bash
ENGINE_DIR=<repo>/AivisSpeech-Engine
TTS_CPP_DIR=<repo>/TTS.cpp
```

### 1. 构建 TTS.cpp runtime

```bash
git clone --recursive https://github.com/clawd20130/TTS.cpp.git "$TTS_CPP_DIR"
git -C "$TTS_CPP_DIR" checkout 0c6678415023c44d52dcf322827c33d36a352cb2
git -C "$TTS_CPP_DIR" submodule update --init --recursive

cmake \
  -S "$TTS_CPP_DIR" \
  -B "$TTS_CPP_DIR/build-aivis-linux-vulkan" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTTS_BUILD_EXAMPLES=OFF \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_BUILD_RPATH='$ORIGIN' \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'

cmake --build "$TTS_CPP_DIR/build-aivis-linux-vulkan" --target tts --parallel
```

成功后需要能找到：

```text
$TTS_CPP_DIR/build-aivis-linux-vulkan/src/libtts.so
$TTS_CPP_DIR/build-aivis-linux-vulkan/ggml/src/libggml*.so*
$TTS_CPP_DIR/build-aivis-linux-vulkan/ggml/src/ggml-vulkan/libggml*.so*
```

### 2. 准备 ONNX Runtime headers

Plugin EP 只需要 ONNX Runtime C/C++ headers 编译，不需要把 release 包里的 `libonnxruntime.so` 打进 Engine：

```bash
cd "$ENGINE_DIR"

ORT_VERSION=1.26.0
ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_DIR="$PWD/build/onnxruntime-${ORT_VERSION}"

mkdir -p download "$ORT_DIR"
curl -sSL \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}" \
  -o "download/${ORT_ARCHIVE}"
tar -xzf "download/${ORT_ARCHIVE}" -C "$ORT_DIR" --strip-components=1 --exclude='*/lib/*'

export ORT_INCLUDE_DIR="$(dirname "$(find "$ORT_DIR" -name onnxruntime_cxx_api.h -type f | head -n 1)")"
test -f "$ORT_INCLUDE_DIR/onnxruntime_cxx_api.h"
```

### 3. 构建 Plugin EP

```bash
cmake \
  -S "$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/native" \
  -B "$ENGINE_DIR/build/onnx-ggml-native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_INCLUDE_DIR="$ORT_INCLUDE_DIR"

cmake --build "$ENGINE_DIR/build/onnx-ggml-native" --config Release --parallel
cmake --install "$ENGINE_DIR/build/onnx-ggml-native" --config Release \
  --prefix "$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/src"
```

成功后需要能找到：

```text
$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/src/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so
```

### 4. 打包 Engine

`run.spec` 会从下面三个环境变量收集 Plugin EP、`libtts.so` 和 ggml 依赖库：

```bash
cd "$ENGINE_DIR"

AIVIS_ONNX_GGML_REQUIRED=1 \
AIVIS_TTS_CPP_LIBRARY_PATH="$TTS_CPP_DIR/build-aivis-linux-vulkan/src/libtts.so" \
AIVIS_TTS_CPP_LIBRARY_DIRS="$TTS_CPP_DIR/build-aivis-linux-vulkan/src:$TTS_CPP_DIR/build-aivis-linux-vulkan/ggml/src:$TTS_CPP_DIR/build-aivis-linux-vulkan/ggml/src/ggml-vulkan" \
uv run --group build pyinstaller --noconfirm run.spec
```

`AIVIS_ONNX_GGML_REQUIRED=1` 很重要：如果 Plugin EP 或 TTS.cpp sidecar 没有被打包，构建会直接失败，而不是生成一个运行时才坏的 Engine。

## 验证

先检查动态库是否都落在 `dist/run` 内：

```bash
test -x dist/run/run
test -f dist/run/lib/libtts.so
test -f dist/run/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so
ldd dist/run/lib/libtts.so
```

`ldd` 输出里的 `libggml*.so*` 应该解析到 `dist/run/lib`。如果解析到 TTS.cpp build 目录，或者显示 `not found`，说明 rpath 没有被正确 patch，需要先安装 `patchelf` 后重新打包。

然后不带 `LD_LIBRARY_PATH` 启动 Engine：

```bash
env -u LD_LIBRARY_PATH ./dist/run/run \
  --host 127.0.0.1 \
  --port 10109 \
  --onnx_provider ggml \
  --ggml_tts_server_backend vulkan \
  --ggml_native_library_path lib/libtts.so \
  --onnx_ep_library_path onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so \
  --disable_sentry
```

另一个终端检查：

```bash
curl -fsS http://127.0.0.1:10109/version
```

日志中应出现类似信息：

```text
Registered ONNX Runtime Plugin EP library aivis_onnx_plugin_ep
Using external ONNX Runtime Plugin EP AivisGgmlExecutionProvider before fallback providers ['CPUExecutionProvider'].
Application startup complete.
```

## App 集成

App 侧保持最小改动：生产配置仍然传 Engine 参数，Linux 下只把默认 Windows 风格 sidecar 路径映射到 Linux 产物名。

Linux 需要确认：

- App 使用的 Engine 目录里有 `run`，不是 `run.exe`。
- TTS.cpp runtime 是 `lib/libtts.so`。
- Plugin EP 是 `onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so`。
- Engine 启动参数包含 `--onnx_provider ggml` 和 `--ggml_tts_server_backend vulkan`。

只要 Engine 目录满足上面的产物结构，App 不需要知道 AIVM/AIVMX 到 GGUF 的转换细节；首次加载模型时由 Engine 的 ONNX GGML cache 逻辑处理。

## 常见问题

`std::format` 编译失败：

使用了未 pin 的 TTS.cpp 或旧编译器不支持 `<format>`。当前验证过的 TTS.cpp pin 是 `0c6678415023c44d52dcf322827c33d36a352cb2`。

`glslc` 找不到或 ggml Vulkan 编译失败：

安装 LunarG Vulkan SDK `1.3.296.0`，并确保 `VULKAN_SDK`、`CMAKE_PREFIX_PATH`、`PATH` 指向 SDK。

Plugin EP 没有被打进 `dist/run`：

确认已经执行 `cmake --install ... --prefix experimental/onnxruntime-ep-aivis-ggml/src`，并在 PyInstaller 构建时设置了 `AIVIS_ONNX_GGML_REQUIRED=1`。

启动时找不到 `libggml*.so*`：

确认安装了 `patchelf` 后重新打包。`run.spec` 会把 TTS.cpp sidecar 的 rpath 设置为 `$ORIGIN`。

本地生成但不应提交的目录：

```text
build/
dist/
download/
```
