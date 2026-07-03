# ONNX GGML ビルド・検証手順

このドキュメントは、ONNX Runtime Plugin EP から TTS.cpp / ggml を呼び出す
AivisSpeech Engine パッケージを、Windows x64・macOS x64/arm64・Linux x64
でビルドして検証するための手順です。

Android / モバイル向けの検証はこの PR の対象外です。

## 対象

この PR で追加する経路は明示的に `--onnx_provider ggml` を指定した場合だけ
有効になります。通常の `auto` / `cuda` / `directml` / CPU 経路は既存のままです。

ビルド成果物には次の sidecar を含めます。

| OS | Engine | TTS.cpp runtime | Plugin EP |
| --- | --- | --- | --- |
| Windows | `dist/run/run.exe` | `dist/run/lib/tts.dll` または `libtts.dll` | `dist/run/onnxruntime_ep_aivis_ggml/lib/aivis_ggml_onnx_ep.dll` |
| macOS | `dist/run/run` | `dist/run/lib/libtts.dylib` | `dist/run/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.dylib` |
| Linux | `dist/run/run` | `dist/run/lib/libtts.so` | `dist/run/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so` |

Linux では `run.spec` が `patchelf` で TTS.cpp / ggml sidecar の rpath を
`$ORIGIN` に変更します。パッケージ検証では、`LD_LIBRARY_PATH` に依存せず
`dist/run/lib` だけで解決できることを確認してください。

## バージョン

CI と同じ値を使うのが再現性のある基準です。

| 項目 | 値 |
| --- | --- |
| ONNX Runtime headers | `1.26.0` |
| TTS.cpp repository | `https://github.com/clawd20130/TTS.cpp.git` |
| TTS.cpp ref | `0c6678415023c44d52dcf322827c33d36a352cb2` |
| Vulkan SDK | `1.3.296.0` |

これらは [.github/workflows/build-engine.yml](../.github/workflows/build-engine.yml)
にも定義されています。

## 依存関係

| OS | 追加依存 |
| --- | --- |
| Linux | `libvulkan-dev`, `patchelf`, `xz-utils`, LunarG Vulkan SDK |
| Windows | MSVC, Chocolatey の `vulkan-sdk` |
| macOS | Xcode Command Line Tools, Homebrew の `gnu-sed`, `coreutils` |

macOS は TTS.cpp を `GGML_METAL=ON` でビルドします。Windows と Linux は
`GGML_VULKAN=ON` でビルドします。

## 共通ビルドフロー

以下の手順は CI の流れと同じです。ローカルパスは例なので、実行環境に合わせて
置き換えてください。

```bash
ENGINE_DIR=<AivisSpeech-Engine の checkout>
TTS_CPP_DIR="$ENGINE_DIR/build/TTS.cpp"
TTS_CPP_BUILD_DIR="$ENGINE_DIR/build/TTS.cpp-build"
ORT_VERSION=1.26.0
```

### 1. ONNX Runtime headers を準備する

Plugin EP のビルドには ONNX Runtime の C/C++ headers だけを使います。
release archive 内の `libonnxruntime` は Engine パッケージには入れません。

```bash
cd "$ENGINE_DIR"

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

Windows の Git Bash で実行する場合、CMake に渡す前に必要に応じて
`cygpath -w "$ORT_INCLUDE_DIR"` で Windows 形式のパスに変換してください。

### 2. Plugin EP をビルドする

```bash
cd "$ENGINE_DIR"

cmake \
  -S experimental/onnxruntime-ep-aivis-ggml/native \
  -B build/onnx-ggml-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_INCLUDE_DIR="$ORT_INCLUDE_DIR"

cmake --build build/onnx-ggml-native --config Release --parallel
cmake --install build/onnx-ggml-native --config Release \
  --prefix experimental/onnxruntime-ep-aivis-ggml/src
```

インストール後、各 OS に対応する native library が
`experimental/onnxruntime-ep-aivis-ggml/src/onnxruntime_ep_aivis_ggml/lib/`
に存在することを確認します。

### 3. TTS.cpp runtime をビルドする

```bash
rm -rf "$TTS_CPP_DIR" "$TTS_CPP_BUILD_DIR"
git clone --recursive https://github.com/clawd20130/TTS.cpp.git "$TTS_CPP_DIR"
git -C "$TTS_CPP_DIR" checkout 0c6678415023c44d52dcf322827c33d36a352cb2
git -C "$TTS_CPP_DIR" submodule update --init --recursive
```

Linux / Windows:

```bash
cmake \
  -S "$TTS_CPP_DIR" \
  -B "$TTS_CPP_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTTS_BUILD_EXAMPLES=OFF \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_BUILD_RPATH='$ORIGIN' \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'

cmake --build "$TTS_CPP_BUILD_DIR" --config Release --target tts --parallel
```

macOS:

```bash
cmake \
  -S "$TTS_CPP_DIR" \
  -B "$TTS_CPP_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTTS_BUILD_EXAMPLES=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_BUILD_RPATH='$ORIGIN' \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'

cmake --build "$TTS_CPP_BUILD_DIR" --config Release --target tts --parallel
```

### 4. Engine を PyInstaller でパッケージする

`run.spec` は次の環境変数から Plugin EP、TTS.cpp runtime、ggml 依存 library
を収集します。

```bash
export AIVIS_ONNX_GGML_REQUIRED=1
export AIVIS_TTS_CPP_LIBRARY_PATH="<libtts / tts.dll のフルパス>"
export AIVIS_TTS_CPP_LIBRARY_DIRS="<TTS.cpp と ggml library を含むディレクトリをパス区切りで列挙>"

uv run --group build pyinstaller --noconfirm run.spec
```

Plugin EP を明示する必要がある場合だけ、次も指定します。

```bash
export AIVIS_ONNX_GGML_EP_LIBRARY_PATH="<libaivis_ggml_onnx_ep / aivis_ggml_onnx_ep.dll のフルパス>"
```

`AIVIS_ONNX_GGML_REQUIRED=1` を付けると、sidecar が不足している場合に
ビルドが失敗します。PR レビュー用の再現では必ず付けてください。

## パッケージ検証

まず成果物が含まれていることを確認します。

```bash
test -e dist/run/onnxruntime_ep_aivis_ggml/lib
test -e dist/run/lib
```

Linux では rpath も確認します。

```bash
ldd dist/run/lib/libtts.so
```

`libggml*.so*` が `dist/run/lib` から解決されていれば正常です。TTS.cpp の
build directory を参照している、または `not found` が出る場合は `patchelf`
を入れてから再パッケージしてください。

## 起動 smoke test

macOS では既定 backend が `metal`、Windows / Linux では `vulkan` です。

Linux:

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

macOS:

```bash
./dist/run/run \
  --host 127.0.0.1 \
  --port 10109 \
  --onnx_provider ggml \
  --ggml_tts_server_backend metal \
  --ggml_native_library_path lib/libtts.dylib \
  --onnx_ep_library_path onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.dylib \
  --disable_sentry
```

Windows PowerShell:

```powershell
.\dist\run\run.exe `
  --host 127.0.0.1 `
  --port 10109 `
  --onnx_provider ggml `
  --ggml_tts_server_backend vulkan `
  --ggml_native_library_path lib\tts.dll `
  --onnx_ep_library_path onnxruntime_ep_aivis_ggml\lib\aivis_ggml_onnx_ep.dll `
  --disable_sentry
```

別の terminal で次を確認します。

```bash
curl -fsS http://127.0.0.1:10109/version
```

ログには次のような内容が出ます。

```text
Registered ONNX Runtime Plugin EP library aivis_onnx_plugin_ep
Using external ONNX Runtime Plugin EP AivisGgmlExecutionProvider before fallback providers ['CPUExecutionProvider'].
Application startup complete.
```

## ベンチマーク再現

パッケージまたはローカルビルド済み sidecar の性能確認には
[ONNX GGML Plugin EP ベンチマーク](onnx-ggml-plugin-benchmark.md) の
`tools/reproduce_onnx_ggml_benchmark.py` を使います。

このスクリプトは AivisHub から取得した AIVMX の SHA-256 を検証し、実行した
バックエンド、Provider 証跡、RTF、PCM 比較、実行コマンドを
`benchmark-artifacts/` 以下に出力します。`benchmark-artifacts/` は git 管理外です。

## よくある問題

`std::format` で TTS.cpp のビルドが失敗する:

検証済みの TTS.cpp ref を使っているか、C++20 対応のコンパイラを使っているかを
確認してください。

`glslc` が見つからない:

Linux / Windows の Vulkan build では Vulkan SDK が必要です。CI と同じ
`1.3.296.0` を使うと差分を減らせます。

Plugin EP がパッケージに入らない:

`cmake --install ... --prefix experimental/onnxruntime-ep-aivis-ggml/src` を実行し、
PyInstaller 実行時に `AIVIS_ONNX_GGML_REQUIRED=1` を設定してください。

起動時に TTS.cpp / ggml library が見つからない:

`AIVIS_TTS_CPP_LIBRARY_PATH` と `AIVIS_TTS_CPP_LIBRARY_DIRS` が build 済み
library を指しているか確認してください。Linux では `patchelf` の有無も確認します。

ONNX Runtime が CPU にフォールバックした結果をベンチマークとして記録してしまう:

再現スクリプトと低レベル benchmark runner は、選択された Provider が期待値と
一致しない場合に失敗します。失敗した場合は raw 結果として採用しないでください。
