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

ローカルレビュー再現では、benchmark 済みの値を使うのが基準です。

| 項目 | 値 |
| --- | --- |
| ONNX Runtime headers | Engine の OS 別 ONNX Runtime pin に合わせる（Windows x64: `1.24.4`、macOS x64: `1.23.2`、macOS arm64: `1.27.0`、Linux x64: `1.26.0`） |
| TTS.cpp repository | `https://github.com/Myoland/TTS.cpp.git` |
| TTS.cpp ref | `94792ed2599656618c1d5eb3934754c391eb2a54` |
| ggml repository | `https://github.com/Myoland/ggml.git`（TTS.cpp submodule） |
| Vulkan SDK | `1.3.296.0`（検証済み。`glslc` / `glslangValidator` が必要） |

## 依存関係

| OS | 追加依存 |
| --- | --- |
| Linux | `libvulkan-dev`, `patchelf`, `xz-utils`, LunarG Vulkan SDK |
| Windows | MSVC, Chocolatey の `vulkan-sdk` |
| macOS | Xcode Command Line Tools, Homebrew の `gnu-sed`, `coreutils` |

macOS は TTS.cpp を `GGML_METAL=ON` でビルドします。Windows と Linux は
`GGML_VULKAN=ON` でビルドします。

## 共通ビルドフロー

以下の手順はローカル再現用の流れです。ローカルパスは例なので、実行環境に合わせて
置き換えてください。

```bash
ENGINE_DIR=<AivisSpeech-Engine の checkout>
TTS_CPP_DIR="$ENGINE_DIR/build/TTS.cpp"
TTS_CPP_BUILD_DIR="$ENGINE_DIR/build/TTS.cpp-build"
```

### 1. ONNX Runtime headers を準備する

Plugin EP のビルドには ONNX Runtime の C/C++ headers だけを使います。
headers は Engine が各 OS で使う ONNX Runtime package の pin に合わせます。
release archive 内の `libonnxruntime` は Engine パッケージには入れません。

```bash
cd "$ENGINE_DIR"

case "$(uname -s):$(uname -m)" in
  MINGW*:x86_64|MSYS*:x86_64|CYGWIN*:x86_64)
    ORT_VERSION=1.24.4
    ORT_ARCHIVE="onnxruntime-win-x64-${ORT_VERSION}.zip"
    ;;
  Darwin:x86_64)
    ORT_VERSION=1.23.2
    ORT_ARCHIVE="onnxruntime-osx-x86_64-${ORT_VERSION}.tgz"
    ;;
  Darwin:arm64)
    ORT_VERSION=1.27.0
    ORT_ARCHIVE="onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
    ;;
  Linux:x86_64)
    ORT_VERSION=1.26.0
    ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    ;;
  *)
    echo "Unsupported platform: $(uname -s):$(uname -m)" >&2
    exit 1
    ;;
esac

ORT_DIR="$PWD/build/onnxruntime-${ORT_VERSION}"

mkdir -p download "$ORT_DIR"
curl -sSL \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}" \
  -o "download/${ORT_ARCHIVE}"
if [[ "$ORT_ARCHIVE" == *.zip ]]; then
  tar -xf "download/${ORT_ARCHIVE}" -C "$ORT_DIR"
  mv "$ORT_DIR"/onnxruntime-*/* "$ORT_DIR"/
  rm -rf "$ORT_DIR"/onnxruntime-*
  rm -rf "$ORT_DIR"/lib
else
  tar -xzf "download/${ORT_ARCHIVE}" -C "$ORT_DIR" --strip-components=1 --exclude='*/lib/*'
fi

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
git clone https://github.com/Myoland/TTS.cpp.git "$TTS_CPP_DIR"
git -C "$TTS_CPP_DIR" checkout 94792ed2599656618c1d5eb3934754c391eb2a54
git -C "$TTS_CPP_DIR" submodule set-url ggml https://github.com/Myoland/ggml.git
git -C "$TTS_CPP_DIR" submodule update --init --recursive
```

この pinned ref の `.gitmodules` は移転前の ggml URL を保持しているため、
`submodule set-url` で移転後の URL を明示します。

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

`run.spec` は環境変数から Plugin EP、TTS.cpp runtime、ggml 依存 library を収集します。
Linux で上記の TTS.cpp CMake build output をそのまま使う場合:

```bash
export AIVIS_ONNX_GGML_REQUIRED=1
export AIVIS_TTS_CPP_LIBRARY_PATH="$TTS_CPP_BUILD_DIR/src/libtts.so"
export AIVIS_TTS_CPP_LIBRARY_DIRS="$TTS_CPP_BUILD_DIR/src:$TTS_CPP_BUILD_DIR/ggml/src:$TTS_CPP_BUILD_DIR/ggml/src/ggml-vulkan"
export AIVIS_ONNX_GGML_EP_LIBRARY_PATH="$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/src/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so"

cd "$ENGINE_DIR"
uv run --group build pyinstaller --noconfirm run.spec
```

macOS で上記の TTS.cpp CMake build output をそのまま使う場合:

```bash
export AIVIS_ONNX_GGML_REQUIRED=1
export AIVIS_TTS_CPP_LIBRARY_PATH="$TTS_CPP_BUILD_DIR/src/libtts.dylib"
export AIVIS_TTS_CPP_LIBRARY_DIRS="$TTS_CPP_BUILD_DIR/src:$TTS_CPP_BUILD_DIR/ggml/src:$TTS_CPP_BUILD_DIR/ggml/src/ggml-blas:$TTS_CPP_BUILD_DIR/ggml/src/ggml-metal"
export AIVIS_ONNX_GGML_EP_LIBRARY_PATH="$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/src/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.dylib"

cd "$ENGINE_DIR"
uv run --group build pyinstaller --noconfirm run.spec
```

`AIVIS_ONNX_GGML_REQUIRED=1` を付けると、sidecar が不足している場合に
ビルドが失敗します。PR レビュー用の再現では必ず付けてください。
Windows / macOS でも同じ環境変数を使い、各 OS の `libtts` / Plugin EP library と
依存 library directory を指定します。`AIVIS_TTS_CPP_LIBRARY_DIRS` は OS のパス区切り
文字（Linux / macOS は `:`、Windows は `;`）で複数指定できます。

## パッケージ検証

まず成果物が含まれていることを確認します。

```bash
test -e dist/run/onnxruntime_ep_aivis_ggml/lib
test -e dist/run/lib
```

Linux では rpath も確認します。

```bash
ldd dist/run/lib/libtts.so
readelf -d dist/run/lib/libtts.so | grep -E 'RPATH|RUNPATH'
```

`libggml*.so*` が `dist/run/lib` から解決されていれば正常です。TTS.cpp の
build directory を参照している、または `not found` が出る場合は `patchelf`
を入れてから再パッケージしてください。

`patchelf` がない環境では `run.spec` が warning を出し、既存の rpath を保持します。
このドキュメントの CMake 設定は `$ORIGIN` rpath を入れるため、その場合でも
`ldd` が `dist/run/lib` から解決できればレビュー用のローカル再現としては有効です。
リリースに近い package 検証では `patchelf` を入れ、再パッケージ後に同じ確認を
行ってください。

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

複数の Vulkan device がある Linux 環境では、driver 側で見せる device を固定してから
Engine 側の device id を渡します。AMD Radeon 780M の例では、上記コマンドに
`MESA_VK_DEVICE_SELECT=1002:1900!` と `--ggml_vulkan_device 0` を追加します。
`--ggml_model_cache_dir` は必須ではありません。benchmark 済みの cache を再利用したい
場合だけ、生成済み artifact 内の `gguf-cache/` を指定してください。

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

Linux / Windows の Vulkan build では、`glslc` / `glslangValidator` を含む
Vulkan SDK が必要です。`1.3.296.0` はこの手順で検証済みのバージョンです。

Plugin EP がパッケージに入らない:

`cmake --install ... --prefix experimental/onnxruntime-ep-aivis-ggml/src` を実行し、
PyInstaller 実行時に `AIVIS_ONNX_GGML_REQUIRED=1` を設定してください。

起動時に TTS.cpp / ggml library が見つからない:

`AIVIS_TTS_CPP_LIBRARY_PATH` と `AIVIS_TTS_CPP_LIBRARY_DIRS` が build 済み
library を指しているか確認してください。Linux では `patchelf` の有無も確認します。

ONNX Runtime が CPU にフォールバックした結果をベンチマークとして記録してしまう:

再現スクリプトと低レベル benchmark runner は、選択された Provider が期待値と
一致しない場合に失敗します。失敗した場合は raw 結果として採用しないでください。
