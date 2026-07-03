# ONNX GGML Engine 接続・検証手順

このドキュメントは、外部 repository で build した ONNX GGML runtime bundle を
AivisSpeech Engine に同梱し、`--onnx_provider ggml` 経路を検証するための手順です。

ONNX Runtime Plugin EP、TTS.cpp、ggml、Vulkan / Metal sidecar の build は
`Myoland/onnxruntime-ep-style-bert-vits2-ggml` 側で行います。Engine 側では、その bundle を
PyInstaller package に取り込むだけです。

Android / モバイル向けの検証はこの PR の対象外です。

## 対象

この PR で追加する経路は明示的に `--onnx_provider ggml` を指定した場合だけ
有効になります。通常の `auto` / `cuda` / `directml` / CPU 経路は既存のままです。

ビルド成果物には次の sidecar を含めます。

| OS | Engine | TTS.cpp runtime | Plugin EP |
| --- | --- | --- | --- |
| Windows | `dist/run/run.exe` | `dist/run/lib/tts.dll` または `libtts.dll` | `dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib/style_bert_vits2_ggml_onnx_ep.dll` |
| macOS | `dist/run/run` | `dist/run/lib/libtts.dylib` | `dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.dylib` |
| Linux | `dist/run/run` | `dist/run/lib/libtts.so` | `dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so` |

Linux では `run.spec` が `patchelf` で TTS.cpp / ggml sidecar の rpath を
`$ORIGIN` に変更します。`patchelf` がない場合でも、bundle 側の library が
`$ORIGIN` rpath を持ち、`ldd` が `dist/run/lib` から解決できればローカル再現としては
有効です。

## 依存関係

| OS | Engine package で必要な追加依存 |
| --- | --- |
| Linux | `patchelf`（推奨） |
| Windows | MSVC runtime / PowerShell |
| macOS | Xcode Command Line Tools, Homebrew の `gnu-sed`, `coreutils` |

Vulkan SDK、ONNX Runtime headers、TTS.cpp runtime build は外部 runtime repository の
手順に従います。

## 共通フロー

以下の手順はローカル再現用の流れです。ローカルパスは例なので、実行環境に合わせて
置き換えてください。

```bash
ENGINE_DIR=<AivisSpeech-Engine の checkout>
PLUGIN_REPO_DIR=<onnxruntime-ep-style-bert-vits2-ggml の checkout>
```

### 1. runtime bundle を準備する

外部 repository で runtime bundle を生成します。

```bash
cd "$PLUGIN_REPO_DIR"
uv run python scripts/build_runtime_bundle.py
```

既に TTS.cpp を build 済みの場合は、Plugin EP だけを再 build して既存の TTS.cpp
runtime を bundle にできます。

```bash
cd "$PLUGIN_REPO_DIR"
uv run python scripts/build_runtime_bundle.py \
  --tts-cpp-build-dir "$ENGINE_DIR/build/TTS.cpp-build" \
  --reuse-tts-cpp-build
```

生成物は既定で次の場所に出力されます。

```text
$PLUGIN_REPO_DIR/dist/style-bert-vits2-ggml-runtime-linux-x64/
```

この bundle の `manifest.json` が ONNX Runtime version、TTS.cpp ref、library checksum
を記録します。macOS / Windows では、runtime repository が出力する対応 platform の
`dist/style-bert-vits2-ggml-runtime-<platform>/` を使います。

### 2. Engine を PyInstaller でパッケージする

Engine package build では bundle directory だけを渡します。

```bash
cd "$ENGINE_DIR"

export STYLE_BERT_VITS2_GGML_REQUIRED=1
export STYLE_BERT_VITS2_GGML_BUNDLE_DIR="$PLUGIN_REPO_DIR/dist/style-bert-vits2-ggml-runtime-linux-x64"

uv run --group build pyinstaller --noconfirm run.spec
```

`STYLE_BERT_VITS2_GGML_REQUIRED=1` を付けると、Plugin EP または TTS.cpp sidecar が
不足している場合にビルドが失敗します。PR レビュー用の再現では必ず付けてください。

`STYLE_BERT_VITS2_GGML_EP_LIBRARY_PATH`、`STYLE_BERT_VITS2_TTS_CPP_LIBRARY_PATH`、
`STYLE_BERT_VITS2_TTS_CPP_LIBRARY_DIRS` も引き続き使えますが、通常は
`STYLE_BERT_VITS2_GGML_BUNDLE_DIR` だけで十分です。

## パッケージ検証

まず成果物が含まれていることを確認します。

```bash
test -e dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib
test -e dist/run/lib
```

Linux では rpath も確認します。

```bash
ldd dist/run/lib/libtts.so
readelf -d dist/run/lib/libtts.so | grep -E 'RPATH|RUNPATH'
```

`libggml*.so*` が `dist/run/lib` から解決されていれば正常です。bundle や TTS.cpp の
build directory を参照している、または `not found` が出る場合は `patchelf` を入れて
から再パッケージしてください。

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
  --onnx_ep_library_path onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so \
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
  --onnx_ep_library_path onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.dylib \
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
  --onnx_ep_library_path onnxruntime_ep_style_bert_vits2_ggml\lib\style_bert_vits2_ggml_onnx_ep.dll `
  --disable_sentry
```

別の terminal で次を確認します。

```bash
curl -fsS http://127.0.0.1:10109/version
```

ログには次のような内容が出ます。

```text
Registered ONNX Runtime Plugin EP library style_bert_vits2_onnx_plugin_ep
Using external ONNX Runtime Plugin EP StyleBertVits2GgmlExecutionProvider before fallback providers ['CPUExecutionProvider'].
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

Plugin EP がパッケージに入らない:

外部 runtime repository で `scripts/build_runtime_bundle.py` が成功しているか、
PyInstaller 実行時に `STYLE_BERT_VITS2_GGML_REQUIRED=1` と
`STYLE_BERT_VITS2_GGML_BUNDLE_DIR` を設定しているか確認してください。

起動時に TTS.cpp / ggml library が見つからない:

`STYLE_BERT_VITS2_GGML_BUNDLE_DIR` が生成済み bundle を指しているか確認してください。
Linux では `patchelf` の有無と `ldd dist/run/lib/libtts.so` の結果も確認します。

ONNX Runtime が CPU にフォールバックした結果をベンチマークとして記録してしまう:

再現スクリプトと低レベル benchmark runner は、選択された Provider が期待値と
一致しない場合に失敗します。失敗した場合は raw 結果として採用しないでください。
