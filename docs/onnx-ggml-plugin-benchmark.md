# ONNX GGML Plugin EP ベンチマーク

このドキュメントは、PR レビュー時に ONNX GGML Plugin EP の動作・性能・
再現性を確認するための手順と、現在の測定結果の要約です。raw JSON や音声
ファイルはリポジトリには含めません。再現スクリプトが
`benchmark-artifacts/` 以下に生成した結果を PR に添付してください。

## レビューで確認する内容

| 観点 | 確認内容 |
| --- | --- |
| Provider 選択 | `AivisGgmlExecutionProvider` が有効化され、CPU フォールバックを GGML 結果として記録していないこと |
| 既存経路への影響 | `--onnx_provider ggml` を指定した時だけ Plugin EP を登録すること |
| モデル再現性 | AivisHub から取得した AIVMX と SHA-256 が一致すること |
| 既定プロファイル | synthesis GGUF は FP32、JP-BERT GGUF は F16 `linear` を使うこと |
| 性能 | Windows / macOS / Linux の代表デバイスで RTF が実時間未満であること |
| 精度 | 決定論的設定で出力サンプル数が保たれ、PCM 差分が許容範囲に収まること |

RTF は `elapsed_seconds / output_duration_seconds` です。小さいほど高速です。
音声ファイルのエンコード時間は測定に含めません。

## モデル入力

ベンチマークに使う AIVMX は [AivisHub](https://hub.aivis-project.com/) から
取得します。再現スクリプトは SHA-256 を検証します。

| model | AivisHub | version | style id | SHA-256 |
| --- | --- | --- | ---: | --- |
| `まお` | <https://hub.aivis-project.com/aivm-models/a59cb814-0083-4369-8542-f51a29e72af7> | `1.2.0` | `888753760` | `f87ccea2e8e2de0e0bfe52e803945af903b4086bf25621a015111628f00e4119` |
| `コハク` | <https://hub.aivis-project.com/aivm-models/22e8ed77-94fe-4ef2-871f-a86f94e9a579> | `1.1.0` | `1878365376` | `3f5c08b52bb8a64efd361268580c81510f96c927cd6905aa7dbae6851333270a` |

現在の PR 用 benchmark は `まお` を標準入力として使います。

## 既定プロファイル

この PR の既定 GGML プロファイルは次の通りです。

| 項目 | 既定値 |
| --- | --- |
| synthesis GGUF | FP32 (`tts-cpp-style-bert-vits2-converter-f32-v1`) |
| JP-BERT GGUF | F16 `linear` |
| Vulkan precision | `fast` |
| Vulkan math mode | `coopmat` |
| runtime Vulkan F16 | 無効 |
| claimed graphs | synthesis と JP-BERT |

synthesis GGUF は保守的な FP32 を既定にします。これはレビュー時の基準線を
明確にし、モデル変換精度を原因とする差分を避けるためです。

一方、JP-BERT GGUF は F16 `linear` を既定の本番プロファイルとして扱います。
F32 基準は `1,314,386,784` bytes、F16 `linear` は `710,407,072` bytes で、
約 `46%` の容量削減になります。決定論的な比較では short / medium / long の
サンプル数が ONNX CPU と一致し、相関も `0.9998` 前後を維持しました。Q8/Q4 は
出力長や音声差分が大きいため、この PR の既定にはしません。

## 再現スクリプト

低レベルの測定処理は [tools/benchmark_onnx_ggml_provider.py](../tools/benchmark_onnx_ggml_provider.py)
が担当します。レビュー用には、入力検証・実行・summary 生成まで行う
[tools/reproduce_onnx_ggml_benchmark.py](../tools/reproduce_onnx_ggml_benchmark.py)
を使ってください。

最小例:

```bash
uv run python tools/reproduce_onnx_ggml_benchmark.py \
  --aivmx-path "<downloaded-mao.aivmx>" \
  --ggml-native-library-path "<libtts.so|libtts.dylib|tts.dll>" \
  --onnx-ep-library-path "<libaivis_ggml_onnx_ep.so|libaivis_ggml_onnx_ep.dylib|aivis_ggml_onnx_ep.dll>" \
  --library-dir "<directory-containing-tts-and-ggml-libraries>"
```

出力先は既定で `benchmark-artifacts/onnx-ggml-<timestamp>/` です。

Linux で [ONNX GGML ビルド・検証手順](onnx-ggml-build.md) の CMake build output を
そのまま使う場合:

```bash
ENGINE_DIR=<AivisSpeech-Engine の checkout>
TTS_CPP_BUILD_DIR="$ENGINE_DIR/build/TTS.cpp-build"

cd "$ENGINE_DIR"
MESA_VK_DEVICE_SELECT=1002:1900! uv run python tools/reproduce_onnx_ggml_benchmark.py \
  --aivmx-path "<downloaded-mao.aivmx>" \
  --backend onnx-cpu \
  --backend onnx-ggml-vulkan \
  --ggml-native-library-path "$TTS_CPP_BUILD_DIR/src/libtts.so" \
  --onnx-ep-library-path "$ENGINE_DIR/experimental/onnxruntime-ep-aivis-ggml/src/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so" \
  --library-dir "$TTS_CPP_BUILD_DIR/src" \
  --library-dir "$TTS_CPP_BUILD_DIR/ggml/src" \
  --library-dir "$TTS_CPP_BUILD_DIR/ggml/src/ggml-vulkan" \
  --ggml-model-cache-dir benchmark-artifacts/local-linux-780m/gguf-cache \
  --ggml-vulkan-device 0 \
  --warmup-runs 1 \
  --runs 1 \
  --output-dir benchmark-artifacts/local-linux-780m
```

`MESA_VK_DEVICE_SELECT=1002:1900!` は AMD Radeon 780M を 1 つ目の Vulkan device
として見せる例です。実行環境に合わせて `vulkaninfo --summary` の `vendorID` /
`deviceID` を確認するか、この指定を外してください。driver 側で device を絞った後、
Engine には `--ggml-vulkan-device 0` を渡します。

`--ggml-model-cache-dir` を固定すると、JP-BERT GGUF download と AIVMX 変換 cache を
後続の benchmark / package smoke test で再利用できます。初回の変換とモデル load を
性能値から外すため、性能確認では `--warmup-runs 1` 以上を使ってください。
`--warmup-runs 0` は provider 登録と合成経路の smoke test 用です。

| 生成物 | 内容 |
| --- | --- |
| `summary.md` | PR に貼り付けやすい RTF 表、Provider 証跡、実行環境、実行コマンド |
| `raw.json` | 全 record、profile、AivisHub のモデル情報、Provider 証跡 |
| `benchmark.log` | benchmark runner の標準出力 |
| `audio/` | 代表 WAV |
| `gguf-cache/` | 生成された GGUF キャッシュ |

プラットフォーム別の既定バックエンド:

| OS | 既定バックエンド |
| --- | --- |
| Linux | `onnx-cpu`, `onnx-ggml-vulkan` |
| Windows | `onnx-cpu`, `onnx-directml`, `onnx-ggml-vulkan` |
| macOS | `onnx-cpu`, `onnx-ggml-metal` |

Linux で CUDA も比較する場合:

```bash
uv run python tools/reproduce_onnx_ggml_benchmark.py \
  --aivmx-path "<downloaded-mao.aivmx>" \
  --ggml-native-library-path "<libtts.so>" \
  --onnx-ep-library-path "<libaivis_ggml_onnx_ep.so>" \
  --library-dir "<tts-cpp-lib-dir>" \
  --library-dir "<cuda12-and-cudnn-lib-dir>" \
  --include-cuda
```

決定論的な PCM 比較を行う場合:

```bash
uv run python tools/reproduce_onnx_ggml_benchmark.py \
  --aivmx-path "<downloaded-mao.aivmx>" \
  --ggml-native-library-path "<libtts.so|libtts.dylib|tts.dll>" \
  --onnx-ep-library-path "<plugin-ep-library>" \
  --library-dir "<native-library-dir>" \
  --deterministic \
  --runs 1
```

`--deterministic` は `tempoDynamicsScale=0.0`, `noise=0.0`, `noise_w=0.0`
を指定し、ONNX CPU を基準に PCM 差分を出力します。音質確認用の自然な
preview では既定のまま `noise` を固定しません。

## 測定テキスト

warmup は測定テキストと異なる文を使います。同じ文で warmup すると frontend、
symbol、graph、runtime cache を過度に温めるためです。

| label | text |
| --- | --- |
| short | `テストです。` |
| medium | `今日はいい天気ですね。` |
| long | `これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。` |

## 現在の測定要約

以下はローカルで取得済みの代表値です。新しいレビュー担当者が同じ手順で再測定した
場合は、`summary.md` を PR に追記してください。

### FP32 synthesis 既定プロファイル

| device | backend | synthesis GGUF | JP-BERT GGUF | short RTF | medium RTF | long RTF |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Linux RTX 3060 | GGML Vulkan | FP32 | F16 `linear` | `0.130` | `0.094` | `0.063` |
| Windows Arc B580 | GGML Vulkan | FP32 | F16 `linear` | `0.108` | `0.091` | `0.056` |
| macOS M1 Pro | GGML Metal | FP32 | F16 `linear` | `0.185` | `0.147` | `0.128` |

macOS M1 Pro の FP32 synthesis / JP-BERT F16 `linear` 決定論的比較では、
short / medium / long のサンプル数差分はすべて `0` でした。

| text | RMSE | max abs diff | correlation |
| --- | ---: | ---: | ---: |
| short | `0.004210` | `0.064331` | `0.999557` |
| medium | `0.008586` | `0.125610` | `0.998728` |
| long | `0.007790` | `0.318207` | `0.998421` |

### JP-BERT F16 `linear` の判断材料

Linux RTX 3060 と Windows Arc B580 で、JP-BERT F16 `linear` と FP32 JP-BERT
を synthesis FP16/FP32 と組み合わせて比較しました。JP-BERT を FP32 に戻しても
RTF は改善しなかったため、容量削減できる F16 `linear` を既定にします。

Linux RTX 3060:

| JP-BERT GGUF | synthesis GGUF | short RTF | medium RTF | long RTF |
| --- | --- | ---: | ---: | ---: |
| F16 `linear` | FP16 | `0.129` | `0.093` | `0.062` |
| F16 `linear` | FP32 | `0.130` | `0.094` | `0.063` |
| FP32 | FP16 | `0.133` | `0.093` | `0.063` |
| FP32 | FP32 | `0.131` | `0.094` | `0.064` |

Windows Arc B580:

| JP-BERT GGUF | synthesis GGUF | short RTF | medium RTF | long RTF |
| --- | --- | ---: | ---: | ---: |
| F16 `linear` | FP16 | `0.108` | `0.090` | `0.055` |
| F16 `linear` | FP32 | `0.108` | `0.091` | `0.056` |
| FP32 | FP16 | `0.109` | `0.091` | `0.056` |
| FP32 | FP32 | `0.109` | `0.094` | `0.056` |

### 追加の互換性確認

Linux RTX 3060 / AMD Radeon 780M / Windows Arc B580 / macOS M1 Pro で
Plugin EP 経路の実行を確認しています。過去の横断測定では、GGML Vulkan は
Windows Arc B580 と Linux RTX 3060 の short / medium / long 全てで ONNX CPU
より高速でした。AMD 780M iGPU でも Linux Vulkan バックエンドが実時間未満で動作し、
short `0.163`, medium `0.135`, long `0.108` RTF を記録しました。

## PR に貼る benchmark 章の推奨形

PR 本文には、次の順で貼るとレビュー担当者が追いやすくなります。

1. 使用モデルと SHA-256。
2. 既定プロファイル: synthesis FP32、JP-BERT F16 `linear`。
3. 代表デバイス別 RTF 表。
4. 決定論的 PCM 比較表。
5. `tools/reproduce_onnx_ggml_benchmark.py` の実行コマンド。
6. raw JSON / 音声生成物へのリンク。

raw JSON、音声、GGUF キャッシュは PR ブランチにコミットしません。
