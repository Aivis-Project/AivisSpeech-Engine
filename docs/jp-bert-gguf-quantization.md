# JP-BERT GGUF 量子化メモ

このメモは、ONNX GGML Plugin EP で使う Style-Bert-VITS2 JP-BERT GGUF の
低精度化検証を記録するものです。目的は、音声品質と出力長を保ったまま
メモリ使用量を下げることです。

結論として、JP-BERT は F16 `linear` を既定の本番プロファイルとして採用します。
synthesis GGUF の既定は保守的な FP32 ですが、JP-BERT は F16 `linear` にしても
精度・出力長・RTF の面で問題がなく、容量を約 `46%` 削減できます。

## 採用する生成物

| 項目 | 値 |
| --- | --- |
| repository | `kevinzhow/style-bert-vits2-gguf` |
| path | `frontend/style-bert-vits2-jp-bert.gguf` |
| commit | `b4678245870b9a74ae8134cb10ebe55cc8fb8181` |
| precision recipe | F16 `linear` |
| size | `710,407,072` bytes |
| sha256 | `93f39f94c42c84ed228d25a40f956fbcfbf895d92a8e64fd3c29d361a64ff664` |

F32 基準は `1,314,386,784` bytes です。F16 `linear` は
`604 MB` 以上小さく、約 `46%` の容量削減になります。

## 量子化対象

`linear` は JP-BERT の attention / FFN の dense weight だけを F16 にします。

- `layers.*.attn.self.{query,key,value}.weight`
- `layers.*.attn.out.dense.weight`
- `layers.*.intermediate.dense.weight`
- `layers.*.output.dense.weight`

embedding、conv、norm、bias は F32 のまま残します。これは Q8/Q4 のような
出力長ずれを避けるための保守的な範囲です。

## 候補比較

最初の候補 sweep は `コハク` version `1.1.0` / style `1878365376` で実施しました。
バックエンドは `StyleBertVits2GgmlExecutionProvider`, `backend=vulkan`,
`precision=accurate` です。決定論的比較のため `tempoDynamicsScale=1.0`,
`noise_scale=0.0`, `noise_scale_w=0.0` を使いました。

| 候補 | size | tensor types | 判断 |
| --- | ---: | --- | --- |
| F32 基準 | `1,314,386,784` bytes | `394 F32` | 基準 |
| F16 `all_weights` | `657,986,464` bytes | `247 F32`, `147 F16` | 不採用: Vulkan の `NORM for f16 to f16` で停止 |
| F16 `linear` | `710,407,072` bytes | `250 F32`, `144 F16` | 採用 |
| Q8_0 `linear` | `427,291,552` bytes | `250 F32`, `144 Q8_0` | 不採用: 出力長がずれる |
| Q4_0 `linear` | `276,296,608` bytes | `250 F32`, `144 Q4_0` | 不採用: 音声差分が大きい |

RTF と出力サンプル数:

| 候補 | short RTF | medium RTF | long RTF | sample counts short / medium / long |
| --- | ---: | ---: | ---: | --- |
| ONNX CPU | `0.269` | `0.277` | `0.241` | `56960 / 90261 / 345994` |
| GGML F32 | `0.118` | `0.091` | `0.062` | `56962 / 90261 / 345482` |
| GGML F16 `linear` | `0.121` | `0.091` | `0.062` | `56960 / 90261 / 345994` |
| GGML Q8_0 `linear` | `0.130` | `0.091` | `0.060` | `56955 / 90765 / 344969` |
| GGML Q4_0 `linear` | `0.115` | `0.085` | `0.060` | `57480 / 90249 / 341385` |

ONNX CPU との PCM 差分:

| 候補 | short RMSE / corr | medium RMSE / corr | long RMSE / corr |
| --- | ---: | ---: | ---: |
| GGML F32 | `0.000686 / 0.999985` | `0.003157 / 0.999883` | `0.157076 / 0.328134` |
| GGML F16 `linear` | `0.001605 / 0.999920` | `0.003362 / 0.999866` | `0.002030 / 0.999887` |
| GGML Q8_0 `linear` | `0.029594 / 0.972538` | `0.095373 / 0.888301` | `0.163228 / 0.275944` |
| GGML Q4_0 `linear` | `0.178028 / -0.000384` | `0.178863 / 0.602183` | `0.189908 / 0.032433` |

F16 `linear` だけが、容量削減と音声 parity を同時に満たしました。
Q8_0 と Q4_0 は、より細かい mixed precision recipe を作るまで既定にはしません。

## synthesis GGUF との組み合わせ

`まお` version `1.2.0` / style `888753760` で、JP-BERT FP32 と
JP-BERT F16 `linear` を synthesis FP16/FP32 と組み合わせて再測定しました。
この結果から、JP-BERT を FP32 に戻しても RTF は改善しないことを確認しています。

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

この PR では、synthesis GGUF は FP32 を既定にし、JP-BERT GGUF は
F16 `linear` を既定にします。synthesis FP16 は明示的なメモリプロファイルとして
検証できますが、レビュー基準にはしません。

## 再現

TTS.cpp の quantizer をビルドします。

```bash
cmake --build <tts-cpp-build-dir> --target quantize --parallel
```

F32 JP-BERT GGUF から候補を生成します。

```bash
<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-f16-linear.gguf> \
  --quantized-type F16 \
  --jp-bert-quantize-scope linear \
  --n-threads <threads>

<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-q8_0-linear.gguf> \
  --quantized-type Q8_0 \
  --jp-bert-quantize-scope linear \
  --n-threads <threads>

<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-q4_0-linear.gguf> \
  --quantized-type Q4_0 \
  --jp-bert-quantize-scope linear \
  --n-threads <threads>
```

Engine 経由の確認は [ONNX GGML Plugin EP ベンチマーク](onnx-ggml-plugin-benchmark.md)
の再現スクリプトを使います。候補 JP-BERT を直接指定する場合は
`--ggml-jp-bert-gguf-path <candidate-jp-bert.gguf>` を渡してください。
