# JP-BERT GGUF Quantization Notes

This note records the current Style-Bert-VITS2 JP-BERT GGUF lower-precision
experiment for the ONNX GGML Plugin EP path. The goal is to reduce memory while
keeping audio parity with ONNX CPU.

The Q8/Q4 candidate sweep below is a historical `コハク` run. The latest Linux
RTX 3060 ONNX GGML benchmark table near the end of this note has been refreshed
with `まお`.

## Scope

- Date: 2026-06-28, Asia/Tokyo
- Model: AIVMX/ONNX `コハク` version `1.1.0`
- Style: `1878365376` (`ノーマル`)
- Backend under test: `AivisGgmlExecutionProvider`, `backend=vulkan`,
  `precision=accurate`
- TTS settings: `tempoDynamicsScale=1.0`, `noise_scale=0.0`,
  `noise_scale_w=0.0`
- Texts: the benchmark short, medium, and long Japanese samples from
  `tools/benchmark_onnx_ggml_provider.py`

Generated GGUF and audio files are local experiment artifacts and should not be
committed. The adopted JP-BERT artifact is published as:

| field | value |
| --- | --- |
| HF repo | `kevinzhow/style-bert-vits2-gguf` |
| HF path | `frontend/style-bert-vits2-jp-bert.gguf` |
| HF commit | `b4678245870b9a74ae8134cb10ebe55cc8fb8181` |
| precision recipe | F16 `linear` |
| size | `710,407,072` bytes |
| sha256 | `93f39f94c42c84ed228d25a40f956fbcfbf895d92a8e64fd3c29d361a64ff664` |

## Quantization Surface

The source JP-BERT GGUF before replacement was all F32:

| candidate | size | tensor types | result |
| --- | ---: | --- | --- |
| F32 baseline | `1,314,386,784` bytes | `394 F32` | Works |
| F16 `all_weights` | `657,986,464` bytes | `247 F32`, `147 F16` | Rejected: Vulkan aborts on `NORM for f16 to f16` |
| F16 `linear` | `710,407,072` bytes | `250 F32`, `144 F16` | Adopted HF default |
| Q8_0 `linear` | `427,291,552` bytes | `250 F32`, `144 Q8_0` | Rejected for now: output duration drifts |
| Q4_0 `linear` | `276,296,608` bytes | `250 F32`, `144 Q4_0` | Rejected: audio divergence is large |

`linear` means only JP-BERT attention and FFN dense weights are quantized:

- `layers.*.attn.self.{query,key,value}.weight`
- `layers.*.attn.out.dense.weight`
- `layers.*.intermediate.dense.weight`
- `layers.*.output.dense.weight`

Embeddings, conv, norm, and bias tensors stay F32 in the `linear` scope.

## Synthesis GGUF F16 Scope

The synthesis GGUF cache is generated locally from AIVM/Safetensors or
AIVMX/ONNX at runtime; it is not published as a fixed artifact. The current
default converter cache key is:

```text
tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1
```

This key is used by both the Engine runtime cache and the experimental Plugin
EP cache planner, so AIVM/AIVMX runtime conversion and offline GGUF cache
planning produce the same precision contract.

The adopted synthesis recipe stores Style-Bert-VITS2 weight tensors as F16
except for tensors that are known to be unsafe or not useful for this Vulkan
path:

- embeddings stay F32
- norms stay F32
- decoder upsample tensors, `style_bert_vits2.decoder.ups.*`, stay F32
- bias tensors and `style_vectors` stay F32

This is a Vulkan-safe refinement of the earlier `weights_no_embed_norm`
experiment. The first version let decoder upsample weights become F16 and
strict backend validation failed because a decoder `CONV_TRANSPOSE_1D` node
landed on CPU instead of `Vulkan0`. Keeping decoder upsample tensors F32 avoids
that fallback while still reducing the synthesis GGUF size.

The 2026-06-28 RTX 3060 validation generated a synthesis GGUF with:

| field | value |
| --- | ---: |
| size | `129,812,864` bytes |
| F32 tensors | `574` |
| F16 tensors | `326` |

The same run used the default JP-BERT GGUF with:

| field | value |
| --- | ---: |
| size | `710,407,072` bytes |
| F32 tensors | `250` |
| F16 tensors | `144` |

With `precision=fast`, `warmup_runs=1`, `runs=3`, and strict provider
validation, the Linux RTX 3060 benchmark measured:

| text length | RTF | output samples |
| --- | ---: | ---: |
| short | `0.119` | `56,962` |
| medium | `0.089` | `90,261` |
| long | `0.061` | `345,994` |

Measured decoder hotspots were `IM2COL`, `MUL_MAT`, and
`CONV_TRANSPOSE_1D`; JP-BERT, duration prediction, and SDP condition/reverse
were not the dominant cost. This means the next performance work should target
decoder graph/kernel behavior instead of pushing the whole synthesis model to
FP16.

## Results

| candidate | short RTF | medium RTF | long RTF | sample counts short / medium / long |
| --- | ---: | ---: | ---: | --- |
| ONNX CPU | `0.269` | `0.277` | `0.241` | `56960 / 90261 / 345994` |
| GGML F32 | `0.118` | `0.091` | `0.062` | `56962 / 90261 / 345482` |
| GGML F16 `linear` | `0.121` | `0.091` | `0.062` | `56960 / 90261 / 345994` |
| GGML Q8_0 `linear` | `0.130` | `0.091` | `0.060` | `56955 / 90765 / 344969` |
| GGML Q4_0 `linear` | `0.115` | `0.085` | `0.060` | `57480 / 90249 / 341385` |

Audio deltas against ONNX CPU:

| candidate | short RMSE / corr | medium RMSE / corr | long RMSE / corr |
| --- | ---: | ---: | ---: |
| GGML F32 | `0.000686 / 0.999985` | `0.003157 / 0.999883` | `0.157076 / 0.328134` |
| GGML F16 `linear` | `0.001605 / 0.999920` | `0.003362 / 0.999866` | `0.002030 / 0.999887` |
| GGML Q8_0 `linear` | `0.029594 / 0.972538` | `0.095373 / 0.888301` | `0.163228 / 0.275944` |
| GGML Q4_0 `linear` | `0.178028 / -0.000384` | `0.178863 / 0.602183` | `0.189908 / 0.032433` |

F16 `linear` is the only candidate from this run that both reduces memory and
keeps audio parity. It is the default JP-BERT GGUF used by the ONNX GGML Plugin
EP cache. Q8_0 and Q4_0 should not be used as a default JP-BERT quantization
path without a more selective mixed-precision recipe.

The latest Linux RTX 3060 ONNX GGML benchmark, rerun with `まお` version
`1.2.0` / style `888753760`, also compares JP-BERT FP32 against the adopted
JP-BERT F16 `linear` artifact while crossing both with voice FP16/FP32 caches.
That refreshed benchmark leaves `noise_scale` and `noise_scale_w` unset so the
audio previews use the natural Style-Bert-VITS2 defaults. JP-BERT FP32 did not
improve RTF there:

| JP-BERT GGUF | voice GGUF | short RTF | medium RTF | long RTF |
| --- | --- | ---: | ---: | ---: |
| F16 `linear` | FP16 voices | `0.129` | `0.093` | `0.062` |
| F16 `linear` | FP32 voices | `0.130` | `0.094` | `0.063` |
| FP32 | FP16 voices | `0.133` | `0.093` | `0.063` |
| FP32 | FP32 voices | `0.131` | `0.094` | `0.064` |

The 2026-06-28 Windows Intel Arc B580 precision-matrix refresh, rerun with
TTS.cpp `94792ed25996`, also showed that JP-BERT FP32 and voice FP32 did not
improve RTF over the default FP16 `linear` JP-BERT plus FP16 voice cache:

| JP-BERT GGUF | voice GGUF | short RTF | medium RTF | long RTF |
| --- | --- | ---: | ---: | ---: |
| F16 `linear` | FP16 voices | `0.108` | `0.090` | `0.055` |
| F16 `linear` | FP32 voices | `0.108` | `0.091` | `0.056` |
| FP32 | FP16 voices | `0.109` | `0.091` | `0.056` |
| FP32 | FP32 voices | `0.109` | `0.094` | `0.056` |

Raw results and audio previews are in
[ONNX GGML Plugin EP Benchmark](onnx-ggml-plugin-benchmark.md).

## Reproduction

Build the TTS.cpp quantizer:

```bash
cmake --build <tts-cpp-build-dir> --target quantize -j$(nproc)
```

Generate candidates from the F32 JP-BERT GGUF:

```bash
<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-f16-linear.gguf> \
  --quantized-type F16 \
  --jp-bert-quantize-scope linear \
  --n-threads $(nproc)

<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-q8_0-linear.gguf> \
  --quantized-type Q8_0 \
  --jp-bert-quantize-scope linear \
  --n-threads $(nproc)

<tts-cpp-build-dir>/bin/quantize \
  --model-path <jp-bert-f32.gguf> \
  --quantized-model-path <jp-bert-q4_0-linear.gguf> \
  --quantized-type Q4_0 \
  --jp-bert-quantize-scope linear \
  --n-threads $(nproc)
```

Run the Engine benchmark with an explicit JP-BERT GGUF candidate:

The fixed `noise_scale=0` and `noise_scale_w=0` settings below are for
deterministic candidate parity checks only. Do not use them to generate
qualitative audio previews.

Warmup texts must be different from the measured benchmark texts so the timed
runs do not reuse text-specific frontend or graph caches.

```bash
GGML_VK_VISIBLE_DEVICES=<vulkan-device-index> \
uv run python tools/benchmark_onnx_ggml_provider.py \
  --aivmx_path <model.aivmx> \
  --style_id 888753760 \
  --backend onnx-ggml-vulkan \
  --warmup_runs 1 \
  --warmup_text "測定用ではない短い文です。" \
  --warmup_text "ウォームアップのために別の文章を読み上げます。" \
  --warmup_text "測定対象とは異なる長めのウォームアップ文章です。バックエンドの初回処理だけを先に済ませます。" \
  --runs 1 \
  --tempo_dynamics_scale 1.0 \
  --noise_scale 0 \
  --noise_scale_w 0 \
  --ggml_native_library_path <libtts.so> \
  --onnx_ep_library_path <libaivis_ggml_onnx_ep.so> \
  --ggml_model_cache_dir <cache-dir> \
  --ggml_jp_bert_gguf_path <candidate-jp-bert.gguf> \
  --ggml_vulkan_device <vulkan-device-index> \
  --ggml_vulkan_precision accurate \
  --output_json <result.json> \
  --audio_output_dir <audio-dir>
```

## Next Step

If memory needs to go below the F16 `linear` size, test a mixed recipe instead
of globally using Q8/Q4. The next useful candidate is likely FFN Q8 with
attention F16, while keeping embedding, conv, norm, and bias tensors F32.
