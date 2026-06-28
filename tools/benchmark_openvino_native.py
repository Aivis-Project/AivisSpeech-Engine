"""
Benchmark direct OpenVINO synthesis graph execution for AIVMX models.

This tool is intentionally separate from benchmark_onnx_ggml_provider.py:
ONNX Runtime's OpenVINO Execution Provider is not used for the OpenVINO
backends here.  The NPU path uses a split graph because the full
Style-Bert-VITS2 synthesis graph has data-dependent output length, which the
current Intel NPU OpenVINO plugin cannot compile as a single graph.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aivmlib
import numpy as np
import onnx
import onnxruntime as ort
import openvino as ov
import soundfile
from aivmlib.schemas.aivm_manifest import AivmMetadata
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper, utils
from style_bert_vits2.constants import (
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer_onnx import get_text_onnx
from style_bert_vits2.nlp import (
    bert_models,
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    onnx_bert_models,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_GGML_PLUGIN_SRC = _REPO_ROOT / "experimental" / "onnxruntime-ep-aivis-ggml" / "src"
if _GGML_PLUGIN_SRC.is_dir() and str(_GGML_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_GGML_PLUGIN_SRC))

from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (  # noqa: E402
    build_graph_initializer_target_overrides,
    map_initializer_name,
)

from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (  # noqa: E402
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.utility.path_utility import get_save_dir  # noqa: E402

_DEFAULT_TEXTS = (
    "テストです。",
    "今日はいい天気ですね。",
    "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。",
)
_SYNTHESIS_INPUT_NAMES = (
    "x_tst",
    "x_tst_lengths",
    "sid",
    "tones",
    "language",
    "bert",
    "style_vec",
    "length_scale",
    "sdp_ratio",
    "noise_scale",
    "noise_scale_w",
)
_DECODER_INPUT_NAMES = ("/Mul_9_output_0", "/Unsqueeze_output_0")
_OUTPUT_NAME = "output"


@dataclass(frozen=True)
class _BenchmarkRecord:
    backend: str
    text_label: str
    text: str
    run_index: int
    elapsed_seconds: float
    output_duration_seconds: float
    output_samples: int
    rtf: float
    peak_abs: float


@dataclass(frozen=True)
class _BackendSummary:
    backend: str
    text_label: str
    rtf_mean: float
    rtf_min: float
    rtf_max: float
    output_duration_seconds_mean: float
    output_samples_last: int


@dataclass(frozen=True)
class _TruthComparison:
    backend: str
    text_label: str
    truth_backend: str
    sample_delta: int
    compared_samples: int
    rmse: float
    max_abs_diff: float
    correlation: float | None


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    decoder_device: str | None
    jpbert_device: str | None
    full_graph_device: str | None = None
    full_graph_variant: str | None = None
    full_graph_precision_hint: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ONNX CPU truth against direct OpenVINO split synthesis "
            "graph execution. The OpenVINO backends do not use ONNX Runtime's "
            "OpenVINO Execution Provider."
        )
    )
    parser.add_argument(
        "--aivmx_path",
        type=Path,
        required=True,
        help="AIVMX/ONNX model path.",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "onnx-cpu",
            "openvino-native-gpu",
            "openvino-native-npu",
            "openvino-native-split-gpu",
            "openvino-native-split-npu",
            "openvino-full-gpu-default",
            "openvino-full-gpu-fp32",
            "openvino-full-gpu-mixed-fp16",
        ),
        action="append",
        default=None,
        help="Backend to benchmark. Repeat to select multiple backends.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text to synthesize. Repeat for short/medium/long benchmark texts.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Warmup syntheses per backend/text before measured runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured syntheses per backend/text.",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Local speaker id in the Style-Bert-VITS2 model.",
    )
    parser.add_argument(
        "--style_id",
        type=int,
        default=0,
        help="Local style id in the Style-Bert-VITS2 model.",
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=DEFAULT_STYLE_WEIGHT,
        help="Style vector weight. For style_id=0 this keeps the neutral vector.",
    )
    parser.add_argument(
        "--sdp_ratio",
        type=float,
        default=DEFAULT_SDP_RATIO,
        help="Style-Bert-VITS2 SDP ratio.",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=1.0,
        help="Style-Bert-VITS2 length scale.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.0,
        help="Style-Bert-VITS2 noise scale. 0.0 is deterministic.",
    )
    parser.add_argument(
        "--noise_scale_w",
        type=float,
        default=0.0,
        help="Style-Bert-VITS2 SDP noise scale. 0.0 is deterministic.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=None,
        help=(
            "Directory for extracted OpenVINO split ONNX artifacts. "
            "Defaults to a temporary directory."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--audio_output_dir",
        type=Path,
        default=None,
        help="Optional directory for representative WAV outputs.",
    )
    parser.add_argument(
        "--probe_full_graph",
        action="store_true",
        help=(
            "Also probe whether the complete synthesis graph can run on "
            "OpenVINO GPU/NPU. This is diagnostic only and is not included "
            "in measured RTF."
        ),
    )
    parser.add_argument(
        "--synthesis_only_rtf",
        action="store_true",
        help=(
            "Prebuild text/JP-BERT inputs outside the measured section. "
            "This measures only synthesis graph execution."
        ),
    )
    return parser.parse_args()


def _cpu_ort_providers() -> list[tuple[str, dict[str, str]]]:
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def _load_jp_bert(
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
) -> Path:
    repo = StyleBertVITS2TTSEngine.BERT_MODEL_REPOSITORY
    revision = StyleBertVITS2TTSEngine.BERT_MODEL_REVISION
    cache_dir = str(get_save_dir() / "BertModelCaches")
    bert_models.load_tokenizer(
        Languages.JP,
        pretrained_model_name_or_path=repo,
        cache_dir=cache_dir,
        revision=revision,
    )
    onnx_bert_models.load_model(
        Languages.JP,
        pretrained_model_name_or_path=repo,
        onnx_providers=onnx_providers,
        cache_dir=cache_dir,
        revision=revision,
    )
    onnx_bert_models.load_tokenizer(
        Languages.JP,
        pretrained_model_name_or_path=repo,
        cache_dir=cache_dir,
        revision=revision,
    )
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename="model_fp16.onnx",
            cache_dir=cache_dir,
            revision=revision,
        )
    )


def _intersperse(items: list[Any], item: Any) -> list[Any]:
    result = [item] * (len(items) * 2 + 1)
    result[1::2] = items
    return result


class _OpenVINOJpBertRunner:
    """Run JP-BERT with direct OpenVINO and cache static-shape compiles."""

    def __init__(
        self,
        *,
        core: ov.Core,
        model_path: Path,
        device: str,
        evidence: dict[str, dict[str, Any]],
    ) -> None:
        self._core = core
        self._model_path = model_path
        self._device = device
        self._compiled_models: dict[
            tuple[tuple[str, tuple[int, ...]], ...], ov.CompiledModel
        ] = {}
        self._evidence = evidence

    def _compile(
        self,
        inputs: dict[str, NDArray[Any]],
    ) -> ov.CompiledModel:
        key = tuple(
            (name, tuple(value.shape)) for name, value in sorted(inputs.items())
        )
        if key in self._compiled_models:
            return self._compiled_models[key]

        model = self._core.read_model(str(self._model_path))
        model.reshape({name: list(value.shape) for name, value in inputs.items()})
        started_at = time.perf_counter()
        compiled = self._core.compile_model(
            model,
            self._device,
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        elapsed = time.perf_counter() - started_at
        try:
            execution_devices: Any = compiled.get_property("EXECUTION_DEVICES")
        except Exception:
            execution_devices = None
        self._evidence[str((self._device, key))] = {
            "seconds": elapsed,
            "execution_devices": execution_devices,
        }
        self._compiled_models[key] = compiled
        return compiled

    def _run_model(self, text: str) -> NDArray[np.float32]:
        tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
        inputs = {
            name: value.astype(np.int64)
            for name, value in tokenizer(text, return_tensors="np").items()
        }
        compiled = self._compile(inputs)
        return np.asarray(compiled(inputs)[compiled.output(0)], dtype=np.float32)

    def extract_feature(
        self,
        *,
        text: str,
        word2ph: list[int],
        assist_text: str | None = None,
        assist_text_weight: float = 0.7,
        sep_text: list[str] | None = None,
        use_nanairo: bool = False,
    ) -> NDArray[np.float32]:
        if sep_text is None:
            from style_bert_vits2.nlp.japanese.bert_feature import text_to_sep_kata

            sep_text = text_to_sep_kata(
                text,
                use_nanairo=use_nanairo,
                raise_yomi_error=False,
            )[0]

        text = "".join(sep_text)
        if assist_text:
            from style_bert_vits2.nlp.japanese.bert_feature import text_to_sep_kata

            assist_text = "".join(
                text_to_sep_kata(
                    assist_text,
                    use_nanairo=use_nanairo,
                    raise_yomi_error=False,
                )[0]
            )

        res = self._run_model(text)
        style_res_mean: NDArray[np.float32] | None = None
        if assist_text:
            style_res = self._run_model(assist_text)
            style_res_mean = np.asarray(np.mean(style_res, axis=0), dtype=np.float32)

        assert len(word2ph) == len(text) + 2, text
        phone_level_feature = []
        for index, phone_count in enumerate(word2ph):
            if assist_text:
                assert style_res_mean is not None
                repeat_feature = (
                    np.tile(res[index], (phone_count, 1)) * (1 - assist_text_weight)
                    + np.tile(style_res_mean, (phone_count, 1)) * assist_text_weight
                )
            else:
                repeat_feature = np.tile(res[index], (phone_count, 1))
            phone_level_feature.append(repeat_feature)

        return np.asarray(
            np.concatenate(phone_level_feature, axis=0).T, dtype=np.float32
        )


def _read_aivmx_metadata(aivmx_path: Path) -> AivmMetadata:
    with aivmx_path.open("rb") as file:
        return aivmlib.read_aivmx_metadata(file)


def _style_vector(
    *,
    style_vectors: NDArray[Any],
    style_id: int,
    style_weight: float,
) -> NDArray[np.float32]:
    mean = style_vectors[0]
    style_vec = style_vectors[style_id]
    return np.asarray(mean + (style_vec - mean) * style_weight, dtype=np.float32)


def _extract_split_models(
    *,
    aivmx_path: Path,
    artifact_dir: Path,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    front_path = artifact_dir / "openvino-front-to-decoder-io.onnx"
    decoder_path = artifact_dir / "openvino-decoder-only.onnx"

    if not front_path.is_file():
        utils.extract_model(
            str(aivmx_path),
            str(front_path),
            list(_SYNTHESIS_INPUT_NAMES),
            list(_DECODER_INPUT_NAMES),
            check_model=False,
        )
    if not decoder_path.is_file():
        utils.extract_model(
            str(aivmx_path),
            str(decoder_path),
            list(_DECODER_INPUT_NAMES),
            [_OUTPUT_NAME],
            check_model=False,
        )
    return front_path, decoder_path


def _store_as_f16_like_ggml(target_name: str, *, enabled: bool = True) -> bool:
    """Mirror the current GGML synthesis GGUF safe-F16 tensor scope."""

    if not enabled:
        return False
    if not target_name.startswith("style_bert_vits2."):
        return False
    if "embedding" in target_name:
        return False
    if ".norm" in target_name or "norm_" in target_name:
        return False
    if target_name.startswith("style_bert_vits2.decoder.ups."):
        return False
    return target_name.endswith(".weight") or target_name.endswith(".w")


def _safe_onnx_node_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _patch_sdp_scatternd_negative_indices(model: onnx.ModelProto) -> dict[str, Any]:
    """
    Work around OpenVINO GPU ScatterNDUpdate negative-index bug.

    OpenVINO 2026.2 GPU maps -1 to rank-1 in scatter_nd_update_opt.cl.  These
    SDP spline boundary indices target the last bin coordinate, whose size is
    11, so -1 must be normalized to 10.
    """

    node_by_name = {node.name: node for node in model.graph.node}
    patched_nodes: list[str] = []
    for flow in (7, 5, 3):
        for expand_index in (9, 15, 21, 24, 27):
            node_name = f"/sdp/flows.{flow}/Expand_{expand_index}"
            node = node_by_name.get(node_name)
            if node is None:
                continue
            replacement_name = f"{node_name}_positive_last_index_10"
            node.input[0] = replacement_name
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.array([10], dtype=np.int64),
                    replacement_name,
                )
            )
            patched_nodes.append(node_name)

    return {
        "description": (
            "Normalize SDP ScatterND -1 boundary indices to 10 as a graph-level "
            "workaround for the current OpenVINO GPU ScatterNDUpdate kernel."
        ),
        "patched_node_count": len(patched_nodes),
        "patched_nodes": patched_nodes,
    }


def _convert_ggml_safe_initializers_to_fp16_storage(
    *,
    model: onnx.ModelProto,
    source_model_path: Path,
) -> dict[str, Any]:
    """Store GGML-safe weights as FP16 while casting them back to FP32 in-graph."""

    target_name_overrides = build_graph_initializer_target_overrides(source_model_path)
    converted: list[dict[str, Any]] = []
    cast_nodes: list[onnx.NodeProto] = []
    source_bytes = 0
    stored_bytes = 0
    for initializer in model.graph.initializer:
        target_name = target_name_overrides.get(
            initializer.name
        ) or map_initializer_name(initializer.name)
        if target_name is None or not _store_as_f16_like_ggml(target_name):
            continue
        if initializer.data_type != TensorProto.FLOAT:
            continue

        source_name = initializer.name
        fp16_initializer_name = f"{source_name}_openvino_mixed_fp16_storage"
        array = numpy_helper.to_array(initializer)
        fp16_array = np.ascontiguousarray(array.astype(np.float16))
        source_bytes += int(array.nbytes)
        stored_bytes += int(fp16_array.nbytes)
        initializer.CopyFrom(numpy_helper.from_array(fp16_array, fp16_initializer_name))
        cast_nodes.append(
            helper.make_node(
                "Cast",
                inputs=[fp16_initializer_name],
                outputs=[source_name],
                name=f"/openvino_mixed_fp16/Cast_{_safe_onnx_node_name(source_name)}",
                to=TensorProto.FLOAT,
            )
        )
        if len(converted) < 50:
            converted.append(
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "shape": list(array.shape),
                }
            )

    for cast_node in reversed(cast_nodes):
        model.graph.node.insert(0, cast_node)

    return {
        "description": (
            "GGML-aligned mixed storage: safe weight tensors are stored as FP16 "
            "but immediately cast to FP32 so OpenVINO GPU math stays FP32."
        ),
        "converted_initializer_count": len(cast_nodes),
        "source_float32_bytes": source_bytes,
        "stored_float16_bytes": stored_bytes,
        "saved_bytes": source_bytes - stored_bytes,
        "sample_converted_initializers": converted,
    }


def _prepare_full_graph_models(
    *,
    aivmx_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    patched_path = artifact_dir / "openvino-full-gpu-negidx-patched.onnx"
    mixed_path = artifact_dir / "openvino-full-gpu-negidx-patched-mixed-fp16.onnx"
    evidence: dict[str, Any] = {
        "source": f"<local-model-dir>/{aivmx_path.name}",
        "variants": {},
    }

    if not patched_path.is_file():
        model = onnx.load(str(aivmx_path), load_external_data=False)
        patch_evidence = _patch_sdp_scatternd_negative_indices(model)
        onnx.save(model, str(patched_path))
    else:
        patch_evidence = {"status": "reused_existing"}
    evidence["variants"]["negidx_patched_fp32_source"] = {
        "path": f"<artifact-dir>/{patched_path.name}",
        "scatternd_negative_index_patch": patch_evidence,
    }

    if not mixed_path.is_file():
        model = onnx.load(str(patched_path), load_external_data=False)
        mixed_evidence = _convert_ggml_safe_initializers_to_fp16_storage(
            model=model,
            source_model_path=aivmx_path,
        )
        onnx.save(model, str(mixed_path))
    else:
        mixed_evidence = {"status": "reused_existing"}
    evidence["variants"]["mixed_fp16_storage"] = {
        "path": f"<artifact-dir>/{mixed_path.name}",
        "mixed_fp16_storage": mixed_evidence,
    }

    return {
        "patched": patched_path,
        "mixed-fp16": mixed_path,
    }, evidence


def _get_text_openvino_jp_bert(
    *,
    text: str,
    hps: HyperParameters,
    jpbert_runner: _OpenVINOJpBertRunner,
) -> tuple[
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
]:
    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()
    if is_nanairo_like_model:
        raise NotImplementedError(
            "Nanairo OpenVINO JP-BERT inference is not supported."
        )

    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        text,
        Languages.JP,
        use_jp_extra=is_jp_extra_like_model,
        use_nanairo=is_nanairo_like_model,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(
        phone,
        tone,
        Languages.JP,
        use_nanairo=is_nanairo_like_model,
    )

    if hps.data.add_blank:
        phone = _intersperse(phone, 0)
        tone = _intersperse(tone, 0)
        language = _intersperse(language, 0)
        for index in range(len(word2ph)):
            word2ph[index] = word2ph[index] * 2
        word2ph[0] += 1

    bert_ori = jpbert_runner.extract_feature(
        text=norm_text,
        word2ph=word2ph,
        sep_text=sep_text,
        use_nanairo=is_nanairo_like_model,
    )
    assert bert_ori.shape[-1] == len(phone), phone

    phone_array = np.array(phone, dtype=np.int64)
    tone_array = np.array(tone, dtype=np.int64)
    language_array = np.array(language, dtype=np.int64)
    return (
        np.zeros((1024, len(phone_array)), dtype=np.float32),
        bert_ori,
        np.zeros((1024, len(phone_array)), dtype=np.float32),
        phone_array,
        tone_array,
        language_array,
    )


def _build_inputs(
    *,
    text: str,
    hps: HyperParameters,
    style_vec: NDArray[np.float32],
    speaker_id: int,
    length_scale: float,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    jpbert_runner: _OpenVINOJpBertRunner | None,
) -> dict[str, NDArray[Any]]:
    if jpbert_runner is None:
        _, ja_bert, _, phones, tones, lang_ids = get_text_onnx(
            text,
            Languages.JP,
            hps,
            onnx_providers=onnx_providers,
        )
    else:
        _, ja_bert, _, phones, tones, lang_ids = _get_text_openvino_jp_bert(
            text=text,
            hps=hps,
            jpbert_runner=jpbert_runner,
        )
    return {
        "x_tst": np.expand_dims(phones, axis=0).astype(np.int64),
        "x_tst_lengths": np.array([phones.shape[0]], dtype=np.int64),
        "sid": np.array([speaker_id], dtype=np.int64),
        "tones": np.expand_dims(tones, axis=0).astype(np.int64),
        "language": np.expand_dims(lang_ids, axis=0).astype(np.int64),
        "bert": np.expand_dims(ja_bert, axis=0).astype(np.float32),
        "style_vec": np.expand_dims(style_vec, axis=0).astype(np.float32),
        "length_scale": np.array(length_scale, dtype=np.float32),
        "sdp_ratio": np.array(sdp_ratio, dtype=np.float32),
        "noise_scale": np.array(noise_scale, dtype=np.float32),
        "noise_scale_w": np.array(noise_scale_w, dtype=np.float32),
    }


def _build_ort_session(aivmx_path: Path) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session_options.log_severity_level = 3
    return ort.InferenceSession(
        str(aivmx_path),
        sess_options=session_options,
        providers=_cpu_ort_providers(),
    )


def _compile_front_model(
    *,
    core: ov.Core,
    front_path: Path,
) -> tuple[ov.CompiledModel, dict[str, Any]]:
    started_at = time.perf_counter()
    compiled = core.compile_model(
        core.read_model(str(front_path)),
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY"},
    )
    elapsed = time.perf_counter() - started_at
    return compiled, {
        "seconds": elapsed,
        "execution_devices": compiled.get_property("EXECUTION_DEVICES"),
    }


def _keep_only_output(model: ov.Model, output_name: str) -> ov.Model:
    for result in list(model.get_results()):
        if output_name not in set(result.output(0).get_names()):
            model.remove_result(result)
    return model


def _run_front(
    *,
    compiled_front: ov.CompiledModel,
    inputs: dict[str, NDArray[Any]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    result = compiled_front(inputs)
    return (
        np.asarray(result[compiled_front.output(_DECODER_INPUT_NAMES[0])]),
        np.asarray(result[compiled_front.output(_DECODER_INPUT_NAMES[1])]),
    )


def _compile_decoder_model(
    *,
    core: ov.Core,
    decoder_path: Path,
    device: str,
    mul9_shape: Sequence[int],
    cond_shape: Sequence[int],
) -> tuple[ov.CompiledModel, dict[str, Any]]:
    model = core.read_model(str(decoder_path))
    model.reshape(
        {
            _DECODER_INPUT_NAMES[0]: list(mul9_shape),
            _DECODER_INPUT_NAMES[1]: list(cond_shape),
        }
    )
    started_at = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    elapsed = time.perf_counter() - started_at
    try:
        execution_devices: Any = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        execution_devices = None
    return compiled, {
        "seconds": elapsed,
        "execution_devices": execution_devices,
    }


def _compile_properties(*, precision_hint: str | None = None) -> dict[str, str]:
    properties = {"PERFORMANCE_HINT": "LATENCY"}
    if precision_hint is not None:
        properties["INFERENCE_PRECISION_HINT"] = precision_hint
    return properties


def _runtime_element_type_counts(compiled: ov.CompiledModel) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in compiled.get_runtime_model().get_ordered_ops():
        try:
            element_type = str(op.get_output_element_type(0))
        except Exception:
            element_type = "unknown"
        counts[element_type] = counts.get(element_type, 0) + 1
    return counts


def _compile_full_graph_model(
    *,
    core: ov.Core,
    model_path: Path,
    device: str,
    precision_hint: str | None,
) -> tuple[ov.CompiledModel, dict[str, Any]]:
    model = _keep_only_output(core.read_model(str(model_path)), _OUTPUT_NAME)
    started_at = time.perf_counter()
    compiled = core.compile_model(
        model,
        device,
        _compile_properties(precision_hint=precision_hint),
    )
    elapsed = time.perf_counter() - started_at
    try:
        execution_devices: Any = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        execution_devices = None
    return compiled, {
        "seconds": elapsed,
        "execution_devices": execution_devices,
        "precision_hint": precision_hint,
        "runtime_element_type_counts": _runtime_element_type_counts(compiled),
    }


def _exception_summary(error: Exception, *, max_chars: int = 3000) -> str:
    message = f"{type(error).__name__}: {error}"
    if len(message) <= max_chars:
        return message
    return message[:max_chars] + "...<truncated>"


def _probe_full_synthesis_graph(
    *,
    core: ov.Core,
    aivmx_path: Path,
    inputs: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    """Probe full-graph OpenVINO support without including it in RTF."""

    probes: dict[str, Any] = {}
    scenarios = (
        {
            "name": "gpu_dynamic_final_output_only",
            "device": "GPU",
            "reshape": False,
            "run": True,
        },
        {
            "name": "gpu_static_final_output_only",
            "device": "GPU",
            "reshape": True,
            "run": True,
        },
        {
            "name": "npu_static_final_output_only",
            "device": "NPU",
            "reshape": True,
            "run": True,
        },
        {
            "name": "auto_gpu_cpu_dynamic_final_output_only",
            "device": "AUTO:GPU,CPU",
            "reshape": False,
            "run": True,
        },
    )
    for scenario in scenarios:
        name = scenario["name"]
        device = scenario["device"]
        try:
            model = _keep_only_output(core.read_model(str(aivmx_path)), _OUTPUT_NAME)
            if scenario["reshape"]:
                model.reshape(
                    {
                        input_name: list(value.shape)
                        for input_name, value in inputs.items()
                    }
                )
            started_at = time.perf_counter()
            compiled = core.compile_model(
                model,
                device,
                {"PERFORMANCE_HINT": "LATENCY"},
            )
            compile_seconds = time.perf_counter() - started_at
            try:
                execution_devices: Any = compiled.get_property("EXECUTION_DEVICES")
            except Exception:
                execution_devices = None
            probe: dict[str, Any] = {
                "status": "compile_ok",
                "device": device,
                "reshape": scenario["reshape"],
                "compile_seconds": compile_seconds,
                "execution_devices": execution_devices,
            }
            if scenario["run"]:
                try:
                    started_at = time.perf_counter()
                    output = np.asarray(
                        compiled(inputs)[compiled.output(_OUTPUT_NAME)],
                        dtype=np.float32,
                    )
                    run_seconds = time.perf_counter() - started_at
                    probe.update(
                        {
                            "status": "run_ok",
                            "run_seconds": run_seconds,
                            "output_shape": list(output.shape),
                            "peak_abs": (
                                float(np.max(np.abs(output)))
                                if output.size > 0
                                else 0.0
                            ),
                        }
                    )
                except Exception as ex:
                    probe.update(
                        {
                            "status": "run_failed",
                            "run_error": _exception_summary(ex),
                        }
                    )
            probes[name] = probe
        except Exception as ex:
            probes[name] = {
                "status": "compile_failed",
                "device": device,
                "reshape": scenario["reshape"],
                "compile_error": _exception_summary(ex),
            }
    return probes


def _summarize(records: Sequence[_BenchmarkRecord]) -> list[_BackendSummary]:
    summaries: list[_BackendSummary] = []
    backends = sorted({record.backend for record in records})
    text_labels = sorted({record.text_label for record in records})
    for backend in backends:
        for text_label in text_labels:
            group = [
                record
                for record in records
                if record.backend == backend and record.text_label == text_label
            ]
            if not group:
                continue
            rtfs = [record.rtf for record in group]
            durations = [record.output_duration_seconds for record in group]
            summaries.append(
                _BackendSummary(
                    backend=backend,
                    text_label=text_label,
                    rtf_mean=float(statistics.mean(rtfs)),
                    rtf_min=float(min(rtfs)),
                    rtf_max=float(max(rtfs)),
                    output_duration_seconds_mean=float(statistics.mean(durations)),
                    output_samples_last=group[-1].output_samples,
                )
            )
    return summaries


def _compare_outputs(
    first_outputs: dict[tuple[str, str], NDArray[np.float32]],
) -> list[_TruthComparison]:
    comparisons: list[_TruthComparison] = []
    text_labels = sorted({text_label for _, text_label in first_outputs})
    backends = sorted(
        {backend for backend, _ in first_outputs if backend != "onnx-cpu"}
    )
    for text_label in text_labels:
        truth = first_outputs.get(("onnx-cpu", text_label))
        if truth is None:
            continue
        truth = truth.reshape(-1)
        for backend in backends:
            candidate = first_outputs.get((backend, text_label))
            if candidate is None:
                continue
            candidate = candidate.reshape(-1)
            compared_samples = min(truth.shape[0], candidate.shape[0])
            truth_slice = truth[:compared_samples]
            candidate_slice = candidate[:compared_samples]
            diff = candidate_slice - truth_slice
            truth_std = float(np.std(truth_slice))
            candidate_std = float(np.std(candidate_slice))
            correlation = (
                None
                if truth_std == 0.0 or candidate_std == 0.0
                else float(np.corrcoef(truth_slice, candidate_slice)[0, 1])
            )
            comparisons.append(
                _TruthComparison(
                    backend=backend,
                    text_label=text_label,
                    truth_backend="onnx-cpu",
                    sample_delta=int(candidate.shape[0] - truth.shape[0]),
                    compared_samples=int(compared_samples),
                    rmse=float(np.sqrt(np.mean(np.square(diff)))),
                    max_abs_diff=float(np.max(np.abs(diff))),
                    correlation=correlation,
                )
            )
    return comparisons


def _text_label(index: int) -> str:
    if index < 3:
        return ("short", "medium", "long")[index]
    return f"text-{index}"


def _backend_specs(backends: Sequence[str]) -> list[_BackendSpec]:
    specs: list[_BackendSpec] = []
    for backend in backends:
        if backend == "onnx-cpu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device=None,
                    jpbert_device=None,
                )
            )
        elif backend == "openvino-native-gpu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device="GPU",
                    jpbert_device="GPU",
                )
            )
        elif backend == "openvino-native-npu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device="NPU",
                    jpbert_device="NPU",
                )
            )
        elif backend == "openvino-native-split-gpu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device="GPU",
                    jpbert_device=None,
                )
            )
        elif backend == "openvino-native-split-npu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device="NPU",
                    jpbert_device=None,
                )
            )
        elif backend == "openvino-full-gpu-default":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device=None,
                    jpbert_device=None,
                    full_graph_device="GPU",
                    full_graph_variant="patched",
                    full_graph_precision_hint=None,
                )
            )
        elif backend == "openvino-full-gpu-fp32":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device=None,
                    jpbert_device=None,
                    full_graph_device="GPU",
                    full_graph_variant="patched",
                    full_graph_precision_hint="f32",
                )
            )
        elif backend == "openvino-full-gpu-mixed-fp16":
            specs.append(
                _BackendSpec(
                    name=backend,
                    decoder_device=None,
                    jpbert_device=None,
                    full_graph_device="GPU",
                    full_graph_variant="mixed-fp16",
                    full_graph_precision_hint="f32",
                )
            )
    return specs


def _run_openvino_native_split_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    backends = args.backend or [
        "onnx-cpu",
        "openvino-native-gpu",
        "openvino-native-npu",
    ]
    specs = _backend_specs(backends)
    texts = tuple(args.text or _DEFAULT_TEXTS)
    onnx_providers = _cpu_ort_providers()
    jpbert_model_path = _load_jp_bert(onnx_providers)

    metadata = _read_aivmx_metadata(args.aivmx_path)
    hps = HyperParameters.model_validate(metadata.hyper_parameters.model_dump())
    assert metadata.style_vectors is not None
    style_vectors = np.load(BytesIO(metadata.style_vectors))
    style_vec = _style_vector(
        style_vectors=style_vectors,
        style_id=args.style_id,
        style_weight=args.style_weight,
    )

    with tempfile.TemporaryDirectory(prefix="aivis-openvino-native-") as tmp_dir:
        artifact_dir = args.artifact_dir or Path(tmp_dir)
        front_path, decoder_path = _extract_split_models(
            aivmx_path=args.aivmx_path,
            artifact_dir=artifact_dir,
        )
        has_full_graph_backend = any(
            spec.full_graph_device is not None for spec in specs
        )
        if has_full_graph_backend:
            full_graph_paths, full_graph_artifact_evidence = _prepare_full_graph_models(
                aivmx_path=args.aivmx_path,
                artifact_dir=artifact_dir,
            )
        else:
            full_graph_paths = {}
            full_graph_artifact_evidence = None

        ort_session = _build_ort_session(args.aivmx_path)
        core = ov.Core()
        compiled_front, front_evidence = _compile_front_model(
            core=core,
            front_path=front_path,
        )
        compiled_decoders: dict[
            tuple[str, tuple[int, ...], tuple[int, ...]],
            ov.CompiledModel,
        ] = {}
        decoder_evidence: dict[str, dict[str, Any]] = {}
        jpbert_evidence: dict[str, dict[str, Any]] = {}
        jpbert_runners: dict[str, _OpenVINOJpBertRunner] = {}
        compiled_full_graphs: dict[str, ov.CompiledModel] = {}
        full_graph_evidence: dict[str, dict[str, Any]] = {}
        prebuilt_inputs: dict[tuple[str, str], dict[str, NDArray[Any]]] = {}

        def get_jpbert_runner(device: str | None) -> _OpenVINOJpBertRunner | None:
            if device is None:
                return None
            if device not in jpbert_runners:
                jpbert_runners[device] = _OpenVINOJpBertRunner(
                    core=core,
                    model_path=jpbert_model_path,
                    device=device,
                    evidence=jpbert_evidence,
                )
            return jpbert_runners[device]

        def get_inputs(
            *,
            spec: _BackendSpec,
            text: str,
            text_label: str,
        ) -> dict[str, NDArray[Any]]:
            key = (spec.name, text_label)
            if args.synthesis_only_rtf and key in prebuilt_inputs:
                return prebuilt_inputs[key]
            inputs = _build_inputs(
                text=text,
                hps=hps,
                style_vec=style_vec,
                speaker_id=args.speaker_id,
                length_scale=args.length_scale,
                sdp_ratio=args.sdp_ratio,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                onnx_providers=onnx_providers,
                jpbert_runner=get_jpbert_runner(spec.jpbert_device),
            )
            if args.synthesis_only_rtf:
                prebuilt_inputs[key] = inputs
            return inputs

        def get_decoder(
            *,
            device: str,
            mul9_shape: Sequence[int],
            cond_shape: Sequence[int],
        ) -> ov.CompiledModel:
            key = (device, tuple(mul9_shape), tuple(cond_shape))
            if key in compiled_decoders:
                return compiled_decoders[key]
            compiled, evidence = _compile_decoder_model(
                core=core,
                decoder_path=decoder_path,
                device=device,
                mul9_shape=mul9_shape,
                cond_shape=cond_shape,
            )
            compiled_decoders[key] = compiled
            decoder_evidence[str(key)] = evidence
            return compiled

        def get_full_graph(spec: _BackendSpec) -> ov.CompiledModel:
            assert spec.full_graph_device is not None
            assert spec.full_graph_variant is not None
            key = spec.name
            if key in compiled_full_graphs:
                return compiled_full_graphs[key]
            compiled, evidence = _compile_full_graph_model(
                core=core,
                model_path=full_graph_paths[spec.full_graph_variant],
                device=spec.full_graph_device,
                precision_hint=spec.full_graph_precision_hint,
            )
            compiled_full_graphs[key] = compiled
            full_graph_evidence[key] = evidence
            return compiled

        # Precompile the static decoder shapes outside measured RTF.
        for spec in specs:
            if spec.decoder_device is None:
                continue
            for text_index, text in enumerate(texts):
                text_label = _text_label(text_index)
                inputs = get_inputs(
                    spec=spec,
                    text=text,
                    text_label=text_label,
                )
                mul9, cond = _run_front(
                    compiled_front=compiled_front,
                    inputs=inputs,
                )
                get_decoder(
                    device=spec.decoder_device,
                    mul9_shape=mul9.shape,
                    cond_shape=cond.shape,
                )

        # Precompile dynamic full-graph models outside measured RTF.
        for spec in specs:
            if spec.full_graph_device is not None:
                get_full_graph(spec)

        records: list[_BenchmarkRecord] = []
        first_outputs: dict[tuple[str, str], NDArray[np.float32]] = {}
        for spec in specs:
            for text_index, text in enumerate(texts):
                text_label = _text_label(text_index)
                for _ in range(args.warmup_runs):
                    inputs = get_inputs(
                        spec=spec,
                        text=text,
                        text_label=text_label,
                    )
                    if spec.full_graph_device is not None:
                        full_graph = get_full_graph(spec)
                        output = full_graph(inputs)[full_graph.output(_OUTPUT_NAME)]
                    elif spec.decoder_device is None:
                        output = ort_session.run([_OUTPUT_NAME], inputs)[0]
                    else:
                        mul9, cond = _run_front(
                            compiled_front=compiled_front,
                            inputs=inputs,
                        )
                        decoder = get_decoder(
                            device=spec.decoder_device,
                            mul9_shape=mul9.shape,
                            cond_shape=cond.shape,
                        )
                        output = decoder(
                            {
                                _DECODER_INPUT_NAMES[0]: mul9,
                                _DECODER_INPUT_NAMES[1]: cond,
                            }
                        )[decoder.output(0)]

                for run_index in range(args.runs):
                    started_at = time.perf_counter()
                    inputs = get_inputs(
                        spec=spec,
                        text=text,
                        text_label=text_label,
                    )
                    if spec.full_graph_device is not None:
                        full_graph = get_full_graph(spec)
                        output = full_graph(inputs)[full_graph.output(_OUTPUT_NAME)]
                    elif spec.decoder_device is None:
                        output = ort_session.run([_OUTPUT_NAME], inputs)[0]
                    else:
                        mul9, cond = _run_front(
                            compiled_front=compiled_front,
                            inputs=inputs,
                        )
                        decoder = get_decoder(
                            device=spec.decoder_device,
                            mul9_shape=mul9.shape,
                            cond_shape=cond.shape,
                        )
                        output = decoder(
                            {
                                _DECODER_INPUT_NAMES[0]: mul9,
                                _DECODER_INPUT_NAMES[1]: cond,
                            }
                        )[decoder.output(0)]
                    elapsed_seconds = time.perf_counter() - started_at
                    output = np.asarray(output, dtype=np.float32)
                    output_samples = int(output.shape[-1])
                    output_duration_seconds = output_samples / hps.data.sampling_rate
                    records.append(
                        _BenchmarkRecord(
                            backend=spec.name,
                            text_label=text_label,
                            text=text,
                            run_index=run_index,
                            elapsed_seconds=elapsed_seconds,
                            output_duration_seconds=output_duration_seconds,
                            output_samples=output_samples,
                            rtf=elapsed_seconds / output_duration_seconds,
                            peak_abs=(
                                float(np.max(np.abs(output)))
                                if output.size > 0
                                else 0.0
                            ),
                        )
                    )
                    if run_index == 0:
                        first_outputs[(spec.name, text_label)] = output.reshape(-1)
                        if args.audio_output_dir is not None:
                            args.audio_output_dir.mkdir(parents=True, exist_ok=True)
                            soundfile.write(
                                args.audio_output_dir / f"{spec.name}_{text_label}.wav",
                                output.reshape(-1),
                                hps.data.sampling_rate,
                                subtype="PCM_16",
                            )

        full_graph_probe: dict[str, Any] | None = None
        if args.probe_full_graph:
            probe_inputs = _build_inputs(
                text=texts[0],
                hps=hps,
                style_vec=style_vec,
                speaker_id=args.speaker_id,
                length_scale=args.length_scale,
                sdp_ratio=args.sdp_ratio,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                onnx_providers=onnx_providers,
                jpbert_runner=None,
            )
            full_graph_probe = _probe_full_synthesis_graph(
                core=core,
                aivmx_path=args.aivmx_path,
                inputs=probe_inputs,
            )

        return {
            "profile": {
                "aivmx_path": f"<local-model-dir>/{args.aivmx_path.name}",
                "backends": list(backends),
                "texts": list(texts),
                "warmup_runs": args.warmup_runs,
                "runs": args.runs,
                "speaker_id": args.speaker_id,
                "style_id": args.style_id,
                "style_weight": args.style_weight,
                "length_scale": args.length_scale,
                "sdp_ratio": args.sdp_ratio,
                "noise_scale": args.noise_scale,
                "noise_scale_w": args.noise_scale_w,
                "synthesis_only_rtf": args.synthesis_only_rtf,
                "rtf_scope": (
                    "synthesis graph only; text/JP-BERT inputs prebuilt; "
                    "OpenVINO compile excluded; raw model waveform duration"
                    if args.synthesis_only_rtf
                    else "g2p + selected JP-BERT backend + synthesis graph; "
                    "OpenVINO compile excluded; raw model waveform duration"
                ),
            },
            "provider_evidence": {
                "front_cpu": front_evidence,
                "jpbert": jpbert_evidence,
                "decoder": decoder_evidence,
                "full_graph_artifacts": full_graph_artifact_evidence,
                "full_graph": full_graph_evidence,
                "full_synthesis_graph_probe": full_graph_probe,
                "native_full_npu": (
                    "The full synthesis graph is not benchmarked as NPU because "
                    "OpenVINO NPU fails to compile its data-dependent duration/"
                    "alignment/output-length subgraph. The split backend runs the "
                    "decoder subgraph on NPU with static shape."
                ),
            },
            "summary": [asdict(summary) for summary in _summarize(records)],
            "truth_comparison": [
                asdict(comparison) for comparison in _compare_outputs(first_outputs)
            ],
            "records": [asdict(record) for record in records],
        }


def main() -> None:
    """Run the direct OpenVINO split benchmark and print JSON summary data."""

    args = _parse_args()
    payload = _run_openvino_native_split_benchmark(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
