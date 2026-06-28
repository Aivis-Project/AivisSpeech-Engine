"""Benchmark ONNX CPU/CUDA against the Aivis GGML ONNX Plugin EP."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import aivmlib
import numpy as np
import soundfile
from style_bert_vits2.nlp import onnx_bert_models

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run import _resolve_default_onnx_ep_library_path
from voicevox_engine.aivm_gguf_cache import (
    DEFAULT_GGUF_CONVERTER_VERSION,
)
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    BuiltinOnnxProvider,
    OnnxPluginExecutionProviderConfig,
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.utility.aivishub_client import (
    AivisHubClient,
    AivisSpeechDefaultModelProperty,
    AivisSpeechForcedRemovalRule,
    AivmModelResponse,
)

_DEFAULT_TEXTS = (
    "テストです。",
    "今日はいい天気ですね。",
    "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。",
)

_DEFAULT_WARMUP_TEXTS = (
    "測定用ではない短い文です。",
    "ウォームアップのために別の文章を読み上げます。",
    "測定対象とは異なる長めのウォームアップ文章です。バックエンドの初回処理だけを先に済ませます。",
)


class _NoNetworkAivisHubClient(AivisHubClient):
    """AivisHub client that prevents benchmark runs from touching the network."""

    def fetch_default_models(self) -> list[AivisSpeechDefaultModelProperty]:
        return []

    def fetch_forced_removal_rules(self) -> list[AivisSpeechForcedRemovalRule]:
        return []

    async def fetch_model_detail(
        self,
        aivm_model_uuid: UUID,
    ) -> AivmModelResponse | None:
        return None

    def send_event(self, *args: Any, **kwargs: Any) -> None:
        return


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    use_gpu: bool
    required_provider: str
    preferred_onnx_provider: BuiltinOnnxProvider | None = None
    onnx_plugin_ep: OnnxPluginExecutionProviderConfig | None = None
    ggml_synthesis_converter_version: str | None = None
    ggml_jp_bert_precision_label: str | None = None
    ggml_jp_bert_gguf_path: Path | None = None


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Aivis Style-Bert-VITS2 ONNX CPU/CUDA and ONNX GGML "
            "Plugin EP Vulkan paths."
        )
    )
    parser.add_argument(
        "--aivmx_path",
        type=Path,
        required=True,
        help="AIVMX/ONNX model path to install into a temporary Models directory.",
    )
    parser.add_argument(
        "--style_id",
        type=int,
        required=True,
        help="Global Aivis style id to synthesize with.",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "onnx-cpu",
            "onnx-directml",
            "onnx-cuda",
            "onnx-ggml-vulkan",
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
        help=(
            "Warmup syntheses per backend/text before measured runs. Warmup "
            "uses --warmup_text, never the measured --text values."
        ),
    )
    parser.add_argument(
        "--warmup_text",
        action="append",
        default=None,
        help=(
            "Text used only for warmup. Repeat for short/medium/long warmup "
            "texts. Values must not match any measured --text."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured syntheses per backend/text.",
    )
    parser.add_argument(
        "--tempo_dynamics_scale",
        type=float,
        default=1.0,
        help=(
            "AudioQuery tempoDynamicsScale used for synthesis. "
            "1.0 matches the Engine /audio_query default."
        ),
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="Optional Style-Bert-VITS2 noise value for deterministic parity checks.",
    )
    parser.add_argument(
        "--noise_scale_w",
        type=float,
        default=None,
        help="Optional Style-Bert-VITS2 SDP noise value for deterministic parity checks.",
    )
    parser.add_argument(
        "--ggml_native_library_path",
        type=Path,
        default=None,
        help="TTS.cpp C API shared library path. Required for onnx-ggml-vulkan.",
    )
    parser.add_argument(
        "--onnx_ep_library_path",
        type=Path,
        default=None,
        help="Aivis GGML ONNX Runtime Plugin EP shared library path.",
    )
    parser.add_argument(
        "--ggml_model_cache_dir",
        type=Path,
        default=None,
        help="Optional GGUF cache directory for the Plugin EP run.",
    )
    parser.add_argument(
        "--ggml_jp_bert_gguf_path",
        type=Path,
        default=None,
        help=(
            "Optional prepared JP-BERT GGUF path for the Plugin EP run. "
            "When omitted, the Engine cache prepares or fetches the default bundle."
        ),
    )
    parser.add_argument(
        "--ggml_vulkan_device",
        default=None,
        help="Provider option device id for Vulkan.",
    )
    parser.add_argument(
        "--ggml_vulkan_precision",
        choices=("accurate", "fast"),
        default="accurate",
        help="Provider option precision for Vulkan.",
    )
    parser.add_argument(
        "--ggml_vulkan_math_mode",
        choices=("f32", "coopmat", "fp16", "fp16-coopmat"),
        default="coopmat",
        help=(
            "Provider option controlling ggml-vulkan F16 and cooperative "
            "matrix use."
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
        "--skip_truth_comparison",
        action="store_true",
        help=(
            "Skip ONNX CPU PCM truth comparison even when audio_output_dir is set. "
            "Use this for natural stochastic preview runs where noise/noise_w are "
            "left at Style-Bert-VITS2 defaults."
        ),
    )
    return parser.parse_args()


def _validate_warmup_texts(
    *,
    texts: Sequence[str],
    warmup_texts: Sequence[str],
) -> None:
    measured = {text.strip() for text in texts}
    overlapping = [text for text in warmup_texts if text.strip() in measured]
    if overlapping:
        raise ValueError(
            "Warmup text must be different from measured benchmark text: "
            + ", ".join(repr(text) for text in overlapping)
        )


def _patch_tts_model_noise(
    *,
    noise_scale: float | None,
    noise_scale_w: float | None,
) -> None:
    if noise_scale is None and noise_scale_w is None:
        return

    from style_bert_vits2.tts_model import TTSModel

    original_infer = TTSModel.infer

    def infer_with_fixed_noise(self: Any, *args: Any, **kwargs: Any) -> Any:
        if noise_scale is not None:
            kwargs.setdefault("noise", noise_scale)
        if noise_scale_w is not None:
            kwargs.setdefault("noise_w", noise_scale_w)
        return original_infer(self, *args, **kwargs)

    TTSModel.infer = infer_with_fixed_noise


def _read_aivmx_model_uuid(aivmx_path: Path) -> str:
    with aivmx_path.open("rb") as file:
        metadata = aivmlib.read_aivmx_metadata(file)
    return str(metadata.manifest.uuid)


def _prepare_models_dir(*, tmp_path: Path, aivmx_path: Path) -> tuple[Path, str]:
    model_uuid = _read_aivmx_model_uuid(aivmx_path)
    models_dir = tmp_path / "Models"
    models_dir.mkdir(parents=True)
    shutil.copyfile(aivmx_path, models_dir / f"{model_uuid}.aivmx")
    return models_dir, model_uuid


def _build_aivm_manager(*, tmp_path: Path, models_dir: Path) -> AivmManager:
    return AivmManager(
        models_dir,
        aivishub_client=_NoNetworkAivisHubClient(
            installation_uuid_path=tmp_path / "installation_uuid.dat",
        ),
        cache_file_path=tmp_path / "aivm_infos_cache.json",
        is_background_scan_enabled=False,
    )


def _build_audio_query(
    *,
    engine: StyleBertVITS2TTSEngine,
    text: str,
    style_id: StyleId,
    tempo_dynamics_scale: float,
) -> AudioQuery:
    accent_phrases = engine.create_accent_phrases(
        text,
        style_id,
        enable_katakana_english=True,
    )
    return AudioQuery(
        accent_phrases=accent_phrases,
        speedScale=1.0,
        intonationScale=1.0,
        tempoDynamicsScale=tempo_dynamics_scale,
        pitchScale=0.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        pauseLength=None,
        pauseLengthScale=1.0,
        outputSamplingRate=engine.default_sampling_rate,
        outputStereo=False,
        kana=text,
    )


def _build_ggml_plugin_config(
    args: argparse.Namespace,
    *,
    jp_bert_gguf_path: Path | None = None,
    inherit_global_jp_bert_gguf_path: bool = True,
) -> OnnxPluginExecutionProviderConfig:
    if args.ggml_native_library_path is None:
        raise RuntimeError("onnx-ggml-vulkan requires --ggml_native_library_path.")
    provider_options = {
        "backend": "vulkan",
        "claim_jp_bert_graph": "1",
        "claim_synthesis_graph": "1",
        "eager_load_model": "1",
        "n_threads": "0",
        "precision": args.ggml_vulkan_precision,
        "vulkan_math_mode": args.ggml_vulkan_math_mode,
        "tts_cpp_library_path": str(args.ggml_native_library_path),
    }
    if args.ggml_vulkan_device is not None:
        provider_options["device"] = args.ggml_vulkan_device
    configured_jp_bert_gguf_path = jp_bert_gguf_path
    if configured_jp_bert_gguf_path is None and inherit_global_jp_bert_gguf_path:
        configured_jp_bert_gguf_path = args.ggml_jp_bert_gguf_path
    if configured_jp_bert_gguf_path is not None:
        provider_options["jp_bert_gguf_path"] = str(configured_jp_bert_gguf_path)
    return OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options=provider_options,
        library_path=_resolve_default_onnx_ep_library_path(args.onnx_ep_library_path),
        strict=True,
    )


def _build_backend_specs(args: argparse.Namespace) -> list[_BackendSpec]:
    requested_backends = args.backend or [
        "onnx-cpu",
        "onnx-cuda",
        "onnx-ggml-vulkan",
    ]
    specs: list[_BackendSpec] = []

    def add_ggml_spec(
        *,
        name: str,
        synthesis_converter_version: str | None = None,
        jp_bert_precision_label: str | None = None,
        jp_bert_gguf_path: Path | None = None,
        inherit_global_jp_bert_gguf_path: bool = True,
    ) -> None:
        configured_jp_bert_gguf_path = jp_bert_gguf_path
        if configured_jp_bert_gguf_path is None and inherit_global_jp_bert_gguf_path:
            configured_jp_bert_gguf_path = args.ggml_jp_bert_gguf_path
        specs.append(
            _BackendSpec(
                name=name,
                use_gpu=False,
                required_provider="AivisGgmlExecutionProvider",
                onnx_plugin_ep=_build_ggml_plugin_config(
                    args,
                    jp_bert_gguf_path=jp_bert_gguf_path,
                    inherit_global_jp_bert_gguf_path=(
                        inherit_global_jp_bert_gguf_path
                    ),
                ),
                ggml_synthesis_converter_version=synthesis_converter_version,
                ggml_jp_bert_precision_label=jp_bert_precision_label,
                ggml_jp_bert_gguf_path=configured_jp_bert_gguf_path,
            )
        )

    for backend in requested_backends:
        if backend == "onnx-cpu":
            specs.append(
                _BackendSpec(
                    name=backend,
                    use_gpu=False,
                    required_provider="CPUExecutionProvider",
                )
            )
        elif backend == "onnx-cuda":
            specs.append(
                _BackendSpec(
                    name=backend,
                    use_gpu=True,
                    required_provider="CUDAExecutionProvider",
                    preferred_onnx_provider="cuda",
                )
            )
        elif backend == "onnx-directml":
            specs.append(
                _BackendSpec(
                    name=backend,
                    use_gpu=True,
                    required_provider="DmlExecutionProvider",
                    preferred_onnx_provider="directml",
                )
            )
        elif backend == "onnx-ggml-vulkan":
            add_ggml_spec(
                name=backend,
                synthesis_converter_version=DEFAULT_GGUF_CONVERTER_VERSION,
                jp_bert_precision_label="fp16-linear",
                inherit_global_jp_bert_gguf_path=False,
            )
    return specs


def _validate_active_provider(
    *,
    engine: StyleBertVITS2TTSEngine,
    model_uuid: str,
    spec: _BackendSpec,
) -> list[str]:
    tts_model = engine.tts_models.get(model_uuid)
    if tts_model is None or tts_model.onnx_session is None:
        raise RuntimeError(f"{spec.name} did not create an ONNX session.")
    providers = list(tts_model.onnx_session.get_providers())
    active_provider = providers[0] if providers else None
    if active_provider != spec.required_provider:
        raise RuntimeError(
            f"{spec.name} expected {spec.required_provider}, but ONNX Runtime "
            f"selected {active_provider}. Full providers: {providers}"
        )
    return providers


def _summarize(records: Sequence[_BenchmarkRecord]) -> list[_BackendSummary]:
    groups: dict[tuple[str, str], list[_BenchmarkRecord]] = {}
    for record in records:
        groups.setdefault((record.backend, record.text_label), []).append(record)

    summaries: list[_BackendSummary] = []
    for (backend, text_label), group_records in sorted(groups.items()):
        rtfs = [record.rtf for record in group_records]
        durations = [record.output_duration_seconds for record in group_records]
        summaries.append(
            _BackendSummary(
                backend=backend,
                text_label=text_label,
                rtf_mean=float(np.mean(rtfs)),
                rtf_min=float(np.min(rtfs)),
                rtf_max=float(np.max(rtfs)),
                output_duration_seconds_mean=float(np.mean(durations)),
                output_samples_last=group_records[-1].output_samples,
            )
        )
    return summaries


def _compare_against_onnx_cpu_truth(
    *,
    audio_output_dir: Path | None,
    summaries: Sequence[_BackendSummary],
) -> list[_TruthComparison]:
    if audio_output_dir is None:
        return []

    text_labels = sorted({summary.text_label for summary in summaries})
    backends = sorted(
        {
            summary.backend
            for summary in summaries
            if summary.backend != "onnx-cpu"
        }
    )
    comparisons: list[_TruthComparison] = []
    for text_label in text_labels:
        truth_path = audio_output_dir / f"onnx-cpu_{text_label}.wav"
        if not truth_path.is_file():
            continue
        truth_wave, truth_sample_rate = soundfile.read(
            truth_path,
            dtype="float32",
            always_2d=False,
        )
        truth_wave = np.asarray(truth_wave, dtype=np.float32).reshape(-1)
        for backend in backends:
            candidate_path = audio_output_dir / f"{backend}_{text_label}.wav"
            if not candidate_path.is_file():
                continue
            candidate_wave, candidate_sample_rate = soundfile.read(
                candidate_path,
                dtype="float32",
                always_2d=False,
            )
            if candidate_sample_rate != truth_sample_rate:
                raise RuntimeError(
                    f"{candidate_path} sample rate {candidate_sample_rate} "
                    f"does not match ONNX CPU truth {truth_sample_rate}."
                )
            candidate_wave = np.asarray(candidate_wave, dtype=np.float32).reshape(-1)
            compared_samples = min(truth_wave.shape[0], candidate_wave.shape[0])
            if compared_samples == 0:
                rmse = 0.0
                max_abs_diff = 0.0
                correlation: float | None = None
            else:
                truth_slice = truth_wave[:compared_samples]
                candidate_slice = candidate_wave[:compared_samples]
                diff = candidate_slice - truth_slice
                rmse = float(np.sqrt(np.mean(np.square(diff))))
                max_abs_diff = float(np.max(np.abs(diff)))
                truth_std = float(np.std(truth_slice))
                candidate_std = float(np.std(candidate_slice))
                if truth_std == 0.0 or candidate_std == 0.0:
                    correlation = None
                else:
                    correlation = float(np.corrcoef(truth_slice, candidate_slice)[0, 1])
            comparisons.append(
                _TruthComparison(
                    backend=backend,
                    text_label=text_label,
                    truth_backend="onnx-cpu",
                    sample_delta=int(candidate_wave.shape[0] - truth_wave.shape[0]),
                    compared_samples=int(compared_samples),
                    rmse=rmse,
                    max_abs_diff=max_abs_diff,
                    correlation=correlation,
                )
            )
    return comparisons


def _benchmark_backend(
    *,
    spec: _BackendSpec,
    tmp_path: Path,
    models_dir: Path,
    model_uuid: str,
    style_id: StyleId,
    texts: Sequence[str],
    warmup_texts: Sequence[str],
    warmup_runs: int,
    runs: int,
    tempo_dynamics_scale: float,
    ggml_model_cache_dir: Path | None,
    audio_output_dir: Path | None,
) -> tuple[list[_BenchmarkRecord], dict[str, Any]]:
    onnx_bert_models.unload_all_models()
    aivm_manager = _build_aivm_manager(tmp_path=tmp_path, models_dir=models_dir)
    engine = StyleBertVITS2TTSEngine(
        aivm_manager,
        use_gpu=spec.use_gpu,
        load_all_models=False,
        preferred_onnx_provider=spec.preferred_onnx_provider,
        onnx_plugin_ep=spec.onnx_plugin_ep,
        ggml_model_cache_dir=ggml_model_cache_dir,
        ggml_synthesis_converter_version=spec.ggml_synthesis_converter_version,
    )

    records: list[_BenchmarkRecord] = []
    provider_evidence: dict[str, Any] = {}
    for text_index, text in enumerate(texts):
        text_label = (
            ("short", "medium", "long")[text_index]
            if text_index < 3
            else f"text-{text_index}"
        )
        warmup_text = warmup_texts[text_index % len(warmup_texts)]
        warmup_query = _build_audio_query(
            engine=engine,
            text=warmup_text,
            style_id=style_id,
            tempo_dynamics_scale=tempo_dynamics_scale,
        )
        for _ in range(warmup_runs):
            engine.synthesize_wave(
                warmup_query,
                style_id,
                enable_interrogative_upspeak=True,
            )

        query = _build_audio_query(
            engine=engine,
            text=text,
            style_id=style_id,
            tempo_dynamics_scale=tempo_dynamics_scale,
        )

        provider_evidence["active_providers"] = _validate_active_provider(
            engine=engine,
            model_uuid=model_uuid,
            spec=spec,
        )
        if spec.ggml_synthesis_converter_version is not None:
            provider_evidence["ggml_synthesis_converter_version"] = (
                spec.ggml_synthesis_converter_version
            )
        if spec.ggml_jp_bert_precision_label is not None:
            provider_evidence["ggml_jp_bert_precision"] = (
                spec.ggml_jp_bert_precision_label
            )
        if spec.ggml_jp_bert_gguf_path is not None:
            provider_evidence["ggml_jp_bert_gguf_path"] = (
                f"<local-gguf-dir>/{spec.ggml_jp_bert_gguf_path.name}"
            )
        for run_index in range(runs):
            started_at = time.perf_counter()
            wave = engine.synthesize_wave(
                query,
                style_id,
                enable_interrogative_upspeak=True,
            )
            elapsed_seconds = time.perf_counter() - started_at
            output_samples = int(wave.shape[0])
            output_duration_seconds = output_samples / query.outputSamplingRate
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
                    peak_abs=float(np.max(np.abs(wave))) if wave.size > 0 else 0.0,
                )
            )
            if run_index == 0 and audio_output_dir is not None:
                audio_output_dir.mkdir(parents=True, exist_ok=True)
                soundfile.write(
                    audio_output_dir / f"{spec.name}_{text_label}.wav",
                    wave,
                    query.outputSamplingRate,
                    subtype="PCM_16",
                )
    return records, provider_evidence


def main() -> None:
    """Run the selected backend benchmarks and emit JSON summary data."""

    args = _parse_args()
    _patch_tts_model_noise(
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
    )
    texts = tuple(args.text or _DEFAULT_TEXTS)
    warmup_texts = tuple(args.warmup_text or _DEFAULT_WARMUP_TEXTS)
    _validate_warmup_texts(texts=texts, warmup_texts=warmup_texts)
    style_id = StyleId(args.style_id)
    specs = _build_backend_specs(args)

    with tempfile.TemporaryDirectory(prefix="aivis-onnx-ggml-bench-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir, model_uuid = _prepare_models_dir(
            tmp_path=tmp_path,
            aivmx_path=args.aivmx_path,
        )

        all_records: list[_BenchmarkRecord] = []
        provider_evidence: dict[str, dict[str, Any]] = {}
        for spec in specs:
            records, evidence = _benchmark_backend(
                spec=spec,
                tmp_path=tmp_path,
                models_dir=models_dir,
                model_uuid=model_uuid,
                style_id=style_id,
                texts=texts,
                warmup_texts=warmup_texts,
                warmup_runs=args.warmup_runs,
                runs=args.runs,
                tempo_dynamics_scale=args.tempo_dynamics_scale,
                ggml_model_cache_dir=args.ggml_model_cache_dir,
                audio_output_dir=args.audio_output_dir,
            )
            all_records.extend(records)
            provider_evidence[spec.name] = evidence

    summaries = _summarize(all_records)
    truth_comparison = (
        []
        if args.skip_truth_comparison
        else _compare_against_onnx_cpu_truth(
            audio_output_dir=args.audio_output_dir,
            summaries=summaries,
        )
    )
    payload = {
        "profile": {
            "aivmx_path": f"<local-model-dir>/{args.aivmx_path.name}",
            "style_id": int(style_id),
            "texts": list(texts),
            "warmup_texts": list(warmup_texts),
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "tempo_dynamics_scale": args.tempo_dynamics_scale,
            "noise_scale": args.noise_scale,
            "noise_scale_w": args.noise_scale_w,
            "truth_comparison_enabled": not args.skip_truth_comparison,
            "ggml_vulkan_precision": args.ggml_vulkan_precision,
            "ggml_vulkan_math_mode": args.ggml_vulkan_math_mode,
            "ggml_default_synthesis_converter_version": DEFAULT_GGUF_CONVERTER_VERSION,
            "ggml_jp_bert_gguf_path": (
                f"<local-gguf-dir>/{args.ggml_jp_bert_gguf_path.name}"
                if args.ggml_jp_bert_gguf_path is not None
                else None
            ),
        },
        "provider_evidence": provider_evidence,
        "summary": [asdict(summary) for summary in summaries],
        "truth_comparison": [
            asdict(comparison) for comparison in truth_comparison
        ],
        "records": [asdict(record) for record in all_records],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
