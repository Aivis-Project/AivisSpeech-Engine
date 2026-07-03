"""Export frontend tensors for the Android GGML mobile benchmark."""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import aivmlib
import numpy as np
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.infer_onnx import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature_onnx,
)
from style_bert_vits2.nlp import onnx_bert_models
from style_bert_vits2.tts_model import TTSModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.benchmark_onnx_ggml_provider import (  # noqa: E402
    _DEFAULT_TEXTS,
    _DEFAULT_WARMUP_TEXTS,
    _build_audio_query,
    _NoNetworkAivisHubClient,
)
from voicevox_engine.aivm_manager import AivmManager  # noqa: E402
from voicevox_engine.metas.metas import StyleId  # noqa: E402
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (  # noqa: E402
    StyleBertVITS2TTSEngine,
)

_MAGIC = b"AIVISMB1"
_VERSION = 2
_T = TypeVar("_T")


@dataclass(frozen=True)
class _CapturedCase:
    role: int
    label: str
    text: str
    tokens: int
    speaker_id: int
    input_sample_rate: int
    sdp_ratio: float
    length_scale: float
    noise_scale: float
    noise_w_scale: float
    phone_ids: np.ndarray
    tone_ids: np.ndarray
    language_ids: np.ndarray
    bert: np.ndarray
    bert_input_ids: np.ndarray
    bert_word2ph: np.ndarray
    style_vec: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export precomputed frontend tensors for Android-side TTS.cpp GGUF "
            "synthesis benchmarking."
        )
    )
    parser.add_argument("--aivmx_path", type=Path, required=True)
    parser.add_argument("--style_id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--warmup_text", action="append", default=None)
    parser.add_argument("--tempo_dynamics_scale", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--noise_scale_w", type=float, default=None)
    return parser.parse_args()


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


def _selected_bert(
    *,
    language: Languages,
    zh_bert: np.ndarray,
    ja_bert: np.ndarray,
    en_bert: np.ndarray,
) -> np.ndarray:
    if language == Languages.JP:
        return ja_bert
    if language == Languages.ZH:
        return zh_bert
    if language == Languages.EN:
        return en_bert
    raise ValueError(f"Unsupported language: {language}")


def _intersperse(items: list[_T], item: _T) -> list[_T]:
    result = [item] * (len(items) * 2 + 1)
    result[1::2] = items
    return result


def _capture_case(
    *,
    model: TTSModel,
    label: str,
    role: int,
    kwargs: dict[str, Any],
    noise_scale: float | None,
    noise_scale_w: float | None,
) -> _CapturedCase:
    language = kwargs.get("language", Languages.JP)
    if not isinstance(language, Languages):
        language = Languages(language)
    style_name = kwargs.get("style", "Neutral")
    style_weight = float(kwargs.get("style_weight", 1.0))
    style_id = model.style2id[style_name]
    style_vec = np.ascontiguousarray(
        model.get_style_vector(style_id, style_weight).astype(np.float32)
    )

    is_jp_extra_like_model = model.hyper_parameters.is_jp_extra_like_model()
    is_nanairo_like_model = model.hyper_parameters.is_nanairo_like_model()
    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        kwargs["text"],
        language,
        given_phone=kwargs.get("given_phone"),
        given_tone=kwargs.get("given_tone"),
        use_jp_extra=is_jp_extra_like_model,
        use_nanairo=is_nanairo_like_model,
        raise_yomi_error=False,
    )
    phone, tone, language_ids = cleaned_text_to_sequence(
        phone,
        tone,
        language,
        use_nanairo=is_nanairo_like_model,
    )

    if model.hyper_parameters.data.add_blank:
        phone = _intersperse(phone, 0)
        tone = _intersperse(tone, 0)
        language_ids = _intersperse(language_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_ori = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        language,
        model.onnx_providers,
        assist_text=kwargs.get("assist_text"),
        assist_text_weight=float(kwargs.get("assist_text_weight", 0.7)),
        sep_text=sep_text,
        use_nanairo=is_nanairo_like_model,
    )
    if language == Languages.ZH:
        zh_bert = bert_ori
        ja_bert = np.zeros((1024, len(phone)), dtype=np.float32)
        en_bert = np.zeros((1024, len(phone)), dtype=np.float32)
    elif language == Languages.JP:
        zh_bert = np.zeros((1024, len(phone)), dtype=np.float32)
        ja_bert = bert_ori
        en_bert = np.zeros((1024, len(phone)), dtype=np.float32)
    elif language == Languages.EN:
        zh_bert = np.zeros((1024, len(phone)), dtype=np.float32)
        ja_bert = np.zeros((1024, len(phone)), dtype=np.float32)
        en_bert = bert_ori
    else:
        raise ValueError(f"Unsupported language: {language}")

    bert = _selected_bert(
        language=language,
        zh_bert=zh_bert,
        ja_bert=ja_bert,
        en_bert=en_bert,
    )
    bert = np.ascontiguousarray(bert.astype(np.float32))
    phone_ids = np.ascontiguousarray(np.array(phone, dtype=np.int32))
    tone_ids = np.ascontiguousarray(np.array(tone, dtype=np.int32))
    language_ids_i32 = np.ascontiguousarray(np.array(language_ids, dtype=np.int32))
    tokens = int(phone_ids.shape[0])
    if bert.shape != (1024, tokens):
        raise ValueError(
            f"{label}: expected BERT shape (1024, {tokens}), got {bert.shape}"
        )

    bert_input_ids = np.empty((0,), dtype=np.int32)
    bert_word2ph = np.empty((0,), dtype=np.int32)
    if language == Languages.JP:
        tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
        bert_text = "".join(sep_text)
        tokenized = tokenizer(bert_text, return_tensors="np")
        bert_input_ids = np.ascontiguousarray(
            tokenized["input_ids"][0].astype(np.int32)  # type: ignore[index]
        )
        bert_word2ph = np.ascontiguousarray(np.array(word2ph, dtype=np.int32))
        if bert_input_ids.shape[0] != bert_word2ph.shape[0]:
            raise ValueError(
                f"{label}: JP-BERT input_ids length {bert_input_ids.shape[0]} "
                f"does not match word2ph length {bert_word2ph.shape[0]}"
            )
        if int(bert_word2ph.sum()) != tokens:
            raise ValueError(
                f"{label}: JP-BERT word2ph sum {int(bert_word2ph.sum())} "
                f"does not match synthesis token count {tokens}"
            )

    return _CapturedCase(
        role=role,
        label=label,
        text=kwargs["text"],
        tokens=tokens,
        speaker_id=int(kwargs.get("speaker_id", 0)),
        input_sample_rate=int(model.hyper_parameters.data.sampling_rate),
        sdp_ratio=float(kwargs.get("sdp_ratio", 0.2)),
        length_scale=float(kwargs.get("length", 1.0)),
        noise_scale=(
            float(noise_scale)
            if noise_scale is not None
            else float(kwargs.get("noise", 0.6))
        ),
        noise_w_scale=(
            float(noise_scale_w)
            if noise_scale_w is not None
            else float(kwargs.get("noise_w", 0.8))
        ),
        phone_ids=phone_ids,
        tone_ids=tone_ids,
        language_ids=language_ids_i32,
        bert=bert,
        bert_input_ids=bert_input_ids,
        bert_word2ph=bert_word2ph,
        style_vec=style_vec,
    )


def _write_string(file: Any, value: str) -> None:
    file.write(value.encode("utf-8"))


def _write_case(file: Any, case: _CapturedCase) -> None:
    label_bytes = case.label.encode("utf-8")
    text_bytes = case.text.encode("utf-8")
    file.write(
        struct.pack(
            "<IIIIiIffffII",
            case.role,
            len(label_bytes),
            len(text_bytes),
            case.tokens,
            case.speaker_id,
            case.input_sample_rate,
            case.sdp_ratio,
            case.length_scale,
            case.noise_scale,
            case.noise_w_scale,
            case.bert_input_ids.size,
            case.bert_word2ph.size,
        )
    )
    _write_string(file, case.label)
    _write_string(file, case.text)
    file.write(case.phone_ids.tobytes(order="C"))
    file.write(case.tone_ids.tobytes(order="C"))
    file.write(case.language_ids.tobytes(order="C"))
    file.write(case.bert.tobytes(order="C"))
    file.write(case.bert_input_ids.tobytes(order="C"))
    file.write(case.bert_word2ph.tobytes(order="C"))
    file.write(case.style_vec.tobytes(order="C"))


def _write_bundle(path: Path, cases: Sequence[_CapturedCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(_MAGIC)
        file.write(struct.pack("<II", _VERSION, len(cases)))
        for case in cases:
            _write_case(file, case)


def main() -> None:
    """Export a mobile benchmark bundle from installed AIVMX models."""

    args = _parse_args()
    texts = tuple(args.text or _DEFAULT_TEXTS)
    warmup_texts = tuple(args.warmup_text or _DEFAULT_WARMUP_TEXTS)
    if {text.strip() for text in texts} & {text.strip() for text in warmup_texts}:
        raise ValueError("Warmup texts must differ from measured texts.")

    captured: list[_CapturedCase] = []
    original_infer = TTSModel.infer
    pending_labels: list[tuple[int, str]] = []

    def capture_infer(
        self: TTSModel, *infer_args: Any, **kwargs: Any
    ) -> tuple[int, np.ndarray]:
        del infer_args
        if not pending_labels:
            raise RuntimeError("No pending label for captured TTSModel.infer call.")
        role, label = pending_labels.pop(0)
        captured.append(
            _capture_case(
                model=self,
                label=label,
                role=role,
                kwargs=kwargs,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
            )
        )
        silence = np.zeros(
            int(self.hyper_parameters.data.sampling_rate * 0.1), dtype=np.int16
        )
        return self.hyper_parameters.data.sampling_rate, silence

    TTSModel.infer = capture_infer
    try:
        onnx_bert_models.unload_all_models()
        with tempfile.TemporaryDirectory(prefix="aivis-mobile-ggml-export-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            models_dir, model_uuid = _prepare_models_dir(
                tmp_path=tmp_path,
                aivmx_path=args.aivmx_path,
            )
            engine = StyleBertVITS2TTSEngine(
                _build_aivm_manager(tmp_path=tmp_path, models_dir=models_dir),
                use_gpu=False,
                load_all_models=False,
            )
            style_id = StyleId(args.style_id)
            labels = ("short", "medium", "long")
            for index, text in enumerate(warmup_texts):
                label = (
                    f"warmup_{labels[index]}"
                    if index < len(labels)
                    else f"warmup_{index}"
                )
                pending_labels.append((0, label))
                query = _build_audio_query(
                    engine=engine,
                    text=text,
                    style_id=style_id,
                    tempo_dynamics_scale=args.tempo_dynamics_scale,
                )
                engine.synthesize_wave(
                    query, style_id, enable_interrogative_upspeak=True
                )
            for index, text in enumerate(texts):
                label = labels[index] if index < len(labels) else f"text_{index}"
                pending_labels.append((1, label))
                query = _build_audio_query(
                    engine=engine,
                    text=text,
                    style_id=style_id,
                    tempo_dynamics_scale=args.tempo_dynamics_scale,
                )
                engine.synthesize_wave(
                    query, style_id, enable_interrogative_upspeak=True
                )
            if pending_labels:
                raise RuntimeError(f"Unconsumed labels: {pending_labels}")
            if not captured:
                raise RuntimeError(f"No cases captured for model {model_uuid}.")
    finally:
        TTSModel.infer = original_infer

    _write_bundle(args.output, captured)
    for case in captured:
        print(
            f"{case.label}: role={case.role} tokens={case.tokens} "
            f"bert={case.bert.shape} bert_tokens={case.bert_input_ids.size} "
            f"sdp={case.sdp_ratio} "
            f"length={case.length_scale} noise={case.noise_scale}/{case.noise_w_scale}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
