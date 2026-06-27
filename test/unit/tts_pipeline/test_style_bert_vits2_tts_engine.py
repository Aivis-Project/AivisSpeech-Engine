"""StyleBertVITS2TTSEngine のテスト。"""

import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    ModelArchitecture,
    ModelFormat,
)
from numpy.typing import NDArray
from style_bert_vits2.constants import DEFAULT_SDP_RATIO, DEFAULT_STYLE_WEIGHT

import voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine as style_bert_vits2_tts_engine
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.core.core_adapter import DeviceSupport
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.model import AccentPhrase, Mora
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    OnnxPluginExecutionProviderConfig,
    StyleBertVITS2TTSEngine,
    _configure_onnx_plugin_execution_provider,
    _onnx_plugin_inference_session_scope,
    _select_onnx_providers,
)


class _RecordingTTSModel:
    """推論直前の引数を記録する TTSModel 互換オブジェクト。"""

    def __init__(self) -> None:
        self.hyper_parameters = SimpleNamespace(
            data=SimpleNamespace(style2id={"ノーマル": 0})
        )
        self.infer_kwargs: dict[str, Any] | None = None

    def infer(self, **kwargs: Any) -> tuple[int, NDArray[np.int16]]:
        """StyleBertVITS2TTSEngine から渡された推論引数を記録する。"""

        self.infer_kwargs = kwargs
        return 44100, np.full(100, 32767, dtype=np.int16)


class _StaticAivmManager:
    """固定の AIVM manifest を返す AivmManager 互換オブジェクト。"""

    def __init__(self) -> None:
        self.aivm_manifest_speaker_style = AivmManifestSpeakerStyle(
            name="ノーマル",
            local_id=0,
            voice_samples=[],
        )
        self.aivm_manifest_speaker = AivmManifestSpeaker(
            name="テスト話者",
            icon="data:image/png;base64,AA==",
            supported_languages=["ja"],
            uuid=uuid.UUID("00000000-0000-4000-8000-000000000101"),
            local_id=0,
            styles=[self.aivm_manifest_speaker_style],
        )
        self.aivm_manifest = AivmManifest(
            manifest_version="1.0",
            name="テストモデル",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=ModelFormat.ONNX,
            uuid=uuid.UUID("00000000-0000-4000-8000-000000000102"),
            version="1.0.0",
            speakers=[self.aivm_manifest_speaker],
        )

    def get_aivm_manifest_from_style_id(
        self,
        style_id: StyleId,
    ) -> tuple[AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle]:
        """任意の style ID に対し、固定の AIVM manifest 情報を返す。"""

        return (
            self.aivm_manifest,
            self.aivm_manifest_speaker,
            self.aivm_manifest_speaker_style,
        )


class _StyleBertVITS2TTSEngineForTest(StyleBertVITS2TTSEngine):
    """推論本体だけを差し替えて前処理を検査する StyleBertVITS2TTSEngine。"""

    def __init__(self, recording_tts_model: _RecordingTTSModel) -> None:
        self.aivm_manager = cast(AivmManager, _StaticAivmManager())
        object.__setattr__(self, "_inference_lock", threading.Lock())
        self.recording_tts_model = recording_tts_model

    def load_model(self, aivm_model_uuid: str) -> Any:
        """記録用 TTSModel 互換オブジェクトを返す。"""

        return self.recording_tts_model


def _generate_style_bert_vits2_tts_engine(
    recording_tts_model: _RecordingTTSModel,
) -> StyleBertVITS2TTSEngine:
    """
    推論本体だけを記録用モデルに差し替えた StyleBertVITS2TTSEngine を生成する。

    Parameters
    ----------
    recording_tts_model : _RecordingTTSModel
        `load_model()` から返す記録用 TTSModel 互換オブジェクト。

    Returns
    -------
    StyleBertVITS2TTSEngine
        `synthesize_wave()` の前処理を直接検査できる TTS エンジン。
    """

    return _StyleBertVITS2TTSEngineForTest(recording_tts_model)


def _generate_audio_query(
    *,
    kana: str | None,
    tempo_dynamics_scale: float = 1.0,
    intonation_scale: float = 1.0,
    pitch_scale: float = 0.0,
) -> AudioQuery:
    """
    StyleBertVITS2TTSEngine の前処理テストで使う AudioQuery を生成する。

    Parameters
    ----------
    kana : str | None
        AudioQuery の `kana` に指定する読み上げテキスト。
    tempo_dynamics_scale : float
        AudioQuery の `tempoDynamicsScale` に指定する値。
    intonation_scale : float
        AudioQuery の `intonationScale` に指定する値。
    pitch_scale : float
        AudioQuery の `pitchScale` に指定する値。

    Returns
    -------
    AudioQuery
        推論直前パラメータの検査に使う AudioQuery。
    """

    return AudioQuery(
        accent_phrases=[
            AccentPhrase(
                moras=[
                    Mora(
                        text="テ",
                        consonant="t",
                        consonant_length=0.0,
                        vowel="e",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ス",
                        consonant="s",
                        consonant_length=0.0,
                        vowel="U",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ト",
                        consonant="t",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=1,
                pause_mora=None,
                is_interrogative=False,
            )
        ],
        speedScale=1.0,
        intonationScale=intonation_scale,
        tempoDynamicsScale=tempo_dynamics_scale,
        pitchScale=pitch_scale,
        volumeScale=1.0,
        prePhonemeLength=0.0,
        postPhonemeLength=0.0,
        pauseLength=None,
        pauseLengthScale=1.0,
        outputSamplingRate=44100,
        outputStereo=False,
        kana=kana,
    )


def _synthesize_and_get_infer_kwargs(
    query: AudioQuery,
) -> dict[str, Any]:
    """
    `synthesize_wave()` を実行し、記録用 TTSModel に渡された推論引数を取得する。

    Parameters
    ----------
    query : AudioQuery
        StyleBertVITS2TTSEngine に渡す AudioQuery。

    Returns
    -------
    dict[str, Any]
        `TTSModel.infer()` に渡された推論引数。
    """

    recording_tts_model = _RecordingTTSModel()
    engine = _generate_style_bert_vits2_tts_engine(recording_tts_model)
    engine.synthesize_wave(query, StyleId(0), enable_interrogative_upspeak=True)

    assert recording_tts_model.infer_kwargs is not None
    return recording_tts_model.infer_kwargs


def test_select_onnx_providers_can_force_directml() -> None:
    """Explicit DirectML selection uses DmlExecutionProvider before CPU fallback."""

    providers = _select_onnx_providers(
        use_gpu=True,
        available_onnx_providers=["CPUExecutionProvider", "DmlExecutionProvider"],
        preferred_onnx_provider="directml",
    )

    assert providers == [
        ("DmlExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_select_onnx_providers_can_force_cuda_without_directml_fallback() -> None:
    """Explicit CUDA selection does not silently switch to DirectML."""

    providers = _select_onnx_providers(
        use_gpu=True,
        available_onnx_providers=[
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
        ],
        preferred_onnx_provider="cuda",
    )

    assert providers == [
        (
            "CUDAExecutionProvider",
            {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_select_onnx_providers_keeps_auto_cuda_directml_fallback() -> None:
    """Auto GPU selection keeps the existing CUDA to DirectML fallback chain."""

    providers = _select_onnx_providers(
        use_gpu=True,
        available_onnx_providers=[
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
        ],
        preferred_onnx_provider=None,
    )

    assert providers == [
        (
            "CUDAExecutionProvider",
            {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ),
        ("DmlExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_configure_onnx_plugin_execution_provider_registers_and_prepends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured Plugin EP is registered and tried before ordinary providers."""

    available_providers = ["CPUExecutionProvider"]
    register_calls: list[tuple[str, str]] = []

    def fake_register_execution_provider_library(
        registration_name: str,
        library_path: str,
    ) -> None:
        register_calls.append((registration_name, library_path))

    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "register_execution_provider_library",
        fake_register_execution_provider_library,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_available_providers",
        lambda: list(available_providers),
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_ep_devices",
        lambda: [SimpleNamespace(ep_name="AivisGgmlExecutionProvider")],
    )

    providers = _configure_onnx_plugin_execution_provider(
        base_providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
        config=OnnxPluginExecutionProviderConfig(
            provider_name="AivisGgmlExecutionProvider",
            provider_options={"backend": "vulkan", "device": "0"},
            library_path=tmp_path / "libaivis_ggml_ep.so",
            registration_name="aivis-ggml",
            strict=True,
        ),
    )

    assert register_calls == [("aivis-ggml", str(tmp_path / "libaivis_ggml_ep.so"))]
    assert providers[0] == (
        "AivisGgmlExecutionProvider",
        {"backend": "vulkan", "device": "0"},
    )
    assert providers[1:] == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_configure_onnx_plugin_execution_provider_raises_when_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict Plugin EP setup fails startup instead of silently falling back."""

    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "register_execution_provider_library",
        lambda _registration_name, _library_path: None,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_ep_devices",
        lambda: [],
    )

    with pytest.raises(RuntimeError) as exc_info:
        _configure_onnx_plugin_execution_provider(
            base_providers=["CPUExecutionProvider"],
            config=OnnxPluginExecutionProviderConfig(
                provider_name="AivisGgmlExecutionProvider",
                provider_options={},
                library_path=tmp_path / "libaivis_ggml_ep.so",
                strict=True,
            ),
        )

    assert "AivisGgmlExecutionProvider" in str(exc_info.value)
    assert "Available providers" in str(exc_info.value)


def test_onnx_plugin_inference_session_scope_uses_ep_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugin EP sessions use OrtEpDevice when available."""

    calls: list[tuple[str, Any]] = []

    class FakeSessionOptions:
        def add_provider_for_devices(
            self,
            ep_devices: Sequence[Any],
            provider_options: dict[str, str],
        ) -> None:
            calls.append(("add_provider_for_devices", (ep_devices, provider_options)))

    def fake_inference_session(
        path_or_bytes: str,
        *,
        sess_options: Any | None = None,
        providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **_kwargs: Any,
    ) -> str:
        calls.append(
            (
                "InferenceSession",
                {
                    "path_or_bytes": path_or_bytes,
                    "sess_options": sess_options,
                    "providers": providers,
                    "provider_options": provider_options,
                },
            )
        )
        return "session"

    ep_device = SimpleNamespace(ep_name="AivisGgmlExecutionProvider")
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_ep_devices",
        lambda: [ep_device],
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "SessionOptions",
        FakeSessionOptions,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "InferenceSession",
        fake_inference_session,
    )

    config = OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options={"backend": "vulkan"},
    )
    providers = [
        ("AivisGgmlExecutionProvider", {"backend": "cpu", "n_threads": "4"}),
        "CPUExecutionProvider",
    ]

    with _onnx_plugin_inference_session_scope(config):
        session = style_bert_vits2_tts_engine.onnxruntime.InferenceSession(
            "model.onnx",
            providers=providers,
        )

    assert session == "session"
    assert calls[0] == (
        "add_provider_for_devices",
        ([ep_device], {"backend": "cpu", "n_threads": "4"}),
    )
    assert calls[1][0] == "InferenceSession"
    assert calls[1][1]["providers"] is None


def test_validate_strict_session_provider_rejects_silent_cpu_fallback() -> None:
    """Strict provider mode fails if ONNX Runtime creates a CPU session."""

    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="AivisGgmlExecutionProvider"):
        StyleBertVITS2TTSEngine._validate_strict_session_provider(  # noqa: SLF001
            session=session,
            required_provider_name="AivisGgmlExecutionProvider",
            context="test session",
        )


def test_prepare_onnx_plugin_jp_bert_provider_options_fills_cache_path(
    tmp_path: Path,
) -> None:
    """JP-BERT GGUF path is prepared before the global BERT ONNX session opens."""

    engine = cast(StyleBertVITS2TTSEngine, object.__new__(StyleBertVITS2TTSEngine))
    engine.onnx_providers = [
        (
            "AivisGgmlExecutionProvider",
            {
                "backend": "vulkan",
                "claim_jp_bert_graph": "1",
                "claim_synthesis_graph": "1",
            },
        ),
        "CPUExecutionProvider",
    ]
    jp_bert_onnx_path = tmp_path / "model_fp16.onnx"
    engine._resolve_jp_bert_onnx_path = lambda: jp_bert_onnx_path  # noqa: SLF001
    jp_bert_gguf_path = tmp_path / "jp-bert.gguf"

    class FakeJpBertGgufCache:
        def ensure(self, *, onnx_path: Path) -> Any:
            assert onnx_path == jp_bert_onnx_path
            return SimpleNamespace(gguf_path=jp_bert_gguf_path)

    config = OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options={
            "backend": "vulkan",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "1",
        },
        strict=True,
    )

    engine._prepare_onnx_plugin_jp_bert_provider_options(  # noqa: SLF001
        config=config,
        jp_bert_gguf_cache=cast(Any, FakeJpBertGgufCache()),
    )

    assert config.provider_options["jp_bert_gguf_path"] == str(jp_bert_gguf_path)
    assert config.provider_options["claim_synthesis_graph"] == "1"
    assert engine.onnx_providers[0] == (
        "AivisGgmlExecutionProvider",
        {
            "backend": "vulkan",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "0",
            "jp_bert_gguf_path": str(jp_bert_gguf_path),
        },
    )


def test_model_specific_onnx_providers_fills_synthesis_gguf_path(
    tmp_path: Path,
) -> None:
    """Synthesis GGUF is prepared per installed AIVMX model before session load."""

    engine = cast(StyleBertVITS2TTSEngine, object.__new__(StyleBertVITS2TTSEngine))
    config = OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options={
            "backend": "vulkan",
            "claim_jp_bert_graph": "0",
            "claim_synthesis_graph": "1",
        },
        strict=True,
    )
    engine._onnx_plugin_ep = config  # noqa: SLF001
    engine.onnx_providers = [
        ("AivisGgmlExecutionProvider", dict(config.provider_options)),
        "CPUExecutionProvider",
    ]
    gguf_path = tmp_path / "model.gguf"

    class FakeAivmGgufCache:
        def ensure(self, *, aivm_file_path: Path, aivm_metadata: Any) -> Any:
            assert aivm_file_path == tmp_path / "model.aivmx"
            assert aivm_metadata is not None
            return SimpleNamespace(gguf_path=gguf_path)

    engine._onnx_plugin_gguf_cache = cast(Any, FakeAivmGgufCache())  # noqa: SLF001
    engine._onnx_plugin_jp_bert_gguf_cache = None  # noqa: SLF001

    providers = engine._model_specific_onnx_providers(  # noqa: SLF001
        onnx_source_path=tmp_path / "model.aivmx",
        aivm_metadata=cast(Any, object()),
    )

    assert providers[0] == (
        "AivisGgmlExecutionProvider",
        {
            "backend": "vulkan",
            "claim_jp_bert_graph": "0",
            "claim_synthesis_graph": "1",
            "gguf_path": str(gguf_path),
        },
    )


def test_supported_devices_reports_onnx_plugin_ep_as_gpu_capable() -> None:
    engine = cast(StyleBertVITS2TTSEngine, object.__new__(StyleBertVITS2TTSEngine))
    config = OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options={
            "backend": "cpu",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "1",
        },
        strict=True,
    )
    engine._onnx_plugin_ep = config  # noqa: SLF001
    engine.onnx_providers = [
        ("AivisGgmlExecutionProvider", dict(config.provider_options)),
        "CPUExecutionProvider",
    ]
    engine.available_onnx_providers = ["CPUExecutionProvider"]

    assert engine.supported_devices == DeviceSupport(cpu=True, cuda=False, dml=True)


def test_supported_devices_ignores_unselected_onnx_plugin_ep() -> None:
    engine = cast(StyleBertVITS2TTSEngine, object.__new__(StyleBertVITS2TTSEngine))
    engine._onnx_plugin_ep = OnnxPluginExecutionProviderConfig(
        provider_name="AivisGgmlExecutionProvider",
        provider_options={
            "backend": "cpu",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "1",
        },
        strict=False,
    )
    engine.onnx_providers = ["CPUExecutionProvider"]
    engine.available_onnx_providers = ["CPUExecutionProvider"]

    assert engine.supported_devices == DeviceSupport(cpu=True, cuda=False, dml=False)


def test_synthesize_wave_uses_trimmed_kana_as_inference_text() -> None:
    """AudioQuery.kana に通常テキストがある場合、前後空白を削除した値が推論テキストになることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(kana="  今日はテストです  ")
    )

    assert infer_kwargs["text"] == "今日はテストです"


@pytest.mark.parametrize("kana", [None, ""])
def test_synthesize_wave_falls_back_to_accent_phrases_when_kana_is_empty(
    kana: str | None,
) -> None:
    """AudioQuery.kana が None または空文字列の場合、アクセント句のモーラ列から推論テキストを生成することを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(_generate_audio_query(kana=kana))

    assert infer_kwargs["text"] == "てすと"


@pytest.mark.parametrize(
    ("tempo_dynamics_scale", "expected_sdp_ratio"),
    [
        (0.0, 0.0),
        (1.0, DEFAULT_SDP_RATIO),
        (2.0, 1.0),
        (-0.1, DEFAULT_SDP_RATIO),
        (2.1, DEFAULT_SDP_RATIO),
    ],
)
def test_synthesize_wave_converts_tempo_dynamics_scale_to_sdp_ratio(
    tempo_dynamics_scale: float,
    expected_sdp_ratio: float,
) -> None:
    """tempoDynamicsScale の境界値と範囲外値が、Style-Bert-VITS2 の sdp_ratio に変換されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            tempo_dynamics_scale=tempo_dynamics_scale,
        )
    )

    assert infer_kwargs["sdp_ratio"] == pytest.approx(expected_sdp_ratio)


@pytest.mark.parametrize(
    ("intonation_scale", "expected_style_weight"),
    [
        (0.0, 0.0),
        (1.0, DEFAULT_STYLE_WEIGHT),
        (2.0, 10.0),
        (-0.1, DEFAULT_STYLE_WEIGHT),
        (2.1, DEFAULT_STYLE_WEIGHT),
    ],
)
def test_synthesize_wave_converts_intonation_scale_to_style_weight(
    intonation_scale: float,
    expected_style_weight: float,
) -> None:
    """intonationScale の境界値と範囲外値が、Style-Bert-VITS2 の style_weight に変換されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            intonation_scale=intonation_scale,
        )
    )

    assert infer_kwargs["style_weight"] == pytest.approx(expected_style_weight)


@pytest.mark.parametrize(
    ("pitch_scale", "expected_pitch_scale"),
    [
        (-1.5, 0.0),
        (-0.5, 0.5),
        (0.0, 1.0),
        (0.5, 1.5),
    ],
)
def test_synthesize_wave_converts_pitch_scale(
    pitch_scale: float,
    expected_pitch_scale: float,
) -> None:
    """pitchScale が Style-Bert-VITS2 の pitch_scale に変換され、負側は 0.0 で下限固定されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            pitch_scale=pitch_scale,
        )
    )

    assert infer_kwargs["pitch_scale"] == pytest.approx(expected_pitch_scale)
