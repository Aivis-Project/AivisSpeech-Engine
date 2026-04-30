"""StyleBertVITS2TTSEngine のテスト。"""

import threading
import uuid
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

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.model import AccentPhrase, Mora
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
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
