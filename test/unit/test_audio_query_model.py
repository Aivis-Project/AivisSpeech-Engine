"""AudioQuery モデルのバリデーションテスト。"""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.model import AccentPhrase, Mora


def _generate_mora(text: str) -> Mora:
    """
    AudioQuery の検証用モーラを生成する。

    Args:
        text (str): `Mora.text` に指定する文字列

    Returns
    -------
    Mora
        音声合成用 AudioQuery に含めるモーラ
    """

    return Mora(
        text=text,
        consonant=None,
        consonant_length=None,
        vowel="a",
        vowel_length=0.0,
        pitch=0.0,
    )


def _generate_audio_query(
    *,
    moras: list[Mora] | None = None,
    accent: int = 1,
    pause_mora: Mora | None = None,
    speed_scale: float = 1.0,
    intonation_scale: float = 1.0,
    tempo_dynamics_scale: float = 1.0,
    pitch_scale: float = 0.0,
    volume_scale: float = 1.0,
    pre_phoneme_length: float = 0.1,
    post_phoneme_length: float = 0.1,
    pause_length: float | None = None,
    pause_length_scale: float = 1.0,
    output_sampling_rate: int = 44100,
) -> AudioQuery:
    """
    AudioQuery モデルの検証に使う最小構成のクエリを生成する。

    Args:
        moras (list[Mora] | None): `accent_phrases[0].moras` に指定するモーラ列
        accent (int): `accent_phrases[0].accent` に指定するアクセント位置
        pause_mora (Mora | None): `accent_phrases[0].pause_mora` に指定する無音モーラ
        speed_scale (float): `speedScale` に指定する話速
        intonation_scale (float): `intonationScale` に指定する感情表現の強さ
        tempo_dynamics_scale (float): `tempoDynamicsScale` に指定するテンポの緩急
        pitch_scale (float): `pitchScale` に指定する音高
        volume_scale (float): `volumeScale` に指定する音量
        pre_phoneme_length (float): `prePhonemeLength` に指定する前無音時間
        post_phoneme_length (float): `postPhonemeLength` に指定する後無音時間
        pause_length (float | None): `pauseLength` に指定する句読点などの無音時間
        pause_length_scale (float): `pauseLengthScale` に指定する無音時間の倍率
        output_sampling_rate (int): `outputSamplingRate` に指定する出力サンプリングレート

    Returns
    -------
    AudioQuery
        検証対象の音声合成クエリ
    """

    return AudioQuery(
        accent_phrases=[
            AccentPhrase(
                moras=moras if moras is not None else [_generate_mora("ア")],
                accent=accent,
                pause_mora=pause_mora,
                is_interrogative=False,
            )
        ],
        speedScale=speed_scale,
        intonationScale=intonation_scale,
        tempoDynamicsScale=tempo_dynamics_scale,
        pitchScale=pitch_scale,
        volumeScale=volume_scale,
        prePhonemeLength=pre_phoneme_length,
        postPhonemeLength=post_phoneme_length,
        pauseLength=pause_length,
        pauseLengthScale=pause_length_scale,
        outputSamplingRate=output_sampling_rate,
        outputStereo=False,
        kana="テスト",
    )


def test_audio_query_accepts_supported_mora_text() -> None:
    """SBV2 の音素変換で扱えるモーラ表記なら AudioQuery として受け付ける。"""

    query = _generate_audio_query(
        moras=[
            _generate_mora("テ"),
            _generate_mora(","),
            _generate_mora("😊"),
        ],
    )

    assert query.accent_phrases[0].moras[0].text == "テ"


def test_audio_query_rejects_unknown_mora_text() -> None:
    """SBV2 の音素表に存在しないモーラ表記は AudioQuery の生成時点で拒否する。"""

    with pytest.raises(ValidationError, match="moras\\[0\\]\\.text"):
        _generate_audio_query(moras=[_generate_mora("??")])


def test_audio_query_rejects_mojibake_mora_text() -> None:
    """文字化けしたモーラ表記は SBV2 の KeyError になる前に拒否する。"""

    with pytest.raises(ValidationError, match="moras\\[0\\]\\.text"):
        _generate_audio_query(moras=[_generate_mora("ã\x83\x86")])


def test_audio_query_rejects_empty_moras_in_accent_phrase() -> None:
    """空のアクセント句は音声合成時に解釈できないため拒否する。"""

    with pytest.raises(ValidationError, match="moras"):
        _generate_audio_query(moras=[])


@pytest.mark.parametrize("accent", [0, 2])
def test_audio_query_rejects_out_of_range_accent(accent: int) -> None:
    """アクセント位置がモーラ数の範囲外なら AudioQuery の生成時点で拒否する。"""

    with pytest.raises(ValidationError, match="accent"):
        _generate_audio_query(accent=accent)


def test_audio_query_allows_legacy_pause_mora_text() -> None:
    """pause_mora は音声合成時に読点へ置き換えるため、互換入力として検証対象から外す。"""

    query = _generate_audio_query(pause_mora=_generate_mora("、"))

    assert query.accent_phrases[0].pause_mora is not None


@pytest.mark.parametrize("speed_scale", [0.0, -1.0, math.nan])
def test_audio_query_rejects_invalid_speed_scale(speed_scale: float) -> None:
    """ゼロ除算や無限長の無音波形につながる話速は AudioQuery の生成時点で拒否する。"""

    with pytest.raises(ValidationError, match="speedScale"):
        _generate_audio_query(speed_scale=speed_scale)


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_error_field"),
    [
        ("intonation_scale", -0.1, "intonationScale"),
        ("intonation_scale", math.nan, "intonationScale"),
        ("intonation_scale", math.inf, "intonationScale"),
        ("tempo_dynamics_scale", -0.1, "tempoDynamicsScale"),
        ("tempo_dynamics_scale", math.nan, "tempoDynamicsScale"),
        ("tempo_dynamics_scale", math.inf, "tempoDynamicsScale"),
        ("volume_scale", -0.1, "volumeScale"),
        ("volume_scale", math.nan, "volumeScale"),
        ("volume_scale", math.inf, "volumeScale"),
        ("pre_phoneme_length", -0.1, "prePhonemeLength"),
        ("pre_phoneme_length", math.nan, "prePhonemeLength"),
        ("pre_phoneme_length", math.inf, "prePhonemeLength"),
        ("post_phoneme_length", -0.1, "postPhonemeLength"),
        ("post_phoneme_length", math.nan, "postPhonemeLength"),
        ("post_phoneme_length", math.inf, "postPhonemeLength"),
        ("pause_length", -0.1, "pauseLength"),
        ("pause_length", math.nan, "pauseLength"),
        ("pause_length", math.inf, "pauseLength"),
        ("pause_length_scale", -0.1, "pauseLengthScale"),
        ("pause_length_scale", math.nan, "pauseLengthScale"),
        ("pause_length_scale", math.inf, "pauseLengthScale"),
    ],
)
def test_audio_query_rejects_invalid_non_negative_finite_number(
    field_name: str,
    field_value: float,
    expected_error_field: str,
) -> None:
    """0以上の有限数が必要なフィールドは不正値を AudioQuery の生成時点で拒否する。"""

    kwargs: dict[str, Any] = {field_name: field_value}
    with pytest.raises(ValidationError, match=expected_error_field):
        _generate_audio_query(**kwargs)


@pytest.mark.parametrize("pitch_scale", [math.nan, math.inf, -math.inf])
def test_audio_query_rejects_non_finite_pitch_scale(pitch_scale: float) -> None:
    """負値を許容する pitchScale でも NaN や無限大は AudioQuery の生成時点で拒否する。"""

    with pytest.raises(ValidationError, match="pitchScale"):
        _generate_audio_query(pitch_scale=pitch_scale)


def test_audio_query_accepts_negative_pitch_scale() -> None:
    """pitchScale は低い声を指定するための負値を受け付ける。"""

    query = _generate_audio_query(pitch_scale=-1.0)

    assert query.pitchScale == -1.0


@pytest.mark.parametrize("output_sampling_rate", [0, -1])
def test_audio_query_rejects_invalid_output_sampling_rate(
    output_sampling_rate: int,
) -> None:
    """0以下の出力サンプリングレートは WAV 書き出しに渡る前に拒否する。"""

    with pytest.raises(ValidationError, match="outputSamplingRate"):
        _generate_audio_query(output_sampling_rate=output_sampling_rate)
