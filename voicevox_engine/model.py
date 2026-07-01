"""
API と ENGINE 内部実装が共有するモデル

このモジュールで定義されるモデル（データ構造）は API と ENGINE の 2 箇所から使われる。そのため
- モデルの変更は API 変更となるため慎重に検討する。
- モデルの docstring や Field は API スキーマとして使われるため、ユーザー向けに丁寧に書く。
- モデルクラスは FastAPI の制約から `BaseModel` を継承しなければならない。
"""

from math import isfinite
from pathlib import Path
from typing import Self

from aivmlib.schemas.aivm_manifest import AivmManifest
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from style_bert_vits2.nlp.japanese.mora_list import MORA_KATA_TO_MORA_PHONEMES
from style_bert_vits2.nlp.nanairo_emoji import is_nanairo_emoji_symbol
from style_bert_vits2.nlp.symbols import PUNCTUATIONS

from voicevox_engine.library.model import LibrarySpeaker
from voicevox_engine.tts_pipeline.model import AccentPhrase, Mora


def _is_supported_synthesis_mora_text(mora_text: str) -> bool:
    """
    音声合成時に Style-Bert-VITS2 へ渡せるモーラ表記かどうかを返す。

    Args:
        mora_text (str): `AudioQuery.accent_phrases[].moras[].text` の値

    Returns
    -------
    bool
        Style-Bert-VITS2 の音素変換処理に渡せる場合は True
    """

    # 音声合成では `Mora.text` だけを基準に SBV2 用の音素列を組み立てる
    ## `Mora.consonant` / `Mora.vowel` は VOICEVOX 互換の表現が入り得るため、SBV2 側の変換処理では採用していない
    return (
        mora_text in MORA_KATA_TO_MORA_PHONEMES
        or mora_text in PUNCTUATIONS
        or is_nanairo_emoji_symbol(mora_text) is True
    )


def _validate_audio_query_mora_text(
    mora: Mora,
    *,
    accent_phrase_index: int,
    mora_index: int,
) -> None:
    """
    音声合成用 `AudioQuery` のモーラ表記を検証する。

    Args:
        mora (Mora): 検証するモーラ
        accent_phrase_index (int): `AudioQuery.accent_phrases` 内の位置
        mora_index (int): `AccentPhrase.moras` 内の位置
    """

    # SBV2 の音素表に存在しない表記は、推論直前の `kata_tone2phone_tone()` で KeyError になる
    ## ここで 422 に変換される Pydantic エラーにしておくことで、外部連携から壊れた AudioQuery が届いても Sentry へ送られない
    if _is_supported_synthesis_mora_text(mora.text) is False:
        raise ValueError(
            f"accent_phrases[{accent_phrase_index}].moras[{mora_index}].text "
            f"に音声合成で利用できないモーラ表記が指定されています: {mora.text}"
        )


def _validate_non_negative_finite_number(value: float, field_name: str) -> None:
    """
    0 以上の有限数でなければならない `AudioQuery` フィールドを検証する。

    Args:
        value (float): 検証する値
        field_name (str): エラーメッセージに含めるフィールド名
    """

    _validate_finite_number(value, field_name)

    # 負数は NumPy の波形処理で ValueError や巨大な配列確保につながる
    if value < 0:
        raise ValueError(f"{field_name} には0以上の有限数を指定してください")


def _validate_finite_number(value: float, field_name: str) -> None:
    """
    有限数でなければならない `AudioQuery` フィールドを検証する。

    Args:
        value (float): 検証する値
        field_name (str): エラーメッセージに含めるフィールド名
    """

    if isfinite(value) is False:
        raise ValueError(f"{field_name} には有限数を指定してください")


class AudioQuery(BaseModel):
    """音声合成用のクエリ。"""

    accent_phrases: list[AccentPhrase] = Field(description="アクセント句のリスト。")
    speedScale: float = Field(
        description=(
            "全体の話速を 0.5 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "2.0 で 2 倍速、0.5 で 0.5 倍速になる。"
        ),
    )
    intonationScale: float = Field(
        description=(
            "選択した話者スタイルの感情表現の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "「全体の抑揚」ではない点で VOICEVOX ENGINE と仕様が異なる。\n"
            "数値が大きいほど、選択した話者スタイルに近い感情表現が込められた声になる。\n"
            "例えば話者スタイルが「上機嫌」なら、数値が大きいほどより嬉しそうな明るい話し方になる。\n"
            "一方で、話者やスタイルによっては、数値を上げすぎると発声がおかしくなったり、棒読みで不自然な声になる場合もある。\n"
            "正しく発声できる上限値は話者やスタイルごとに異なる。必要に応じて最適な値を見つけて調整すること。\n"
            "全スタイルの平均であるノーマルスタイルでは自動で適切な感情表現が選択されるため、この値を指定しても無視される。"
        ),
    )
    tempoDynamicsScale: float = Field(
        default=1.0,
        description=(
            "話す速さ（テンポ）の緩急の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "AivisSpeech Engine 固有のフィールドで、VOICEVOX ENGINE には存在しない。\n"
            "数値が大きいほどより早口で生っぽい抑揚がついた声になる。\n"
            "VOICEVOX ENGINE との互換性のため、未指定時はデフォルト値が適用される。"
        ),
    )
    pitchScale: float = Field(
        description=(
            "全体の音高を -0.15 ~ 0.15 の範囲で指定する (デフォルト: 0.0) 。\n"
            "数値が大きいほど高い声になる。\n"
            "VOICEVOX ENGINE と異なり、この値を 0.0 から変更すると音質が劣化するため注意が必要。"
        ),
    )
    volumeScale: float = Field(
        description=(
            "全体の音量を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "数値が大きいほど大きな声になる。"
        ),
    )
    prePhonemeLength: float = Field(description="音声の前の無音時間 (秒)。")
    postPhonemeLength: float = Field(description="音声の後の無音時間 (秒)。")
    pauseLength: float | None = Field(
        default=None,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間。null のときは無視される。デフォルト値は null 。",
    )
    pauseLengthScale: float = Field(
        default=1,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間（倍率）。デフォルト値は 1 。",
    )
    outputSamplingRate: int = Field(description="音声データの出力サンプリングレート。")
    outputStereo: bool = Field(description="音声データをステレオ出力するか否か。")
    kana: str | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "読み上げるテキストを指定する。「読みの AquesTalk 風記法テキスト」ではない点で VOICEVOX ENGINE と仕様が異なる。\n"
            "VOICEVOX ENGINE では AquesTalk 風記法テキストが入る読み取り専用フィールドだが (音声合成時には無視される) 、"
            "AivisSpeech Engine では音声合成時に漢字や記号が含まれた通常の読み上げテキストも必要なため、"
            "苦肉の策で読み上げテキスト指定用のフィールドとして転用した。\n"
            "VOICEVOX ENGINE との互換性のため None や空文字列が指定された場合も動作するが、"
            "その場合はアクセント句から自動生成されたひらがな文字列が読み上げテキストになるため、不自然なイントネーションになってしまう。\n"
            "可能な限り kana に通常の読み上げテキストを指定した上で音声合成 API に渡すことを推奨する。"
        ),
    )

    @model_validator(mode="after")
    def _validate_for_synthesis(self) -> Self:
        # Sentry で確認された壊れた AudioQuery は、型としては正しくても SBV2 の音素変換で落ちる
        ## `/synthesis` の入口で Pydantic エラーに変換し、クライアント起因の不正値を 422 として返す
        for accent_phrase_index, accent_phrase in enumerate(self.accent_phrases):
            # 空のアクセント句はアクセント位置もモーラ表記も解釈できないため不正な入力として扱う
            if len(accent_phrase.moras) == 0:
                raise ValueError(
                    f"accent_phrases[{accent_phrase_index}].moras は1つ以上指定してください"
                )

            # `accent` は 1-indexed のアクセント核位置として synthesize_wave() で扱われる
            ## 0 やモーラ数を超える値を通すと、全モーラが低音になるなど入力意図と異なる音高列が作られる
            if not 1 <= accent_phrase.accent <= len(accent_phrase.moras):
                raise ValueError(
                    f"accent_phrases[{accent_phrase_index}].accent には "
                    f"1 以上 {len(accent_phrase.moras)} 以下の値を指定してください"
                )

            # `pause_mora` は音声合成時に存在有無だけを見て固定の読点に置き換えるため、ここでは moras のみ検証する
            for mora_index, mora in enumerate(accent_phrase.moras):
                _validate_audio_query_mora_text(
                    mora,
                    accent_phrase_index=accent_phrase_index,
                    mora_index=mora_index,
                )

        # 話速 0 は後処理の無音追加でゼロ除算になり、負数は無音波形の長さが負になる
        if isfinite(self.speedScale) is False or self.speedScale <= 0:
            raise ValueError("speedScale には0より大きい有限数を指定してください")

        _validate_non_negative_finite_number(
            self.intonationScale,
            "intonationScale",
        )
        _validate_non_negative_finite_number(
            self.tempoDynamicsScale,
            "tempoDynamicsScale",
        )
        # pitchScale は負値も仕様上許可されるため、NaN / Inf だけを拒否する
        _validate_finite_number(self.pitchScale, "pitchScale")
        _validate_non_negative_finite_number(self.volumeScale, "volumeScale")
        _validate_non_negative_finite_number(
            self.prePhonemeLength,
            "prePhonemeLength",
        )
        _validate_non_negative_finite_number(
            self.postPhonemeLength,
            "postPhonemeLength",
        )
        _validate_non_negative_finite_number(self.pauseLengthScale, "pauseLengthScale")

        # pauseLength は AivisSpeech Engine では無視されるが、互換 API の入力として受けるため値の妥当性だけ確認する
        if self.pauseLength is not None:
            _validate_non_negative_finite_number(self.pauseLength, "pauseLength")

        # resample() と WAV 書き出しに渡る値なので、0 や負数は API 入力時点で止める
        if self.outputSamplingRate <= 0:
            raise ValueError("outputSamplingRate には0より大きい値を指定してください")

        return self

    def __hash__(self) -> int:
        """内容に対して一意なハッシュ値を返す。"""
        # NOTE: lru_cache がユースケースのひとつ
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class AivmInfo(BaseModel):
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルのメタデータ情報。

    AIVM マニフェストには、音声合成モデルに関連する全てのメタデータが含まれる。
    speakers フィールド内の話者情報は、VOICEVOX ENGINE との API 互換性のために、
    AIVM マニフェストを基に Speaker / SpeakerStyle / SpeakerInfo / StyleInfo モデルに変換したもの。
    """

    is_loaded: bool = Field(description="この音声合成モデルがロードされているかどうか")
    is_update_available: bool = Field(
        description="この音声合成モデルの新しいバージョンが AivisHub で公開されているかどうか"
    )
    is_private_model: bool = Field(
        description="AivisHub で公開されておらず、ユーザーがローカルからインストールしたモデルの場合は True (ネットワークエラーなどで AivisHub から情報を取得できなかった場合も True を返す)",
    )
    is_default_model: bool = Field(
        description="AivisHub がデフォルトインストール対象として指定した音声合成モデルかどうか",
        default=False,  # 旧バージョンで生成されたモデル情報キャッシュの読み込みエラー回避のためデフォルト値を指定
    )
    latest_version: str = Field(
        description="この音声合成モデルの AivisHub で公開されている最新バージョン (AivisHub で公開されていない場合は AIVM マニフェスト記載のバージョン)"
    )
    file_path: Path = Field(description="AIVMX ファイルのインストール先パス")
    file_size: int = Field(
        description="AIVMX ファイルのインストールサイズ (バイト単位)"
    )
    manifest: AivmManifest = Field(description="AIVM マニフェスト")
    speakers: list[LibrarySpeaker] = Field(
        description="話者情報のリスト (VOICEVOX ENGINE 互換)"
    )
