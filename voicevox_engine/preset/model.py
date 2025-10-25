"""
プリセット機能に関して API と ENGINE 内部実装が共有するモデル（データ構造）

モデルの注意点は `voicevox_engine/model.py` の module docstring を確認すること。
"""

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.metas.Metas import StyleId


class Preset(BaseModel):
    """プリセット情報。"""

    id: int = Field(description="プリセット ID 。")
    name: str = Field(description="プリセット名。")
    speaker_uuid: str = Field(description="キャラクターの UUID 。")
    style_id: StyleId = Field(description="スタイル ID 。")
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
    pauseLength: float | SkipJsonSchema[None] = Field(
        default=None,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間。null のときは無視される。デフォルト値は null 。",
    )
    pauseLengthScale: float = Field(
        default=1,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間（倍率）。デフォルト値は 1 。",
    )
