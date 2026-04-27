"""キャラクター情報とキャラクターメタ情報"""

from typing import Literal, NewType

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

# NOTE: 循環importを防ぐためにとりあえずここに書いている
# FIXME: 他のmodelに依存せず、全modelから参照できる場所に配置する
StyleId = NewType("StyleId", int)
StyleType = Literal["talk", "singing_teacher", "frame_decode", "sing"]


class SpeakerStyle(BaseModel):
    """キャラクターのスタイル情報"""

    name: str = Field(description="スタイル名")
    id: StyleId = Field(description="スタイル ID")
    type: StyleType = Field(
        default="talk",
        description=(
            "スタイルの種類。"
            "talk: 音声合成クエリの作成と音声合成が可能。"
            "singing_teacher: 歌唱音声合成用のクエリの作成が可能。"
            "frame_decode: 歌唱音声合成が可能。"
            "sing: 歌唱音声合成用のクエリの作成と歌唱音声合成が可能。"
        ),
    )


class SpeakerSupportedFeatures(BaseModel):
    """キャラクターの対応機能の情報"""

    permitted_synthesis_morphing: Literal["ALL", "SELF_ONLY", "NOTHING"] = Field(
        description=(
            "モーフィング機能への対応。"
            "'ALL' は「全て許可」、'SELF_ONLY' は「同じキャラクター内でのみ許可」、'NOTHING' は「全て禁止」"
        ),
        default="ALL",
    )


class Speaker(BaseModel):
    """キャラクター情報"""

    name: str = Field(description="名前")
    speaker_uuid: str = Field(description="キャラクターの UUID")
    styles: list[SpeakerStyle] = Field(description="スタイルの一覧")
    version: str = Field(description="キャラクターのバージョン")
    supported_features: SpeakerSupportedFeatures = Field(
        description="キャラクターの対応機能", default_factory=SpeakerSupportedFeatures
    )


class StyleInfo(BaseModel):
    """スタイルの追加情報"""

    id: StyleId = Field(description="スタイル ID")
    icon: str = Field(
        description="このスタイルのアイコンを base64 エンコードしたもの、あるいは URL"
    )
    portrait: str | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "AivisSpeech Engine では常に None を返す "
            "(「このスタイルの立ち絵画像を base64 エンコードしたもの」ではない点で VOICEVOX ENGINE と異なる)"
        ),
    )
    voice_samples: list[str] = Field(
        description="ボイスサンプルの音声データを base64 エンコードしたもの、あるいは URL"
    )
    voice_sample_transcripts: list[str] = Field(
        default=[],
        description="ボイスサンプルの書き起こしテキスト (voice_samples の配列インデックスと対応し、存在しない場合は空文字列)",
    )


class SpeakerInfo(BaseModel):
    """キャラクターの追加情報"""

    policy: str = Field(description="policy.md")
    portrait: str = Field(
        description=(
            "アイコンを base64 エンコードしたもの、あるいは URL "
            "(「立ち絵画像を base64 エンコードしたもの」ではない点で VOICEVOX ENGINE と異なる)"
        ),
    )
    style_infos: list[StyleInfo] = Field(description="スタイルの追加情報")
