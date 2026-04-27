"""音声合成機能を提供する API Router"""

import io
import zipfile
from traceback import print_exception
from typing import Annotated, Self

import soundfile
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.core.core_adapter import DeviceSupport
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.preset.preset_manager import (
    PresetInputError,
    PresetInternalError,
    PresetManager,
)
from voicevox_engine.tts_pipeline.connect_base64_waves import (
    ConnectBase64WavesException,
    connect_base64_waves,
)
from voicevox_engine.tts_pipeline.kana_converter import ParseKanaError, parse_kana
from voicevox_engine.tts_pipeline.model import (
    AccentPhrase,
    FrameAudioQuery,
    ParseKanaErrorCode,
    Score,
)
from voicevox_engine.tts_pipeline.song_engine import SongEngineManager
from voicevox_engine.tts_pipeline.tts_engine import LATEST_VERSION, TTSEngineManager


class ParseKanaBadRequest(BaseModel):
    """読み仮名のパースに失敗した。"""

    text: str = Field(description="エラーメッセージ")
    error_name: str = Field(
        description="エラー名\n\n"
        "|name|description|\n|---|---|\n"
        + "\n".join([f"| {e.name} | {e.value} |" for e in list(ParseKanaErrorCode)]),
    )
    error_args: dict[str, str] = Field(description="エラーを起こした箇所")

    def __init__(self, e: ParseKanaError):
        super().__init__(text=e.text, error_name=e.errname, error_args=e.kwargs)


class SupportedDevicesInfo(BaseModel):
    """対応しているデバイスの情報。"""

    cpu: bool = Field(description="CPU に対応しているか")
    cuda: bool = Field(description="CUDA (NVIDIA GPU) に対応しているか")
    dml: bool = Field(
        description="DirectML (NVIDIA GPU/Radeon GPU 等) に対応しているか"
    )

    @classmethod
    def generate_from(cls, device_support: DeviceSupport) -> Self:
        """`DeviceSupport` インスタンスからこのインスタンスを生成する。"""
        return cls(
            cpu=device_support.cpu,
            cuda=device_support.cuda,
            dml=device_support.dml,
        )


def generate_tts_pipeline_router(
    tts_engines: TTSEngineManager,
    song_engines: SongEngineManager,
    preset_manager: PresetManager,
    cancellable_engine: CancellableEngine | None,
) -> APIRouter:
    """音声合成 API Router を生成する"""
    router = APIRouter()

    @router.post(
        "/audio_query",
        tags=["クエリ作成"],
        summary="音声合成用のクエリを作成する",
    )
    def audio_query(
        text: str,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        enable_katakana_english: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> AudioQuery:
        """
        音声合成用のクエリの初期値を得ます。ここで得られたクエリはそのまま音声合成に利用できます。<br>
        各値の意味は `Schemas` を参照してください。

        AivisSpeech Engine では英単語は常に自動的に自然なカタカナ読みに変換されて読み上げられる仕様のため、enable_katakana_english は指定不要です（常に無視されます）。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        accent_phrases = engine.create_accent_phrases(
            text, style_id, enable_katakana_english
        )
        return AudioQuery(
            accent_phrases=accent_phrases,
            speedScale=1.0,
            intonationScale=1.0,
            tempoDynamicsScale=1.0,
            pitchScale=0.0,
            volumeScale=1.0,
            prePhonemeLength=0.1,
            postPhonemeLength=0.1,
            pauseLength=None,
            pauseLengthScale=1,
            outputSamplingRate=engine.default_sampling_rate,
            outputStereo=False,
            # kana=create_kana(accent_phrases),
            kana=text,  # AivisSpeech Engine では音声合成時に読み上げテキストも必要なため、kana に読み上げテキストをそのまま入れて返す
        )

    @router.post(
        "/audio_query_from_preset",
        tags=["クエリ作成"],
        summary="音声合成用のクエリをプリセットを用いて作成する",
    )
    def audio_query_from_preset(
        text: str,
        preset_id: int,
        enable_katakana_english: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> AudioQuery:
        """
        音声合成用のクエリの初期値を得ます。ここで得られたクエリはそのまま音声合成に利用できます。<br>
        各値の意味は `Schemas` を参照してください。

        AivisSpeech Engine では英単語は常に自動的に自然なカタカナ読みに変換されて読み上げられる仕様のため、enable_katakana_english は指定不要です（常に無視されます）。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        try:
            presets = preset_manager.load_presets()
        except PresetInputError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except PresetInternalError as e:
            print_exception(e)
            raise HTTPException(status_code=500) from e
        for preset in presets:
            if preset.id == preset_id:
                selected_preset = preset
                break
        else:
            raise HTTPException(
                status_code=422, detail="該当するプリセットIDが見つかりません"
            )

        accent_phrases = engine.create_accent_phrases(
            text, selected_preset.style_id, enable_katakana_english
        )
        return AudioQuery(
            accent_phrases=accent_phrases,
            speedScale=selected_preset.speedScale,
            intonationScale=selected_preset.intonationScale,
            tempoDynamicsScale=selected_preset.tempoDynamicsScale,
            pitchScale=selected_preset.pitchScale,
            volumeScale=selected_preset.volumeScale,
            prePhonemeLength=selected_preset.prePhonemeLength,
            postPhonemeLength=selected_preset.postPhonemeLength,
            pauseLength=selected_preset.pauseLength,
            pauseLengthScale=selected_preset.pauseLengthScale,
            outputSamplingRate=engine.default_sampling_rate,
            outputStereo=False,
            # kana=create_kana(accent_phrases),
            kana=text,  # AivisSpeech Engine では音声合成時に読み上げテキストも必要なため、kana に読み上げテキストをそのまま入れて返す
        )

    @router.post(
        "/accent_phrases",
        tags=["クエリ編集"],
        summary="テキストからアクセント句を得る",
        responses={
            400: {
                "description": "読み仮名のパースに失敗",
                "model": ParseKanaBadRequest,
            }
        },
    )
    def accent_phrases(
        text: str,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        is_kana: bool = False,
        enable_katakana_english: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[AccentPhrase]:
        """
        テキストからアクセント句を得ます。<br>
        is_kanaが`true`のとき、テキストは次の AquesTalk 風記法で解釈されます。デフォルトは`false`です。
        * 全てのカナはカタカナで記述される
        * アクセント句は`/`または`、`で区切る。`、`で区切った場合に限り無音区間が挿入される。
        * カナの手前に`_`を入れるとそのカナは無声化される
        * アクセント位置を`'`で指定する。全てのアクセント句にはアクセント位置を1つ指定する必要がある。
        * アクセント句末に`？`(全角)を入れることにより疑問文の発音ができる。

        AivisSpeech Engine では英単語は常に自動的に自然なカタカナ読みに変換されて読み上げられる仕様のため、enable_katakana_english は指定不要です（常に無視されます）。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        if is_kana:
            try:
                return engine.create_accent_phrases_from_kana(text, style_id)
            except ParseKanaError as e:
                raise HTTPException(
                    status_code=400, detail=ParseKanaBadRequest(e).model_dump()
                ) from e
        else:
            return engine.create_accent_phrases(text, style_id, enable_katakana_english)

    @router.post(
        "/mora_data",
        tags=["クエリ編集"],
        summary="アクセント句から音高・音素長を得る (AivisSpeech Engine では常にダミーの値が返されます)",
    )
    def mora_data(
        accent_phrases: list[AccentPhrase],
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[AccentPhrase]:
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        return engine.update_length_and_pitch(accent_phrases, style_id)

    @router.post(
        "/mora_length",
        tags=["クエリ編集"],
        summary="アクセント句から音素長を得る (AivisSpeech Engine では常にダミーの値が返されます)",
    )
    def mora_length(
        accent_phrases: list[AccentPhrase],
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[AccentPhrase]:
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        return engine.update_length(accent_phrases, style_id)

    @router.post(
        "/mora_pitch",
        tags=["クエリ編集"],
        summary="アクセント句から音高を得る (AivisSpeech Engine では常にダミーの値が返されます)",
    )
    def mora_pitch(
        accent_phrases: list[AccentPhrase],
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[AccentPhrase]:
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        return engine.update_pitch(accent_phrases, style_id)

    @router.post(
        "/synthesis",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="音声合成する",
    )
    def synthesis(
        query: AudioQuery,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        enable_interrogative_upspeak: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> Response:
        """
        指定されたスタイル ID に紐づく音声合成モデルを用いて音声合成を行います。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        wave = engine.synthesize_wave(
            query, style_id, enable_interrogative_upspeak=enable_interrogative_upspeak
        )

        buffer = io.BytesIO()
        soundfile.write(
            file=buffer, data=wave, samplerate=query.outputSamplingRate, format="WAV"
        )

        return Response(buffer.getvalue(), media_type="audio/wav")

    @router.post(
        "/cancellable_synthesis",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        # summary="音声合成する（キャンセル可能）",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def cancellable_synthesis(
        query: AudioQuery,
        request: Request,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        enable_interrogative_upspeak: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> FileResponse:
        raise HTTPException(
            status_code=501,
            detail="Cancelable synthesis is not supported in AivisSpeech Engine.",
        )
        """
        if cancellable_engine is None:
            raise HTTPException(
                status_code=404,
                detail="実験的機能はデフォルトで無効になっています。使用するには引数を指定してください。",
            )
        try:
            version = core_version or LATEST_VERSION
            f_name = cancellable_engine.synthesize_wave(
                query,
                style_id,
                enable_interrogative_upspeak=enable_interrogative_upspeak,
                request=request,
                version=version,
            )
        except CancellableEngineInternalError as e:
            print_exception(e)
            raise HTTPException(status_code=500) from e

        if f_name == "":
            raise HTTPException(status_code=422, detail="不明なバージョンです")

        return FileResponse(
            f_name,
            media_type="audio/wav",
            background=BackgroundTask(try_delete_file, f_name),
        )
        """

    @router.post(
        "/multi_synthesis",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
        tags=["音声合成"],
        summary="複数まとめて音声合成する",
    )
    def multi_synthesis(
        queries: list[AudioQuery],
        style_id: Annotated[StyleId, Query(alias="speaker")],
        enable_interrogative_upspeak: Annotated[
            bool,
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = True,  # fmt: skip # noqa
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> Response:
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        sampling_rate = queries[0].outputSamplingRate

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="a") as zip_file:
            for i in range(len(queries)):
                if queries[i].outputSamplingRate != sampling_rate:
                    raise HTTPException(
                        status_code=422,
                        detail="サンプリングレートが異なるクエリがあります",
                    )

                wav_file_buffer = io.BytesIO()
                wave = engine.synthesize_wave(
                    queries[i],
                    style_id,
                    enable_interrogative_upspeak=enable_interrogative_upspeak,
                )
                soundfile.write(
                    file=wav_file_buffer,
                    data=wave,
                    samplerate=sampling_rate,
                    format="WAV",
                )
                zip_file.writestr(
                    f"{str(i + 1).zfill(3)}.wav", wav_file_buffer.getvalue()
                )

        return Response(buffer.getvalue(), media_type="application/zip")

    @router.post(
        "/sing_frame_audio_query",
        tags=["クエリ作成"],
        # summary="歌唱音声合成用のクエリを作成する",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def sing_frame_audio_query(
        score: Score,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> FrameAudioQuery:
        raise HTTPException(
            status_code=501,
            detail="Sing frame audio query is not supported in AivisSpeech Engine.",
        )
        '''
        """歌唱音声合成用のクエリの初期値を得ます。ここで得られたクエリはそのまま歌唱音声合成に利用できます。各値の意味は`Schemas`を参照してください。"""
        version = core_version or LATEST_VERSION
        engine = song_engines.get_song_engine(version)
        try:
            phonemes, f0, volume = engine.create_phoneme_and_f0_and_volume(
                score, style_id
            )
        except SongInvalidInputError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return FrameAudioQuery(
            f0=f0,
            volume=volume,
            phonemes=phonemes,
            volumeScale=1,
            outputSamplingRate=engine.default_sampling_rate,
            outputStereo=False,
        )
        '''

    @router.post(
        "/sing_frame_f0",
        tags=["クエリ編集"],
        # summary="楽譜・歌唱音声合成用のクエリからフレームごとの基本周波数を得る",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def sing_frame_f0(
        score: Score,
        frame_audio_query: FrameAudioQuery,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[float]:
        raise HTTPException(
            status_code=501,
            detail="Sing frame f0 is not supported in AivisSpeech Engine.",
        )
        """
        version = core_version or LATEST_VERSION
        engine = song_engines.get_song_engine(version)
        try:
            return engine.create_f0_from_phoneme(
                score, frame_audio_query.phonemes, style_id
            )
        except SongInvalidInputError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        """

    @router.post(
        "/sing_frame_volume",
        tags=["クエリ編集"],
        # summary="楽譜・歌唱音声合成用のクエリからフレームごとの音量を得る",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def sing_frame_volume(
        score: Score,
        frame_audio_query: FrameAudioQuery,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[float]:
        raise HTTPException(
            status_code=501,
            detail="Sing frame volume is not supported in AivisSpeech Engine.",
        )
        """
        version = core_version or LATEST_VERSION
        engine = song_engines.get_song_engine(version)
        try:
            return engine.create_volume_from_phoneme_and_f0(
                score, frame_audio_query.phonemes, frame_audio_query.f0, style_id
            )
        except SongInvalidInputError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        """

    @router.post(
        "/frame_synthesis",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def frame_synthesis(
        query: FrameAudioQuery,
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> Response:
        raise HTTPException(
            status_code=501,
            detail="Frame synthesis is not supported in AivisSpeech Engine.",
        )
        '''
        """歌唱音声合成を行います。"""
        version = core_version or LATEST_VERSION
        engine = song_engines.get_song_engine(version)
        try:
            wave = engine.frame_synthesize_wave(query, style_id)
        except SongInvalidInputError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        with NamedTemporaryFile(delete=False) as f:
            soundfile.write(
                file=f, data=wave, samplerate=query.outputSamplingRate, format="WAV"
            )

        return FileResponse(
            f.name,
            media_type="audio/wav",
            background=BackgroundTask(try_delete_file, f.name),
        )
        '''

    @router.post(
        "/connect_waves",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="Base64 エンコードされた複数の WAV データを一つに結合する",
    )
    def connect_waves(waves: list[str]) -> Response:
        """
        Base64 エンコードされた WAV データを一つに結合し、WAV ファイルで返します。
        """
        try:
            waves_nparray, sampling_rate = connect_base64_waves(waves)
        except ConnectBase64WavesException as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        buffer = io.BytesIO()
        soundfile.write(
            file=buffer,
            data=waves_nparray,
            samplerate=sampling_rate,
            format="WAV",
        )

        return Response(buffer.getvalue(), media_type="audio/wav")

    @router.post(
        "/validate_kana",
        tags=["クエリ作成"],
        summary="テキストが AquesTalk 風記法に従っているか判定する",
        responses={
            400: {
                "description": "テキストが不正です",
                "model": ParseKanaBadRequest,
            }
        },
    )
    async def validate_kana(
        text: Annotated[str, Query(description="判定する対象の文字列")],
    ) -> bool:
        """
        テキストが AquesTalk 風記法に従っているかどうかを判定します。従っていない場合はエラーが返ります。
        """
        try:
            parse_kana(text)
            return True
        except ParseKanaError as e:
            raise HTTPException(
                status_code=400,
                detail=ParseKanaBadRequest(e).model_dump(),
            ) from e

    @router.post(
        "/initialize_speaker",
        status_code=204,
        tags=["音声合成モデル管理"],
        summary="指定されたスタイル ID に紐づく音声合成モデルをロードする",
    )
    def initialize_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        skip_reinit: Annotated[
            bool,
            Query(
                description="既にロード済みの音声合成モデルの再ロードをスキップするかどうか"
            ),
        ] = False,
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> None:
        """
        指定されたスタイル ID に紐づく音声合成モデルをロードします。<br>
        実行しなくても他の API は利用できますが、音声合成の初回実行時に時間がかかることがあります。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        engine.initialize_synthesis(style_id, skip_reinit=skip_reinit)

    @router.get(
        "/is_initialized_speaker",
        tags=["音声合成モデル管理"],
        summary="指定されたスタイル ID に紐づく音声合成モデルがロードされているかを確認する",
    )
    def is_initialized_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> bool:
        """
        指定されたスタイル ID に紐づく音声合成モデルがロードされているかどうかを返します。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)
        return engine.is_synthesis_initialized(style_id)

    @router.get(
        "/supported_devices",
        tags=["音声合成モデル管理"],
        summary="このビルドでサポートされている、音声合成モデルの推論デバイスを取得する",
    )
    def supported_devices(
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> SupportedDevicesInfo:
        """
        このビルドでサポートされている、音声合成モデルの推論デバイスを返します。<br>
        通常、下記の値が返されます。true であっても実際に推論デバイスが利用可能とは限りません。
        - Windows: `{"cpu": true, "cuda": false, "dml": true}`
        - macOS: `{"cpu": true, "cuda": false, "dml": false}`
        - Linux: `{"cpu": true, "cuda": true, "dml": false}`
        """
        version = core_version or LATEST_VERSION
        supported_devices = tts_engines.get_tts_engine(version).supported_devices
        if supported_devices is None:
            raise HTTPException(status_code=422, detail="非対応の機能です。")
        return SupportedDevicesInfo.generate_from(supported_devices)

    return router
