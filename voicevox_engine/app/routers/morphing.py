"""モーフィング機能を提供する API Router"""

import io
from functools import lru_cache
from typing import Annotated

import soundfile
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.morphing.model import MorphableTargetInfo
from voicevox_engine.morphing.morphing import (
    StyleIdNotFoundError,
    get_morphable_targets,
    is_morphable,
    synthesize_morphed_wave,
)
from voicevox_engine.morphing.morphing import (
    synthesis_morphing_parameter as _synthesis_morphing_parameter,
)
from voicevox_engine.tts_pipeline.tts_engine import LATEST_VERSION, TTSEngineManager

# キャッシュを有効化
# モジュール側でlru_cacheを指定するとキャッシュを制御しにくいため、HTTPサーバ側で指定する
# TODO: キャッシュを管理するモジュール側API・HTTP側APIを用意する
synthesis_morphing_parameter = lru_cache(maxsize=4)(_synthesis_morphing_parameter)


def generate_morphing_router(
    tts_engines: TTSEngineManager,
    aivm_manager: AivmManager,
) -> APIRouter:
    """モーフィング API Router を生成する"""
    router = APIRouter(tags=["音声合成"])

    @router.post(
        "/morphable_targets",
        summary="指定したスタイルに対してエンジン内のキャラクターがモーフィングが可能か判定する",
    )
    def morphable_targets(
        base_style_ids: list[StyleId],
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[dict[str, MorphableTargetInfo]]:
        """
        指定されたベーススタイルに対してエンジン内の各キャラクターがモーフィング機能を利用可能か返します。<br>
        モーフィングの許可/禁止は `/speakers `の `speaker.supported_features.synthesis_morphing` に記載されています。<br>
        プロパティが存在しない場合は、モーフィングが許可されているとみなします。<br>
        返り値のスタイル ID は string 型なので注意。<br>
        AivisSpeech Engine では話者ごとに発声タイミングが異なる関係で実装不可能なため (動作こそするが聴くに耐えない) 、
        全ての話者でモーフィングが禁止されています。
        """  # noqa
        characters = aivm_manager.get_characters()
        try:
            morphable_targets = get_morphable_targets(characters, base_style_ids)
        except StyleIdNotFoundError as e:
            msg = f"該当するスタイル(style_id={e.style_id})が見つかりません"
            raise HTTPException(status_code=404, detail=msg) from e
        # NOTE: jsonはint型のキーを持てないので、string型に変換する
        return [
            {str(k): v for k, v in morphable_target.items()}
            for morphable_target in morphable_targets
        ]

    @router.post(
        "/synthesis_morphing",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        summary="2種類のスタイルでモーフィングした音声を合成する",
    )
    def _synthesis_morphing(
        query: AudioQuery,
        base_style_id: Annotated[StyleId, Query(alias="base_speaker")],
        target_style_id: Annotated[StyleId, Query(alias="target_speaker")],
        morph_rate: Annotated[float, Query(ge=0.0, le=1.0)],
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
        指定された 2 種類のスタイルで音声を合成、指定した割合でモーフィングした音声を得ます。<br>
        モーフィングの割合は `morph_rate` で指定でき、0.0 でベースのスタイル、1.0 でターゲットのスタイルに近づきます。<br>
        AivisSpeech Engine では話者ごとに発声タイミングが異なる関係で実装不可能なため (動作こそするが聴くに耐えない) 、
        常に 400 Bad Request を返します。
        """
        version = core_version or LATEST_VERSION
        engine = tts_engines.get_tts_engine(version)

        # モーフィングが許可されないキャラクターペアを拒否する
        characters = aivm_manager.get_characters()
        try:
            morphable = is_morphable(characters, base_style_id, target_style_id)
        except StyleIdNotFoundError as e:
            msg = f"該当するスタイル(style_id={e.style_id})が見つかりません"
            raise HTTPException(status_code=404, detail=msg) from e
        if not morphable:
            msg = "指定されたスタイルペアでのモーフィングはできません"
            raise HTTPException(status_code=400, detail=msg)

        # 生成したパラメータはキャッシュされる
        morph_param = synthesis_morphing_parameter(
            engine=engine,
            query=query,
            base_style_id=base_style_id,
            target_style_id=target_style_id,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
        )

        morph_wave = synthesize_morphed_wave(
            morph_param=morph_param,
            morph_rate=morph_rate,
            output_fs=query.outputSamplingRate,
            output_stereo=query.outputStereo,
        )

        buffer = io.BytesIO()
        soundfile.write(
            file=buffer,
            data=morph_wave,
            samplerate=query.outputSamplingRate,
            format="WAV",
        )

        return Response(buffer.getvalue(), media_type="audio/wav")

    return router
