"""キャラクター情報機能を提供する API Router"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.metas import Speaker, SpeakerInfo
from voicevox_engine.metas.metas_store import (
    Character,
    ResourceFormat,
)
from voicevox_engine.resource_manager import ResourceManager, ResourceManagerError

RESOURCE_ENDPOINT = "_resources"


async def _get_resource_baseurl(request: Request) -> str:
    """
    リソースのベースURLを生成する。

    リクエストのスキームとホスト名を使用して、リソースのベースURLを生成する。
    例: `http://localhost:50021/_resources`
    """
    return f"{request.url.scheme}://{request.url.netloc}/{RESOURCE_ENDPOINT}"


def _characters_to_speakers(characters: list[Character]) -> list[Speaker]:
    """キャラクターのリストを `Speaker` のリストへキャストする。"""
    return list(
        map(
            lambda character: Speaker(
                name=character.name,
                speaker_uuid=character.uuid,
                styles=character.talk_styles + character.sing_styles,
                version=character.version,
                supported_features=character.supported_features,
            ),
            characters,
        )
    )


def generate_character_router(
    resource_manager: ResourceManager,
    aivm_manager: AivmManager,
) -> APIRouter:
    """キャラクター情報 API Router を生成する"""
    router = APIRouter(tags=["話者情報"])

    @router.get(
        "/speakers",
        summary="話者情報の一覧を取得する",
    )
    def speakers(
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        """話者情報の一覧を返します。"""
        # AivisSpeech Engine では常に AivmManager から Speaker を取得する
        return aivm_manager.get_speakers()
        """
        characters = metas_store.talk_characters(core_version)
        return _characters_to_speakers(characters)
        """

    @router.get(
        "/speaker_info",
        summary="UUID で指定された話者の情報を取得する",
    )
    def speaker_info(
        resource_baseurl: Annotated[str, Depends(_get_resource_baseurl)],
        speaker_uuid: str,
        resource_format: ResourceFormat = "base64",
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> SpeakerInfo:
        """
        UUID で指定された話者の情報を返します。
        画像や音声は resource_format で指定した形式で返されます。
        """
        # AivisSpeech Engine では常に AivmManager から SpeakerInfo を取得する
        return aivm_manager.get_speaker_info(speaker_uuid)
        """
        return metas_store.character_info(
            character_uuid=speaker_uuid,
            talk_or_sing="talk",
            core_version=core_version,
            resource_baseurl=resource_baseurl,
            resource_format=resource_format,
        )
        """

    @router.get(
        "/singers",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def singers(
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        raise HTTPException(
            status_code=501,
            detail="Singers is not supported in AivisSpeech Engine.",
        )
        '''
        """歌えるキャラクターの情報の一覧を返します。"""
        characters = metas_store.sing_characters(core_version)
        return _characters_to_speakers(characters)
        '''

    @router.get(
        "/singer_info",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def singer_info(
        resource_baseurl: Annotated[str, Depends(_get_resource_baseurl)],
        speaker_uuid: str,
        resource_format: ResourceFormat = "base64",
        core_version: Annotated[
            str | SkipJsonSchema[None],
            Query(
                description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。"
            ),
        ] = None,  # fmt: skip # noqa
    ) -> SpeakerInfo:
        raise HTTPException(
            status_code=501,
            detail="Singer info is not supported in AivisSpeech Engine.",
        )
        '''
        """
        UUID で指定された歌えるキャラクターの情報を返します。

        画像や音声はresource_formatで指定した形式で返されます。
        """
        return metas_store.character_info(
            character_uuid=speaker_uuid,
            talk_or_sing="sing",
            core_version=core_version,
            resource_baseurl=resource_baseurl,
            resource_format=resource_format,
        )
        '''

    # リソースはAPIとしてアクセスするものではないことを表明するためOpenAPIスキーマーから除外する
    @router.get(f"/{RESOURCE_ENDPOINT}/{{resource_hash}}", include_in_schema=False)
    async def resources(resource_hash: str) -> FileResponse:
        """ResourceManagerから発行されたハッシュ値に対応するリソースファイルを返す。"""
        try:
            resource_path = resource_manager.resource_path(resource_hash)
        except ResourceManagerError as e:
            raise HTTPException(status_code=404) from e
        return FileResponse(
            resource_path,
            headers={"Cache-Control": "max-age=2592000"},  # 30日
        )

    return router
