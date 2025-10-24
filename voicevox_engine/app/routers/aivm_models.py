"""音声合成モデル管理機能を提供する API Router"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.model import AivmInfo
from voicevox_engine.tts_pipeline.tts_engine import LATEST_VERSION, TTSEngineManager

from ..dependencies import VerifyMutabilityAllowed


def generate_aivm_models_router(
    aivm_manager: AivmManager,
    tts_engines: TTSEngineManager,
    verify_mutability: VerifyMutabilityAllowed,
) -> APIRouter:
    """音声合成モデル管理 API Router を生成する"""

    # ごく稀に style_bert_vits2_tts_engine.py (が依存する onnxruntime) のインポート自体に失敗し
    # 例外が発生する環境があるようなので、例外をキャッチしてエラーログに出力できるよう、敢えてルーター初期化時にインポートする
    from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
        StyleBertVITS2TTSEngine,
    )

    router = APIRouter(
        prefix="/aivm_models",
        tags=["音声合成モデル管理"],
    )

    @router.get(
        "",
        summary="インストール済みのすべての音声合成モデルの情報を取得する",
        response_description="インストール済みのすべての音声合成モデルの情報",
    )
    def get_installed_aivm_infos() -> dict[str, AivmInfo]:
        """
        インストール済みのすべての音声合成モデルの情報を返します。
        """

        return aivm_manager.get_installed_aivm_infos()

    @router.post(
        "/install",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="音声合成モデルをインストールする",
    )
    def install_model(
        file: Annotated[
            UploadFile | None,
            File(description="AIVMX ファイル (`.aivmx`)"),
        ] = None,
        url: Annotated[
            str | None,
            Form(description="AIVMX ファイルの URL"),
        ] = None,
    ) -> None:
        """
        音声合成モデルをインストールします。<br>
        ファイルからインストールする場合は `file` を指定してください。<br>
        URL からインストールする場合は `url` を指定してください。
        """

        if file is not None:
            # ファイルから音声合成モデルをインストール
            aivm_manager.install_model(file.file)
        elif url is not None:
            # URL から音声合成モデルをダウンロードしてインストール
            aivm_manager.install_model_from_url(url)
        else:
            raise HTTPException(
                status_code=422,
                detail="Either file or url must be provided.",
            )

    @router.get(
        "/{ aivm_model_uuid}",
        summary="指定された音声合成モデルの情報を取得する",
        response_description="指定された音声合成モデルの情報",
    )
    def get_aivm_info(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmInfo:
        """
        指定された音声合成モデルの情報を取得します。
        """

        return aivm_manager.get_aivm_info(aivm_model_uuid)

    @router.post(
        "/{ aivm_model_uuid}/load",
        status_code=204,
        summary="指定された音声合成モデルをロードする",
    )
    def load_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをロードします。すでにロード済みの場合は何も行われません。<br>
        実行しなくても他の API は利用できますが、音声合成の初回実行時に時間がかかることがあります。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをロード
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.load_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{ aivm_model_uuid}/unload",
        status_code=204,
        summary="指定された音声合成モデルをアンロードする",
    )
    def unload_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをアンロードします。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをアンロード
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.unload_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{ aivm_model_uuid}/update",
        status_code=204,
        summary="指定された音声合成モデルを更新する",
    )
    def update_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        AivisHub から指定された音声合成モデルの一番新しいバージョンをダウンロードし、
        インストール済みの音声合成モデルへ上書き更新します。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # AivisHub からダウンロードした新しいバージョンの音声合成モデルを上書きインストール
        aivm_manager.update_model(str(aivm_info.manifest.uuid))

    @router.delete(
        "/{ aivm_model_uuid}/uninstall",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="指定された音声合成モデルをアンインストールする",
    )
    def uninstall_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをアンインストールします。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをアンロード
        # アンインストール前にアンロードしておかないと、既にロードされている場合に
        # プロセス終了までモデルがアンロードされなくなってしまう
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.unload_model(str(aivm_info.manifest.uuid))

        # インストール済みの音声合成モデルをアンインストール
        aivm_manager.uninstall_model(str(aivm_info.manifest.uuid))

    return router
