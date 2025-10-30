"""AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルと AIVM マニフェストを管理するクラス"""

import re
import time
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import aivmlib
import httpx
import semver
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
)
from fastapi import HTTPException
from semver.version import Version

from voicevox_engine.aivm_infos_repository import AivmInfosRepository
from voicevox_engine.logging import logger
from voicevox_engine.metas.Metas import Speaker, SpeakerInfo, StyleId
from voicevox_engine.metas.MetasStore import Character
from voicevox_engine.model import AivmInfo
from voicevox_engine.utility.aivishub_client import AivisHubClient
from voicevox_engine.utility.user_agent_utility import generate_user_agent

__all__ = ["AivmManager"]


class AivmManager:
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルと AIVM マニフェストを管理するクラス。

    VOICEVOX ENGINE における MetasStore の役割を代替する。(AivisSpeech Engine では MetasStore は無効化されている)
    AivisSpeech はインストールサイズを削減するため、AIVMX ファイルにのみ対応している。
    ref: https://github.com/Aivis-Project/aivmlib#aivm-specification
    """

    def __init__(self, installed_models_dir: Path):
        """
        AivmManager のコンストラクタ。

        Parameters
        ----------
        installed_models_dir : Path
            AIVMX ファイルのインストール先ディレクトリ
        """

        # インストール先ディレクトリが存在しなければここで作成
        self.installed_models_dir = installed_models_dir
        self.installed_models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models directory: {self.installed_models_dir}")

        # リポジトリを初期化
        ## この時点で前回起動時に作成したキャッシュがあればそこから即座に情報が読み込まれる
        ## キャッシュがない場合はコンストラクタで同期的にスキャンを行い、スキャン完了次第リポジトリの初期化が完了する
        ## コンストラクタの実行が完了した時点で、確実にインストール済み音声合成モデルの情報が存在している
        self._repository = AivmInfosRepository(self.installed_models_dir)

        # 強制削除ルールに基づいて、該当する音声合成モデルを自動的に削除する
        self._apply_forced_removal_rules()

        # AivisHub から取得したデフォルトモデル構成に基づいて、インストールまたは自動更新を行う
        self._install_or_update_default_models()

        # インストール済み音声合成モデルの一覧をログ出力
        logger.info("Installed models:")
        for aivm_info in self._repository.get_installed_aivm_infos().values():
            logger.info(f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid})")

    def _apply_forced_removal_rules(self) -> None:
        """
        AivisHub から取得した強制削除ルールに基づいて、該当する音声合成モデルを自動的に削除する。
        このメソッドはコンストラクタ実行時に自動的に実行される。
        """

        # AivisHub から強制削除ルールを取得
        # ネットワークエラーなどで取得できなかった場合、フォールバックとしてハードコードされた値が返される
        forced_removal_rules = AivisHubClient.fetch_forced_removal_rules()

        # ルールごとに該当するモデルがないかをチェック
        for rule in forced_removal_rules:
            target_model_uuid = str(rule.model_uuid)

            # 最新のインストール済みモデル一覧を再取得
            # ループ中に強制削除に該当したモデルの情報が削除される可能性があるため、常に最新の情報を取得してから判定すべき
            current_infos = self.get_installed_aivm_infos()

            # 対象の音声合成モデルがインストール済みでない場合はスキップ
            if target_model_uuid not in current_infos:
                continue

            # version_specifiers が null の場合は指定されたモデル UUID に対応する全バージョンが削除対象
            if rule.version_specifiers is None:
                should_remove = True

            # すべての条件を満たす場合のみ強制削除とみなす
            else:
                try:
                    ## semver.match() で各バージョン指定子と一致するか確認し、all() で全てのバージョン指定子と一致する場合のみ True
                    should_remove = all(
                        semver.match(
                            current_infos[target_model_uuid].manifest.version,
                            specifier,
                        )
                        for specifier in rule.version_specifiers
                    )
                except ValueError as ex:
                    # Semver 評価に失敗した場合は安全のためスキップ
                    logger.warning(
                        "Failed to evaluate version specifiers. "
                        f"(model_uuid: {target_model_uuid}, version_specifiers: {rule.version_specifiers})",
                        exc_info=ex,
                    )
                    continue

            # 強制削除ルールに一致したモデルを強制的にアンインストール
            if should_remove is True:
                logger.info(
                    f"Removing model {target_model_uuid} as forced removal rule matched. "
                )
                ## force=True の指定により、バリデーションをスキップし強制的にアンインストールする
                self.uninstall_model(target_model_uuid, force=True)

    def _install_or_update_default_models(self) -> None:
        """
        AivisHub から取得したデフォルトモデル構成に基づいて、インストールまたは自動更新を行う。
        エンジン起動時に必ずデフォルトモデルがインストールされていることを保証する。
        このメソッドはコンストラクタ実行時に自動的に実行される。
        """

        # AivisHub からデフォルトモデル構成を取得
        # ネットワークエラーなどで取得できなかった場合、フォールバックとしてハードコードされた値が返される
        default_model_properties = AivisHubClient.fetch_default_models()

        # デフォルトモデルの UUID セットを構築
        # この情報は後で self._repository.mark_default_models() に渡すために保持しておく
        default_model_uuid_set = {
            model_property.model_uuid for model_property in default_model_properties
        }

        # 指定されたデフォルトモデルごとにインストールまたは更新を行う
        for model_property in default_model_properties:
            model_uuid_str = str(model_property.model_uuid)
            latest_version_str = model_property.latest_version

            # 現在のインストール済みモデル一覧を取得
            current_infos = self.get_installed_aivm_infos()
            current_info = current_infos.get(model_uuid_str)

            # 未インストールの場合は新規インストール
            if current_info is None:
                logger.info(
                    f"Installing default model {model_uuid_str} (v{latest_version_str}) specified by AivisHub."
                )
                try:
                    # AivisHub API のダウンロード URL を構築して install_model_from_url() に渡す
                    download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{model_uuid_str}/download?model_type=AIVMX"
                    self.install_model_from_url(download_url)
                except Exception as ex:
                    logger.error(
                        "Failed to install default model.",
                        exc_info=ex,
                    )
                continue

            # バージョン比較のため semver を用いて厳密に判定
            try:
                current_version = Version.parse(current_info.manifest.version)
                latest_version = Version.parse(latest_version_str)
            except ValueError as ex:
                logger.warning(
                    "Failed to parse version information. "
                    f"(model_uuid: {model_uuid_str}, local_version: {current_info.manifest.version}, latest_version: {latest_version_str})",
                    exc_info=ex,
                )
                continue

            # 既存モデルが古い場合はサーバー側の最新バージョンへ更新
            if latest_version > current_version:
                logger.info(
                    f"Updating default model {model_uuid_str} (v{current_version}) to latest version (v{latest_version}). "
                )
                try:
                    # AivisHub API のダウンロード URL を構築して install_model_from_url() に渡す
                    download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{model_uuid_str}/download?model_type=AIVMX"
                    self.install_model_from_url(download_url)
                except Exception as ex:
                    logger.error("Failed to update default model.", exc_info=ex)

        # デフォルトモデル指定フラグを更新
        ## AivisHub から取得したデフォルトモデル一覧に基づいて、各モデルの is_default_model フラグを更新する
        ## この処理はデフォルトモデルすべてのインストールまたは更新が終わった後に実行する必要がある（インストール前だと情報が取得できていないため）
        self._repository.mark_default_models(default_model_uuid_set)

    def get_characters(self) -> list[Character]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を Character 型で取得する (MetasStore 互換用)。

        Returns
        -------
        characters : list[Character]
            インストール済み音声合成モデル内の話者の一覧
        """

        speakers = self.get_speakers()
        characters: list[Character] = []
        for speaker in speakers:
            character = Character(
                name=speaker.name,
                uuid=speaker.speaker_uuid,
                # AivisSpeech Engine では talk スタイルのみがサポートされる
                talk_styles=speaker.styles,
                # AivisSpeech Engine では歌唱音声合成はサポートされていない
                sing_styles=[],
                version=speaker.version,
                supported_features=speaker.supported_features,
            )
            characters.append(character)

        # 既に get_speakers() で話者名でソートされているのでそのまま返す
        return characters

    def get_speakers(self) -> list[Speaker]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を取得する。

        Returns
        -------
        speakers : list[Speaker]
            インストール済み音声合成モデル内の話者の一覧
        """

        aivm_infos = self.get_installed_aivm_infos()
        speakers: list[Speaker] = []
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                speakers.append(aivm_info_speaker.speaker)

        # 話者名でソートしてから返す
        return sorted(speakers, key=lambda x: x.name)

    def get_speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        """
        インストール済み音声合成モデル内の話者の追加情報を取得する。

        Parameters
        ----------
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        speaker_info : SpeakerInfo
            話者の追加情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                if aivm_info_speaker.speaker.speaker_uuid == speaker_uuid:
                    return aivm_info_speaker.speaker_info

        logger.error(f"Speaker {speaker_uuid} is not installed.")
        raise HTTPException(
            status_code=404,
            detail=f"話者 {speaker_uuid} はインストールされていません。",
        )

    def get_aivm_info(self, aivm_model_uuid: str) -> AivmInfo:
        """
        音声合成モデルの UUID から AIVMX ファイルの情報を取得する。

        Parameters
        ----------
         aivm_model_uuid : str
            AIVM マニフェスト記載の音声合成モデルの UUID

        Returns
        -------
        aivm_info : AivmInfo
            AIVMX ファイルの情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            if str(aivm_info.manifest.uuid) == aivm_model_uuid:
                return aivm_info

        logger.error(f"Model {aivm_model_uuid} is not installed.")
        raise HTTPException(
            status_code=404,
            detail=f"音声合成モデル {aivm_model_uuid} はインストールされていません。",
        )

    def get_aivm_manifest_from_style_id(
        self, style_id: StyleId
    ) -> tuple[AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle]:
        """
        スタイル ID に対応する AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle を取得する。

        Parameters
        ----------
        style_id : StyleId
            スタイル ID

        Returns
        -------
        aivm_manifest : AivmManifest
            AIVM マニフェスト
        aivm_manifest_speaker : AivmManifestSpeaker
            AIVM マニフェスト内の話者
        aivm_manifest_style : AivmManifestSpeakerStyle
            AIVM マニフェスト内のスタイル
        """

        # fmt: off
        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                for aivm_info_speaker_style in aivm_info_speaker.speaker.styles:
                    if aivm_info_speaker_style.id == style_id:
                        # ここでスタイル ID が示す音声合成モデルに対応する AivmManifest を特定
                        aivm_manifest = aivm_info.manifest
                        for aivm_manifest_speaker in aivm_manifest.speakers:
                            # ここでスタイル ID が示す話者に対応する AivmManifestSpeaker を特定
                            if str(aivm_manifest_speaker.uuid) == aivm_info_speaker.speaker.speaker_uuid:
                                for aivm_manifest_style in aivm_manifest_speaker.styles:
                                    # ここでスタイル ID が示すスタイルに対応する AivmManifestSpeakerStyle を特定
                                    local_style_id = self._repository.style_id_to_local_style_id(style_id)
                                    if aivm_manifest_style.local_id == local_style_id:
                                        # すべて取得できたので値を返す
                                        return aivm_manifest, aivm_manifest_speaker, aivm_manifest_style

        logger.error(f"Style {style_id} is not found.")
        raise HTTPException(
            status_code=404,
            detail=f"スタイル {style_id} は存在しません。",
        )

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        すべてのインストール済み音声合成モデルの情報を取得する。

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報 (キー: 音声合成モデルの UUID, 値: AivmInfo)
        """

        # リポジトリの現在の状態を返す
        return self._repository.get_installed_aivm_infos()

    def update_model_load_state(self, aivm_model_uuid: str, is_loaded: bool) -> None:
        """
        音声合成モデルのロード状態を更新する。
        このメソッドは StyleBertVITS2TTSEngine 上でロード/アンロードが行われた際に呼び出される。

        Parameters
        ----------
         aivm_model_uuid : str
            AIVM マニフェスト記載の音声合成モデルの UUID
        is_loaded : bool
            モデルがロードされているかどうか
        """

        # リポジトリの現在のモデルロード状態を更新する
        self._repository.update_model_load_state(aivm_model_uuid, is_loaded)

    def install_model(self, file: BinaryIO) -> None:
        """
        AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をインストールする。

        Parameters
        ----------
        file : BinaryIO
            AIVMX ファイルのバイナリ
        """

        # AIVMX ファイルからから AIVM メタデータを取得
        try:
            aivm_metadata = aivmlib.read_aivmx_metadata(file)
            aivm_manifest = aivm_metadata.manifest
        except aivmlib.AivmValidationError as ex:
            logger.error("AIVMX file is invalid:", exc_info=ex)
            raise HTTPException(
                status_code=422,
                detail=f"指定された AIVMX ファイルの形式が正しくありません。({ex})",
            ) from ex

        # すでに同一 UUID のファイルがインストール済みの場合、同じファイルを更新する
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしていた (ファイル名が UUID と一致しない) 場合も更新できるよう、
        ## この場合のみ特別に更新先ファイル名を現在保存されているファイル名に変更する
        aivm_file_path = self.installed_models_dir / f"{aivm_manifest.uuid}.aivmx"
        aivm_infos = self.get_installed_aivm_infos()
        if str(aivm_manifest.uuid) in aivm_infos:
            logger.info(f"Model {aivm_manifest.uuid} is already installed. Updating...")
            # aivm_file_path を現在保存されているファイル名に変更
            aivm_file_path = aivm_infos[str(aivm_manifest.uuid)].file_path

        # マニフェストバージョンのバリデーション
        if (
            aivm_manifest.manifest_version
            not in self._repository.SUPPORTED_MANIFEST_VERSIONS
        ):
            logger.error(
                f"AIVM manifest version {aivm_manifest.manifest_version} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f"AIVM マニフェストバージョン {aivm_manifest.manifest_version} には対応していません。",
            )

        # 音声合成モデルのアーキテクチャのバリデーション
        if (
            aivm_manifest.model_architecture
            not in self._repository.SUPPORTED_MODEL_ARCHITECTURES
        ):
            logger.error(
                f"Model architecture {aivm_manifest.model_architecture} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f'モデルアーキテクチャ "{aivm_manifest.model_architecture}" には対応していません。',
            )

        # BinaryIO のシークをリセット
        # ここでリセットしないとファイルの内容を読み込めない
        file.seek(0)

        # AIVMX ファイルをインストール
        ## 通常は重複防止のため "(音声合成モデルの UUID).aivmx" のフォーマットのファイル名でインストールされるが、
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしても一通り動作するように考慮している
        logger.info("Installing AIVMX file ...")
        try:
            with open(aivm_file_path, mode="wb") as f:
                f.write(file.read())
            logger.info(f"Installed AIVMX file to {aivm_file_path}.")
        except OSError as ex:
            logger.error(
                f"Failed to write AIVMX file to {aivm_file_path}:", exc_info=ex
            )
            error_message = str(ex).lower()
            if "no space" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。ストレージ容量が不足しています。({ex})"
            elif "permission denied" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダへのアクセス権限が不足しています。({ex})"
            elif "read-only" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダが読み取り専用権限になっています。({ex})"
            else:
                detail = f"AIVMX ファイルの書き込みに失敗しました。({ex})"
            raise HTTPException(
                status_code=500,
                detail=detail,
            ) from ex

        # リポジトリのメタデータを更新
        self._repository.upsert_model_from_metadata(aivm_metadata, aivm_file_path)

    def install_model_from_url(self, download_url: str) -> None:
        """
        指定された URL から AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をダウンロードしてインストールする。

        Parameters
        ----------
        download_url : str
            AIVMX ファイルのダウンロード先 URL
        """

        # AivisHub の音声合成モデル詳細ページの URL が渡された場合、特別に AivisHub API を使い AIVMX ファイルをダウンロードする
        if download_url.startswith("https://hub.aivis-project.com/aivm-models/"):
            # URL から AIVM の UUID を抽出
            uuid_match = re.search(
                r"/aivm-models/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                download_url.lower(),
            )
            if not uuid_match:
                logger.error(f"Invalid AivisHub URL: {download_url}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid AivisHub URL: {download_url}",
                )
            # group(0) は一致した文字列全体なので、group(1) で UUID 部分のみを取得
            aivm_model_uuid = uuid_match.group(1)
            # AIVMX ダウンロード用の API の URL に置き換え
            download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{aivm_model_uuid}/download?model_type=AIVMX"
            logger.info(
                f"Detected AivisHub model page URL. Using download API URL: {download_url}"
            )

        # URL から AIVMX ファイルをダウンロード
        max_retries = 3
        retry_count = 0
        last_exception: httpx.HTTPError | None = None
        while retry_count < max_retries:
            try:
                logger.info(
                    f"Downloading AIVMX file from {download_url} (Attempt {retry_count + 1}/{max_retries})..."
                )
                response = httpx.get(
                    download_url,
                    headers={"User-Agent": generate_user_agent()},
                    # リダイレクトを追跡する
                    follow_redirects=True,
                    # 接続タイムアウト10秒 / 読み取りタイムアウト300秒
                    timeout=httpx.Timeout(10.0, read=300.0),
                )
                response.raise_for_status()
                logger.info("Downloaded AIVMX file.")
                # ダウンロードした AIVMX ファイルの内容を渡してインストール処理を行う
                self.install_model(BytesIO(response.content))
                return
            except httpx.HTTPStatusError as ex:
                last_exception = ex
                # 403 Forbidden や 404 Not Found の場合はリトライしない
                if ex.response.status_code in [403, 404]:
                    logger.error(
                        f"Failed to download AIVMX file from {download_url} (HTTP Error {ex.response.status_code}). No retry.",
                        exc_info=ex,
                    )
                    raise HTTPException(
                        status_code=500,  # 4xx 系エラーでもサーバー側の問題として 500 を返す
                        detail=f"AIVMX ファイルのダウンロードに失敗しました。({ex})",
                    ) from ex
                logger.warning(
                    f"Failed to download AIVMX file from {download_url} (Attempt {retry_count + 1}/{max_retries}). Retrying...",
                    exc_info=ex,
                )
            except httpx.HTTPError as ex:
                last_exception = ex
                logger.warning(
                    f"Failed to download AIVMX file from {download_url} (Attempt {retry_count + 1}/{max_retries}). Retrying...",
                    exc_info=ex,
                )

            retry_count += 1
            if retry_count < max_retries:
                # リトライ前に1秒待機
                time.sleep(1)

        # リトライ上限に達しても成功しなかった場合
        logger.error(
            f"Failed to download AIVMX file from {download_url} after {max_retries} attempts.",
            exc_info=last_exception,
        )
        raise HTTPException(
            status_code=500,
            detail=f"AIVMX ファイルのダウンロードに失敗しました。({last_exception})",
        ) from last_exception

    def update_model(self, aivm_model_uuid: str) -> None:
        """
        AivisHub から指定された音声合成モデルの一番新しいバージョンをダウンロードし、
        インストール済みの音声合成モデルへ上書き更新する。

        Parameters
        ----------
         aivm_model_uuid : str
            AIVM マニフェスト記載の音声合成モデルの UUID
        """

        # 対象の音声合成モデルがインストール済みかを確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_model_uuid not in installed_aivm_infos.keys():
            logger.error(f"Model {aivm_model_uuid} is not installed.")
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_model_uuid} はインストールされていません。",
            )

        # アップデートが利用可能かを確認
        aivm_info = installed_aivm_infos[aivm_model_uuid]
        if not aivm_info.is_update_available:
            logger.error(f"Model {aivm_model_uuid} has no new update.")
            raise HTTPException(
                status_code=422,
                detail=f"音声合成モデル {aivm_model_uuid} の新しいアップデートはありません。",
            )

        ## AivisHub API のダウンロード URL を構築して install_model_from_url() に渡す
        ## install_model_from_url() により既存のモデルファイルが AivisHub からダウンロードした新しいモデルファイルに上書き更新される
        download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{aivm_model_uuid}/download?model_type=AIVMX"
        logger.info(
            f"Updating model {aivm_model_uuid} to version {aivm_info.latest_version}..."
        )
        self.install_model_from_url(download_url)
        logger.info(
            f"Updated model {aivm_model_uuid} to version {aivm_info.latest_version}."
        )

    def uninstall_model(self, aivm_model_uuid: str, force: bool = False) -> None:
        """
        インストール済み音声合成モデルをアンインストールする。

        Parameters
        ----------
        aivm_model_uuid : str
            AIVM マニフェスト記載の音声合成モデルの UUID
        force : bool
            強制削除フラグ。True の場合、バリデーションをスキップし強制的にアンインストールする（デフォルト: False）
        """

        # 対象の音声合成モデルがインストール済みかを確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_model_uuid not in installed_aivm_infos.keys():
            logger.error(f"Model {aivm_model_uuid} is not installed.")
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_model_uuid} はインストールされていません。",
            )

        # 対象の音声合成モデル情報を取得
        target_info = installed_aivm_infos[aivm_model_uuid]

        # デフォルトモデルかどうかをチェック (強制削除フラグが False の場合のみ)
        if force is False and target_info.is_default_model is True:
            logger.error(
                f"Model {aivm_model_uuid} is a default model. It cannot be uninstalled."
            )
            raise HTTPException(
                status_code=400,
                detail="指定された音声合成モデルはデフォルトモデルのためアンインストールできません。",
            )

        # インストール済みの音声合成モデルの数を確認 (強制削除フラグが False の場合のみ)
        if force is False and len(installed_aivm_infos) <= 1:
            logger.error("AivisSpeech Engine must have at least one installed model.")
            raise HTTPException(
                status_code=400,
                detail="AivisSpeech Engine には必ず 1 つ以上の音声合成モデルがインストールされている必要があります。",
            )

        # AIVMX ファイルをアンインストール
        ## AIVMX ファイルのファイル名は必ずしも "(音声合成モデルの UUID).aivmx" になるとは限らないため、
        ## AivmInfo 内に格納されているファイルパスを使って削除する
        ## 万が一 AIVMX ファイルが存在しない場合は無視する
        if force is True:
            logger.info(
                f"Force uninstalling model {aivm_model_uuid} from {target_info.file_path}..."
            )
        else:
            logger.info(
                f"Uninstalling model {aivm_model_uuid} from {target_info.file_path}..."
            )
        target_info.file_path.unlink(missing_ok=True)
        logger.info(f"Uninstalled model {aivm_model_uuid}.")

        # リポジトリから当該モデルの情報を削除
        self._repository.remove_model(aivm_model_uuid)
