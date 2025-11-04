"""
AivisHub API との通信を行うクライアントと、API スキーマを表現する Pydantic モデルを提供するモジュール。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from voicevox_engine.logging import logger
from voicevox_engine.utility.path_utility import get_save_dir
from voicevox_engine.utility.user_agent_utility import (
    AivisSpeechRuntimeEnvironment,
    generate_user_agent,
)


class UserResponse(BaseModel):
    """
    ユーザー情報のレスポンスモデル。
    """

    handle: str
    name: str
    description: str
    icon_url: str
    account_type: str
    account_status: str
    social_links: list[dict[str, Any]]


class AivmModelResponse(BaseModel):
    """
    音声合成モデル情報のレスポンスモデル。
    """

    aivm_model_uuid: uuid.UUID
    user: UserResponse
    name: str
    description: str
    detailed_description: str
    category: str
    voice_timbre: str
    visibility: str
    is_tag_locked: bool
    total_download_count: int
    model_files: list[AivmModelFile]
    tags: list[AivmModelTag]
    like_count: int
    is_liked: bool
    speakers: list[AivmModelSpeaker]
    created_at: str
    updated_at: str

    model_config = ConfigDict(protected_namespaces=())


class AivmModelFile(BaseModel):
    """
    音声合成モデルのファイル情報。
    """

    aivm_model_uuid: uuid.UUID
    manifest_version: str
    name: str
    description: str
    creators: list[str]
    license_type: str
    license_text: str | None
    model_type: str
    model_architecture: str
    model_format: str
    training_epochs: int | None
    training_steps: int | None
    version: str
    file_size: int
    checksum: str
    download_count: int
    created_at: str
    updated_at: str

    model_config = ConfigDict(protected_namespaces=())


class AivmModelTag(BaseModel):
    """
    音声合成モデルのタグ。
    """

    name: str


class AivmModelSpeaker(BaseModel):
    """
    音声合成モデルの話者情報。
    """

    aivm_speaker_uuid: uuid.UUID
    name: str
    icon_url: str
    supported_languages: list[str]
    local_id: int
    styles: list[AivmModelSpeakerStyle]


class AivmModelSpeakerStyle(BaseModel):
    """
    音声合成モデルの話者スタイル情報。
    """

    name: str
    icon_url: str | None
    local_id: int
    voice_samples: list[AivmModelVoiceSample]


class AivmModelVoiceSample(BaseModel):
    """
    音声サンプル情報。
    """

    audio_url: str
    transcript: str


class AivisSpeechDefaultModelsResponse(BaseModel):
    """
    AivisSpeech Engine が起動時に自動的にインストールする音声合成モデルの一覧を表すレスポンスモデル。
    """

    aivm_models: Annotated[
        list[AivisSpeechDefaultModelProperty],
        Field(
            description="AivisSpeech Engine が起動時に自動的にインストールすべき音声合成モデルの一覧",
            min_length=0,
        ),
    ]


class AivisSpeechDefaultModelProperty(BaseModel):
    """
    AivisSpeech Engine が起動時に自動的にインストールする音声合成モデルの情報。
    """

    model_uuid: Annotated[uuid.UUID, Field(description="音声合成モデルの UUID")]
    latest_version: Annotated[str, Field(description="音声合成モデルの最新バージョン")]

    model_config = ConfigDict(protected_namespaces=())


class AivisSpeechForcedRemovalRulesResponse(BaseModel):
    """
    AivisSpeech Engine が適用すべき強制削除ルールの一覧を表すレスポンスモデル。
    """

    rules: Annotated[
        list[AivisSpeechForcedRemovalRule],
        Field(
            description="""
                強制削除対象の音声合成モデルの UUID とバージョン条件の組み合わせ。
                ルールは上から順に評価され、条件を満たした時点で音声合成モデルを削除する実装を想定している。
            """.strip().replace("\n", "<br>"),
            min_length=0,
        ),
    ]


class AivisSpeechForcedRemovalRule(BaseModel):
    """
    AivisSpeech Engine で強制的に削除すべき音声合成モデルのバージョン条件。
    """

    model_uuid: Annotated[
        uuid.UUID, Field(description="強制削除対象の音声合成モデルの UUID")
    ]
    version_specifiers: Annotated[
        list[str] | None,
        Field(
            description="""
                リスト内の条件式をカンマ区切りで連結し、Semver ライブラリの `match` 関数に渡して評価する。
                すべての条件を満たした場合に強制削除対象と判断する。
                例: `['>=1.1.0', '<2.0.0', '!=1.5.*']`
                全バージョンが対象の場合は `null` が指定される。
            """.strip().replace("\n", "<br>"),
        ),
    ] = None

    model_config = ConfigDict(protected_namespaces=())


class AivisSpeechEventRequest(BaseModel):
    """
    AivisSpeech Engine が任意のイベントを送信する際のリクエストボディ。
    """

    installation_uuid: Annotated[uuid.UUID, Field(description="インストール UUID")]
    event_type: Annotated[
        Literal["Startup", "StartupFailed"],
        Field(description="イベントタイプ"),
    ]
    runtime_environment: Annotated[
        AivisSpeechRuntimeEnvironment,
        Field(description="イベント発生時点の実行環境情報"),
    ]
    stack_trace: Annotated[
        str | None,
        Field(description="イベントに紐づくスタックトレース (起動失敗時のみ)"),
    ] = None


class AivisHubClient:
    """
    AivisHub API へのアクセスを集約し、HTTP 通信の詳細を隠蔽するクライアント。
    """

    # API ベース URL
    BASE_URL = "https://api.aivis-project.com/v1"

    # インストール UUID の保存先パス
    INSTALLATION_UUID_PATH = get_save_dir() / "installation_uuid.dat"

    @staticmethod
    def _request(
        method: str,
        path: str,
        request_body: BaseModel | None = None,
    ) -> httpx.Response:
        """
        共通の HTTP リクエスト処理を行う。

        Parameters
        ----------
        method : str
            HTTP メソッド（例: "GET", "POST"）
        path : str
            API パス（例: "/aivm-models/{uuid}"）
        request_body : BaseModel | None
            リクエストボディ（デフォルト: None）

        Returns
        -------
        httpx.Response
            HTTP レスポンスオブジェクト

        Raises
        ------
        httpx.HTTPError
            HTTP リクエストに失敗した場合
        """

        return httpx.request(
            method=method,
            url=f"{AivisHubClient.BASE_URL}/{path.lstrip('/')}",
            # 環境情報から生成したユーザーエージェントを設定
            headers={"User-Agent": generate_user_agent()},
            # リクエストボディが指定されていればシリアライズ化して設定
            json=(
                request_body.model_dump(mode="json")
                if request_body is not None
                else None
            ),
            # API が死んでる時に接続を待ち続けないようにタイムアウトを設定
            # 接続タイムアウト: 10秒 / 読み取りタイムアウト: 30秒
            timeout=httpx.Timeout(10.0, read=30.0),
            # リダイレクトを追跡する
            follow_redirects=True,
        )

    @staticmethod
    def fetch_default_models() -> list[AivisSpeechDefaultModelProperty]:
        """
        AivisSpeech Engine が起動時に自動的にインストールする音声合成モデルの一覧を取得する。
        ネットワークエラーなどで取得できなかった場合、フォールバックとしてハードコードされた値が返される。

        Returns
        -------
        list[AivisSpeechDefaultModelProperty]
            AivisSpeech Engine が起動時に自動的にインストールする音声合成モデルの一覧
        """

        # プリインストール対象の音声合成モデルの定義
        ## 何らかの理由で AivisHub API から情報が取得できなかった場合にフォールバックとして利用する
        ## 通常は AivisHub API から取得した最新の情報を優先する
        DEFAULT_MODEL_PROPERTIES: list[AivisSpeechDefaultModelProperty] = [
            AivisSpeechDefaultModelProperty(
                model_uuid=uuid.UUID("22e8ed77-94fe-4ef2-871f-a86f94e9a579"),
                latest_version="1.1.0",
            ),
            AivisSpeechDefaultModelProperty(
                model_uuid=uuid.UUID("a59cb814-0083-4369-8542-f51a29e72af7"),
                latest_version="1.2.0",
            ),
        ]

        try:
            response = AivisHubClient._request(
                method="GET",
                path="/aivisspeech/default-models",
            )
            response.raise_for_status()
            return AivisSpeechDefaultModelsResponse.model_validate(
                response.json()
            ).aivm_models
        except ValidationError as ex:
            logger.warning(
                "[AivisHubClient] Failed to parse default models.",
                exc_info=ex,
            )
            return DEFAULT_MODEL_PROPERTIES.copy()
        except httpx.HTTPStatusError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch default models from AivisHub. (HTTP Error: {ex.response.status_code})",
                exc_info=ex,
            )
            return DEFAULT_MODEL_PROPERTIES.copy()
        except httpx.TimeoutException as ex:
            logger.warning(
                "[AivisHubClient] Timeout while fetching default models from AivisHub.",
                exc_info=ex,
            )
            return DEFAULT_MODEL_PROPERTIES.copy()
        except httpx.HTTPError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch default models from AivisHub. ({type(ex).__name__}: {ex})",
                exc_info=ex,
            )
            return DEFAULT_MODEL_PROPERTIES.copy()

    @staticmethod
    def fetch_forced_removal_rules() -> list[AivisSpeechForcedRemovalRule]:
        """
        AivisSpeech Engine が強制削除対象とする音声合成モデルとバージョン条件のルールの一覧を取得する。
        ネットワークエラーなどで取得できなかった場合、フォールバックとしてハードコードされた値が返される。

        Returns
        -------
        list[AivisSpeechForcedRemovalRule]
            AivisSpeech Engine で強制的に削除すべき音声合成モデルのバージョン条件
        """

        # 音声合成モデルの強制削除ルールの定義
        ## 何らかの理由で AivisHub API から情報が取得できなかった場合にフォールバックとして利用する
        ## 通常は AivisHub API から取得した最新の情報を優先する
        FORCED_REMOVAL_RULES: list[AivisSpeechForcedRemovalRule] = [
            AivisSpeechForcedRemovalRule(
                model_uuid=uuid.UUID("a59cb814-0083-4369-8542-f51a29e72af7"),
                version_specifiers=["<1.1.0"],
            ),
            AivisSpeechForcedRemovalRule(
                model_uuid=uuid.UUID("1e5ae2f1-b3bc-4d90-b618-d891a3d7383b"),
                version_specifiers=None,
            ),
            AivisSpeechForcedRemovalRule(
                model_uuid=uuid.UUID("4cf3e1d8-5583-41a9-a554-b2d2cda2c569"),
                version_specifiers=None,
            ),
        ]

        try:
            response = AivisHubClient._request(
                method="GET",
                path="/aivisspeech/forced-removal-rules",
            )
            response.raise_for_status()
            return AivisSpeechForcedRemovalRulesResponse.model_validate(
                response.json()
            ).rules
        except ValidationError as ex:
            logger.warning(
                "[AivisHubClient] Failed to parse forced removal rules.",
                exc_info=ex,
            )
            return FORCED_REMOVAL_RULES.copy()
        except httpx.HTTPStatusError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch forced removal rules from AivisHub. (HTTP Error: {ex.response.status_code})",
                exc_info=ex,
            )
            return FORCED_REMOVAL_RULES.copy()
        except httpx.TimeoutException as ex:
            logger.warning(
                "[AivisHubClient] Timeout while fetching forced removal rules from AivisHub.",
                exc_info=ex,
            )
            return FORCED_REMOVAL_RULES.copy()
        except httpx.HTTPError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch forced removal rules from AivisHub. ({type(ex).__name__}: {ex})",
                exc_info=ex,
            )
            return FORCED_REMOVAL_RULES.copy()

    @staticmethod
    async def fetch_model_detail(
        aivm_model_uuid: uuid.UUID,
    ) -> AivmModelResponse | None:
        """
        指定された音声合成モデルの詳細情報を取得する。
        呼び出し元関数が非同期関数であるため、このメソッドのみ非同期関数として実装している。

        Parameters
        ----------
        aivm_model_uuid : uuid.UUID
            音声合成モデルの UUID

        Returns
        -------
        AivmModelResponse | None
            音声合成モデルの詳細情報。取得に失敗した場合やモデルが存在しない場合は None
        """

        # API リクエストを送信
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method="GET",
                    url=f"{AivisHubClient.BASE_URL}/aivm-models/{aivm_model_uuid}",
                    # 環境情報から生成したユーザーエージェントを設定
                    headers={"User-Agent": generate_user_agent()},
                    # API が死んでる時に接続を待ち続けないようにタイムアウトを設定
                    # 接続タイムアウト: 10秒 / 読み取りタイムアウト: 30秒
                    timeout=httpx.Timeout(10.0, read=30.0),
                    # リダイレクトを追跡する
                    follow_redirects=True,
                )
            if response.status_code == 404:
                # 404 の場合は単に当該モデルが AivisHub に公開されていないだけなので、エラーは出さずに None を返す
                return None
            response.raise_for_status()
            return AivmModelResponse.model_validate(response.json())
        except ValidationError as ex:
            logger.warning(
                "[AivisHubClient] Failed to parse model detail.",
                exc_info=ex,
            )
            return None
        except httpx.HTTPStatusError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch model detail from AivisHub. (HTTP Error: {ex.response.status_code})",
                exc_info=ex,
            )
            return None
        except httpx.TimeoutException as ex:
            logger.warning(
                "[AivisHubClient] Timeout while fetching model detail from AivisHub.",
                exc_info=ex,
            )
            return None
        except httpx.HTTPError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to fetch model detail from AivisHub. ({type(ex).__name__}: {ex})",
                exc_info=ex,
            )
            return None

    @staticmethod
    def _ensure_installation_uuid() -> uuid.UUID:
        """
        installation_uuid.dat を読み込み、存在しなければ新しく生成して保存する。

        Returns
        -------
        uuid.UUID
            インストール UUID

        Raises
        ------
        OSError
            installation_uuid.dat の保存に失敗した場合
        """

        # インストール UUID を保存するファイルが存在するか確認
        if AivisHubClient.INSTALLATION_UUID_PATH.exists():
            try:
                # 正常に読み込めればそれをそのまま返す
                content = AivisHubClient.INSTALLATION_UUID_PATH.read_text(
                    encoding="utf-8"
                ).strip()
                parsed = uuid.UUID(content)
                return parsed
            except (ValueError, OSError) as ex:
                # インストール UUID が正常に書き込まれていない場合はしょうがないので新しく生成
                logger.warning(
                    "[AivisHubClient] Invalid installation UUID detected. Regenerating.",
                    exc_info=ex,
                )

        # 新しいインストール UUID を生成
        installation_uuid = uuid.uuid4()

        # インストール UUID をファイルに保存
        try:
            get_save_dir().mkdir(parents=True, exist_ok=True)
            AivisHubClient.INSTALLATION_UUID_PATH.write_text(
                str(installation_uuid), encoding="utf-8"
            )
        except OSError as ex:
            logger.error(
                "[AivisHubClient] Failed to persist installation UUID.", exc_info=ex
            )
            raise ex
        return installation_uuid

    @staticmethod
    def send_event(
        event_type: Literal["Startup", "StartupFailed"],
        runtime_environment: AivisSpeechRuntimeEnvironment,
        stack_trace: str | None = None,
    ) -> None:
        """
        AivisSpeech Engine の起動イベントを記録する。
        イベント送信に失敗しても起動を継続できるよう、例外を発生させずにログのみ出力する。

        Parameters
        ----------
        event_type : Literal["Startup", "StartupFailed"]
            イベントタイプ
        runtime_environment : AivisSpeechRuntimeEnvironment
            実行環境情報
        stack_trace : str | None
            スタックトレース（エラー時のみ、デフォルト: None）
        """

        # ファイル保存先ディレクトリ内にある installation_uuid.dat を読み込み、存在しなければ新しく生成して保存
        installation_uuid = AivisHubClient._ensure_installation_uuid()

        # イベントを送信
        try:
            response = AivisHubClient._request(
                method="POST",
                path="/aivisspeech/events",
                request_body=AivisSpeechEventRequest(
                    installation_uuid=installation_uuid,
                    event_type=event_type,
                    runtime_environment=runtime_environment,
                    stack_trace=stack_trace,
                ),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as ex:
            logger.warning(
                f"[AivisHubClient] Failed to send event to AivisHub. (HTTP Error: {ex.response.status_code})",
                exc_info=ex,
            )
        except httpx.TimeoutException as ex:
            logger.warning(
                "[AivisHubClient] Timeout while sending event to AivisHub.",
                exc_info=ex,
            )
        except Exception as ex:
            logger.warning(
                f"[AivisHubClient] Failed to send event to AivisHub. ({type(ex).__name__}: {ex})",
                exc_info=ex,
            )
