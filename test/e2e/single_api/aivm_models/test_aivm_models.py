"""/aivm_models API のテスト。"""

import copy
import io
import uuid
from pathlib import Path
from typing import Any, cast

import aivmlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import hash_long_string
from voicevox_engine.aivm_infos_repository import AivmInfosRepository
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.routers.aivm_models import generate_aivm_models_router
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager
from voicevox_engine.utility.aivishub_client import AivisHubClient
from voicevox_engine.utility.core_version_utility import MOCK_CORE_VERSION

_NON_EXISTENT_AIVM_MODEL_UUID = "00000000-0000-4000-8000-000000000000"
_AIVISHUB_DOWNLOAD_TEST_MODEL_UUID = "22e8ed77-94fe-4ef2-871f-a86f94e9a579"
_AIVMX_FILE_INSTALL_TEST_MODEL_UUID = "00000000-0000-4000-8000-000000000001"


class _AivmModelsRouterStyleBertVITS2TTSEngine(StyleBertVITS2TTSEngine):
    """`/aivm_models` API Router の状態変更だけを扱う StyleBertVITS2TTSEngine。"""

    def __init__(self, aivm_manager: AivmManager) -> None:
        self.aivm_manager = aivm_manager

    def load_model(self, aivm_model_uuid: str) -> Any:
        """モデルロード状態だけを AivmManager に反映する。"""

        self.aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=True)
        return None

    def unload_model(self, aivm_model_uuid: str) -> None:
        """モデルアンロード状態だけを AivmManager に反映する。"""

        self.aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=False)


@pytest.fixture
def isolated_aivm_models_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """
    実ユーザーのモデル保存先に触れない `/aivm_models` 専用クライアントを生成する。

    Parameters
    ----------
    tmp_path : Path
        pytest が生成するテスト専用一時ディレクトリ。
    monkeypatch : pytest.MonkeyPatch
        AivisHub の起動時処理とキャッシュパスを隔離するための monkeypatch fixture。

    Returns
    -------
    TestClient
        `/aivm_models` API Router だけを持つテストクライアント。
    """

    monkeypatch.setattr(
        AivmInfosRepository,
        "CACHE_FILE_PATH",
        tmp_path / "aivm_infos_cache.json",
    )
    monkeypatch.setattr(
        AivisHubClient,
        "fetch_default_models",
        staticmethod(lambda: []),
    )
    monkeypatch.setattr(
        AivisHubClient,
        "fetch_forced_removal_rules",
        staticmethod(lambda: []),
    )

    async def fetch_model_detail(aivm_model_uuid: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(
        AivisHubClient,
        "fetch_model_detail",
        staticmethod(fetch_model_detail),
    )

    aivm_manager = AivmManager(tmp_path / "Models")
    tts_engines = TTSEngineManager()
    tts_engines.register_engine(
        _AivmModelsRouterStyleBertVITS2TTSEngine(aivm_manager),
        MOCK_CORE_VERSION,
    )

    app = FastAPI()

    async def verify_mutability() -> None:
        return None

    app.include_router(
        generate_aivm_models_router(
            aivm_manager=aivm_manager,
            tts_engines=tts_engines,
            verify_mutability=verify_mutability,
        )
    )
    return TestClient(app)


def _normalize_aivm_info_for_snapshot(aivm_info: dict[str, Any]) -> dict[str, Any]:
    """
    環境ごとに変わる値を置き換え、AIVM 情報を snapshot に適した形へ変換する。

    Parameters
    ----------
    aivm_info : dict[str, Any]
        `/aivm_models` API から返された AIVM 情報。

    Returns
    -------
    dict[str, Any]
        snapshot 比較に利用する AIVM 情報。
    """

    normalized_aivm_info = aivm_info.copy()
    normalized_aivm_info["file_path"] = Path(aivm_info["file_path"]).name
    normalized_aivm_info["file_size"] = "<positive>" if aivm_info["file_size"] > 0 else aivm_info["file_size"]  # fmt: skip
    normalized_aivm_info["latest_version"] = "<version>"
    normalized_aivm_info["is_private_model"] = "<bool>"
    normalized_aivm_info["is_update_available"] = "<bool>"
    return cast(dict[str, Any], hash_long_string(normalized_aivm_info))


def _normalize_aivm_infos_for_snapshot(
    aivm_infos: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    デフォルト AIVM 情報一覧を snapshot に適した形へ変換する。

    Parameters
    ----------
    aivm_infos : dict[str, dict[str, Any]]
        `/aivm_models` API から返された AIVM 情報一覧。

    Returns
    -------
    dict[str, dict[str, Any]]
        snapshot 比較に利用するデフォルト AIVM 情報一覧。
    """

    return {
        aivm_model_uuid: _normalize_aivm_info_for_snapshot(aivm_info)
        for aivm_model_uuid, aivm_info in aivm_infos.items()
        if aivm_info["is_default_model"] is True
    }


def _get_first_default_aivm_info(client: TestClient) -> tuple[str, dict[str, Any]]:
    """
    インストール済み AIVM 情報のうち、先頭のデフォルトモデル 1 件を取得する。

    Parameters
    ----------
    client : TestClient
        AivisSpeech Engine へ HTTP リクエストを送信するクライアント。

    Returns
    -------
    tuple[str, dict[str, Any]]
        デフォルト AIVM モデル UUID と AIVM 情報。
    """

    response = client.get("/aivm_models")
    assert response.status_code == 200

    aivm_infos = response.json()
    assert len(aivm_infos) > 0

    for aivm_model_uuid, aivm_info in aivm_infos.items():
        if aivm_info["is_default_model"] is True:
            return aivm_model_uuid, aivm_info

    raise AssertionError("Default AIVM model is not installed.")


def _build_aivmx_file_install_variant(
    installed_aivm_info: dict[str, Any],
) -> bytes:
    """
    URL install 済み AIVMX から、別 UUID の file install 用 AIVMX を生成する。

    Parameters
    ----------
    installed_aivm_info : dict[str, Any]
        `/aivm_models/{uuid}` API から返された、URL install 済み AIVM 情報。

    Returns
    -------
    bytes
        `/aivm_models/install` の `file` に渡す AIVMX バイト列。
    """

    with open(installed_aivm_info["file_path"], mode="rb") as installed_aivmx_file:
        aivm_metadata = aivmlib.read_aivmx_metadata(installed_aivmx_file)
        variant_aivm_metadata = copy.deepcopy(aivm_metadata)
        variant_aivm_metadata.manifest.uuid = uuid.UUID(
            _AIVMX_FILE_INSTALL_TEST_MODEL_UUID
        )
        variant_aivm_metadata.manifest.name = "AIVMX file install test model"

        return cast(
            bytes,
            aivmlib.write_aivmx_metadata(
                installed_aivmx_file,
                variant_aivm_metadata,
            ),
        )  # pyright: ignore[reportUnnecessaryCast]


def test_get_aivm_models_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """デフォルトモデルがインストール済み一覧に含まれ、AIVM 情報が API 互換形式で返ることを確認する。"""

    response = client.get("/aivm_models")
    assert response.status_code == 200

    aivm_infos = response.json()
    assert len(aivm_infos) > 0
    assert any(aivm_info["is_default_model"] is True for aivm_info in aivm_infos.values())  # fmt: skip
    assert snapshot_json == _normalize_aivm_infos_for_snapshot(aivm_infos)


def test_get_aivm_model_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """インストール済みデフォルトモデルの UUID を指定すると、単一の AIVM 情報を取得できることを確認する。"""

    aivm_model_uuid, _ = _get_first_default_aivm_info(client)

    response = client.get(f"/aivm_models/{aivm_model_uuid}")
    assert response.status_code == 200
    assert snapshot_json == _normalize_aivm_info_for_snapshot(response.json())


def test_get_aivm_model_404(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """存在しない AIVM モデル UUID を指定すると、404 とエラー内容が返ることを確認する。"""

    response = client.get(f"/aivm_models/{_NON_EXISTENT_AIVM_MODEL_UUID}")
    assert response.status_code == 404
    assert snapshot_json == response.json()


def test_post_install_aivm_model_without_file_or_url_422(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """AIVMX file と URL をどちらも指定しない install リクエストが 422 になることを確認する。"""

    response = client.post("/aivm_models/install")
    assert response.status_code == 422
    assert snapshot_json == response.json()


def test_post_load_and_unload_aivm_model_204(client: TestClient) -> None:
    """インストール済みデフォルトモデルの load / unload で `is_loaded` が切り替わることを確認する。"""

    aivm_model_uuid, _ = _get_first_default_aivm_info(client)

    load_response = client.post(f"/aivm_models/{aivm_model_uuid}/load")
    assert load_response.status_code == 204

    loaded_aivm_info_response = client.get(f"/aivm_models/{aivm_model_uuid}")
    assert loaded_aivm_info_response.status_code == 200
    assert loaded_aivm_info_response.json()["is_loaded"] is True

    unload_response = client.post(f"/aivm_models/{aivm_model_uuid}/unload")
    assert unload_response.status_code == 204

    unloaded_aivm_info_response = client.get(f"/aivm_models/{aivm_model_uuid}")
    assert unloaded_aivm_info_response.status_code == 200
    assert unloaded_aivm_info_response.json()["is_loaded"] is False


def test_post_load_aivm_model_404(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """存在しない AIVM モデル UUID を load しようとすると、404 とエラー内容が返ることを確認する。"""

    response = client.post(f"/aivm_models/{_NON_EXISTENT_AIVM_MODEL_UUID}/load")
    assert response.status_code == 404
    assert snapshot_json == response.json()


def test_post_update_aivm_model_without_update_422(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """更新可能フラグが立っていないデフォルトモデルを update しようとすると、422 になることを確認する。"""

    aivm_model_uuid, aivm_info = _get_first_default_aivm_info(client)
    assert aivm_info["is_update_available"] is False

    response = client.post(f"/aivm_models/{aivm_model_uuid}/update")
    assert response.status_code == 422
    assert snapshot_json == response.json()


def test_delete_default_aivm_model_400(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """デフォルトモデルを uninstall しようとすると、保護されて 400 になることを確認する。"""

    response = client.get("/aivm_models")
    assert response.status_code == 200

    default_aivm_model_uuid = next(
        aivm_model_uuid
        for aivm_model_uuid, aivm_info in response.json().items()
        if aivm_info["is_default_model"] is True
    )

    response = client.delete(f"/aivm_models/{default_aivm_model_uuid}/uninstall")
    assert response.status_code == 400
    assert snapshot_json == response.json()


def test_delete_aivm_model_404(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """存在しない AIVM モデル UUID を uninstall しようとすると、404 とエラー内容が返ることを確認する。"""

    response = client.delete(f"/aivm_models/{_NON_EXISTENT_AIVM_MODEL_UUID}/uninstall")
    assert response.status_code == 404
    assert snapshot_json == response.json()


def test_post_install_aivm_model_from_aivishub_url_and_file_then_delete_204(
    isolated_aivm_models_client: TestClient,
) -> None:
    """AivisHub API からの URL install、AIVMX file install、非デフォルトモデル uninstall が一連で成功することを確認する。"""

    download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{_AIVISHUB_DOWNLOAD_TEST_MODEL_UUID}/download?model_type=AIVMX"
    url_install_response = isolated_aivm_models_client.post(
        "/aivm_models/install",
        data={"url": download_url},
    )
    assert url_install_response.status_code == 204

    installed_response = isolated_aivm_models_client.get(
        f"/aivm_models/{_AIVISHUB_DOWNLOAD_TEST_MODEL_UUID}"
    )
    assert installed_response.status_code == 200
    installed_aivm_info = installed_response.json()
    assert installed_aivm_info["file_size"] > 0
    assert installed_aivm_info["is_default_model"] is False

    variant_aivmx = _build_aivmx_file_install_variant(installed_aivm_info)
    file_install_response = isolated_aivm_models_client.post(
        "/aivm_models/install",
        files={
            "file": (
                f"{_AIVMX_FILE_INSTALL_TEST_MODEL_UUID}.aivmx",
                io.BytesIO(variant_aivmx),
                "application/octet-stream",
            )
        },
    )
    assert file_install_response.status_code == 204

    file_installed_response = isolated_aivm_models_client.get(
        f"/aivm_models/{_AIVMX_FILE_INSTALL_TEST_MODEL_UUID}"
    )
    assert file_installed_response.status_code == 200
    assert file_installed_response.json()["file_size"] > 0

    delete_response = isolated_aivm_models_client.delete(
        f"/aivm_models/{_AIVMX_FILE_INSTALL_TEST_MODEL_UUID}/uninstall"
    )
    assert delete_response.status_code == 204

    deleted_response = isolated_aivm_models_client.get(
        f"/aivm_models/{_AIVMX_FILE_INSTALL_TEST_MODEL_UUID}"
    )
    assert deleted_response.status_code == 404

    remaining_response = isolated_aivm_models_client.get(
        f"/aivm_models/{_AIVISHUB_DOWNLOAD_TEST_MODEL_UUID}"
    )
    assert remaining_response.status_code == 200


def test_post_install_broken_aivmx_file_422(
    isolated_aivm_models_client: TestClient,
    snapshot_json: SnapshotAssertion,
) -> None:
    """AIVMX ではない壊れた file を install しようとすると、422 とエラー内容が返ることを確認する。"""

    response = isolated_aivm_models_client.post(
        "/aivm_models/install",
        files={
            "file": (
                "broken.aivmx",
                io.BytesIO(b"broken aivmx"),
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 422
    assert snapshot_json == response.json()
