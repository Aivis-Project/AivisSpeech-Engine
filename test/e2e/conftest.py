"""E2E テスト共通の pytest 用 fixtures。"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.application import generate_app
from voicevox_engine.core.core_initializer import initialize_cores
from voicevox_engine.engine_manifest import load_manifest
from voicevox_engine.library.library_manager import LibraryManager
from voicevox_engine.preset.preset_manager import PresetManager
from voicevox_engine.setting.setting_manager import SettingHandler
from voicevox_engine.tts_pipeline.song_engine import make_song_engines_from_cores
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager
from voicevox_engine.user_dict.user_dict_manager import UserDictionary
from voicevox_engine.utility.aivishub_client import (
    AivisHubClient,
    AivisSpeechDefaultModelProperty,
    AivisSpeechForcedRemovalRule,
    AivmModelResponse,
)
from voicevox_engine.utility.core_version_utility import MOCK_CORE_VERSION
from voicevox_engine.utility.path_utility import engine_manifest_path
from voicevox_engine.utility.user_agent_utility import generate_user_agent

# デフォルトモデルの UUID
_DEFAULT_MODEL_UUIDS = [
    "22e8ed77-94fe-4ef2-871f-a86f94e9a579",  # コハク
    "a59cb814-0083-4369-8542-f51a29e72af7",  # まお
]


class _NoopAivisHubClient(AivisHubClient):
    """ネットワーク呼び出しを一切行わないテスト用 AivisHubClient。"""

    def __init__(self, installation_uuid_path: Path | None = None) -> None:
        """
        テスト用 AivisHubClient を初期化する。

        Parameters
        ----------
        installation_uuid_path : Path | None
            インストール UUID の保存先パス（テスト用一時ディレクトリ下を推奨）。
        """

        super().__init__(installation_uuid_path=installation_uuid_path)

    def fetch_default_models(self) -> list[AivisSpeechDefaultModelProperty]:
        return []

    def fetch_forced_removal_rules(self) -> list[AivisSpeechForcedRemovalRule]:
        return []

    async def fetch_model_detail(
        self,
        aivm_model_uuid: uuid.UUID,
    ) -> AivmModelResponse | None:
        return None

    def send_event(self, *args: Any, **kwargs: Any) -> None:
        pass


class _DefaultModelAivisHubClient(_NoopAivisHubClient):
    """デフォルトモデルの UUID のみを返すテスト用 AivisHubClient。"""

    def fetch_default_models(self) -> list[AivisSpeechDefaultModelProperty]:
        return [
            AivisSpeechDefaultModelProperty(
                model_uuid=uuid.UUID(model_uuid),
                # 既にインストール済みの AIVMX を利用するため、バージョンは低く設定して再 DL を防ぐ
                latest_version="0.0.0",
            )
            for model_uuid in _DEFAULT_MODEL_UUIDS
        ]


@pytest.fixture(scope="session")
def _shared_default_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    セッション中に 1 回だけデフォルトモデルの AIVMX ファイルを共有ディレクトリに配置して返す。
    全テストがこのディレクトリを読み取り専用で共有することで、テストごとのコピーを不要にする。

    開発環境のモデルディレクトリに AIVMX ファイルが存在すればそこからコピーする。
    存在しない場合（CI 環境など）は AivisHub から DL にフォールバックする。
    """

    from voicevox_engine.utility.path_utility import get_save_dir

    shared_dir = tmp_path_factory.mktemp("shared_models")
    for model_uuid in _DEFAULT_MODEL_UUIDS:
        target_path = shared_dir / f"{model_uuid}.aivmx"
        # 開発環境のモデルディレクトリからのコピーを試みる
        local_path = get_save_dir() / "Models" / f"{model_uuid}.aivmx"
        if local_path.exists() is True:
            shutil.copyfile(local_path, target_path)
        else:
            # ローカルにない場合は AivisHub から DL（CI 環境など）
            download_url = f"{AivisHubClient.BASE_URL}/aivm-models/{model_uuid}/download?model_type=AIVMX"
            response = httpx.get(
                download_url,
                headers={"User-Agent": generate_user_agent()},
                timeout=httpx.Timeout(10.0, read=120.0),
                follow_redirects=True,
            )
            response.raise_for_status()
            target_path.write_bytes(response.content)
    return shared_dir


def _build_app_params(
    tmp_path: Path,
    aivishub_client: AivisHubClient | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """
    generate_app に渡す引数辞書を構築する共通ヘルパー。
    AivmManager の構築も含め、テスト用アプリに必要な全依存を隔離して生成する。

    Parameters
    ----------
    tmp_path : Path
        テスト用一時ディレクトリ。
    aivishub_client : AivisHubClient | None
        テスト用の AivisHubClient インスタンス。
        None の場合はネットワーク呼び出しを行わない _NoopAivisHubClient が使われる。
    models_dir : Path | None
        AIVMX ファイルのインストール先ディレクトリ。None の場合は tmp_path / "Models" が使われる。

    Returns
    -------
    dict[str, Any]
        generate_app に渡す引数辞書。
    """

    if aivishub_client is None:
        aivishub_client = _NoopAivisHubClient(
            installation_uuid_path=tmp_path / "installation_uuid.dat",
        )
    if models_dir is None:
        models_dir = tmp_path / "Models"

    # aivishub_client にテスト用クライアントを渡すことでネットワーク呼び出しを防止し、
    # cache_file_path に tmp_path 配下のパスを渡すことで実データディレクトリへのキャッシュ書き込みを防止する
    ## models_dir が指定されていない場合は tmp_path 配下に空のモデルディレクトリを作成する
    aivm_manager = AivmManager(
        models_dir,
        aivishub_client=aivishub_client,
        cache_file_path=tmp_path / "aivm_infos_cache.json",
        is_background_scan_enabled=False,
    )

    core_manager = initialize_cores(use_gpu=False, enable_mock=True, cpu_num_threads=1)
    tts_engines = TTSEngineManager()
    tts_engines.register_engine(
        # BERT モデルのキャッシュディレクトリは実データディレクトリのものをそのまま利用する
        ## BERT モデルは ~650MB あり tmp_path にリダイレクトすると毎回ダウンロードが走るため、
        ## 読み取り専用で破壊リスクのない実キャッシュを許容する
        StyleBertVITS2TTSEngine(aivm_manager, False, False),
        MOCK_CORE_VERSION,
    )
    song_engines = make_song_engines_from_cores(core_manager)
    setting_loader = SettingHandler(tmp_path / "not_exist.yaml")

    # テスト用に隔離されたプリセットを生成する
    preset_path = tmp_path / "presets.yaml"
    _generate_preset(preset_path)
    preset_manager = PresetManager(preset_path)

    # テスト用に隔離されたユーザー辞書を生成する
    # デフォルトユーザー辞書は Windows の pytest 実行時には適用されないため、
    # E2E テストのアクセント句生成が OS ごとに変わらないよう空の辞書ディレクトリを使う
    default_dict_dir_path = tmp_path / "default_dictionaries"
    default_dict_dir_path.mkdir()
    user_dict = UserDictionary(
        default_dict_dir_path=default_dict_dir_path,
        user_dict_path=_generate_user_dict(tmp_path),
    )

    engine_manifest = load_manifest(engine_manifest_path())
    # LibraryManager も tmp_path 配下で隔離する
    library_manager = LibraryManager(
        tmp_path / "libraries",
        engine_manifest.supported_vvlib_manifest_version,
        engine_manifest.brand_name,
        engine_manifest.name,
        engine_manifest.uuid,
    )

    return {
        "tts_engines": tts_engines,
        "song_engines": song_engines,
        "aivm_manager": aivm_manager,
        "core_manager": core_manager,
        "setting_loader": setting_loader,
        "preset_manager": preset_manager,
        "user_dict": user_dict,
        "engine_manifest": engine_manifest,
        "library_manager": library_manager,
    }


@pytest.fixture
def app_params(tmp_path: Path) -> dict[str, Any]:
    """`generate_app` の全ての引数を生成する。"""
    return _build_app_params(tmp_path)


@pytest.fixture
def app(app_params: dict[str, Any]) -> FastAPI:
    """app インスタンスを生成する。"""
    return generate_app(**app_params)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """HTTP リクエストを AivisSpeech Engine へ送信するクライアントを生成する。"""
    return TestClient(app)


@pytest.fixture
def client_with_default_model(
    tmp_path: Path,
    _shared_default_models_dir: Path,
) -> TestClient:
    """デフォルトモデルがインストール済みのフルアプリ TestClient を生成する。"""

    # セッションで共有されたモデルディレクトリを直接参照する（テストごとのコピーは行わない）
    # デフォルトモデルの UUID を返す AivisHubClient を利用し、モデルをデフォルトとしてマークさせる
    ## latest_version が "0.0.0" なので再 DL は発生しない
    aivishub_client = _DefaultModelAivisHubClient(
        installation_uuid_path=tmp_path / "installation_uuid.dat",
    )
    return TestClient(
        generate_app(
            **_build_app_params(tmp_path, aivishub_client, _shared_default_models_dir)
        )
    )


def _generate_preset(preset_path: Path) -> None:
    """指定パス下にプリセットファイルを生成する。"""
    contents = [
        {
            "id": 1,
            "name": "サンプルプリセット",
            "speaker_uuid": "7ffcb7ce-00ec-4bdc-82cd-45a8889e43ff",
            "style_id": 0,
            "speedScale": 1.0,
            "intonationScale": 1.0,
            "tempoDynamicsScale": 1.0,
            "pitchScale": 0.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1,
        }
    ]
    with open(preset_path, mode="w", encoding="utf-8") as f:
        yaml.safe_dump(contents, f, allow_unicode=True, sort_keys=False)


def _generate_user_dict(dir_path: Path) -> Path:
    """指定されたディレクトリ下にユーザー辞書ファイルを生成し、生成されたファイルのパスを返す。"""
    contents = {
        "a89596ad-caa8-4f4e-8eb3-3d2261c798fd": {
            "surface": "テスト１",
            "context_id": 1348,
            "priority": 5,
            "part_of_speech": "名詞",
            "part_of_speech_detail_1": "固有名詞",
            "part_of_speech_detail_2": "一般",
            "part_of_speech_detail_3": "*",
            "inflectional_type": "*",
            "inflectional_form": "*",
            "stem": ["*"],
            "yomi": ["テストイチ"],
            "pronunciation": ["テストイチ"],
            "accent_type": [1],
            "accent_associative_rule": "*",
        },
        "c89596ad-caa8-4f4e-8eb3-3d2261c798fd": {
            "surface": "テスト２",
            "context_id": 1348,
            "priority": 5,
            "part_of_speech": "名詞",
            "part_of_speech_detail_1": "固有名詞",
            "part_of_speech_detail_2": "一般",
            "part_of_speech_detail_3": "*",
            "inflectional_type": "*",
            "inflectional_form": "*",
            "stem": ["*"],
            "yomi": ["テストニ"],
            "pronunciation": ["テストニ"],
            "accent_type": [1],
            "accent_associative_rule": "*",
        },
    }
    contents_json = json.dumps(contents, ensure_ascii=False)

    file_path = dir_path / "user_dict_for_test.json"
    file_path.write_text(contents_json, encoding="utf-8")

    return file_path
