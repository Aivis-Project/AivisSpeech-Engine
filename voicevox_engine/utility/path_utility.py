"""パスに関する utility"""

import sys
from datetime import datetime
from pathlib import Path

from platformdirs import user_data_dir

from voicevox_engine.logging import logger
from voicevox_engine.utility.runtime_utility import is_development


def engine_root() -> Path:
    """エンジンのルートディレクトリを指すパスを取得する。"""
    if is_development():
        # git レポジトリのルートを指している
        root_dir = Path(__file__).parents[2]
    else:
        root_dir = Path(sys.executable).parent

    return root_dir.resolve(strict=True)


def resource_root() -> Path:
    """リソースのルートディレクトリを指すパスを取得する。"""
    return engine_root() / "resources"


def engine_manifest_path() -> Path:
    """エンジンマニフェストのパスを取得する。"""
    # NOTE: VOICEVOX API の規定によりエンジンマニフェストファイルは必ず `<engine_root>/engine_manifest.json` に存在する
    return engine_root() / "engine_manifest.json"


def get_save_dir() -> Path:
    """ファイルの保存先ディレクトリを指すパスを取得する。"""
    # FIXME: ファイル保存場所をエンジン固有のIDが入ったものにする
    if is_development():
        app_name = "AivisSpeech-Engine-Dev"
    else:
        app_name = "AivisSpeech-Engine"
    return Path(user_data_dir(app_name, appauthor=False, roaming=True))


def ensure_directory_exists(directory: Path, *, create_parents: bool = False) -> None:
    """指定されたディレクトリが存在することを確認し、競合するファイルがあればリネームする。"""
    try:
        directory.mkdir(parents=create_parents, exist_ok=True)
        return
    except FileExistsError as ex:
        if directory.exists():
            is_directory = directory.is_dir()
            if is_directory is True:
                return
            conflict_path = _generate_conflict_path(directory)
            directory.rename(conflict_path)
            logger.warning(
                "Renamed conflicting file %s to %s before creating directory.",
                directory,
                conflict_path,
            )
            directory.mkdir(parents=create_parents, exist_ok=True)
            return
        raise ex
    except OSError as ex:
        if directory.exists() and directory.is_dir():
            return
        raise ex


def _generate_conflict_path(directory: Path) -> Path:
    """競合するファイルのリネーム先を生成する。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = f"{directory.name}.conflict-{timestamp}"
    candidate = directory.with_name(base_name)
    counter = 1
    while candidate.exists():
        candidate = directory.with_name(
            f"{directory.name}.conflict-{timestamp}-{counter}"
        )
        counter += 1
    return candidate
