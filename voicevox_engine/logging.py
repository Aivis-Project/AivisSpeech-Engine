"""ロギングの設定"""

import logging
import logging.config
from typing import Any

from voicevox_engine.utility.path_utility import ensure_directory_exists, get_save_dir

ENGINE_LOG_DIR = get_save_dir() / "Logs"
ENGINE_LOG_PATH = ENGINE_LOG_DIR / "AivisSpeech-Engine.log"
ENGINE_ACCESS_LOG_PATH = ENGINE_LOG_DIR / "AivisSpeech-Engine-Access.log"

# ログディレクトリを作成
ensure_directory_exists(ENGINE_LOG_DIR, create_parents=True)

# 前回の起動時に作成したログファイルを削除
for log_file in [ENGINE_LOG_PATH, ENGINE_ACCESS_LOG_PATH]:
    try:
        if log_file.exists():
            log_file.unlink()
    except PermissionError:
        pass

# Uvicorn のロギング設定
## この dictConfig を Uvicorn の起動時に渡す
## Uvicorn のもとの dictConfig を参考にして作成した
## ref: https://github.com/encode/uvicorn/blob/0.18.2/uvicorn/config.py#L95-L126
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # サーバーログ用のログフォーマッター
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "datefmt": "%Y/%m/%d %H:%M:%S",
            "format": "[%(asctime)s.%(msecs)03d] %(levelprefix)s %(message)s",
        },
        "default_file": {
            "()": "uvicorn.logging.DefaultFormatter",
            "datefmt": "%Y/%m/%d %H:%M:%S",
            "format": "[%(asctime)s.%(msecs)03d] %(levelprefix)s %(message)s",
            "use_colors": False,  # ANSI エスケープシーケンスを出力しない
        },
        # アクセスログ用のログフォーマッター
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "datefmt": "%Y/%m/%d %H:%M:%S",
            "format": '[%(asctime)s.%(msecs)03d] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
        "access_file": {
            "()": "uvicorn.logging.AccessFormatter",
            "datefmt": "%Y/%m/%d %H:%M:%S",
            "format": '[%(asctime)s.%(msecs)03d] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": False,  # ANSI エスケープシーケンスを出力しない
        },
    },
    "handlers": {
        # サーバーログは標準出力と logs/AivisSpeech-Engine.log の両方に出力する
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "default_file": {
            "formatter": "default_file",
            "class": "logging.FileHandler",
            "filename": str(ENGINE_LOG_PATH),
            "mode": "a",
            "encoding": "utf-8",
        },
        # アクセスログは標準出力と logs/AivisSpeech-Engine-Access.log の両方に出力する
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access_file": {
            "formatter": "access_file",
            "class": "logging.FileHandler",
            "filename": str(ENGINE_ACCESS_LOG_PATH),
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default", "default_file"]},
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["access", "access_file"],
            "propagate": False,
        },
        "uvicorn.error": {"level": "INFO"},
    },
}

# Uvicorn を起動する前に Uvicorn のロガーを使えるようにする
logging.config.dictConfig(LOGGING_CONFIG)

# ロガーを取得
logger = logging.getLogger("uvicorn")
