"""Sentry 送信前フィルタリング用ユーティリティのテスト。"""

from __future__ import annotations

import pytest
from sentry_sdk.types import Event, Hint

from voicevox_engine.utility.sentry_utility import filter_sentry_event


def _generate_exception_event(exception_type: str, exception_value: str) -> Event:
    """
    Sentry SDK が before_send に渡す例外イベントを生成する。

    Args:
        exception_type (str): Sentry の exception.values に入る例外型名
        exception_value (str): Sentry の exception.values に入る例外メッセージ

    Returns
    -------
        Event: 例外情報を含む Sentry イベント
    """

    return {
        "exception": {
            "values": [
                {
                    "type": exception_type,
                    "value": exception_value,
                }
            ]
        }
    }


def _generate_hint(exception: BaseException) -> Hint:
    """
    Sentry SDK が before_send に渡す exc_info 付きヒントを生成する。

    Args:
        exception (BaseException): ヒントへ入れる例外

    Returns
    -------
        Hint: exc_info を含む Sentry ヒント
    """

    return {"exc_info": (type(exception), exception, exception.__traceback__)}


@pytest.mark.parametrize(
    ("event", "hint"),
    [
        (
            _generate_exception_event(
                "ConnectionResetError",
                "[WinError 10054] 既存の接続はリモート ホストに強制的に切断されました。",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 48] error while attempting to bind on address "
                "('0.0.0.0', 10101): address already in use",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 10048] error while attempting to bind on address "
                "('127.0.0.1', 10101): only one usage of each socket address is normally permitted",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 10048] error while attempting to bind on address "
                "('::1', 10101, 0, 0): 通常、各ソケット アドレスに対してプロトコル、"
                "ネットワーク アドレス、またはポートのどれか 1 つのみを使用できます。",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 10048] error while attempting to bind on address "
                "('127.0.0.1', 10101): 각 소켓 주소(프로토콜/네트워크 주소/포트)는 하나만 사용할 수 있습니다",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 13] error while attempting to bind on address "
                "('::1', 10101, 0, 0): アクセス許可で禁じられた方法でソケットにアクセスしようとしました。",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeError",
                "Failed to CreateFileW for "
                "C:\\Users\\user\\AppData\\Local\\AivisSpeech-Engine\\user.dict_compiled-abc.tmp",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeException",
                "Non-zero status code returned while running Conv node. "
                "Error in execution: bad allocation",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeError",
                "onnxruntime::BFCArena::AllocateRawInternal "
                "Failed to allocate memory for requested buffer of size 38825944064",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeException",
                "DmlExecutionProvider failed at PooledUploadHeap.cpp with HRESULT 887A0005. "
                "The GPU device instance has been suspended.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeError",
                "CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR ; expr=cudnnCreate(&cudnn_handle_);",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "OSError",
                "[Errno 28] No space left on device",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "404: スタイル 2 は存在しません。",
            ),
            {},
        ),
        (
            {"message": "Model 7ffcb7ce-00ec-4bdc-82cd-45a8889e43ff is not installed."},
            {},
        ),
        ({"message": "Model recommended is not installed."}, {}),
        ({"message": "Speaker morioki-uuid is not installed."}, {}),
        (
            _generate_exception_event(
                "ValidationError",
                "1 validation error for AudioQuery\n"
                "accent_phrases.0.moras.0.text\n"
                "  Value error, mora text must be one katakana mora",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "NoSuchFile",
                "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from "
                "C:\\Users\\user\\AppData\\Roaming\\AivisSpeech-Engine\\BertModelCaches\\"
                "models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\\"
                "snapshots\\hash\\model_fp16.onnx failed. File doesn't exist",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "LocalEntryNotFoundError",
                "An error happened while trying to locate the file on the Hub and "
                "we cannot find the requested files in the local cache.",
            ),
            {},
        ),
        ({}, _generate_hint(MemoryError("Unable to allocate 1024 bytes"))),
    ],
)
def test_filter_sentry_event_drops_known_unrecoverable_errors(
    event: Event, hint: Hint
) -> None:
    """`filter_sentry_event()` は既知の破棄対象エラーを送信しない。"""
    # Tests
    assert filter_sentry_event(event, hint) is None


@pytest.mark.parametrize(
    ("event", "hint"),
    [
        (
            _generate_exception_event(
                "ValueError",
                "Input must be katakana only: サ*チャン",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "UnicodeDecodeError",
                "'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "AssertionError",
                "34 != 83",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "InvalidPhoneError",
                "Invalid phone: e",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValueError",
                "Style ID 100 not found in hyper parameters.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeError",
                "The browser zoom level was changed in the control room",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValidationError",
                "1 validation error for Setting\n"
                "allow_origin\n"
                "  Input should be a valid string [type=string_type, input_value=[], input_type=list]",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValidationError",
                "1 validation error for ParseKanaBadRequest\n"
                "error_args.position\n"
                "  Input should be a valid string [type=string_type, input_value=1, input_type=int]",
            ),
            {},
        ),
    ],
)
def test_filter_sentry_event_keeps_text_processing_errors(
    event: Event, hint: Hint
) -> None:
    """`filter_sentry_event()` は音素・g2p・テキスト前処理の疑いがあるエラーを残す。"""
    # Inputs
    original_event = event.copy()

    # Outputs
    filtered_event = filter_sentry_event(event, hint)

    # Tests
    assert filtered_event == original_event


def test_filter_sentry_event_keeps_unknown_event() -> None:
    """`filter_sentry_event()` は未知のイベントをそのまま残す。"""
    # Inputs
    event: Event = {"message": "Unexpected production error"}

    # Outputs
    filtered_event = filter_sentry_event(event, {})

    # Tests
    assert filtered_event == event


def test_filter_sentry_event_uses_outer_exception_type_without_hint() -> None:
    """`filter_sentry_event()` は hint がない場合でも外側の例外型で判定する。"""
    # Inputs
    event: Event = {
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": "inner error",
                },
                {
                    "type": "MemoryError",
                    "value": "",
                },
            ]
        }
    }

    # Tests
    assert filter_sentry_event(event, {}) is None
