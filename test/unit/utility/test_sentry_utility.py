"""Sentry 送信前フィルタリング用ユーティリティのテスト。"""

from __future__ import annotations

import pytest
from sentry_sdk.types import Event, Hint

from voicevox_engine.utility.sentry_utility import filter_sentry_event

_WINDOWS_ENGINE_ROOT = "C:\\Program Files\\AivisSpeech\\AivisSpeech-Engine"
_WINDOWS_ENGINE_DATA_ROOT = "C:\\Users\\Taro\\AppData\\Roaming\\AivisSpeech-Engine"
_POSIX_ENGINE_ROOT = "/opt/AivisSpeech-Engine"


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
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\user.dict_compiled-abc.tmp",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "FileNotFoundError",
                "No such file or directory: "
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\"
                "user.dict_compiled-a96ccc50-022a-43ee-9bae-d9b018f198a9.tmp",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "FileNotFoundError",
                "[WinError 2] 指定されたファイルが見つかりません。: "
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\"
                "user.dict_compiled-a96ccc50-022a-43ee-9bae-d9b018f198a9.tmp",
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
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\BertModelCaches\\"
                "models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\\"
                "snapshots\\hash\\model_fp16.onnx failed. File doesn't exist",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "Fail",
                "[ONNXRuntimeError] : 1 : FAIL : Load model from "
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\BertModelCaches\\"
                "models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\\"
                "snapshots\\hash\\model_fp16.onnx failed: bad allocation",
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
        (
            _generate_exception_event(
                "FileNotFoundError",
                "No such file or directory: "
                f"{_POSIX_ENGINE_ROOT}/BertModelCaches/"
                "models--tsukumijima--deberta-v2-large-japanese-char-wwm-onnx\\"
                "snapshots\\hash\\tokenizer_config.json",
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
        (
            _generate_exception_event(
                "ValueError",
                "Exceeds the limit (4300 digits) for integer string conversion: "
                "value has 7620 digits; use sys.set_int_max_str_digits() to increase the limit",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeError",
                "Input text is too long after normalization",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "Model a59cb814-0083-4369-8542-f51a29e72af7 is a default model. "
                "It cannot be uninstalled.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "AivisSpeech Engine must have at least one installed model.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "Singer info is not supported in AivisSpeech Engine.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "Singers is not supported in AivisSpeech Engine.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "AIVMX ファイルのダウンロードに失敗しました。(timed out)",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ConnectTimeout",
                "timed out",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ReadTimeout",
                "The read operation timed out",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ReadError",
                "The operation did not complete (read) (_ssl.c:2580)",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPStatusError",
                "Client error '404 Not Found' for url "
                "'https://api.aivis-project.com/v1/aivm-models/model/download?model_type=AIVMX'",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RemoteProtocolError",
                "peer closed connection without sending complete message body "
                "(received 86376448 bytes, expected 258037076)",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ConnectionError",
                "Connection aborted. ConnectionResetError(10054, "
                "'既存の接続はリモート ホストに強制的に切断されました。')",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ProxyError",
                "407 Proxy Authentication Required",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "UnsupportedProtocol",
                "Request URL has an unsupported protocol 'aivmx://'.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ChunkedEncodingError",
                "Connection broken: IncompleteRead(231734924 bytes read, 421340775 more expected)",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HfHubHTTPError",
                "503 Server Error: Service Unavailable for url: "
                "https://cas-bridge.xethub.hf.co/xet-bridge-us/model",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "AivmValidationError",
                "Failed to decode AIVM metadata. This file is not an AIVMX (ONNX) file.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "HTTPException",
                "ユーザー辞書のインポートに失敗しました。",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "PermissionError",
                "Permission denied: "
                f"{_WINDOWS_ENGINE_DATA_ROOT}\\Models\\"
                "a59cb814-0083-4369-8542-f51a29e72af7.aivmx",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValueError",
                "Cannot load file containing pickled data when allow_pickle=False",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValidationError",
                "540 validation errors for dict[str,SaveFormatUserDictWord]",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "FileNotFoundError",
                "No such file or directory: "
                f"{_WINDOWS_ENGINE_ROOT}\\"
                "resources\\engine_manifest_assets\\icon.png",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "RuntimeException",
                "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during initialization: "
                "No mapping for the Unicode character exists in the target multi-byte code page.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "ValueError",
                "Unable to compare versions for regex>=2025.10.22: need=2025.10.22 "
                "found=None. This is unusual. Consider reinstalling regex.",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "PermissionError",
                "Permission denied: 'dmesg'",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "LocalProtocolError",
                "can't handle event type Response when role=SERVER and state=MUST_CLOSE",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "KeyboardInterrupt",
                "",
            ),
            {},
        ),
        (
            _generate_exception_event(
                "CancelledError",
                "",
            ),
            {},
        ),
        (
            {
                "message": "Traceback (most recent call last):\n"
                '  File "uvicorn\\server.py", line 339, in capture_signals\n'
                '  File "asyncio\\runners.py", line 157, in _on_sigint\n'
                "KeyboardInterrupt\n"
                "\n"
                "During handling of the above exception, another exception occurred:\n"
                "\n"
                "asyncio.exceptions.CancelledError\n"
            },
            {},
        ),
        ({}, _generate_hint(MemoryError("Unable to allocate 1024 bytes"))),
        ({}, _generate_hint(KeyboardInterrupt())),
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
