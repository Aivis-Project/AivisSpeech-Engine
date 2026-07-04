"""Sentry 送信前フィルタリング用ユーティリティ。"""

from __future__ import annotations

import re
from collections.abc import Callable
from types import TracebackType
from typing import Any

from sentry_sdk.types import Event, Hint

_ExceptionPredicate = Callable[[str, str], bool]
_ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]

_PATH_SEPARATOR_PATTERN = r"[\\/]"
_USER_DICTIONARY_TEMP_FILE_PATTERN = r"user\.dict_compiled-[^\\/\s'\"]+\.tmp"
_AIVIS_ENGINE_MODEL_PATTERN = (
    rf"AivisSpeech-Engine{_PATH_SEPARATOR_PATTERN}.*"
    rf"Models{_PATH_SEPARATOR_PATTERN}.+\.aivmx"
)
_BERT_CACHE_PATH_PATTERN = (
    rf"BertModelCaches{_PATH_SEPARATOR_PATTERN}.+"
    rf"tsukumijima--deberta-v2-large-japanese-char-wwm-onnx"
)
_ENGINE_MANIFEST_ICON_PATTERN = (
    rf"resources{_PATH_SEPARATOR_PATTERN}"
    rf"engine_manifest_assets{_PATH_SEPARATOR_PATTERN}icon\.png"
)


_ADDRESS_ALREADY_IN_USE_PATTERNS = (
    re.compile(r"\[Errno (?:48|10048)\].*attempting to bind", re.IGNORECASE),
    re.compile(r"\[Errno 13\].*attempting to bind", re.IGNORECASE),
    re.compile(r"address already in use", re.IGNORECASE),
    re.compile(r"only one usage of each socket address", re.IGNORECASE),
    re.compile(r"通常、各ソケット アドレス"),
    re.compile(r"각 소켓 주소"),
    re.compile(r"could not bind on any address", re.IGNORECASE),
)

_USER_DICTIONARY_TEMP_FILE_PATTERNS = (
    re.compile(rf"Failed to CreateFileW for .+{_USER_DICTIONARY_TEMP_FILE_PATTERN}"),
    re.compile(rf"No such file or directory: .+{_USER_DICTIONARY_TEMP_FILE_PATTERN}"),
    re.compile(rf"\[WinError 2\].+{_USER_DICTIONARY_TEMP_FILE_PATTERN}"),
)

_MEMORY_OR_GPU_RESOURCE_PATTERNS = (
    re.compile(r"bad allocation", re.IGNORECASE),
    re.compile(r"Failed to allocate memory", re.IGNORECASE),
    re.compile(r"Not enough memory resources", re.IGNORECASE),
    re.compile(r"unable to allocate", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
    re.compile(r"No space left on device", re.IGNORECASE),
    re.compile(r"\[WinError 1450\]"),
    re.compile(r"_ArrayMemoryError"),
    re.compile(r"\bMemoryError\b"),
    re.compile(r"887A0005"),
    re.compile(r"device instance has been suspended", re.IGNORECASE),
    re.compile(r"PooledUploadHeap"),
    re.compile(r"DmlExecutionProvider"),
    re.compile(r"CUDNN_STATUS_INTERNAL_ERROR"),
    re.compile(r"cudnnCreate"),
)

_MISSING_AIVM_RESOURCE_PATTERNS = (
    re.compile(r"Speaker [^\n]+ is not installed\."),
    re.compile(r"Model [^\n]+ is not installed\."),
    re.compile(r"Style \d+ is not found\."),
    re.compile(r"話者 [0-9a-fA-F-]{36} はインストールされていません"),
    re.compile(r"音声合成モデル [0-9a-fA-F-]{36} はインストールされていません"),
    re.compile(r"スタイル \d+ は存在しません"),
)

_AUDIO_QUERY_VALIDATION_PATTERNS = (
    re.compile(r"validation error for AudioQuery", re.IGNORECASE),
)

_CLIENT_INPUT_OR_REQUEST_ERROR_PATTERNS = (
    re.compile(r"validation error for Setting", re.IGNORECASE),
    re.compile(r"validation error for ParseKanaBadRequest", re.IGNORECASE),
    re.compile(r"Exceeds the limit \(\d+ digits\) for integer string conversion"),
    re.compile(r"Input text is too long after normalization"),
    re.compile(
        r"Model [0-9a-fA-F-]{36} is a default model\. It cannot be uninstalled\."
    ),
    re.compile(r"AivisSpeech Engine must have at least one installed model\."),
    re.compile(r"Singer info is not supported in AivisSpeech Engine\."),
    re.compile(r"Singers is not supported in AivisSpeech Engine\."),
)

_LOCAL_MODEL_OR_USER_DATA_ERROR_PATTERNS = (
    re.compile(r"Failed to decode AIVM metadata"),
    re.compile(r"This file is not an AIVMX \(ONNX\) file"),
    re.compile(r"Failed to read AIVM metadata"),
    re.compile(r"指定された AIVMX ファイルの形式が正しくありません"),
    re.compile(r"AIVMX ファイルの書き込みに失敗しました"),
    re.compile(rf"No such file or directory: .+{_AIVIS_ENGINE_MODEL_PATTERN}"),
    re.compile(rf"Permission denied: .+{_AIVIS_ENGINE_MODEL_PATTERN}"),
    re.compile(r"Cannot load file containing pickled data"),
    re.compile(r"validation error for EngineManifestJson", re.IGNORECASE),
    re.compile(r"validation error for dict\[str,function-after", re.IGNORECASE),
    re.compile(
        r"validation errors for dict\[str,SaveFormatUserDictWord", re.IGNORECASE
    ),
    re.compile(r"ユーザー辞書のインポートに失敗しました"),
    re.compile(r"辞書の読み込みに失敗しました"),
)

_LOCAL_BERT_CACHE_ERROR_PATTERNS = (
    re.compile(r"BertModelCaches.+File doesn't exist"),
    re.compile(r"BertModelCaches.+No such file or directory"),
    re.compile(r"No such file or directory: .+BertModelCaches"),
    re.compile(rf"Load model from .+{_BERT_CACHE_PATH_PATTERN}"),
    re.compile(
        r"Can't load tokenizer for 'tsukumijima/deberta-v2-large-japanese-char-wwm-onnx'"
    ),
    re.compile(r"tsukumijima/deberta-v2-large-japanese-char-wwm-onnx.+config\.json"),
    re.compile(r"cannot find the requested files in the local cache"),
)

_BROKEN_INSTALL_RESOURCE_PATTERNS = (
    re.compile(rf"No such file or directory: .+{_ENGINE_MANIFEST_ICON_PATTERN}"),
)

_EXTERNAL_DOWNLOAD_OR_PROXY_ERROR_PATTERNS = (
    re.compile(r"AIVMX ファイルのダウンロードに失敗しました"),
    re.compile(r"\bConnectTimeout\b[\s\S]*\btimed out\b", re.IGNORECASE),
    re.compile(r"\bReadTimeout\b[\s\S]*timed out", re.IGNORECASE),
    re.compile(r"\bReadError\b[\s\S]*_ssl\.c", re.IGNORECASE),
    re.compile(r"\bHTTPStatusError\b[\s\S]*api\.aivis-project\.com.+/download"),
    re.compile(r"\bRemoteProtocolError\b[\s\S]*peer closed connection", re.IGNORECASE),
    re.compile(r"ConnectionResetError\(10054"),
    re.compile(r"Proxy Authentication Required", re.IGNORECASE),
    re.compile(r"Max retries exceeded with url", re.IGNORECASE),
    re.compile(r"Request URL has an unsupported protocol 'aivmx://'", re.IGNORECASE),
    re.compile(r"UNEXPECTED_EOF_WHILE_READING"),
    re.compile(r"ChunkedEncodingError"),
    re.compile(r"IncompleteRead\("),
    re.compile(r"HfHubHTTPError: 503 Server Error"),
    re.compile(r"https?://(?:huggingface\.co|[^\s/]+\.xethub\.hf\.co)/"),
)

_INTERRUPTED_SHUTDOWN_PATTERNS = (
    re.compile(r"\bKeyboardInterrupt\b"),
    re.compile(r"\bCancelledError\b"),
)

_OS_OR_BROKEN_RUNTIME_ERROR_PATTERNS = (
    re.compile(
        r"can't handle event type Response when role=SERVER and state=MUST_CLOSE"
    ),
    re.compile(r"Permission denied: 'dmesg'"),
    re.compile(
        r"No mapping for the Unicode character exists in the target multi-byte code page"
    ),
    re.compile(r"Unable to compare versions for regex.+Consider reinstalling regex"),
)


def filter_sentry_event(event: Event, hint: Hint) -> Event | None:
    """
    Sentry へ送る必要がない既知の環境依存エラーを破棄する。

    Args:
        event (Event): Sentry SDK が送信しようとしているイベント
        hint (Hint): Sentry SDK が付与する例外情報などの補足情報

    Returns
    -------
        Event | None: 送信するイベント、または送信を止める場合は None
    """

    exception_type, exception_text = _collect_exception_text(event, hint)

    # 音素・g2p・テキスト前処理の不具合は残したいため、例外型だけの広い判定は避ける
    ## 過去の Sentry 実測で「ユーザー環境・OS・外部 API 呼び出し側の問題」と判断できた署名だけを落とす
    if any(
        predicate(exception_type, exception_text) is True
        for predicate in _SENTRY_DROP_PREDICATES
    ):
        return None

    return event


def _collect_exception_text(event: Event, hint: Hint) -> tuple[str, str]:
    """
    Sentry イベントとヒントから判定に使う文字列を集める。

    Args:
        event (Event): Sentry SDK が送信しようとしているイベント
        hint (Hint): Sentry SDK が付与する例外情報などの補足情報

    Returns
    -------
        tuple[str, str]: 例外型名と、イベント内の文字列を連結したテキスト
    """

    texts: list[str] = []
    exception_type = ""

    # `hint["exc_info"]` は実例外の型名とメッセージを一番素直に取れる
    ## Sentry イベント側で加工された表示文字列だけに頼ると、SDK の変更で判定が外れやすい
    exception_info = _extract_exception_info(hint.get("exc_info"))
    if exception_info is not None:
        exc_type, exc_value, _traceback = exception_info
        exception_type = exc_type.__name__
        texts.append(exception_type)
        texts.append(str(exc_value))

    # FastAPI / Uvicorn 経由のイベントでは、例外情報が event 側にだけ入る場合がある
    ## values の末尾が最も外側の例外なので、型名がまだ取れていない場合だけ採用する
    exception = event.get("exception")
    if isinstance(exception, dict):
        values = exception.get("values")
        if isinstance(values, list):
            for value in values:
                _append_text(texts, value.get("type"))
                _append_text(texts, value.get("value"))

            # 連鎖例外では末尾が外側の例外なので、フィルタ判定の型名も末尾から採用する
            ## 文字列は全例外分を拾いつつ、型名だけはユーザーに見える一番外側の例外へ寄せる
            if exception_type == "":
                for value in reversed(values):
                    if isinstance(value, dict):
                        event_exception_type = value.get("type")
                        if isinstance(event_exception_type, str):
                            exception_type = event_exception_type
                            break

    # logger 経由やメッセージイベントでは、例外ブロック以外に本文が入る
    ## 既知の文字列署名だけを見るため、ここではタグなどの広いメタ情報は含めない
    _append_text(texts, event.get("message"))
    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        _append_text(texts, logentry.get("message"))
        _append_text(texts, logentry.get("formatted"))

    return exception_type, "\n".join(texts)


def _extract_exception_info(value: Any) -> _ExceptionInfo | None:
    """
    Sentry の exc_info 形式として扱える値だけを取り出す。

    Args:
        value (Any): hint から取り出した値

    Returns
    -------
        _ExceptionInfo | None: exc_info として扱える値、または None
    """

    if not isinstance(value, tuple) or len(value) != 3:
        return None

    exc_type, exc_value, traceback = value
    if (
        isinstance(exc_type, type)
        and issubclass(exc_type, BaseException)
        and isinstance(exc_value, BaseException)
        and (traceback is None or isinstance(traceback, TracebackType))
    ):
        return exc_type, exc_value, traceback

    return None


def _append_text(texts: list[str], value: Any) -> None:
    """
    文字列だけを判定用テキストへ追加する。

    Args:
        texts (list[str]): 判定に使う文字列のリスト
        value (Any): Sentry イベント内の値
    """

    if isinstance(value, str):
        texts.append(value)


def _is_client_disconnect_error(exception_type: str, exception_text: str) -> bool:
    """
    クライアント側の切断で発生する接続リセットかどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: クライアント切断として破棄する場合は True
    """

    return (
        exception_type == "ConnectionResetError" or "[WinError 10054]" in exception_text
    )


def _matches_any_pattern(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    """
    既知の文字列署名に一致するかどうかを判定する。

    Args:
        patterns (tuple[re.Pattern[str], ...]): 判定に使う正規表現
        text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: いずれかの正規表現に一致する場合は True
    """

    return any(pattern.search(text) is not None for pattern in patterns)


def _is_port_already_in_use_error(exception_type: str, exception_text: str) -> bool:
    """
    ローカルポートが既に使われている起動失敗かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: ポート競合として破棄する場合は True
    """

    return _matches_any_pattern(_ADDRESS_ALREADY_IN_USE_PATTERNS, exception_text)


def _is_user_dictionary_temp_file_error(
    exception_type: str, exception_text: str
) -> bool:
    """
    Windows の一時ユーザー辞書ファイル削除失敗かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 一時ファイルのハンドル競合として破棄する場合は True
    """

    return _matches_any_pattern(_USER_DICTIONARY_TEMP_FILE_PATTERNS, exception_text)


def _is_memory_or_gpu_resource_error(exception_type: str, exception_text: str) -> bool:
    """
    メモリ不足または DirectML の GPU リソース喪失かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 実行環境のリソース不足として破棄する場合は True
    """

    if exception_type in {"MemoryError", "_ArrayMemoryError"}:
        return True

    return _matches_any_pattern(_MEMORY_OR_GPU_RESOURCE_PATTERNS, exception_text)


def _is_missing_aivm_resource_error(exception_type: str, exception_text: str) -> bool:
    """
    インストール済み AIVM に存在しない話者・モデル・スタイル参照かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: API 呼び出し側の古い ID 参照として破棄する場合は True
    """

    return _matches_any_pattern(_MISSING_AIVM_RESOURCE_PATTERNS, exception_text)


def _is_audio_query_validation_error(exception_type: str, exception_text: str) -> bool:
    """
    音声合成 API の入力検証で止めた AudioQuery エラーかどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 不正な外部入力として破棄する場合は True
    """

    return _matches_any_pattern(_AUDIO_QUERY_VALIDATION_PATTERNS, exception_text)


def _is_client_input_or_request_error(exception_type: str, exception_text: str) -> bool:
    """
    不正な設定値や未対応リクエストを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 外部入力や未対応 API 呼び出しとして破棄する場合は True
    """

    return _matches_any_pattern(_CLIENT_INPUT_OR_REQUEST_ERROR_PATTERNS, exception_text)


def _is_local_model_or_user_data_error(
    exception_type: str, exception_text: str
) -> bool:
    """
    ローカルのモデルファイルやユーザーデータ破損かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 利用者環境のファイル破損や権限問題として破棄する場合は True
    """

    return _matches_any_pattern(
        _LOCAL_MODEL_OR_USER_DATA_ERROR_PATTERNS, exception_text
    )


def _is_local_bert_cache_error(exception_type: str, exception_text: str) -> bool:
    """
    ローカルの BERT モデルキャッシュ欠損や取得失敗かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: ローカルキャッシュやネットワーク状態の問題として破棄する場合は True
    """

    return _matches_any_pattern(_LOCAL_BERT_CACHE_ERROR_PATTERNS, exception_text)


def _is_broken_install_resource_error(exception_type: str, exception_text: str) -> bool:
    """
    インストール済みリソースの欠損かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 壊れたインストール状態として破棄する場合は True
    """

    return _matches_any_pattern(_BROKEN_INSTALL_RESOURCE_PATTERNS, exception_text)


def _is_external_download_or_proxy_error(
    exception_type: str, exception_text: str
) -> bool:
    """
    外部サービスやプロキシによるダウンロード失敗かどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: ネットワーク環境や外部サービス側の問題として破棄する場合は True
    """

    return _matches_any_pattern(
        _EXTERNAL_DOWNLOAD_OR_PROXY_ERROR_PATTERNS, exception_text
    )


def _is_interrupted_shutdown_error(exception_type: str, exception_text: str) -> bool:
    """
    ユーザー操作やプロセス終了で中断されたエラーかどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: 通常の終了操作に伴う中断として破棄する場合は True
    """

    # KeyboardInterrupt は Ctrl+C や親プロセスからの終了で発生し、アプリ側で直せる例外ではない
    ## Uvicorn の終了処理では KeyboardInterrupt の後続として CancelledError だけが送られる場合もある
    if exception_type in {"KeyboardInterrupt", "CancelledError"}:
        return True

    return _matches_any_pattern(_INTERRUPTED_SHUTDOWN_PATTERNS, exception_text)


def _is_os_or_broken_runtime_error(exception_type: str, exception_text: str) -> bool:
    """
    OS の制約や壊れた実行環境に由来するエラーかどうかを判定する。

    Args:
        exception_type (str): 例外型名
        exception_text (str): イベント内の文字列を連結したテキスト

    Returns
    -------
        bool: アプリ側から修正できない実行環境の問題として破棄する場合は True
    """

    return _matches_any_pattern(_OS_OR_BROKEN_RUNTIME_ERROR_PATTERNS, exception_text)


_SENTRY_DROP_PREDICATES: tuple[_ExceptionPredicate, ...] = (
    _is_client_disconnect_error,
    _is_interrupted_shutdown_error,
    _is_port_already_in_use_error,
    _is_user_dictionary_temp_file_error,
    _is_memory_or_gpu_resource_error,
    _is_missing_aivm_resource_error,
    _is_audio_query_validation_error,
    _is_client_input_or_request_error,
    _is_local_model_or_user_data_error,
    _is_local_bert_cache_error,
    _is_broken_install_resource_error,
    _is_external_download_or_proxy_error,
    _is_os_or_broken_runtime_error,
)
