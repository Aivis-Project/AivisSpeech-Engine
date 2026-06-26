"""AivisSpeech Engine の実行"""

# truststore を適用し、HTTPS 通信時にシステムにインストールされた証明書ストアを使う
## 企業内 LAN など HTTPS プロキシが導入されている環境で、システムにインストールされた自己署名証明書を信頼するために必要
## requests からの HTTPS 通信には certifi が使われるため、HTTPS プロキシ導入環境では truststore を適用しない限り通信エラーが発生する
## また、一部の非常に古い社内用 HTTPS プロキシ / TLS 中継装置では RFC 5746 非対応の legacy renegotiation を要求するため、
## Python 3.11.5 以降に内蔵される OpenSSL 3.x 系では [SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED] が発生することがある
## そのため truststore の SSLContext を inject 前に差し替え、SSL_OP_LEGACY_SERVER_CONNECT 相当のオプションを常に有効化する
## ref: https://github.com/psf/requests/issues/2966
## ref: https://truststore.readthedocs.io/en/latest/
## ref: https://docs.openssl.org/3.0/man3/SSL_CTX_set_options/
## ref: https://yamori-jp.blogspot.com/2022/09/python-ssl-unsafelegacyrenegotiationdis.html
# fmt: off
import truststore._api as truststore_api  # isort: skip
import truststore  # isort: skip
class _PatchedTrustStoreSSLContext(truststore.SSLContext):
    def __init__(self, protocol: int | None = None) -> None:
        super().__init__(protocol)  # type: ignore
        # TLS ハンドシェイク拒否を回避するため、SSL_OP_LEGACY_SERVER_CONNECT 相当のビットを常に有効化する
        ## Python 3.11 では ssl.OP_LEGACY_SERVER_CONNECT が公開されていないため、
        ## OpenSSL の定義値 (SSL_OP_BIT(2) == 1 << 2) を直接利用する
        _SSL_OP_LEGACY_SERVER_CONNECT = 1 << 2
        self.options |= _SSL_OP_LEGACY_SERVER_CONNECT
truststore.SSLContext = _PatchedTrustStoreSSLContext  # type: ignore
truststore_api.SSLContext = _PatchedTrustStoreSSLContext  # type: ignore
truststore.inject_into_ssl()
# fmt: on
# flake8: noqa: E402

import argparse
import gc
import importlib
import multiprocessing
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Literal, TextIO, TypeVar

import sentry_sdk
import uvicorn
from pydantic import TypeAdapter

from voicevox_engine import __version__
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.application import generate_app
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.core.core_initializer import initialize_cores
from voicevox_engine.engine_manifest import load_manifest
from voicevox_engine.library.library_manager import LibraryManager
from voicevox_engine.logging import LOGGING_CONFIG, logger
from voicevox_engine.preset.preset_manager import PresetManager
from voicevox_engine.setting.model import CorsPolicyMode
from voicevox_engine.setting.setting_manager import USER_SETTING_PATH, SettingHandler
from voicevox_engine.tts_pipeline.song_engine import make_song_engines_from_cores
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager
from voicevox_engine.user_dict.user_dict_manager import UserDictionary
from voicevox_engine.utility.aivishub_client import AivisHubClient
from voicevox_engine.utility.core_version_utility import MOCK_CORE_VERSION
from voicevox_engine.utility.path_utility import (
    engine_manifest_path,
    engine_root,
    get_save_dir,
)
from voicevox_engine.utility.user_agent_utility import collect_runtime_environment

# Uvicorn でバインドするアドレスを "localhost" にすることで IPv4 (127.0.0.1) と IPv6 ([::1]) の両方でリッスンできます.
# これは Uvicorn のドキュメントに記載されていない挙動です; 将来のアップデートにより動作しなくなる可能性があります.
# ref: https://github.com/VOICEVOX/voicevox_engine/pull/647#issuecomment-1540204653
_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 10101
_DEFAULT_ONNX_PLUGIN_EP_NAME = "AivisGgmlExecutionProvider"
_ONNX_PROVIDER_CHOICES = ("auto", "cuda", "directml", "ggml")
_ONNX_EP_LIBRARY_BASENAMES = (
    "libaivis_ggml_onnx_ep.so",
    "libaivis_ggml_onnx_ep.dylib",
    "aivis_ggml_onnx_ep.dll",
)


def parse_key_value_options(raw_options: list[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI options into a dictionary."""

    parsed_options: dict[str, str] = {}
    for raw_option in raw_options or []:
        key, separator, value = raw_option.partition("=")
        if separator == "" or key == "":
            raise ValueError(f"Expected KEY=VALUE, got: {raw_option}")
        parsed_options[key] = value
    return parsed_options


def decide_onnx_provider_from_env(env_name: str) -> str | None:
    """Read an explicit ONNX provider selection from the environment."""

    env = os.getenv(env_name)
    if env is None or env == "":
        return None
    provider = env.lower()
    if provider in _ONNX_PROVIDER_CHOICES:
        return provider
    msg = (
        f"Invalid environment variable value: {env_name}={env}. "
        f"Expected one of: {', '.join(_ONNX_PROVIDER_CHOICES)}"
    )
    warnings.warn(msg, stacklevel=1)
    return None


def decide_boolean_from_env(env_name: str) -> bool:
    """
    環境変数からbool値を返す。

    * 環境変数が"1"ならTrueを返す
    * 環境変数が"0"か空白か存在しないならFalseを返す
    * それ以外はwarningを出してFalseを返す
    """
    env = os.getenv(env_name, default="")
    match env:
        case "1":
            return True
        case "" | "0":
            return False
        case _:
            msg = f"Invalid environment variable value: {env_name}={env}"
            warnings.warn(msg, stacklevel=1)
            return False


def decide_port_from_env(env_name: str) -> int | None:
    """
    環境変数からポート番号を返す。

    * 環境変数が0から65535の範囲の整数と解釈可能ならその数を返す
    * 環境変数が空白か存在しないならNoneを返す
    * それ以外はwarningを出してNoneを返す
    """
    env = os.getenv(env_name)
    if env is None or env == "":
        return None
    try:
        port = int(env)
        if 0 <= port <= 65535:
            return port
    except ValueError:
        pass
    msg = f"Invalid environment variable value: {env_name}={env}"
    warnings.warn(msg, stacklevel=1)
    return None


@dataclass(frozen=True)
class Envs:
    """環境変数の集合"""

    output_log_utf8: bool
    cpu_num_threads: str | None
    env_preset_path: str | None
    disable_mutable_api: bool
    host: str | None
    port: int | None
    use_gpu: bool
    onnx_provider: str | None


_env_adapter = TypeAdapter(Envs)


def read_environment_variables() -> Envs:
    """環境変数を読み込む。"""
    envs = Envs(
        output_log_utf8=decide_boolean_from_env("VV_OUTPUT_LOG_UTF8"),
        cpu_num_threads=os.getenv("VV_CPU_NUM_THREADS"),
        env_preset_path=os.getenv("VV_PRESET_FILE"),
        disable_mutable_api=decide_boolean_from_env("VV_DISABLE_MUTABLE_API"),
        host=os.getenv("VV_HOST"),
        port=decide_port_from_env("VV_PORT"),
        use_gpu=decide_boolean_from_env("VV_USE_GPU"),
        onnx_provider=decide_onnx_provider_from_env("VV_ONNX_PROVIDER"),
    )
    return _env_adapter.validate_python(asdict(envs))


def set_output_log_utf8() -> None:
    """標準出力と標準エラー出力の出力形式を UTF-8 ベースに切り替える"""

    # NOTE: for 文で回せないため関数内関数で実装している
    def _prepare_utf8_stdio(stdio: TextIO) -> TextIO:
        """UTF-8 ベースの標準入出力インターフェイスを用意する"""
        CODEC = "utf-8"  # locale に依存せず UTF-8 コーデックを用いる
        ERR = "backslashreplace"  # 不正な形式のデータをバックスラッシュ付きのエスケープシーケンスに置換する

        # 既定の `TextIOWrapper` 入出力インターフェイスを UTF-8 へ再設定して返す
        if isinstance(stdio, TextIOWrapper):
            stdio.reconfigure(encoding=CODEC)
            return stdio
        else:
            # 既定インターフェイスのバッファを全て出力しきった上で UTF-8 設定の `TextIOWrapper` を生成して返す
            stdio.flush()
            try:
                return TextIOWrapper(stdio.buffer, encoding=CODEC, errors=ERR)
            except AttributeError:
                # バッファへのアクセスに失敗した場合、設定変更をおこなわず返す
                return stdio

    # NOTE:
    # `sys.std*` はコンソールがない環境だと `None` をとる (出典: https://docs.python.org/ja/3/library/sys.html#sys.__stdin__ )
    # これは Python インタープリタが標準入出力へ接続されていないことを意味するため、設定不要とみなす

    if sys.stdout is None:
        pass
    else:
        sys.stdout = _prepare_utf8_stdio(sys.stdout)

    if sys.stderr is None:
        pass
    else:
        sys.stderr = _prepare_utf8_stdio(sys.stderr)


T = TypeVar("T")


def select_first_not_none(candidates: list[T | None]) -> T:
    """None でない最初の値を取り出す。全て None の場合はエラーを送出する。"""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("すべての候補値が None です")


S = TypeVar("S")


def select_first_not_none_or_none(candidates: list[S | None]) -> S | None:
    """None でない最初の値を取り出そうとし、全て None の場合は None を返す。"""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


@dataclass(frozen=True)
class _CLIArgs:
    host: str | None
    port: int | None
    use_gpu: bool | None
    onnx_provider: str | None
    ggml_model_cache_dir: Path | None
    ggml_native_library_path: Path | None
    ggml_tts_server_backend: str
    ggml_vulkan_device: str | None
    ggml_vulkan_precision: str
    onnx_ep_library_path: Path | None
    onnx_ep_registration_name: str
    onnx_ep_name: str | None
    onnx_ep_options: dict[str, str]
    load_all_models: bool
    output_log_utf8: bool
    cors_policy_mode: CorsPolicyMode | None
    allow_origins: list[str] | None
    setting_file: Path
    preset_file: Path | None
    disable_mutable_api: bool
    disable_sentry: bool
    # 以下は極力 VOICEVOX ENGINE との差分を最小限にするための互換用
    # 対応する引数は AivisSpeech Engine では常に無効化されている
    voicevox_dir: Path | None = None  # 常に None
    voicelib_dirs: list[Path] | None = None  # 常に None
    runtime_dirs: list[Path] | None = None  # 常に None
    enable_mock: bool = True  # 常にモック版 VOICEVOX CORE を利用する
    enable_cancellable_synthesis: bool = False  # 常に False
    init_processes: int = 2  # 常に 2
    cpu_num_threads: int | None = 4  # 常に 4


_cli_args_adapter = TypeAdapter(_CLIArgs)


def _default_ggml_cpu_threads() -> int:
    """Return a positive thread count for the TTS.cpp CPU backend."""

    return max(os.cpu_count() or 1, 1)


def _add_local_onnx_ep_package_src_to_path() -> None:
    package_src_path = (
        engine_root() / "experimental" / "onnxruntime-ep-aivis-ggml" / "src"
    )
    if not package_src_path.exists():
        return
    package_src_text = str(package_src_path)
    if package_src_text not in sys.path:
        sys.path.insert(0, package_src_text)


def _resolve_engine_root_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (engine_root() / path).resolve()


def _resolve_default_onnx_ep_library_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return _resolve_engine_root_relative_path(explicit_path)

    package_error: Exception | None = None
    try:
        try:
            plugin_ep = importlib.import_module("onnxruntime_ep_aivis_ggml")
        except ModuleNotFoundError:
            _add_local_onnx_ep_package_src_to_path()
            plugin_ep = importlib.import_module("onnxruntime_ep_aivis_ggml")
        return Path(plugin_ep.get_library_path())
    except Exception as ex:
        package_error = ex

    build_dir = engine_root() / "experimental" / "onnxruntime-ep-aivis-ggml" / "build"
    for basename in _ONNX_EP_LIBRARY_BASENAMES:
        for candidate in (
            build_dir / "native" / basename,
            build_dir / basename,
        ):
            if candidate.exists():
                return candidate

    raise RuntimeError(
        "--onnx_provider ggml requires --onnx_ep_library_path or a packaged "
        "onnxruntime_ep_aivis_ggml library."
    ) from package_error


def _build_ggml_onnx_ep_options(args: _CLIArgs) -> dict[str, str]:
    provider_options = dict(args.onnx_ep_options)
    provider_options.setdefault("backend", args.ggml_tts_server_backend)
    provider_options.setdefault("claim_jp_bert_graph", "1")
    provider_options.setdefault("claim_synthesis_graph", "1")
    provider_options.setdefault("eager_load_model", "1")
    provider_options.setdefault("n_threads", "0")
    provider_options.setdefault("precision", args.ggml_vulkan_precision)
    if args.ggml_native_library_path is not None:
        provider_options.setdefault(
            "tts_cpp_library_path",
            str(_resolve_engine_root_relative_path(args.ggml_native_library_path)),
        )
    tts_cpp_library_path = provider_options.get("tts_cpp_library_path")
    if tts_cpp_library_path is not None and tts_cpp_library_path != "":
        provider_options["tts_cpp_library_path"] = str(
            _resolve_engine_root_relative_path(Path(tts_cpp_library_path))
        )
    if args.ggml_vulkan_device is not None and provider_options.get("backend") != "cpu":
        provider_options.setdefault("device", args.ggml_vulkan_device)
    if (
        provider_options.get("backend") == "cpu"
        and provider_options.get("n_threads") == "0"
    ):
        provider_options["n_threads"] = str(_default_ggml_cpu_threads())

    if provider_options.get("tts_cpp_library_path") in {None, ""}:
        raise RuntimeError(
            "--onnx_provider ggml requires --ggml_native_library_path or "
            "--onnx_ep_option tts_cpp_library_path=<path-to-libtts>."
        )
    return provider_options


def read_cli_arguments(envs: Envs) -> _CLIArgs:
    """コマンドライン引数を読み込む。"""
    parser = argparse.ArgumentParser(
        description="AivisSpeech Engine: AI Voice Imitation System - Text to Speech Engine"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="接続を受け付けるホストアドレスです。指定しない場合、代わりに環境変数 VV_HOST の値が使われます。",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="接続を受け付けるポート番号です。指定しない場合、代わりに環境変数 VV_PORT の値が使われます。",
    )
    parser.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        help=(
            "GPUを使って音声合成するか設定します。指定しない場合、代わりに環境変数 VV_USE_GPU の値が使われます。"
            "VV_USE_GPU の値が1の場合はGPUを使用し、0または空文字、値がない場合は使用されません。"
        ),
    )
    parser.add_argument(
        "--onnx_provider",
        choices=_ONNX_PROVIDER_CHOICES,
        default=None,
        help=(
            "Style-Bert-VITS2 の ONNX 推論で使う provider を指定します。"
            "auto は既存の --use_gpu に従い、cuda / directml は組み込み Execution Provider を明示的に使い、"
            "ggml は外部 ONNX Runtime Plugin EP を使います。"
            "指定しない場合、代わりに環境変数 VV_ONNX_PROVIDER の値が使われます。"
        ),
    )
    parser.add_argument(
        "--ggml_model_cache_dir",
        type=Path,
        default=None,
        help="ONNX GGML Plugin EP 用に変換した GGUF モデルのキャッシュディレクトリです。",
    )
    parser.add_argument(
        "--ggml_native_library_path",
        type=Path,
        default=None,
        help=(
            "ONNX GGML Plugin EP が利用する TTS.cpp C API 共有ライブラリのパスです。"
            "--onnx_provider ggml では必須です。"
        ),
    )
    parser.add_argument(
        "--ggml_tts_server_backend",
        choices=("vulkan", "metal", "cpu"),
        default="vulkan",
        help="ONNX GGML Plugin EP 内の TTS.cpp backend を指定します。",
    )
    parser.add_argument(
        "--ggml_vulkan_device",
        default=None,
        help="Vulkan/Metal backend で利用する device id を provider option として渡します。",
    )
    parser.add_argument(
        "--ggml_vulkan_precision",
        choices=("accurate", "fast"),
        default="accurate",
        help="ONNX GGML Plugin EP の Vulkan precision を指定します。",
    )
    parser.add_argument(
        "--onnx_ep_library_path",
        type=Path,
        default=None,
        help="ONNX Runtime Plugin EP 共有ライブラリのパスです。省略時はパッケージまたはローカル build から探します。",
    )
    parser.add_argument(
        "--onnx_ep_registration_name",
        default="aivis_onnx_plugin_ep",
        help="ONNX Runtime Plugin EP library の登録名です。",
    )
    parser.add_argument(
        "--onnx_ep_name",
        default=None,
        help=(
            "優先する ONNX Runtime Plugin EP 名です。省略時は "
            f"{_DEFAULT_ONNX_PLUGIN_EP_NAME} を使います。"
        ),
    )
    parser.add_argument(
        "--onnx_ep_option",
        action="append",
        default=[],
        help="ONNX Runtime Plugin EP に渡す追加オプションです。KEY=VALUE 形式で複数指定できます。",
    )
    # parser.add_argument(
    #     "--voicevox_dir",
    #     type=Path,
    #     default=None,
    #     help="VOICEVOXのディレクトリパスです。",
    # )
    # parser.add_argument(
    #     "--voicelib_dir",
    #     type=Path,
    #     default=None,
    #     action="append",
    #     help="VOICEVOX COREのディレクトリパスです。",
    # )
    # parser.add_argument(
    #     "--runtime_dir",
    #     type=Path,
    #     default=None,
    #     action="append",
    #     help="VOICEVOX COREで使用するライブラリのディレクトリパスです。",
    # )
    # parser.add_argument(
    #     "--enable_mock",
    #     action="store_true",
    #     help="VOICEVOX COREを使わずモックで音声合成を行います。",
    # )
    # parser.add_argument(
    #     "--enable_cancellable_synthesis",
    #     action="store_true",
    #     help="音声合成を途中でキャンセルできるようになります。",
    # )
    # parser.add_argument(
    #     "--init_processes",
    #     type=int,
    #     default=2,
    #     help="cancellable_synthesis機能の初期化時に生成するプロセス数です。",
    # )
    parser.add_argument(
        "--load_all_models",
        action="store_true",
        help="起動時に全ての音声合成モデルを読み込みます。",
    )

    # 引数へcpu_num_threadsの指定がなければ、環境変数をロールします。
    # 環境変数にもない場合は、Noneのままとします。
    # VV_CPU_NUM_THREADSが空文字列でなく数値でもない場合、エラー終了します。
    # parser.add_argument(
    #     "--cpu_num_threads",
    #     type=int,
    #     default=envs.cpu_num_threads,
    #     help=(
    #         "音声合成を行うスレッド数です。指定しない場合、代わりに環境変数 VV_CPU_NUM_THREADS の値が使われます。"
    #         "VV_CPU_NUM_THREADS が空文字列でなく数値でもない場合はエラー終了します。"
    #     ),
    # )

    parser.add_argument(
        "--output_log_utf8",
        action="store_true",
        help=(
            "ログ出力を UTF-8 で行います。指定しない場合、代わりに環境変数 VV_OUTPUT_LOG_UTF8 の値が使われます。"
            "VV_OUTPUT_LOG_UTF8 の値が 1 の場合は UTF-8 で、0 または空文字、値がない場合は環境によって自動的に決定されます。"
        ),
    )

    parser.add_argument(
        "--cors_policy_mode",
        type=CorsPolicyMode,
        choices=[i.value for i in CorsPolicyMode],
        default=None,
        help=(
            "CORS の許可モード。all または localapps が指定できます。all はすべてを許可します。"
            "localapps はオリジン間リソース共有ポリシーを、app://. と localhost 関連、ブラウザ拡張 URI に限定します。"
            "その他のオリジンは allow_origin オプションで追加できます。デフォルトは localapps です。"
            "このオプションは --setting_file で指定される設定ファイルよりも優先されます。"
        ),
    )

    parser.add_argument(
        "--allow_origin",
        nargs="*",
        help=(
            "許可するオリジンを指定します。スペースで区切ることで複数指定できます。"
            "このオプションは --setting_file で指定される設定ファイルよりも優先されます。"
        ),
    )

    parser.add_argument(
        "--setting_file",
        type=Path,
        default=USER_SETTING_PATH,
        help="設定ファイルを指定できます。",
    )

    parser.add_argument(
        "--preset_file",
        type=Path,
        default=None,
        help=(
            "プリセットファイルを指定できます。"
            "指定がない場合、環境変数 VV_PRESET_FILE 、ユーザーディレクトリの presets.yaml を順に探します。"
        ),
    )

    parser.add_argument(
        "--disable_mutable_api",
        action="store_true",
        help=(
            "辞書登録や設定変更など、音声合成エンジンの静的なデータを変更する API を無効化します。"
            "指定しない場合、代わりに環境変数 VV_DISABLE_MUTABLE_API の値が使われます。"
            "VV_DISABLE_MUTABLE_API の値が 1 の場合は無効化で、0 または空文字、値がない場合は無視されます。"
        ),
    )

    parser.add_argument(
        "--disable_sentry",
        action="store_true",
        help="Sentry によるエラーログ収集を無効化します。",
    )

    args_dict = vars(parser.parse_args())

    # NOTE: 複数個の同名引数に基づいてリスト化されるため `CLIArgs` で複数形にリネームされている
    # args_dict["voicelib_dirs"] = args_dict.pop("voicelib_dir")
    # args_dict["runtime_dirs"] = args_dict.pop("runtime_dir")
    args_dict["allow_origins"] = args_dict.pop("allow_origin")
    try:
        args_dict["onnx_ep_options"] = parse_key_value_options(
            args_dict.pop("onnx_ep_option")
        )
    except ValueError as ex:
        parser.error(str(ex))

    # --host に 127.0.0.1 が指定されたとき、Windows 上で localhost でアクセスした際に
    # IPv6 でバインドされないことによる接続遅延を防ぐために、代わりに localhost を指定し IPv4 と IPv6 の両方でバインドする
    # ref: https://github.com/VOICEVOX/voicevox_engine/issues/1480
    if args_dict["host"] == "127.0.0.1":
        args_dict["host"] = "localhost"

    args = _cli_args_adapter.validate_python(args_dict)

    return args


def main() -> None:
    """AivisSpeech Engine を実行する"""

    # Windows でも multiprocessing を利用するために必要
    multiprocessing.freeze_support()

    # ユーザーの環境変数で hf_transfer が有効化されている場合に備え、事前に無効化しておく
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    envs = read_environment_variables()
    if envs.output_log_utf8:
        set_output_log_utf8()

    args = read_cli_arguments(envs)
    if args.output_log_utf8:
        set_output_log_utf8()

    use_gpu = select_first_not_none([args.use_gpu, envs.use_gpu])
    onnx_provider = (
        select_first_not_none_or_none([args.onnx_provider, envs.onnx_provider])
        or "auto"
    )
    builtin_onnx_provider: Literal["cuda", "directml"] | None = None
    if onnx_provider == "cuda":
        builtin_onnx_provider = "cuda"
    elif onnx_provider == "directml":
        builtin_onnx_provider = "directml"
    if onnx_provider in {"cuda", "directml", "ggml"}:
        use_gpu = True

    # この PC の動作環境情報を取得
    # 起動時の可能な限り早い段階で実行結果をキャッシュしておくのが重要
    runtime_environment = collect_runtime_environment(
        inference_type="GPU" if use_gpu is True else "CPU"
    )

    # AivisHub API クライアントを初期化
    ## except ブロックからも参照されるため、try の外で生成する
    aivishub_client = AivisHubClient()

    try:
        # Sentry によるエラートラッキングを開始 (production 環境のみ有効)
        # ref: https://docs.sentry.io/platforms/python/integrations/fastapi/
        if not args.disable_sentry and __version__ != "latest":
            sentry_sdk.init(
                dsn="https://ebdf5cc288b3ab31a262186329ff3a95@o4508551725383680.ingest.us.sentry.io/4508555159470080",
                release=f"AivisSpeech-Engine@{__version__}",
                environment="production",
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for tracing.
                traces_sample_rate=1.0,
                _experiments={
                    # Set continuous_profiling_auto_start to True
                    # to automatically start the profiler on when
                    # possible.
                    "continuous_profiling_auto_start": True,
                },
            )

        logger.info(f"AivisSpeech Engine version {__version__}")
        if args.disable_sentry:
            logger.info("Sentry error tracking is disabled.")
        logger.info(f"Engine root directory: {engine_root()}")
        logger.info(f"User data directory: {get_save_dir()}")

        # AivmManager を初期化
        aivm_manager = AivmManager(
            get_save_dir() / "Models",
            aivishub_client=aivishub_client,
        )

        # ごく稀に style_bert_vits2_tts_engine.py (が依存する onnxruntime) のインポート自体に失敗し
        # 例外が発生する環境があるようなので、例外をキャッチしてエラーログに出力できるよう、敢えてルーター初期化時にインポートする
        from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
            OnnxPluginExecutionProviderConfig,
            StyleBertVITS2TTSEngine,
        )

        onnx_plugin_ep = None
        if onnx_provider == "ggml":
            onnx_plugin_ep = OnnxPluginExecutionProviderConfig(
                library_path=_resolve_default_onnx_ep_library_path(
                    args.onnx_ep_library_path
                ),
                registration_name=args.onnx_ep_registration_name,
                provider_name=args.onnx_ep_name or _DEFAULT_ONNX_PLUGIN_EP_NAME,
                provider_options=_build_ggml_onnx_ep_options(args),
                strict=True,
            )

        # AivisSpeech Engine 独自の StyleBertVITS2TTSEngine を通常の TTSEngine の代わりに利用
        tts_engines = TTSEngineManager()
        tts_engines.register_engine(
            StyleBertVITS2TTSEngine(
                aivm_manager,
                use_gpu,
                args.load_all_models,
                preferred_onnx_provider=builtin_onnx_provider,
                onnx_plugin_ep=onnx_plugin_ep,
                ggml_model_cache_dir=args.ggml_model_cache_dir,
            ),
            MOCK_CORE_VERSION,
        )

        core_manager = initialize_cores(
            use_gpu=use_gpu,
            voicelib_dirs=args.voicelib_dirs,
            voicevox_dir=args.voicevox_dir,
            runtime_dirs=args.runtime_dirs,
            cpu_num_threads=args.cpu_num_threads,
            enable_mock=args.enable_mock,
            load_all_models=args.load_all_models,
        )
        # tts_engines = make_tts_engines_from_cores(core_manager)
        song_engines = make_song_engines_from_cores(core_manager)
        # assert len(tts_engines.versions()) != 0, "音声合成エンジンがありません。"
        assert len(song_engines.versions()) != 0, "音声合成エンジンがありません。"

        cancellable_engine: CancellableEngine | None = None
        if args.enable_cancellable_synthesis:
            cancellable_engine = CancellableEngine(
                init_processes=args.init_processes,
                use_gpu=use_gpu,
                voicelib_dirs=args.voicelib_dirs,
                voicevox_dir=args.voicevox_dir,
                runtime_dirs=args.runtime_dirs,
                cpu_num_threads=args.cpu_num_threads,
                enable_mock=args.enable_mock,
            )

        setting_loader = SettingHandler(args.setting_file)
        settings = setting_loader.load()

        # 複数方式で指定可能な場合、優先度は上から「引数」「環境変数」「設定ファイル」「デフォルト値」

        host = select_first_not_none([args.host, envs.host, _DEFAULT_HOST])
        port = select_first_not_none([args.port, envs.port, _DEFAULT_PORT])

        cors_policy_mode = select_first_not_none(
            [args.cors_policy_mode, settings.cors_policy_mode]
        )

        setting_allow_origins = None
        if settings.allow_origin is not None:
            setting_allow_origins = settings.allow_origin.split(" ")
        allow_origin = select_first_not_none_or_none(
            [args.allow_origins, setting_allow_origins]
        )

        if envs.env_preset_path is not None and len(envs.env_preset_path) != 0:
            env_preset_path = Path(envs.env_preset_path)
        else:
            env_preset_path = None
        default_preset_path = get_save_dir() / "presets.yaml"
        preset_path = select_first_not_none(
            [args.preset_file, env_preset_path, default_preset_path]
        )
        preset_manager = PresetManager(preset_path)

        user_dict = UserDictionary()

        engine_manifest = load_manifest(engine_manifest_path())

        library_manager = LibraryManager(
            # get_save_dir() / "installed_libraries",
            # AivisSpeech では利用しない LibraryManager によるディレクトリ作成を防ぐため、get_save_dir() 直下を指定
            get_save_dir(),
            engine_manifest.supported_vvlib_manifest_version,
            engine_manifest.brand_name,
            engine_manifest.name,
            engine_manifest.uuid,
        )

        root_dir = select_first_not_none([args.voicevox_dir, engine_root()])
        character_info_dir = root_dir / "resources" / "character_info"
        # NOTE: ENGINE v0.19 以前向けに後方互換性を確保する
        if not character_info_dir.exists():
            character_info_dir = root_dir / "speaker_info"

        disable_mutable_api = args.disable_mutable_api or envs.disable_mutable_api

        # ASGI に準拠した AivisSpeech Engine アプリケーションを生成する
        app = generate_app(
            tts_engines,
            song_engines,
            aivm_manager,
            core_manager,
            setting_loader,
            preset_manager,
            user_dict,
            engine_manifest,
            library_manager,
            cancellable_engine,
            character_info_dir,
            cors_policy_mode,
            allow_origin,
            disable_mutable_api=disable_mutable_api,
        )

        # 起動処理にのみに要したメモリを明示的に解放
        gc.collect()

        # エンジンが正常に起動したことを AivisHub へ通知する
        # イベント送信時にいかなるエラーが発生してもエラーはメソッド内で吸収されるため、起動処理には影響を与えない
        if runtime_environment is not None:
            aivishub_client.send_event(
                event_type="Startup",
                runtime_environment=runtime_environment,
            )

        # AivisSpeech Engine サーバーを起動
        # NOTE: デフォルトは ASGI に準拠した HTTP/1.1 サーバー
        uvicorn.run(app, host=host, port=port, log_config=LOGGING_CONFIG)

    except Exception as ex:
        # 起動時にエラーが発生した場合、スタックトレースを取得した上で起動失敗イベントを AivisHub へ通知する
        aivishub_client.send_event(
            event_type="StartupFailed",
            runtime_environment=runtime_environment,
            stack_trace=traceback.format_exc(),
        )
        logger.error("Unexpected error occurred during engine startup:", exc_info=ex)
        raise


if __name__ == "__main__":
    main()
