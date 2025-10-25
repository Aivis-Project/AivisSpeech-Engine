"""AivisSpeech Engine の実行"""

# truststore を適用し、HTTPS 通信時にシステムにインストールされた証明書ストアを使う
# 企業内 LAN など HTTPS プロキシが導入されている環境で、システムにインストールされた自己署名証明書を信頼するために必要
# requests からの HTTPS 通信には certifi が使われるため、HTTPS プロキシ導入環境では truststore を適用しない限り通信エラーが発生する
# ref: https://github.com/psf/requests/issues/2966
# ref: https://truststore.readthedocs.io/en/latest/
# fmt: off
import truststore  # isort: skip
truststore.inject_into_ssl()
# fmt: on
# flake8: noqa: E402

import argparse
import gc
import multiprocessing
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO, TypeVar

import sentry_sdk
import uvicorn
from pydantic import TypeAdapter

from voicevox_engine import __version__
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.application import generate_app
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.core.core_initializer import MOCK_VER, initialize_cores
from voicevox_engine.engine_manifest import load_manifest
from voicevox_engine.library.library_manager import LibraryManager
from voicevox_engine.logging import LOGGING_CONFIG, logger
from voicevox_engine.preset.preset_manager import PresetManager
from voicevox_engine.setting.model import CorsPolicyMode
from voicevox_engine.setting.setting_manager import USER_SETTING_PATH, SettingHandler
from voicevox_engine.tts_pipeline.song_engine import make_song_engines_from_cores
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager
from voicevox_engine.user_dict.user_dict_manager import UserDictionary
from voicevox_engine.utility.path_utility import (
    engine_manifest_path,
    engine_root,
    get_save_dir,
)
from voicevox_engine.utility.user_agent_utility import collect_runtime_environment


def decide_boolean_from_env(env_name: str) -> bool:
    """
    環境変数からbool値を返す。

    * 環境変数が"1"ならTrueを返す
    * 環境変数が"0"か空白か存在しないならFalseを返す
    * それ以外はwarningを出してFalseを返す
    """
    env = os.getenv(env_name, default="")
    if env == "1":
        return True
    elif env == "" or env == "0":
        return False
    else:
        warnings.warn(
            f"Invalid environment variable value: {env_name}={env}",
            stacklevel=1,
        )
        return False


@dataclass(frozen=True)
class Envs:
    """環境変数の集合"""

    output_log_utf8: bool
    cpu_num_threads: str | None
    env_preset_path: str | None
    disable_mutable_api: bool


_env_adapter = TypeAdapter(Envs)


def read_environment_variables() -> Envs:
    """環境変数を読み込む。"""
    envs = Envs(
        output_log_utf8=decide_boolean_from_env("VV_OUTPUT_LOG_UTF8"),
        cpu_num_threads=os.getenv("VV_CPU_NUM_THREADS"),
        env_preset_path=os.getenv("VV_PRESET_FILE"),
        disable_mutable_api=decide_boolean_from_env("VV_DISABLE_MUTABLE_API"),
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
    host: str
    port: int
    use_gpu: bool
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


def read_cli_arguments(envs: Envs) -> _CLIArgs:
    """コマンドライン引数を読み込む。"""
    parser = argparse.ArgumentParser(
        description="AivisSpeech Engine: AI Voice Imitation System - Text to Speech Engine"
    )
    # Uvicorn でバインドするアドレスを "localhost" にすることで IPv4 (127.0.0.1) と IPv6 ([::1]) の両方でリッスンできます.
    # これは Uvicorn のドキュメントに記載されていない挙動です; 将来のアップデートにより動作しなくなる可能性があります.
    # ref: https://github.com/VOICEVOX/voicevox_engine/pull/647#issuecomment-1540204653
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="接続を受け付けるホストアドレスです。",
    )
    parser.add_argument(
        "--port", type=int, default=10101, help="接続を受け付けるポート番号です。"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="対応している場合、GPU を使い音声合成処理を行います。",
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
        choices=list(CorsPolicyMode),
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

    # --host に 127.0.0.1 が指定されたとき、Windows 上で localhost でアクセスした際に
    # IPv6 でバインドされないことによる接続遅延を防ぐために、代わりに localhost を指定し IPv4 と IPv6 の両方でバインドする
    # ref: https://github.com/VOICEVOX/voicevox_engine/issues/1480
    if args_dict["host"] == "127.0.0.1":
        args_dict["host"] = "localhost"

    args = _cli_args_adapter.validate_python(args_dict)

    return args


def main() -> None:
    """AivisSpeech Engine を実行する"""
    try:
        multiprocessing.freeze_support()

        # ユーザーの環境変数で hf_transfer が有効化されている場合に備え、事前に無効化しておく
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        envs = read_environment_variables()
        if envs.output_log_utf8:
            set_output_log_utf8()

        args = read_cli_arguments(envs)
        if args.output_log_utf8:
            set_output_log_utf8()

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

        # 起動時の可能な限り早い段階で実行結果をキャッシュしておくのが重要
        collect_runtime_environment(
            inference_type="GPU" if args.use_gpu is True else "CPU"
        )

        logger.info(f"AivisSpeech Engine version {__version__}")
        if args.disable_sentry:
            logger.info("Sentry error tracking is disabled.")
        logger.info(f"Engine root directory: {engine_root()}")
        logger.info(f"User data directory: {get_save_dir()}")

        # AivmManager を初期化
        aivm_manager = AivmManager(get_save_dir() / "Models")

        # ごく稀に style_bert_vits2_tts_engine.py (が依存する onnxruntime) のインポート自体に失敗し
        # 例外が発生する環境があるようなので、例外をキャッチしてエラーログに出力できるよう、敢えてルーター初期化時にインポートする
        from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
            StyleBertVITS2TTSEngine,
        )

        # AivisSpeech Engine 独自の StyleBertVITS2TTSEngine を通常の TTSEngine の代わりに利用
        tts_engines = TTSEngineManager()
        tts_engines.register_engine(
            StyleBertVITS2TTSEngine(aivm_manager, args.use_gpu, args.load_all_models),
            MOCK_VER,
        )

        core_manager = initialize_cores(
            use_gpu=args.use_gpu,
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
                use_gpu=args.use_gpu,
                voicelib_dirs=args.voicelib_dirs,
                voicevox_dir=args.voicevox_dir,
                runtime_dirs=args.runtime_dirs,
                cpu_num_threads=args.cpu_num_threads,
                enable_mock=args.enable_mock,
            )

        setting_loader = SettingHandler(args.setting_file)
        settings = setting_loader.load()

        # 複数方式で指定可能な場合、優先度は上から「引数」「環境変数」「設定ファイル」「デフォルト値」

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

        if args.disable_mutable_api:
            disable_mutable_api = True
        else:
            disable_mutable_api = envs.disable_mutable_api

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

        # 起動処理にのみに要したメモリを開放
        gc.collect()

        # AivisSpeech Engine サーバーを起動
        # NOTE: デフォルトは ASGI に準拠した HTTP/1.1 サーバー
        uvicorn.run(app, host=args.host, port=args.port, log_config=LOGGING_CONFIG)

    except Exception as e:
        logger.error("Unexpected error occurred during engine startup:", exc_info=e)
        raise e


if __name__ == "__main__":
    main()
