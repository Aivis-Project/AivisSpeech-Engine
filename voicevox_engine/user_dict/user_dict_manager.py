"""ユーザー辞書関連の処理"""

import gc
import json
import sys
import threading
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pyopenjtalk
from pydantic import TypeAdapter, ValidationError

from voicevox_engine.logging import logger
from voicevox_engine.user_dict.constants import (
    PART_OF_SPEECH_DATA,
    WordProperty,
    WordTypes,
    cost2priority,
    priority2cost,
)
from voicevox_engine.user_dict.model import UserDictInputError, UserDictWord
from voicevox_engine.utility.path_utility import get_save_dir, resource_root

# リソースディレクトリと保存ディレクトリのパス
_resource_dir = resource_root()
_save_dir = get_save_dir()
if not _save_dir.is_dir():
    _save_dir.mkdir(parents=True)

# ビルド済みデフォルトユーザー辞書ディレクトリのパス
DEFAULT_DICT_DIR_PATH = _resource_dir / "dictionaries"
# ユーザー辞書保存ファイルのパス
_USER_DICT_PATH = _save_dir / "user_dict.json"


class UserDictionaryRepository:
    """
    JSON ファイルを SSoT として扱う、ユーザー辞書リポジトリ。
    ユーザー辞書保存ファイルの読み書きと排他制御を行う。
    """

    def __init__(self, user_dict_path: Path) -> None:
        """
        UserDictionaryRepository のコンストラクタ。

        Parameters
        ----------
        user_dict_path : Path
            ユーザー辞書保存ファイルのパス
        """

        self.user_dict_path = user_dict_path
        self._user_dict_adapter = TypeAdapter(dict[str, UserDictWord])
        self._lock = threading.Lock()

    def load(self) -> dict[str, UserDictWord]:
        """
        ユーザー辞書データを SSoT からロードする。

        Returns
        -------
        dict[str, UserDictWord]
            ユーザー辞書データ
        """

        with self._lock:
            if not self.user_dict_path.exists():
                return {}
            try:
                # まず通常のバリデーションを試みる
                with open(self.user_dict_path, encoding="utf-8") as f:
                    # 中身が空なら {} を返す
                    content = f.read().strip()
                    if not content:
                        return {}
                    user_dict = self._user_dict_adapter.validate_json(content)
                # バリデーションに成功した場合でも、stem の値が要素数1で、[0] の中身が "*" なら surface に変更する
                for word_uuid, word in user_dict.items():
                    if len(word.stem) == 1 and word.stem[0] == "*" and word.surface:
                        logger.info(
                            f"UserDictionaryRepository: [{word_uuid}] stem ({word.stem}) is '*'. "  # noqa
                            + f"Use surface ({word.surface}) instead."
                        )
                        word.stem = [word.surface]
                return user_dict
            except JSONDecodeError as ex:
                # ファイルの中身が JSON 形式でない場合、空の辞書を返す
                logger.warning(
                    "UserDictionaryRepository: JSON decode error occurred.",
                    exc_info=ex,
                )
                return {}
            except ValidationError as ex:
                # バリデーションエラーが発生した場合、マイグレーション処理を行う
                logger.warning(
                    "UserDictionaryRepository: Validation error occurred. Trying migration...",
                    exc_info=ex,
                )
                with open(self.user_dict_path, encoding="utf-8") as f:
                    # マイグレーション処理のために一旦自前で JSON をパースする
                    raw_word_dict: dict[str, Any] = json.load(f)
                for word_uuid, raw_word in raw_word_dict.items():
                    # VOICEVOX ENGINE 0.12 以前の辞書は、context_id がハードコーディングされていたためにユーザー辞書内に保管されていない
                    # ハードコーディングされていた context_id は固有名詞を意味するものなので、固有名詞の context_id を補完する
                    context_id_p_noun = PART_OF_SPEECH_DATA[
                        WordTypes.PROPER_NOUN
                    ].context_id
                    if "context_id" not in raw_word or raw_word["context_id"] is None:
                        raw_word["context_id"] = context_id_p_noun
                    # AivisSpeech Engine 1.1.0 以前の辞書では priority (項目の優先度) ではなく
                    # cost (MeCab 辞書上の生起コスト) を保存していたため、priority が
                    # 設定されていない場合は cost を元に priority を計算する
                    if "priority" not in raw_word or raw_word["priority"] is None:
                        assert "context_id" in raw_word
                        assert raw_word["context_id"] is not None
                        assert "cost" in raw_word
                        assert raw_word["cost"] is not None
                        raw_word["priority"] = cost2priority(
                            raw_word["context_id"], raw_word["cost"]
                        )
                        logger.warning(
                            f"UserDictionaryRepository: [{word_uuid}] priority is not set. "
                            + "Use cost to calculate priority. "
                            + f"(cost: {raw_word['cost']} -> priority: {raw_word['priority']})"
                        )
                    # AivisSpeech Engine 1.1.0 以前の辞書では stem, yomi, pronunciation,
                    # accent_type, mora_count が文字列として保存されていたため、リストに変換する
                    if "stem" in raw_word and isinstance(raw_word["stem"], str):
                        # もし stem が "*" の場合は stem の代わりに surface を使う
                        # 上記仕様変更により、stem の値は省略せず surface の値が入るように変更されたため
                        if (
                            raw_word["stem"] == "*"
                            and "surface" in raw_word
                            and isinstance(raw_word["surface"], str)
                        ):
                            raw_word["stem"] = [raw_word["surface"]]
                        else:
                            raw_word["stem"] = [raw_word["stem"]]
                    if "yomi" in raw_word and isinstance(raw_word["yomi"], str):
                        raw_word["yomi"] = [raw_word["yomi"]]
                    if "pronunciation" in raw_word and isinstance(raw_word["pronunciation"], str):  # fmt: skip # noqa
                        raw_word["pronunciation"] = [raw_word["pronunciation"]]
                    if "accent_type" in raw_word and isinstance(raw_word["accent_type"], int):  # fmt: skip # noqa
                        raw_word["accent_type"] = [raw_word["accent_type"]]
                    if "mora_count" in raw_word and isinstance(raw_word["mora_count"], int):  # fmt: skip # noqa
                        raw_word["mora_count"] = [raw_word["mora_count"]]
                # マイグレーション処理が完了した辞書データをバリデーション
                migrated_data = self._user_dict_adapter.validate_python(raw_word_dict)
                # ファイルに書き込む (デッドロックになるので自前で書き込む)
                with open(self.user_dict_path, mode="w", encoding="utf-8") as f:
                    f.write(self._user_dict_adapter.dump_json(migrated_data, indent=4).decode("utf-8"))  # fmt: skip # noqa
                logger.info(
                    "UserDictionaryRepository: Dictionary migration completed and saved."
                )
                return migrated_data

    def save(self, data: dict[str, UserDictWord]) -> None:
        """
        ユーザー辞書データを SSoT へ保存する。

        Parameters
        ----------
        data : dict[str, UserDictWord]
            保存するユーザー辞書データ
        """

        with self._lock, open(self.user_dict_path, mode="w", encoding="utf-8") as f:
            f.write(self._user_dict_adapter.dump_json(data, indent=4).decode("utf-8"))


class UserDictionary:
    """ユーザー辞書を管理するクラス。"""

    def __init__(
        self,
        default_dict_dir_path: Path = DEFAULT_DICT_DIR_PATH,
        user_dict_path: Path = _USER_DICT_PATH,
    ) -> None:
        """
        UserDictionary のコンストラクタ。

        Parameters
        ----------
        default_dict_dir_path : Path
            ビルド済みデフォルトユーザー辞書ディレクトリのパス
        user_dict_path : Path
            ユーザー辞書保存ファイルのパス
        """

        self._default_dict_dir_path = default_dict_dir_path
        self._user_dict_path = user_dict_path
        self._repository = UserDictionaryRepository(user_dict_path)
        self._lock = threading.Lock()

        # pytest から実行されているかどうか
        self._is_pytest = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]

        # 起動時に一時的な辞書バイナリファイルを削除
        try:
            # 一時ファイルのパターンに一致するファイルを取得
            parent_dir = self._user_dict_path.parent
            # バイナリ辞書と CSV の両方のパターンを処理
            tmp_patterns = ["user.dict_compiled-*.tmp", "user.dict_csv-*.tmp"]
            for pattern in tmp_patterns:
                tmp_files = parent_dir.glob(pattern)
                for tmp_file in tmp_files:
                    try:
                        if tmp_file.exists():
                            _delete_file_on_close(tmp_file)
                    except Exception as ex:
                        logger.warning(
                            f"Failed to delete temporary file {tmp_file}:",
                            exc_info=ex,
                        )
        except Exception as ex:
            logger.warning("Failed to cleanup temporary files:", exc_info=ex)

        # 初回起動時などまだユーザー辞書 JSON が存在しない場合、辞書登録例として「担々麺」の辞書エントリを書き込む
        # pytest から実行されている場合は書き込まない
        if not self._user_dict_path.is_file() and not self._is_pytest:
            self._repository.save({
                "dc94a187-9881-43c9-a9c1-cebbf774a96d": UserDictWord.from_word_property(
                    WordProperty(
                        surface=["担々麺"],
                        pronunciation=["タンタンメン"],
                        accent_type=[3],
                        word_type=WordTypes.PROPER_NOUN,
                        priority=5,
                    )
                ),
            })  # fmt: skip

        # 現在 UserDictionaryRepository に保持されているユーザー辞書データを OpenJTalk に適用
        self.apply_jtalk_dictionary()

    def get_all_words(self) -> dict[str, UserDictWord]:
        """
        ユーザー辞書に登録されているすべての単語を取得する。

        Returns
        -------
        dict[str, UserDictWord]
            ユーザー辞書に登録されているすべての単語の情報
        """

        return self._repository.load()

    def add_word(self, word_property: WordProperty) -> str:
        """
        新規単語をユーザー辞書に追加し、その単語に割り当てられた UUID を返す。

        Parameters
        ----------
        word_property : WordProperty
            追加する単語の情報

        Returns
        -------
        str
            追加した単語の UUID
        """

        # 新規単語の追加による辞書データの更新
        user_dict = self._repository.load()
        word_uuid = str(uuid4())
        user_dict[word_uuid] = UserDictWord.from_word_property(word_property)

        # 更新された辞書データを保存・適用
        self._repository.save(user_dict)
        self.apply_jtalk_dictionary()

        # 追加した単語の UUID を返す
        return word_uuid

    def update_word(self, word_uuid: str, word_property: WordProperty) -> None:
        """
        単語 UUID で指定された単語をユーザー辞書内で上書き更新する。

        Parameters
        ----------
        word_uuid : str
            更新する単語の UUID
        word_property : WordProperty
            更新する単語の情報
        """

        # 既存単語の上書きによる辞書データの更新
        user_dict = self._repository.load()
        if word_uuid not in user_dict:
            raise UserDictInputError("UUID に該当する単語が見つかりませんでした。")
        user_dict[word_uuid] = UserDictWord.from_word_property(word_property)

        # 更新された辞書データを保存・適用
        self._repository.save(user_dict)
        self.apply_jtalk_dictionary()

    def delete_word(self, word_uuid: str) -> None:
        """
        単語 UUID で指定された単語をユーザー辞書から削除する。

        Parameters
        ----------
        word_uuid : str
            削除する単語の UUID
        """

        # 既存単語の削除による辞書データの更新
        user_dict = self._repository.load()
        if word_uuid not in user_dict:
            raise UserDictInputError("UUID に該当する単語が見つかりませんでした。")
        del user_dict[word_uuid]

        # 更新された辞書データを保存・適用
        self._repository.save(user_dict)
        self.apply_jtalk_dictionary()

    def import_dictionary(
        self, dict_data: dict[str, UserDictWord], override: bool = False
    ) -> None:
        """
        ユーザー辞書をインポートする。

        Parameters
        ----------
        dict_data : dict[str, UserDictWord]
            インポートするユーザー辞書のデータ
        override : bool
            重複したエントリがあった場合、上書きするかどうか
        """

        # インポートする辞書データのバリデーション
        for word_uuid, word in dict_data.items():
            UUID(word_uuid)
            for pos_detail in PART_OF_SPEECH_DATA.values():
                if word.context_id == pos_detail.context_id:
                    assert word.part_of_speech == pos_detail.part_of_speech
                    assert (
                        word.part_of_speech_detail_1
                        == pos_detail.part_of_speech_detail_1
                    )
                    assert (
                        word.part_of_speech_detail_2
                        == pos_detail.part_of_speech_detail_2
                    )
                    assert (
                        word.part_of_speech_detail_3
                        == pos_detail.part_of_speech_detail_3
                    )
                    assert (
                        word.accent_associative_rule
                        in pos_detail.accent_associative_rules
                    )
                    break
            else:
                raise UserDictInputError("対応していない品詞です。")

        # 既存辞書の読み出し
        old_dict = self._repository.load()

        # 辞書データの更新
        # 重複エントリの上書き
        if override:
            new_dict = {**old_dict, **dict_data}
        # 重複エントリの保持
        else:
            new_dict = {**dict_data, **old_dict}

        # 更新された辞書データを保存・適用
        self._repository.save(new_dict)
        self.apply_jtalk_dictionary()

    def apply_jtalk_dictionary(self) -> None:
        """
        現在 UserDictionaryRepository に保持されているユーザー辞書データを OpenJTalk (MeCab) 用の辞書データに変換し、
        現在のプロセスで実行される全ての pyopenjtalk 呼び出しに、設定したユーザー辞書データを反映させる。
        """

        default_dict_dir_path = self._default_dict_dir_path
        user_dict_path = self._user_dict_path

        # pytest 実行時かつ Windows ではなぜか辞書更新時に MeCab の初期化に失敗するので、辞書更新自体を無効化する
        if self._is_pytest and sys.platform == "win32":
            return

        # 一時保存ファイルのパスを生成
        random_string = uuid4()
        tmp_csv_path = user_dict_path.with_name(f"user.dict_csv-{random_string}.tmp")
        tmp_compiled_path = user_dict_path.with_name(
            f"user.dict_compiled-{random_string}.tmp"
        )

        # 排他制御を行う
        with self._lock:
            start_time = time.time()
            try:
                # 現在のユーザー辞書データをロード
                user_dict = self._repository.load()

                # CSV 形式の OpenJTalk (MeCab) 用辞書データを作成
                logger.info("Current user dictionary (CSV format):")
                csv_text = ""
                for word in user_dict.values():
                    # アクセント位置とモーラ数はスラッシュで結合する
                    accent_type_mora_count: list[str] = []
                    for accent_type, mora_count in zip(
                        word.accent_type, word.mora_count, strict=False
                    ):
                        accent_type_mora_count.append(f"{accent_type}/{mora_count}")
                    # UserDictWord 内の情報を CSV のフォーマットに変換
                    csv_row = (
                        "{surface},{context_id},{context_id},{cost},{part_of_speech},"
                        + "{part_of_speech_detail_1},{part_of_speech_detail_2},"
                        + "{part_of_speech_detail_3},{inflectional_type},"
                        + "{inflectional_form},{stem},{yomi},{pronunciation},"
                        + "{accent_type_mora_count},{accent_associative_rule}\n"
                    ).format(
                        surface=word.surface,
                        context_id=word.context_id,
                        cost=priority2cost(word.context_id, word.priority),
                        part_of_speech=word.part_of_speech,
                        part_of_speech_detail_1=word.part_of_speech_detail_1,
                        part_of_speech_detail_2=word.part_of_speech_detail_2,
                        part_of_speech_detail_3=word.part_of_speech_detail_3,
                        inflectional_type=word.inflectional_type,
                        inflectional_form=word.inflectional_form,
                        # 以下、アクセント句が複数ある場合は半角コロンで結合する（アクセント結合規則を除く）
                        stem=":".join(word.stem),
                        yomi=":".join(word.yomi),
                        pronunciation=":".join(word.pronunciation),
                        accent_type_mora_count=":".join(accent_type_mora_count),
                        accent_associative_rule=word.accent_associative_rule,
                    )
                    logger.info("- " + csv_row.strip())
                    csv_text += csv_row

                # この時点で csv_text が空文字列のとき、ユーザー辞書が空なため処理を終了する
                # ユーザー辞書 CSV が空の状態で継続すると pyopenjtalk.mecab_dict_index() 実行時に
                # Segmentation Fault が発生するのを回避する
                if not csv_text:
                    logger.info("User dictionary is empty. Skipping dictionary update.")
                    return

                # 辞書データを辞書.csv へ一時保存
                tmp_csv_path.write_text(csv_text, encoding="utf-8")

                # 辞書.csv を OpenJTalk 用にビルド
                pyopenjtalk.mecab_dict_index(str(tmp_csv_path), str(tmp_compiled_path))
                if not tmp_compiled_path.is_file():
                    raise RuntimeError("辞書のビルド時にエラーが発生しました。")

                # ユーザー辞書の適用を解除
                pyopenjtalk.unset_user_dict()

                # デフォルトユーザー辞書ディレクトリにある *.dic ファイルを名前順に取得
                dict_files = sorted(list(default_dict_dir_path.glob("**/*.dic")))
                # 先ほどビルドしたユーザー辞書ファイルのパスを追加
                dict_files.append(tmp_compiled_path)

                # ユーザー辞書を pyopenjtalk に適用
                # デフォルトのユーザー辞書ファイルと、先ほどビルドしたユーザー辞書ファイルの両方を指定する
                # NOTE: resolve() によりコンパイル実行時でも相対パスを正しく認識できる
                dict_paths = [str(p.resolve(strict=True)) for p in dict_files]
                if dict_paths:  # 辞書ファイルが1つ以上存在する場合のみ実行
                    pyopenjtalk.update_global_jtalk_with_user_dict(dict_paths)

                logger.info(
                    f"User dictionary applied. ({time.time() - start_time:.2f}s)"
                )

            except Exception as ex:
                logger.error(
                    f"Failed to apply user dictionary. ({time.time() - start_time:.2f}s)",
                    exc_info=ex,
                )
                raise

            finally:
                # 後処理
                if tmp_csv_path.exists():
                    tmp_csv_path.unlink()
                if tmp_compiled_path.exists():
                    _delete_file_on_close(tmp_compiled_path)

                # 強制的にメモリを開放
                gc.collect()


def _delete_file_on_close(file_path: Path) -> None:
    """
    ファイルのハンドルが全て閉じたときにファイルを削除する。OpenJTalk 用のカスタム辞書用。

    Windowsでは CreateFileW() 関数で `FILE_FLAG_DELETE_ON_CLOSE` を付けてすぐに閉じることで、
    `FILE_SHARE_DELETE` を付けて開かれているファイルのハンドルが全て閉じた時に削除されるようにする。

    Windows 以外では即座にファイルを削除する。

    Parameters
    ----------
    file_path : Path
        削除するファイルのパス
    """

    if sys.platform == "win32":
        import ctypes
        from ctypes.wintypes import DWORD, HANDLE, LPCWSTR

        _CreateFileW = ctypes.windll.kernel32.CreateFileW
        _CreateFileW.argtypes = [
            LPCWSTR,
            DWORD,
            DWORD,
            ctypes.c_void_p,
            DWORD,
            DWORD,
            HANDLE,
        ]
        _CreateFileW.restype = HANDLE
        _CloseHandle = ctypes.windll.kernel32.CloseHandle
        _CloseHandle.argtypes = [HANDLE]

        _FILE_SHARE_DELETE = 0x00000004
        _FILE_SHARE_READ = 0x00000001
        _OPEN_EXISTING = 3
        _FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
        _INVALID_HANDLE_VALUE = HANDLE(-1).value

        h_file = _CreateFileW(
            str(file_path),
            0,
            _FILE_SHARE_DELETE | _FILE_SHARE_READ,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_DELETE_ON_CLOSE,
            None,
        )
        if h_file == _INVALID_HANDLE_VALUE:
            raise RuntimeError(
                f"Failed to CreateFileW for {file_path}"
            ) from ctypes.WinError()

        result = _CloseHandle(h_file)
        if result == 0:
            raise RuntimeError(
                f"Failed to CloseHandle for {file_path}"
            ) from ctypes.WinError()
    else:
        file_path.unlink()
