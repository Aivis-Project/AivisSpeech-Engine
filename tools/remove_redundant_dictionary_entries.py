# flake8: noqa
"""
Usage: python remove_redundant_dictionary_entries.py

resources/dictionaries ディレクトリにある辞書データのうち、pyopenjtalk デフォルト辞書で得られる読みと一致する単語を削除するツール。
"""

import csv
import re
import shutil
from pathlib import Path

import pyopenjtalk


def get_default_reading_pronunciation(text: str) -> tuple[str, str]:
    """pyopenjtalk のデフォルト辞書から読みと発音を取得する"""
    njd_features = pyopenjtalk.run_frontend(text)

    reads: list[str] = []
    prons: list[str] = []
    for n in njd_features:
        if n["pos"] == "記号":
            r = n["string"]
            p = n["string"]
        else:
            r = n["read"]
            p = n["pron"]
        # remove special chars
        for c in "’":
            r = r.replace(c, "")
            p = p.replace(c, "")
        reads.append(r)
        prons.append(p)
    return "".join(reads), "".join(prons)


def process_csv_file(file_path: str) -> tuple[int, list[list[str]]]:
    """CSV ファイルを処理し、デフォルト辞書の読みと一致する行を削除する"""
    unique_rows: list[list[str]] = []
    removed_rows: list[list[str]] = []
    total_rows = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
    processed_rows = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            processed_rows += 1
            if processed_rows % 100 == 0:  # 100行ごとに進捗を表示
                print(f"Processing... {processed_rows}/{total_rows} rows")

            if not row:  # 空行をスキップ
                continue

            # CSV の形式は 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音 を想定
            # : は除去してから比較
            surface = row[0]
            reading = row[11].replace(":", "")
            pronunciation = row[12].replace(":", "")

            # surface がひらがな・カタカナのみで構成される場合、かつ3文字以上の場合は、pyopenjtalk が苦手とする
            # ひらがな・カタカナ単語の分かち書き強化のために意図的に残す
            if re.match(r"^[\u3040-\u309F\u30A0-\u30FF]{3,}$", surface):
                unique_rows.append(row)
                continue

            # デフォルト辞書のみ適用した pyopenjtalk から読みと発音を取得
            default_reading, default_pronunciation = get_default_reading_pronunciation(surface)  # fmt: skip
            default_reading_without_special_chars = default_reading.replace(
                "・", ""
            ).replace("　", "")
            default_pronunciation_without_special_chars = default_pronunciation.replace(
                "・", ""
            ).replace("　", "")

            # デフォルト辞書のみ適用した pyopenjtalk の発音と完全一致する場合は削除
            ## pyopenjtalk から取得した発音には「・」や全角スペースが含まれることがあるが、Mecab 辞書データの発音には含まれていないことが多いので、
            ## 除去した状態でも一致する場合は削除する
            if (
                default_pronunciation == pronunciation
                or default_pronunciation_without_special_chars == pronunciation
            ):
                removed_rows.append(row)
                print(
                    f'Removed: {surface} → {default_pronunciation} (Pron) | \033[91m{",".join(row)}\033[0m'
                )
            # そうでないが、デフォルト辞書の読みと完全一致する場合は削除
            elif (
                default_reading == reading
                or default_reading_without_special_chars == reading
            ):
                removed_rows.append(row)
                print(
                    f'Removed: {surface} → {default_reading} (Read) | \033[91m{",".join(row)}\033[0m'
                )
            else:
                unique_rows.append(row)

    print(f"\nProcessed all {total_rows} rows")

    # 重複を削除した結果を書き込む
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(unique_rows)

    # ファイルの末尾の改行を削除
    with open(file_path, "rb+") as f:
        f.seek(-1, 2)
        last_char = f.read(1)
        if last_char == b"\n":
            f.seek(-1, 2)
            f.truncate()

    return len(unique_rows), removed_rows


def print_removed_entries(file_name: str, removed_rows: list[list[str]]) -> None:
    """削除された行を赤色で出力する"""
    print(f"\nSummary of removed entries from {file_name}:")
    print("-" * shutil.get_terminal_size().columns)
    for row in removed_rows:
        print(f'\033[91m{",".join(row)}\033[0m')  # 赤色で出力
        print("-" * shutil.get_terminal_size().columns)


def remove_redundant_dictionary_entries() -> None:
    dict_dir = Path("resources/dictionaries")
    total_removed = 0

    for file_path in sorted(dict_dir.glob("*.csv")):
        # 01_ から始まる辞書は手動生成された辞書なのでスキップ
        if file_path.name.startswith("01_"):
            continue

        print(f"\nProcessing {file_path.name}...")
        original_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
        new_count, removed_rows = process_csv_file(str(file_path))
        removed = original_count - new_count
        total_removed += removed

        print(f"Finished {file_path.name}: Removed {removed} entries")
        print_removed_entries(file_path.name, removed_rows)

    print(f"\nTotal removed entries: {total_removed}")


if __name__ == "__main__":
    remove_redundant_dictionary_entries()