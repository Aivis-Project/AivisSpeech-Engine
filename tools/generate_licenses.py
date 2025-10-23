"""
AivisSpeech Engine の実行に必要なライブラリのライセンス一覧を作成する。

実行環境にインストールされている Python ライブラリのライセンス一覧を取得する。
一覧に対し、ハードコードされたライセンス一覧を追加する。
ライセンス一覧をファイルとして出力する。
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from pydantic import TypeAdapter


def _get_license_text(text_url: str) -> str:
    """URL が指すテキストを取得する。"""
    with urllib.request.urlopen(text_url) as res:
        # NOTE: `urlopen` 返り値の型が貧弱なため型チェックを無視する
        return res.read().decode()  # type: ignore


@dataclass
class _PipLicense:
    """`pip-license` により得られる依存ライブラリの情報"""

    License: str
    Name: str
    URL: str
    Version: str
    LicenseText: str


_pip_licenses_adapter = TypeAdapter(list[_PipLicense])


def _acquire_licenses_of_pip_managed_libraries() -> list[_PipLicense]:
    """pip-licenses で管理されている依存ライブラリのライセンス情報を取得する。"""
    # ライセンス一覧を取得する
    try:
        pip_licenses_json = subprocess.run(
            [
                sys.executable,
                "-m",
                "piplicenses",
                "--from=mixed",
                "--format=json",
                "--with-urls",
                "--with-license-file",
                "--no-license-path",
            ],
            capture_output=True,
            check=True,
        ).stdout.decode()
    except subprocess.CalledProcessError as e:
        raise Exception(f"command output:\n{e.stderr and e.stderr.decode()}") from e
    # ライセンス情報の形式をチェックする
    licenses = _pip_licenses_adapter.validate_json(pip_licenses_json)
    return licenses


class _License:
    def __init__(
        self,
        package_name: str,
        package_version: str | None,
        license_name: str | None,
        license_text: str,
        license_text_type: Literal["raw", "local_address", "remote_address"],
    ):
        self.package_name = package_name
        self.package_version = package_version
        self.license_name = license_name

        match license_text_type:
            case "raw":
                self.license_text = license_text
            case "local_address":
                # ライセンステキストをローカルのライセンスファイルから抽出する
                self.license_text = Path(license_text).read_text(encoding="utf8")
            case "remote_address":
                self.license_text = _get_license_text(license_text)
            case unreachable:
                assert_never(unreachable)


def _update_licenses(pip_licenses: list[_PipLicense]) -> list[_License]:
    """pip から取得したライセンス情報の抜けを補完する。"""
    package_to_license_url: dict[str, str] = {
        # "package_name": "https://license.adress.com/v0.0.0/LICENSE.txt",
        "gputil": "https://raw.githubusercontent.com/anderskm/gputil/refs/heads/master/LICENSE.txt",
        "jieba": "https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/LICENSE",
        "loguru": "https://raw.githubusercontent.com/Delgan/loguru/0.7.3/LICENSE",
        "safetensors": "https://raw.githubusercontent.com/huggingface/safetensors/v0.4.3/LICENSE",
        "sudachipy": "https://raw.githubusercontent.com/WorksApplications/sudachi.rs/v0.6.8/LICENSE",
        "tokenizers": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.19.1/LICENSE",
    }

    updated_licenses = []

    for pip_license in pip_licenses:
        package_name = pip_license.Name.lower()

        # 開発時のみ利用するライブラリを無視
        if package_name in [
            "annotated-doc",
            "license-expression",
            "packageurl-python",
            "pip-requirements-parser",
            "pyinstaller",
            "pyinstaller-hooks-contrib",
            "typos",
        ]:
            continue

        # ライセンスファイルがリポジトリに存在しないため無視
        if package_name in ["pywin32", "wmi"]:
            continue

        # ライセンス文が pip から取得できていない場合、pip 外から補う
        if pip_license.LicenseText == "UNKNOWN":
            if package_name not in package_to_license_url:
                # ライセンスが PyPI に存在しない
                raise Exception(f"No License info provided for {package_name}")
            text_url = package_to_license_url[package_name]
            pip_license.LicenseText = _get_license_text(text_url)

        updated_licenses.append(
            _License(
                package_name=pip_license.Name,
                package_version=pip_license.Version,
                license_name=pip_license.License,
                license_text=pip_license.LicenseText,
                license_text_type="raw",
            )
        )

    return updated_licenses


def _add_licenses_manually(licenses: list[_License]) -> None:
    python_version = "3.11.9"

    licenses = [
        _License(
            package_name="VOICEVOX ENGINE",
            package_version=None,
            license_name="LGPL license",
            license_text="https://raw.githubusercontent.com/VOICEVOX/voicevox_engine/master/LGPL_LICENSE",
            license_text_type="remote_address",
        ),
        # https://sourceforge.net/projects/open-jtalk/files/Open%20JTalk/open_jtalk-1.11/
        _License(
            package_name="Open JTalk",
            package_version="1.11",
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/COPYING",
            license_text_type="local_address",
        ),
        _License(
            package_name="MeCab",
            package_version=None,
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/mecab/COPYING",
            license_text_type="local_address",
        ),
        _License(
            package_name="NAIST Japanese Dictionary",
            package_version=None,
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/mecab-naist-jdic/COPYING",
            license_text_type="local_address",
        ),
        _License(
            package_name="World",
            package_version=None,
            license_name="Modified BSD license",
            license_text="https://raw.githubusercontent.com/mmorise/World/master/LICENSE.txt",
            license_text_type="remote_address",
        ),
        _License(
            package_name="Python",
            package_version=python_version,
            license_name="Python Software Foundation License",
            license_text=f"https://raw.githubusercontent.com/python/cpython/v{python_version}/LICENSE",
            license_text_type="remote_address",
        ),
    ] + licenses  # 前方に追加


def _generate_licenses() -> list[_License]:
    pip_licenses = _acquire_licenses_of_pip_managed_libraries()
    licenses = _update_licenses(pip_licenses)
    _add_licenses_manually(licenses)

    return licenses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    output_path = args.output_path

    licenses = _generate_licenses()

    # dump
    out = Path(output_path).open("w") if output_path else sys.stdout
    json.dump(
        [
            {
                "name": license.package_name,
                "version": license.package_version,
                "license": license.license_name,
                "text": license.license_text,
            }
            for license in licenses
        ],
        out,
    )
