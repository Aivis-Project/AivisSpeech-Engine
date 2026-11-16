"""proxy_utility のテスト。"""

import os

import httpx2 as httpx
import pytest

from voicevox_engine.utility.proxy_utility import (
    remove_invalid_proxy_environment_variables,
)

_PROXY_ENVIRONMENT_VARIABLE_NAMES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _clear_proxy_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # テストを実行する環境のプロキシ設定の影響を受けないよう、事前にすべて取り除く
    for name in _PROXY_ENVIRONMENT_VARIABLE_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "proxy_value",
    [
        # 実際の利用環境で起動失敗の原因になった値
        "::1]",
        # socksio が同梱されていないため httpx が ImportError を送出する値
        "socks5://127.0.0.1:1080",
        # httpx が対応していないスキーム
        "socks4://127.0.0.1:1080",
    ],
)
def test_remove_invalid_proxy_environment_variables_removes_unusable_value(
    monkeypatch: pytest.MonkeyPatch, proxy_value: str
) -> None:
    """httpx が利用できないプロキシの環境変数は取り除かれ、その後は httpx のクライアントを生成できる"""

    monkeypatch.setenv("ALL_PROXY", proxy_value)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")

    # 取り除く前は、環境変数を参照する httpx のクライアントの生成自体が失敗する
    with pytest.raises((httpx.InvalidURL, ValueError, ImportError)):
        httpx.Client()

    removed_names = remove_invalid_proxy_environment_variables()

    assert removed_names == ["ALL_PROXY"]
    # 正しい値のプロキシ設定は残す
    assert "HTTPS_PROXY" in os.environ
    httpx.Client().close()


def test_remove_invalid_proxy_environment_variables_removes_unusable_no_proxy_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx が解析できない NO_PROXY の項目だけが取り除かれ、残りの項目とプロキシ設定は維持される"""

    # 角括弧付きの IPv6 アドレスは一般的な書き方だが、httpx は解析できずクライアントの生成が失敗する
    monkeypatch.setenv("NO_PROXY", "[::1],localhost,192.168.0.0/16,.example.com")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")

    with pytest.raises(httpx.InvalidURL):
        httpx.Client()

    removed_names = remove_invalid_proxy_environment_variables()

    assert "NO_PROXY" in removed_names
    assert os.environ["NO_PROXY"] == "localhost,192.168.0.0/16,.example.com"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.example.com:8080"
    httpx.Client().close()


def test_remove_invalid_proxy_environment_variables_keeps_valid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正しい値のプロキシの環境変数は取り除かれない"""

    monkeypatch.setenv("ALL_PROXY", "[::1]:7890")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:8080")
    monkeypatch.setenv("NO_PROXY", "localhost,::1,*.example.com")

    assert remove_invalid_proxy_environment_variables() == []
    assert os.environ["NO_PROXY"] == "localhost,::1,*.example.com"
