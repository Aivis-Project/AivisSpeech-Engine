"""プロキシ設定に関する utility"""

import os

import httpx2 as httpx

from ..logging import logger

# httpx (urllib.request.getproxies()) が参照するプロキシの環境変数
_PROXY_ENVIRONMENT_VARIABLE_NAMES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)

# httpx (urllib.request.getproxies()) が参照する、プロキシを経由しないホストの環境変数
_NO_PROXY_ENVIRONMENT_VARIABLE_NAMES = ("NO_PROXY", "no_proxy")


def _is_valid_no_proxy_entry(entry: str) -> bool:
    """
    NO_PROXY の 1 項目を httpx が解析できるかを返す。

    httpx は NO_PROXY の各項目を all://[::1] のような URL パターンに変換して解析するため、
    `[::1]` のような角括弧付きの IPv6 アドレスなどはクライアントの生成時に InvalidURL になる。
    判定を httpx の解釈と一致させるため、その項目だけを NO_PROXY に設定した状態で実際にクライアントを生成して確認する。

    Parameters
    ----------
    entry : str
        NO_PROXY の 1 項目

    Returns
    -------
    bool
        httpx が解析できる場合は True
    """

    # 検証対象の項目以外のプロキシ設定の影響を受けないよう、一時的に退避する
    ## NO_PROXY が設定されていれば urllib.request.getproxies() は OS のプロキシ設定を参照しないため、検証中に OS の設定は読まれない
    saved_environment = {
        name: os.environ.pop(name)
        for name in (
            *_PROXY_ENVIRONMENT_VARIABLE_NAMES,
            *_NO_PROXY_ENVIRONMENT_VARIABLE_NAMES,
        )
        if name in os.environ
    }
    os.environ["NO_PROXY"] = entry
    try:
        httpx.Client(trust_env=True).close()
        return True
    except (httpx.InvalidURL, ValueError):
        return False
    finally:
        os.environ.pop("NO_PROXY", None)
        os.environ.update(saved_environment)


def remove_invalid_proxy_environment_variables() -> list[str]:
    """
    httpx が解析できないプロキシの環境変数を取り除き、取り除いた環境変数名のリストを返す。

    httpx は環境変数のプロキシ設定をクライアントの生成時に解析するため、`ALL_PROXY=::1]` のような壊れた値や、
    socksio が同梱されていない SOCKS プロキシの指定があると、AivisHub API や Hugging Face Hub へのリクエストのたびに
    InvalidURL / ValueError / ImportError が発生し、エンジンが起動できなくなる。
    これらは HTTP 通信のエラーではなく利用者の環境設定の問題のため、起動時に 1 回だけ検証し、利用できない値はプロキシなしとして扱う。
    NO_PROXY に `[::1]` のような httpx が解析できない項目がある場合も同様に起動できなくなるため、その項目だけを取り除く。

    Returns
    -------
    list[str]
        取り除いた環境変数名と、項目を取り除いた NO_PROXY の環境変数名のリスト
    """

    removed_names: list[str] = []
    for name in _PROXY_ENVIRONMENT_VARIABLE_NAMES:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            continue

        # httpx と同じく、スキームが省略されている場合は http:// を補って解釈する
        proxy_url = value if "://" in value else f"http://{value}"
        try:
            # trust_env=False にして、検証対象の値以外の環境変数の影響を受けないようにする
            httpx.Client(proxy=proxy_url, trust_env=False).close()
        except (httpx.InvalidURL, ValueError, ImportError) as ex:
            # プロキシの値や例外メッセージには認証情報を含む URL が入りうるため、環境変数名と例外の型だけを記録する
            logger.warning(
                f"Ignoring invalid proxy environment variable {name}. ({type(ex).__name__})"
            )
            del os.environ[name]
            removed_names.append(name)

    # NO_PROXY は項目ごとに解析されるため、解析できない項目だけを取り除き、残りの項目は維持する
    for name in _NO_PROXY_ENVIRONMENT_VARIABLE_NAMES:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            continue

        entries = [entry.strip() for entry in value.split(",")]
        valid_entries = [
            entry for entry in entries if entry == "" or _is_valid_no_proxy_entry(entry)
        ]
        if len(valid_entries) != len(entries):
            # 項目の値は記録せず、環境変数名だけを記録する
            logger.warning(
                f"Ignoring invalid entries in proxy environment variable {name}."
            )
            os.environ[name] = ",".join(valid_entries)
            removed_names.append(name)

    return removed_names
