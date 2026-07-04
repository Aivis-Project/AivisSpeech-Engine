"""AivisSpeech Engine の Sentry issue をローカルフィルタへ通す確認スクリプト。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _request_json(url: str, token: str) -> tuple[Any, str]:
    """
    Sentry API から JSON と Link ヘッダーを取得する

    Args:
        url (str): 取得する Sentry API URL
        token (str): Sentry API トークン

    Returns
    -------
        tuple[Any, str]: JSON と Link ヘッダー
    """

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8")), response.headers.get(
            "Link", ""
        )


def _extract_next_url(link_header: str) -> str | None:
    """
    Sentry の Link ヘッダーから次ページ URL を取り出す

    Args:
        link_header (str): Sentry API の Link ヘッダー

    Returns
    -------
        str | None: 次ページ URL、または次ページがない場合は None
    """

    for link_part in link_header.split(","):
        # Sentry は次ページがあっても results="false" を返すため、実データがあるページだけ進む
        if 'rel="next"' not in link_part or 'results="true"' not in link_part:
            continue

        start = link_part.find("<")
        end = link_part.find(">")
        if start != -1 and end != -1:
            return link_part[start + 1 : end]

    return None


def _iter_issues(
    *,
    org: str,
    project_id: str,
    query: str,
    stats_period: str,
    token: str,
) -> Iterator[dict[str, Any]]:
    """
    organization issues API から issue をページング取得する

    Args:
        org (str): Sentry organization slug
        project_id (str): Sentry project ID
        query (str): Sentry issue 検索クエリ
        stats_period (str): Sentry statsPeriod
        token (str): Sentry API トークン

    Yields
    ------
        dict[str, Any]: Sentry issue
    """

    params = urllib.parse.urlencode(
        {
            "project": project_id,
            "query": query,
            "limit": "100",
            "statsPeriod": stats_period,
        }
    )
    url: str | None = f"https://sentry.io/api/0/organizations/{org}/issues/?{params}"

    while url is not None:
        issues, link_header = _request_json(url, token)
        if not isinstance(issues, list):
            raise RuntimeError("Sentry issues response was not a list")

        for issue in issues:
            if isinstance(issue, dict):
                yield issue

        url = _extract_next_url(link_header)


def _build_title_only_event(issue: dict[str, Any]) -> dict[str, Any]:
    """
    issue 一覧だけから `before_send` へ渡せる最小イベントを組み立てる

    Args:
        issue (dict[str, Any]): Sentry issue

    Returns
    -------
        dict[str, Any]: `filter_sentry_event()` に渡す疑似 Sentry イベント
    """

    texts = [
        issue.get("title"),
        issue.get("culprit"),
    ]

    return {
        "message": "\n".join(
            str(text) for text in texts if isinstance(text, str) and text != ""
        )
    }


def main() -> int:
    """
    Sentry issue 一覧をローカルフィルタへ通した結果を JSONL で出力する

    Returns
    -------
        int: 終了コード
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--org", default="aivis-project")
    parser.add_argument("--project-id", default="4508555159470080")
    parser.add_argument("--query", default="is:unresolved")
    parser.add_argument("--stats-period", default="90d")
    args = parser.parse_args()

    token = os.environ.get("SENTRY_AUTH_TOKEN")
    if token is None or token == "":
        print("SENTRY_AUTH_TOKEN is not set", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root))

    # 対象 repo の現在の実装を使うため、import は `sys.path` 設定後に行う
    from voicevox_engine.utility.sentry_utility import filter_sentry_event

    for issue in _iter_issues(
        org=args.org,
        project_id=args.project_id,
        query=args.query,
        stats_period=args.stats_period,
        token=token,
    ):
        event = _build_title_only_event(issue)
        decision = "DROP" if filter_sentry_event(event, {}) is None else "KEEP"
        print(
            json.dumps(
                {
                    "shortId": issue.get("shortId"),
                    "decision": decision,
                    "status": issue.get("status"),
                    "substatus": issue.get("substatus"),
                    "title": issue.get("title"),
                    "culprit": issue.get("culprit"),
                    "count": issue.get("count"),
                    "lastSeen": issue.get("lastSeen"),
                },
                ensure_ascii=False,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
