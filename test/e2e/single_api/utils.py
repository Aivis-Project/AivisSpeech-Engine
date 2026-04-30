"""単独 API に対する E2E テスト共通のユーティリティ。"""

from typing import Any, TypedDict

from fastapi.testclient import TestClient


class _MoraForTest(TypedDict):
    text: str
    consonant: str
    consonant_length: float
    vowel: str
    vowel_length: float
    pitch: float


def get_first_style_id(client: TestClient) -> int:
    """
    インストール済み話者の先頭のスタイル ID を取得する。

    Parameters
    ----------
    client : TestClient
        AivisSpeech Engine へ HTTP リクエストを送信するクライアント。

    Returns
    -------
    int
        先頭のスタイル ID。
    """

    response = client.get("/speakers", params={})
    assert response.status_code == 200
    speakers: list[dict[str, Any]] = response.json()
    assert len(speakers) > 0
    assert len(speakers[0]["styles"]) > 0, (
        f"Speaker {speakers[0]['name']} has no styles."
    )
    style_id: int = speakers[0]["styles"][0]["id"]
    return style_id


def gen_mora(
    text: str,
    consonant: str,
    consonant_length: float,
    vowel: str,
    vowel_length: float,
    pitch: float,
) -> _MoraForTest:
    """モーラを生成する。"""
    return {
        "text": text,
        "consonant": consonant,
        "consonant_length": consonant_length,
        "vowel": vowel,
        "vowel_length": vowel_length,
        "pitch": pitch,
    }
