"""/speaker_info API のテスト。"""

from typing import Any

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import hash_long_string


def _get_first_speaker_uuid(client: TestClient) -> str:
    """
    インストール済み話者の先頭の speaker_uuid を動的に取得する。

    Parameters
    ----------
    client : TestClient
        AivisSpeech Engine へ HTTP リクエストを送信するクライアント。

    Returns
    -------
    str
        先頭の話者の speaker_uuid。
    """

    response = client.get("/speakers", params={})
    assert response.status_code == 200
    speakers: list[dict[str, Any]] = response.json()
    assert len(speakers) > 0
    speaker_uuid: str = speakers[0]["speaker_uuid"]
    return speaker_uuid


def test_get_speaker_info_200(
    client_with_default_model: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    client = client_with_default_model
    speaker_uuid = _get_first_speaker_uuid(client)
    response = client.get("/speaker_info", params={"speaker_uuid": speaker_uuid})
    assert response.status_code == 200
    assert snapshot_json == hash_long_string(response.json())


def test_get_speaker_info_with_url_200(
    client_with_default_model: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    client = client_with_default_model
    speaker_uuid = _get_first_speaker_uuid(client)
    response = client.get(
        "/speaker_info",
        params={
            "speaker_uuid": speaker_uuid,
            "resource_format": "url",
        },
    )
    assert response.status_code == 200
    assert snapshot_json == hash_long_string(response.json())


def test_get_speaker_info_404(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    response = client.get(
        "/speaker_info", params={"speaker_uuid": "111a111a-1a11-1aa1-1a1a-1a11a1aa11a1"}
    )
    assert response.status_code == 404
    assert snapshot_json == hash_long_string(response.json())
