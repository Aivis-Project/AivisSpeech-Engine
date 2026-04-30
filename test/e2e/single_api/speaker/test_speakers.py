"""/speakers API のテスト。"""

from typing import Any

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import hash_long_string


def _get_default_speaker_uuids(client: TestClient) -> set[str]:
    """
    デフォルトモデルの話者 UUID 一覧を `/aivm_models` API から取得する。

    Parameters
    ----------
    client : TestClient
        AivisSpeech Engine へ HTTP リクエストを送信するクライアント。

    Returns
    -------
    set[str]
        デフォルトモデルに含まれる話者 UUID の集合。
    """

    response = client.get("/aivm_models")
    assert response.status_code == 200

    speaker_uuids: set[str] = set()
    for aivm_info in response.json().values():
        if aivm_info["is_default_model"] is True:
            for speaker in aivm_info["manifest"]["speakers"]:
                speaker_uuids.add(speaker["uuid"])
    return speaker_uuids


def _filter_default_speakers(
    speakers: list[dict[str, Any]],
    default_speaker_uuids: set[str],
) -> list[dict[str, Any]]:
    """
    デフォルトモデル由来の話者のみにフィルタする。

    Parameters
    ----------
    speakers : list[dict[str, Any]]
        `/speakers` API から返された話者一覧。
    default_speaker_uuids : set[str]
        デフォルトモデルに含まれる話者 UUID の集合。

    Returns
    -------
    list[dict[str, Any]]
        デフォルトモデル由来の話者のみを含む一覧。
    """

    return [
        speaker
        for speaker in speakers
        if speaker["speaker_uuid"] in default_speaker_uuids
    ]


def test_get_speakers_200(client: TestClient, snapshot_json: SnapshotAssertion) -> None:
    """デフォルトモデル由来の話者一覧が Speaker API 互換形式で返ることを確認する。"""

    response = client.get("/speakers", params={})
    assert response.status_code == 200

    # デフォルトモデル由来の話者のみにフィルタしてスナップショット比較する
    default_speaker_uuids = _get_default_speaker_uuids(client)
    default_speakers = _filter_default_speakers(response.json(), default_speaker_uuids)
    assert len(default_speakers) > 0
    assert snapshot_json == hash_long_string(default_speakers)
