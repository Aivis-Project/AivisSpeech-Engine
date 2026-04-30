"""/audio_query API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import get_first_style_id
from test.utility import round_floats


def test_post_audio_query_200(
    client_with_default_model: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    client = client_with_default_model
    style_id = get_first_style_id(client)
    response = client.post(
        "/audio_query", params={"text": "テストです", "speaker": style_id}
    )
    assert response.status_code == 200
    assert snapshot_json == round_floats(response.json(), round_value=2)


def test_post_audio_query_enable_katakana_english_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    response = client.post(
        "/audio_query",
        params={"text": "Voivo", "speaker": 0, "enable_katakana_english": True},
    )
    assert response.status_code == 200
    assert snapshot_json == round_floats(response.json(), round_value=2)
