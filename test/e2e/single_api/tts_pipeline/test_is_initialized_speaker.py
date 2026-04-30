"""/is_initialized_speaker API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import get_first_style_id


def test_get_is_initialized_speaker_200(
    client_with_default_model: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    style_id = get_first_style_id(client_with_default_model)
    response = client_with_default_model.get(
        "/is_initialized_speaker", params={"speaker": style_id}
    )
    assert response.status_code == 200
    assert snapshot_json == response.json()
