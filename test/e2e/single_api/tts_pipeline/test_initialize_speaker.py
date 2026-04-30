"""/initialize_speaker API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import get_first_style_id


def test_post_initialize_speaker_204(
    client_with_default_model: TestClient, snapshot: SnapshotAssertion
) -> None:
    style_id = get_first_style_id(client_with_default_model)
    response = client_with_default_model.post(
        "/initialize_speaker", params={"speaker": style_id}
    )
    assert response.status_code == 204
    assert snapshot == response.content
