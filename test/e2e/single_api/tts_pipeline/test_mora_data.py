"""/mora_data API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import gen_mora, get_first_style_id
from test.utility import round_floats


def test_post_mora_data_200(
    client_with_default_model: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    client = client_with_default_model
    style_id = get_first_style_id(client)
    accent_phrases = [
        {
            "moras": [
                gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
            ],
            "accent": 1,
            "pause_mora": None,
            "is_interrogative": False,
        }
    ]
    response = client.post(
        "/mora_data", params={"speaker": style_id}, json=accent_phrases
    )
    assert response.status_code == 200
    assert snapshot_json == round_floats(response.json(), 2)
