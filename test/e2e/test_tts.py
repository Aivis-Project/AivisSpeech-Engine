"""TTSのテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import get_first_style_id

# from test.utility import hash_wave_floats_from_wav_bytes


def test_テキストとキャラクターIDから音声を合成できる(
    client_with_default_model: TestClient, snapshot: SnapshotAssertion
) -> None:
    client = client_with_default_model
    style_id = get_first_style_id(client)

    # テキストとキャラクター ID から AudioQuery を生成する
    audio_query_res = client.post(
        "/audio_query", params={"text": "テストです", "speaker": style_id}
    )
    audio_query = audio_query_res.json()

    # AudioQuery から音声波形を生成する
    synthesis_res = client.post(
        "/synthesis", params={"speaker": style_id}, json=audio_query
    )

    # リクエストが成功している
    assert synthesis_res.status_code == 200

    # FileResponse 内の .wav から抽出された音声波形が一致する
    assert synthesis_res.headers["content-type"] == "audio/wav"
    # AivisSpeech Engine の音声合成は常にある程度のランダム性があるため、テストではハッシュ値の比較は行わない
    # assert snapshot == hash_wave_floats_from_wav_bytes(synthesis_res.read())
