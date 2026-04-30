"""/synthesis API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import gen_mora, get_first_style_id

# from test.utility import hash_wave_floats_from_wav_bytes


def test_post_synthesis_200(
    client_with_default_model: TestClient, snapshot: SnapshotAssertion
) -> None:
    client = client_with_default_model
    style_id = get_first_style_id(client)
    query = {
        "accent_phrases": [
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
        ],
        "speedScale": 1.0,
        "pitchScale": 1.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "pauseLength": None,
        "pauseLengthScale": 1.0,
        "outputSamplingRate": 44100,
        "outputStereo": False,
        "kana": "テスト",
    }
    response = client.post("/synthesis", params={"speaker": style_id}, json=query)
    assert response.status_code == 200

    # 音声波形が一致する
    assert response.headers["content-type"] == "audio/wav"
    # AivisSpeech Engine の音声合成は常にある程度のランダム性があるため、テストではハッシュ値の比較は行わない
    # assert snapshot == hash_wave_floats_from_wav_bytes(response.read())


def test_post_synthesis_old_audio_query_200(
    client_with_default_model: TestClient, snapshot: SnapshotAssertion
) -> None:
    """古いバージョンの audio_query でもエラーなく合成できる"""

    client = client_with_default_model
    style_id = get_first_style_id(client)
    query = {
        "accent_phrases": [
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
        ],
        "speedScale": 1.0,
        "pitchScale": 1.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "outputSamplingRate": 44100,
        "outputStereo": False,
    }
    response = client.post("/synthesis", params={"speaker": style_id}, json=query)
    assert response.status_code == 200

    # 音声波形が一致する
    assert response.headers["content-type"] == "audio/wav"
    # AivisSpeech Engine の音声合成は常にある程度のランダム性があるため、テストではハッシュ値の比較は行わない
    # assert snapshot == hash_wave_floats_from_wav_bytes(response.read())
