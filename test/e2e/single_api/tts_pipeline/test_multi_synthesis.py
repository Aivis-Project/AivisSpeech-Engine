"""/multi_synthesis API のテスト。"""

# import io
# import zipfile

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import gen_mora

# from test.utility import hash_wave_floats_from_wav_bytes


def test_post_multi_synthesis_200(
    client: TestClient, snapshot: SnapshotAssertion
) -> None:
    """同じサンプリングレートの複数 AudioQuery をまとめて音声合成すると、zip が返ることを確認する。"""

    queries = [
        {
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
        },
        {
            "accent_phrases": [
                {
                    "moras": [
                        gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                        gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
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
            "prePhonemeLength": 0.2,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": 44100,
            "outputStereo": False,
            "kana": "テストト",
        },
    ]
    response = client.post(
        "/multi_synthesis", params={"speaker": 888753760}, json=queries
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    # zip 内の全ての wav の波形がスナップショットと一致する
    # AivisSpeech Engine の音声合成は常にある程度のランダム性があるため、テストではハッシュ値の比較は行わない
    # zip_bytes = io.BytesIO(response.read())
    # with zipfile.ZipFile(zip_bytes, "r") as zip_file:
    #     wav_files = (zip_file.read(name) for name in zip_file.namelist())
    #     for wav in wav_files:
    #         assert snapshot == hash_wave_floats_from_wav_bytes(wav)


def test_post_multi_synthesis_empty_queries_422(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    """空配列の音声合成クエリで `/multi_synthesis` を呼ぶと、422 とエラー内容が返ることを確認する。"""

    response = client.post("/multi_synthesis", params={"speaker": 888753760}, json=[])
    assert response.status_code == 422
    assert snapshot_json == response.json()
