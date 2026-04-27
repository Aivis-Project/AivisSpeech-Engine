"""/synthesis_morphing API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.e2e.single_api.utils import gen_mora

# AivisSpeech Engine では話者ごとに発声タイミングが異なる関係でモーフィングは実装不可能なため
# (動作こそするが聴くに耐えない) 、モーフィング可能なスタイルのペアが存在しない。
# そのため下記の 200 系テストは常に `/synthesis_morphing` の前段で 400 Bad Request を返し、
# 期待値と一致しないためデッドコード化している。将来モーフィング機能を再有効化する際の参照用に残置。
"""
def test_post_synthesis_morphing_200(client: TestClient) -> None:
    queries = {
        "accent_phrases": [
            {
                "moras": [
                    gen_mora("テ", "t", 2.3, "e", 0.8, 3.3),
                    gen_mora("ス", "s", 2.1, "U", 0.3, 0.0),
                    gen_mora("ト", "t", 2.3, "o", 1.8, 4.1),
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
        "outputSamplingRate": 24000,
        "outputStereo": False,
        "kana": "テ'_スト",
    }
    response = client.post(
        "/synthesis_morphing",
        params={"base_speaker": 0, "target_speaker": 0, "morph_rate": 0.8},
        json=queries,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    # FIXME: LinuxとMacOSで計算結果が一致しないためスナップショットテストがコケる（原因不明）
    # from test.utility import summarize_wav_bytes
    # from syrupy.assertion import SnapshotAssertion
    # # FileResponse 内の .wav から抽出された音声波形が一致する
    # assert response.headers["content-type"] == "audio/wav"
    # assert snapshot == summarize_wav_bytes(response.read())
"""


def test_post_synthesis_morphing_422(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    too_much_rate = 100
    queries = {
        "accent_phrases": [
            {
                "moras": [gen_mora("テ", "t", 2.3, "e", 0.8, 3.3)],
                "accent": 1,
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
        "outputSamplingRate": 24000,
        "outputStereo": False,
        "kana": "テ'_スト",
    }
    response = client.post(
        "/synthesis_morphing",
        params={"base_speaker": 0, "target_speaker": 0, "morph_rate": too_much_rate},
        json=queries,
    )
    assert response.status_code == 422
    assert snapshot_json == response.json()
