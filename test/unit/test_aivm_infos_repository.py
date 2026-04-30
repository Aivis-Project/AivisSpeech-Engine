"""AIVM 情報リポジトリのテスト。"""

import pytest

from voicevox_engine.aivm_infos_repository import AivmInfosRepository


def test_style_id_conversion_keeps_local_style_id() -> None:
    """ローカルスタイル ID 0〜31 が、互換 style ID に変換後も下位 5 bit から復元できることを確認する。"""

    speaker_uuid = "5680ac39-43c9-487a-bc3e-018c0d29cc38"

    for local_style_id in range(32):
        style_id = AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=local_style_id,
            speaker_uuid=speaker_uuid,
        )

        assert isinstance(style_id, int)
        assert 0 <= style_id <= 0x7FFFFFFF
        assert AivmInfosRepository.style_id_to_local_style_id(style_id) == local_style_id  # fmt: skip


def test_style_id_conversion_generates_different_speaker_id() -> None:
    """同じローカルスタイル ID でも、話者 UUID が異なれば別の互換 style ID になることを確認する。"""

    local_style_id = 1

    first_style_id = AivmInfosRepository.local_style_id_to_style_id(
        local_style_id=local_style_id,
        speaker_uuid="5680ac39-43c9-487a-bc3e-018c0d29cc38",
    )
    second_style_id = AivmInfosRepository.local_style_id_to_style_id(
        local_style_id=local_style_id,
        speaker_uuid="e756b8e4-b606-4e15-99b1-3f9c6a1b2317",
    )

    assert first_style_id != second_style_id
    assert AivmInfosRepository.style_id_to_local_style_id(first_style_id) == local_style_id  # fmt: skip
    assert AivmInfosRepository.style_id_to_local_style_id(second_style_id) == local_style_id  # fmt: skip


@pytest.mark.parametrize("local_style_id", [-1, 32])
def test_style_id_conversion_rejects_out_of_range_local_style_id(
    local_style_id: int,
) -> None:
    """ローカルスタイル ID が 0〜31 の範囲外の場合、ValueError で拒否されることを確認する。"""

    with pytest.raises(ValueError, match="local_style_id"):
        AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=local_style_id,
            speaker_uuid="5680ac39-43c9-487a-bc3e-018c0d29cc38",
        )


def test_style_id_conversion_rejects_empty_speaker_uuid() -> None:
    """話者 UUID が空文字列の場合、互換 style ID を生成せず ValueError で拒否することを確認する。"""

    with pytest.raises(ValueError, match="speaker_uuid"):
        AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=0,
            speaker_uuid="",
        )


def test_extract_base64_from_data_url() -> None:
    """Data URL から Base64 本体だけを取り出せることを確認する。"""

    base64 = AivmInfosRepository.extract_base64_from_data_url(
        "data:image/png;base64,Zm9vYmFy"
    )

    assert base64 == "Zm9vYmFy"


@pytest.mark.parametrize(
    "data_url",
    [
        "",
        "https://example.com/icon.png",
        "data:image/png;base64",
    ],
)
def test_extract_base64_from_data_url_rejects_invalid_data_url(
    data_url: str,
) -> None:
    """空文字列・通常 URL・カンマのない Data URL を ValueError で拒否することを確認する。"""

    with pytest.raises(ValueError, match="Data URL|data URL"):
        AivmInfosRepository.extract_base64_from_data_url(data_url)
