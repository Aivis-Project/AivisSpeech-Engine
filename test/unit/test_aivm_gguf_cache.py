"""Tests for GGUF cache entry retention."""

from __future__ import annotations

import json
from pathlib import Path

from voicevox_engine.aivm_gguf_cache import AivmGgufCache


def _write_entry(path: Path, *, converter_version: str) -> None:
    path.write_bytes(b"gguf")
    path.with_suffix(".json").write_text(
        json.dumps({"converter_version": converter_version}),
        encoding="utf-8",
    )


def test_stale_cleanup_keeps_other_converter_versions(tmp_path: Path) -> None:
    model_uuid = "00000000-0000-4000-8000-000000000102"
    cache = AivmGgufCache(
        cache_dir=tmp_path,
        converter_version="tts-cpp-style-bert-vits2-converter-f16-v1",
    )
    keep_path = tmp_path / f"{model_uuid}-keep.gguf"
    stale_same_converter_path = tmp_path / f"{model_uuid}-old-f16.gguf"
    stale_other_converter_path = tmp_path / f"{model_uuid}-old-f32.gguf"

    _write_entry(
        keep_path,
        converter_version="tts-cpp-style-bert-vits2-converter-f16-v1",
    )
    _write_entry(
        stale_same_converter_path,
        converter_version="tts-cpp-style-bert-vits2-converter-f16-v1",
    )
    _write_entry(
        stale_other_converter_path,
        converter_version="tts-cpp-style-bert-vits2-converter-f32-v1",
    )

    cache._delete_stale_entries(  # noqa: SLF001
        aivm_model_uuid=model_uuid,
        keep_gguf_path=keep_path,
    )

    assert keep_path.exists()
    assert keep_path.with_suffix(".json").exists()
    assert not stale_same_converter_path.exists()
    assert not stale_same_converter_path.with_suffix(".json").exists()
    assert stale_other_converter_path.exists()
    assert stale_other_converter_path.with_suffix(".json").exists()
