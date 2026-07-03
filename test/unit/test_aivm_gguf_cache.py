"""Tests for GGUF cache entry retention."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aivmlib.schemas.aivm_manifest import ModelFormat

from voicevox_engine.aivm_gguf_cache import AivmGgufCache, JpBertGgufCache


def _write_entry(path: Path, *, converter_version: str) -> None:
    path.write_bytes(b"gguf")
    path.with_suffix(".json").write_text(
        json.dumps({"converter_version": converter_version}),
        encoding="utf-8",
    )


def _metadata(model_format: ModelFormat = ModelFormat.ONNX) -> Any:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            uuid="00000000-0000-4000-8000-000000000102",
            version="1.0.0",
            model_architecture="Style-Bert-VITS2",
            model_format=model_format,
        )
    )


def test_aivm_cache_key_uses_file_content_not_path_or_mtime(tmp_path: Path) -> None:
    cache = AivmGgufCache(cache_dir=tmp_path)
    first_path = tmp_path / "first.aivmx"
    second_path = tmp_path / "second.aivmx"
    first_path.write_bytes(b"same model bytes")
    second_path.write_bytes(b"same model bytes")

    first_inputs = cache._build_cache_key_inputs(first_path, _metadata())  # noqa: SLF001
    second_inputs = cache._build_cache_key_inputs(second_path, _metadata())  # noqa: SLF001

    assert first_inputs["aivm_file_sha256"] == second_inputs["aivm_file_sha256"]
    assert first_inputs == second_inputs
    assert "aivm_file_path" not in first_inputs
    assert "aivm_file_mtime_ns" not in first_inputs


def test_jp_bert_cache_key_uses_file_content_not_path_or_mtime(tmp_path: Path) -> None:
    cache = JpBertGgufCache(cache_dir=tmp_path)
    first_path = tmp_path / "first.onnx"
    second_path = tmp_path / "second.onnx"
    first_path.write_bytes(b"same jp bert bytes")
    second_path.write_bytes(b"same jp bert bytes")

    first_inputs = cache._build_cache_key_inputs(first_path)  # noqa: SLF001
    second_inputs = cache._build_cache_key_inputs(second_path)  # noqa: SLF001

    assert first_inputs["jp_bert_onnx_sha256"] == second_inputs["jp_bert_onnx_sha256"]
    assert first_inputs == second_inputs
    assert "jp_bert_onnx_path" not in first_inputs
    assert "jp_bert_onnx_mtime_ns" not in first_inputs


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
