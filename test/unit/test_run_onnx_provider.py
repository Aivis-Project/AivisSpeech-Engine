"""run.py ONNX GGML provider option tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from run import (
    _build_ggml_onnx_ep_options,
    decide_onnx_provider_from_env,
    engine_root,
)


def test_decide_onnx_provider_from_env_accepts_ggml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VV_ONNX_PROVIDER can explicitly select the ggml Plugin EP route."""

    monkeypatch.setenv("VV_ONNX_PROVIDER", "ggml")

    assert decide_onnx_provider_from_env("VV_ONNX_PROVIDER") == "ggml"


@pytest.mark.parametrize("provider", ["cuda", "directml"])
def test_decide_onnx_provider_from_env_accepts_builtin_gpu_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """VV_ONNX_PROVIDER can explicitly select built-in GPU providers."""

    monkeypatch.setenv("VV_ONNX_PROVIDER", provider)

    assert decide_onnx_provider_from_env("VV_ONNX_PROVIDER") == provider


def test_decide_onnx_provider_from_env_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown ONNX provider names are ignored with a warning."""

    monkeypatch.setenv("VV_ONNX_PROVIDER", "tensorrt")

    with pytest.warns(UserWarning, match="Expected one of"):
        assert decide_onnx_provider_from_env("VV_ONNX_PROVIDER") is None


def test_build_ggml_onnx_ep_options_defaults_to_claiming_supported_graphs() -> None:
    """--onnx_provider ggml creates the production Plugin EP defaults."""

    library_path = Path("opt/tts.cpp/libtts.so")
    args = SimpleNamespace(
        onnx_ep_options={},
        ggml_tts_server_backend="vulkan",
        ggml_vulkan_precision="accurate",
        ggml_native_library_path=library_path,
        ggml_vulkan_device="0",
    )

    provider_options = _build_ggml_onnx_ep_options(cast(Any, args))

    assert provider_options == {
        "backend": "vulkan",
        "claim_jp_bert_graph": "1",
        "claim_synthesis_graph": "1",
        "device": "0",
        "eager_load_model": "1",
        "n_threads": "0",
        "precision": "accurate",
        "tts_cpp_library_path": str((engine_root() / library_path).resolve()),
    }


def test_build_ggml_onnx_ep_options_sets_positive_threads_for_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTS.cpp CPU backend requires a positive thread count."""

    monkeypatch.setattr("run.os.cpu_count", lambda: 8)
    args = SimpleNamespace(
        onnx_ep_options={},
        ggml_tts_server_backend="cpu",
        ggml_vulkan_precision="accurate",
        ggml_native_library_path=Path("/opt/tts.cpp/libtts.so"),
        ggml_vulkan_device="0",
    )

    provider_options = _build_ggml_onnx_ep_options(cast(Any, args))

    assert provider_options["backend"] == "cpu"
    assert provider_options["n_threads"] == "8"
    assert "device" not in provider_options


def test_build_ggml_onnx_ep_options_requires_tts_cpp_library() -> None:
    """The ggml Plugin EP cannot claim graphs without the TTS.cpp C API."""

    args = SimpleNamespace(
        onnx_ep_options={},
        ggml_tts_server_backend="vulkan",
        ggml_vulkan_precision="accurate",
        ggml_native_library_path=None,
        ggml_vulkan_device=None,
    )

    with pytest.raises(RuntimeError, match="tts_cpp_library_path"):
        _build_ggml_onnx_ep_options(cast(Any, args))
