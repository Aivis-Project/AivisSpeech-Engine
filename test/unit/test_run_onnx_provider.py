"""run.py ONNX GGML provider option tests."""

import pytest

from run import decide_onnx_provider_from_env


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
