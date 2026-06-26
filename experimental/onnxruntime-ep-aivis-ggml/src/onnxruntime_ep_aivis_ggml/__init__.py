"""Discovery helpers for the Aivis ggml ONNX Runtime Plugin EP."""

from __future__ import annotations

from pathlib import Path

_EP_NAME = "AivisGgmlExecutionProvider"
_LIBRARY_BASENAMES = (
    "libaivis_ggml_onnx_ep.so",
    "libaivis_ggml_onnx_ep.dylib",
    "aivis_ggml_onnx_ep.dll",
)


def get_ep_name() -> str:
    """Return the ONNX Runtime provider name exported by this package."""

    return _EP_NAME


def get_ep_names() -> list[str]:
    """Return all ONNX Runtime provider names exported by this package."""

    return [_EP_NAME]


def get_default_provider_options() -> dict[str, str]:
    """Return conservative provider options for an accurate default run."""

    return {
        "backend": "vulkan",
        "claim_jp_bert_graph": "0",
        "claim_synthesis_graph": "0",
        "eager_load_model": "0",
        "n_threads": "0",
        "precision": "accurate",
    }


def get_library_path() -> str:
    """Return the packaged native Plugin EP library path."""

    library_dir = Path(__file__).resolve().parent / "lib"
    for basename in _LIBRARY_BASENAMES:
        library_path = library_dir / basename
        if library_path.exists():
            return str(library_path)

    candidates = ", ".join(
        str(library_dir / basename) for basename in _LIBRARY_BASENAMES
    )
    raise FileNotFoundError(
        "Aivis ggml ONNX Runtime Plugin EP library was not found. "
        f"Expected one of: {candidates}"
    )
