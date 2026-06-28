"""Tests for ONNX GGML tensor mapping edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PLUGIN_SRC = (
    Path(__file__).resolve().parents[2]
    / "experimental"
    / "onnxruntime-ep-aivis-ggml"
    / "src"
)
sys.path.insert(0, str(_PLUGIN_SRC))

from onnxruntime_ep_aivis_ggml.gguf_writer import (  # noqa: E402
    _store_as_f16,
    prepare_mapped_tensor_array,
)
from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import map_initializer_name  # noqa: E402


def test_sdp_flow_exp_initializer_maps_to_logs() -> None:
    assert (
        map_initializer_name("/sdp/flows.0/Exp_output_0")
        == "style_bert_vits2.sdp.flows.0.logs"
    )


def test_sdp_flow_exp_initializer_reconstructs_logs() -> None:
    exp_negative_logs = np.asarray([[2.0], [0.5]], dtype=np.float32)

    logs = prepare_mapped_tensor_array(
        source_name="/sdp/flows.0/Exp_output_0",
        target_name="style_bert_vits2.sdp.flows.0.logs",
        array=exp_negative_logs,
    )

    np.testing.assert_allclose(logs, -np.log(exp_negative_logs))


def test_synthesis_gguf_writer_uses_safe_f16_weight_scope() -> None:
    assert _store_as_f16("style_bert_vits2.decoder.conv_pre.weight")
    assert _store_as_f16("style_bert_vits2.te.enc.ffn.0.c1.w")

    assert not _store_as_f16(
        "style_bert_vits2.decoder.conv_pre.weight",
        enabled=False,
    )
    assert not _store_as_f16("style_bert_vits2.text_encoder.token_embedding.weight")
    assert not _store_as_f16("style_bert_vits2.text_encoder.norm_layers_1.0.gamma")
    assert not _store_as_f16("style_bert_vits2.decoder.ups.0.weight")
    assert not _store_as_f16("style_bert_vits2.decoder.conv_pre.bias")
    assert not _store_as_f16("style_bert_vits2.style_vectors")
