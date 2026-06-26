"""Style-Bert-VITS2 ONNX graph signature helpers for the Aivis GGML Plugin EP."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

SIGNATURE_CONTRACT_VERSION = "aivis-ggml-signature-contract-v1"


@dataclass(frozen=True)
class TensorSignature:
    """Stable tensor interface details used by the graph gate."""

    name: str
    elem_type: str
    shape: tuple[str | int, ...]


@dataclass(frozen=True)
class OnnxGraphSignature:
    """Stable ONNX graph details used before a Plugin EP claims a graph."""

    ir_version: int
    producer_name: str
    producer_version: str
    graph_name: str
    opsets: tuple[tuple[str, int], ...]
    inputs: tuple[TensorSignature, ...]
    outputs: tuple[TensorSignature, ...]
    node_count: int
    initializer_count: int
    op_counts: tuple[tuple[str, int], ...]
    op_sequence_sha256: str
    initializer_names_sha256: str
    metadata_model_architecture: str | None = None
    metadata_model_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class SignatureMatch:
    """Result of checking whether a graph is safe for the Aivis GGML EP."""

    supported: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS = {
    "ir_version": 8,
    "opsets": (("", 18),),
    "graph_name": "main_graph",
    "input_names": (
        "x_tst",
        "x_tst_lengths",
        "sid",
        "tones",
        "language",
        "bert",
        "style_vec",
        "length_scale",
        "sdp_ratio",
        "noise_scale",
        "noise_scale_w",
    ),
    "input_elem_types": (
        "INT64",
        "INT64",
        "INT64",
        "INT64",
        "INT64",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    ),
    "first_output_name": "output",
    "output_count": 7,
    "node_count": 5334,
    "initializer_count": 948,
    "op_sequence_sha256": (
        "4e1290e04ede4ffd3dd7b795ab1c651fee6f3f5ee20200081cebc01a082448cc"
    ),
    "initializer_names_sha256": (
        "88cba3cdb9a7b3c147f23ef3aa69ccc64fbb6c6bccc7b1586d2e226c7d4d1fad"
    ),
    "metadata_model_architectures": (
        "Style-Bert-VITS2",
        "Style-Bert-VITS2 (JP-Extra)",
    ),
    "metadata_model_format": "ONNX",
}

SUPPORTED_STYLE_BERT_VITS2_JP_BERT = {
    "ir_version": 8,
    "opsets": (("", 17),),
    "input_names": ("input_ids", "attention_mask"),
    "input_elem_types": ("INT64", "INT64"),
    "output_name": "output",
    "output_elem_type": "FLOAT",
    "node_counts": (3619, 3092, 3180),
    "initializer_counts": (432, 521, 543),
    "required_op_types": (
        "Gather",
        "LayerNormalization",
        "MatMul",
        "Reshape",
        "Where",
    ),
}


def load_onnx_graph_signature(model_path: str | Path) -> OnnxGraphSignature:
    """Load an ONNX or AIVMX model and compute its stable graph signature."""

    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return graph_signature_from_onnx_model(model)


def graph_signature_from_onnx_model(model: Any) -> OnnxGraphSignature:
    """Compute a stable graph signature from an ONNX ModelProto-like object."""

    import onnx

    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    aivm_manifest = _parse_metadata_json(metadata.get("aivm_manifest"))

    return OnnxGraphSignature(
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        graph_name=graph.name,
        opsets=tuple((op.domain, op.version) for op in model.opset_import),
        inputs=tuple(_tensor_signature(onnx, value_info) for value_info in graph.input),
        outputs=tuple(
            _tensor_signature(onnx, value_info) for value_info in graph.output
        ),
        node_count=len(graph.node),
        initializer_count=len(graph.initializer),
        op_counts=tuple(sorted(op_counts.items())),
        op_sequence_sha256=_sha256_lines(node.op_type for node in graph.node),
        initializer_names_sha256=_sha256_lines(
            initializer.name for initializer in graph.initializer
        ),
        metadata_model_architecture=_metadata_str(aivm_manifest, "model_architecture"),
        metadata_model_format=_metadata_str(aivm_manifest, "model_format"),
    )


def match_supported_style_bert_vits2_synthesis(
    signature: OnnxGraphSignature,
) -> SignatureMatch:
    """Check whether the graph matches the currently supported synthesis export."""

    expected = SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS
    reasons: list[str] = []

    if signature.ir_version != expected["ir_version"]:
        reasons.append(f"ir_version {signature.ir_version} != {expected['ir_version']}")
    if signature.opsets != expected["opsets"]:
        reasons.append(f"opsets {signature.opsets} != {expected['opsets']}")
    if signature.graph_name != expected["graph_name"]:
        reasons.append(
            f"graph_name {signature.graph_name!r} != {expected['graph_name']!r}"
        )

    input_names = tuple(input_signature.name for input_signature in signature.inputs)
    if input_names != expected["input_names"]:
        reasons.append("input names do not match Style-Bert-VITS2 synthesis")

    input_elem_types = tuple(
        input_signature.elem_type for input_signature in signature.inputs
    )
    if input_elem_types != expected["input_elem_types"]:
        reasons.append("input element types do not match Style-Bert-VITS2 synthesis")

    if len(signature.outputs) != expected["output_count"]:
        reasons.append(
            f"output_count {len(signature.outputs)} != {expected['output_count']}"
        )
    elif signature.outputs[0].name != expected["first_output_name"]:
        reasons.append(
            f"first output {signature.outputs[0].name!r} "
            f"!= {expected['first_output_name']!r}"
        )

    if signature.node_count != expected["node_count"]:
        reasons.append(f"node_count {signature.node_count} != {expected['node_count']}")
    if signature.initializer_count != expected["initializer_count"]:
        reasons.append(
            f"initializer_count {signature.initializer_count} "
            f"!= {expected['initializer_count']}"
        )
    if signature.op_sequence_sha256 != expected["op_sequence_sha256"]:
        reasons.append("op sequence hash does not match supported synthesis graph")
    if signature.initializer_names_sha256 != expected["initializer_names_sha256"]:
        reasons.append("initializer name hash does not match supported synthesis graph")

    architecture = signature.metadata_model_architecture
    expected_architectures = expected["metadata_model_architectures"]
    if architecture is not None and architecture not in expected_architectures:
        reasons.append(f"metadata model_architecture {architecture!r} is unsupported")

    model_format = signature.metadata_model_format
    expected_format = expected["metadata_model_format"]
    if model_format is not None and model_format != expected_format:
        reasons.append(f"metadata model_format {model_format!r} != {expected_format!r}")

    return SignatureMatch(supported=len(reasons) == 0, reasons=tuple(reasons))


def match_supported_style_bert_vits2_jp_bert(
    signature: OnnxGraphSignature,
) -> SignatureMatch:
    """Check whether the graph matches the supported JP-BERT ONNX contract."""

    expected = SUPPORTED_STYLE_BERT_VITS2_JP_BERT
    reasons: list[str] = []

    if signature.ir_version != expected["ir_version"]:
        reasons.append(f"ir_version {signature.ir_version} != {expected['ir_version']}")
    if signature.opsets != expected["opsets"]:
        reasons.append(f"opsets {signature.opsets} != {expected['opsets']}")

    input_names = tuple(input_signature.name for input_signature in signature.inputs)
    if input_names != expected["input_names"]:
        reasons.append("input names do not match Style-Bert-VITS2 JP-BERT")

    input_elem_types = tuple(
        input_signature.elem_type for input_signature in signature.inputs
    )
    if input_elem_types != expected["input_elem_types"]:
        reasons.append("input element types do not match Style-Bert-VITS2 JP-BERT")

    if len(signature.outputs) != 1:
        reasons.append(f"output_count {len(signature.outputs)} != 1")
    elif signature.outputs[0].name != expected["output_name"]:
        reasons.append(
            f"output {signature.outputs[0].name!r} != {expected['output_name']!r}"
        )
    elif signature.outputs[0].elem_type != expected["output_elem_type"]:
        reasons.append(
            f"output element type {signature.outputs[0].elem_type!r} "
            f"!= {expected['output_elem_type']!r}"
        )

    if signature.node_count not in expected["node_counts"]:
        reasons.append(f"node_count {signature.node_count} is not accepted")
    if signature.initializer_count not in expected["initializer_counts"]:
        reasons.append(
            f"initializer_count {signature.initializer_count} is not accepted"
        )

    op_types = {op_type for op_type, _count in signature.op_counts}
    for required_op_type in expected["required_op_types"]:
        if required_op_type not in op_types:
            reasons.append(f"required op type {required_op_type!r} is missing")

    return SignatureMatch(supported=len(reasons) == 0, reasons=tuple(reasons))


def signature_structural_sha256(signature: OnnxGraphSignature) -> str:
    """Hash the stable structural fields that form the provider contract."""

    payload = {
        "version": SIGNATURE_CONTRACT_VERSION,
        "ir_version": signature.ir_version,
        "graph_name": signature.graph_name,
        "opsets": signature.opsets,
        "inputs": tuple(input_signature for input_signature in signature.inputs),
        "outputs": tuple(output_signature for output_signature in signature.outputs),
        "node_count": signature.node_count,
        "initializer_count": signature.initializer_count,
        "op_counts": signature.op_counts,
        "op_sequence_sha256": signature.op_sequence_sha256,
        "initializer_names_sha256": signature.initializer_names_sha256,
        "metadata_model_architecture": signature.metadata_model_architecture,
        "metadata_model_format": signature.metadata_model_format,
    }
    raw = json.dumps(
        payload,
        default=asdict,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(raw.encode()).hexdigest()


def build_signature_contract(signature: OnnxGraphSignature) -> dict[str, Any]:
    """Return the versioned graph contract used by cache manifests."""

    synthesis_match = match_supported_style_bert_vits2_synthesis(signature)
    jp_bert_match = match_supported_style_bert_vits2_jp_bert(signature)
    if synthesis_match.supported:
        graph_kind = "style_bert_vits2_synthesis"
        match = synthesis_match
    elif jp_bert_match.supported:
        graph_kind = "style_bert_vits2_jp_bert"
        match = jp_bert_match
    else:
        graph_kind = "unsupported"
        match = SignatureMatch(
            supported=False,
            reasons=(
                "synthesis={" + "; ".join(synthesis_match.reasons) + "}",
                "jp_bert={" + "; ".join(jp_bert_match.reasons) + "}",
            ),
        )

    return {
        "version": SIGNATURE_CONTRACT_VERSION,
        "graph_kind": graph_kind,
        "supported": match.supported,
        "reasons": match.reasons,
        "structural_sha256": signature_structural_sha256(signature),
        "op_sequence_sha256": signature.op_sequence_sha256,
        "initializer_names_sha256": signature.initializer_names_sha256,
        "node_count": signature.node_count,
        "initializer_count": signature.initializer_count,
    }


def _tensor_signature(onnx_module: Any, value_info: Any) -> TensorSignature:
    tensor_type = value_info.type.tensor_type
    return TensorSignature(
        name=value_info.name,
        elem_type=onnx_module.TensorProto.DataType.Name(tensor_type.elem_type),
        shape=tuple(_dim_signature(dim) for dim in tensor_type.shape.dim),
    )


def _dim_signature(dim: Any) -> str | int:
    if dim.dim_param:
        return dim.dim_param
    if dim.HasField("dim_value"):
        return dim.dim_value
    return "?"


def _sha256_lines(values: Any) -> str:
    return sha256("\n".join(values).encode()).hexdigest()


def _parse_metadata_json(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None
