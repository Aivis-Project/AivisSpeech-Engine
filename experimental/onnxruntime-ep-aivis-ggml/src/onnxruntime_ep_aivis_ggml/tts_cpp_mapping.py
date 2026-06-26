"""Map Aivis ONNX initializer names to TTS.cpp Style-Bert-VITS2 GGUF keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from onnxruntime_ep_aivis_ggml.initializer_pack import (
    InitializerTensorRecord,
    write_initializer_tensor_pack,
)

MAPPING_CONTRACT_VERSION = "tts-cpp-style-bert-vits2-mapping-v1"


@dataclass(frozen=True)
class TensorMappingRecord:
    """A single source initializer's planned TTS.cpp tensor mapping."""

    source_name: str
    status: str
    target_name: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class TensorMappingReport:
    """Conservative initializer mapping report for the future GGUF writer."""

    version: str
    mapped_count: int
    ignored_count: int
    transform_source_count: int
    requires_transform_count: int
    unsupported_count: int
    records: tuple[TensorMappingRecord, ...]
    required_external_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "version": self.version,
            "mapped_count": self.mapped_count,
            "ignored_count": self.ignored_count,
            "transform_source_count": self.transform_source_count,
            "requires_transform_count": self.requires_transform_count,
            "unsupported_count": self.unsupported_count,
            "records": tuple(record.to_dict() for record in self.records),
            "required_external_artifacts": self.required_external_artifacts,
        }


def build_tts_cpp_mapping_report(
    records: tuple[InitializerTensorRecord, ...],
    *,
    available_external_artifacts: tuple[str, ...] = (),
    target_name_overrides: dict[str, str] | None = None,
) -> TensorMappingReport:
    """Map ONNX initializer records to TTS.cpp Style-Bert-VITS2 GGUF tensor keys."""

    source_names = frozenset(record.name for record in records)
    target_name_overrides = target_name_overrides or {}
    mapping_records = tuple(
        map_initializer_record(
            record,
            source_names=source_names,
            target_name_overrides=target_name_overrides,
        )
        for record in records
    )
    mapped_count = sum(record.status == "mapped" for record in mapping_records)
    ignored_count = sum(record.status == "ignored" for record in mapping_records)
    transform_source_count = sum(
        record.status == "transform_source" for record in mapping_records
    )
    requires_transform_count = sum(
        record.status == "requires_transform" for record in mapping_records
    )
    unsupported_count = sum(
        record.status == "unsupported" for record in mapping_records
    )

    required_external_artifacts = []
    if "style_vectors.npy" not in available_external_artifacts and not any(
        record.target_name == "style_bert_vits2.style_vectors"
        for record in mapping_records
    ):
        required_external_artifacts.append("style_vectors.npy")

    return TensorMappingReport(
        version=MAPPING_CONTRACT_VERSION,
        mapped_count=mapped_count,
        ignored_count=ignored_count,
        transform_source_count=transform_source_count,
        requires_transform_count=requires_transform_count,
        unsupported_count=unsupported_count,
        records=mapping_records,
        required_external_artifacts=tuple(required_external_artifacts),
    )


def map_initializer_record(
    record: InitializerTensorRecord,
    *,
    source_names: frozenset[str] | None = None,
    target_name_overrides: dict[str, str] | None = None,
) -> TensorMappingRecord:
    """Map one ONNX initializer to a TTS.cpp tensor key when it is safe."""

    source_names = source_names or frozenset()
    target_name_overrides = target_name_overrides or {}
    source_name = record.name
    if source_name in target_name_overrides:
        return TensorMappingRecord(
            source_name=source_name,
            status="mapped",
            target_name=target_name_overrides[source_name],
            reason="ONNX graph consumer path maps anonymous initializer",
        )

    if _is_onnx_graph_constant_name(source_name):
        return TensorMappingRecord(
            source_name=source_name,
            status="ignored",
            target_name=None,
            reason="ONNX graph constant, not a model weight",
        )

    target_name = map_initializer_name(source_name)
    if target_name is not None:
        return TensorMappingRecord(
            source_name=source_name,
            status="mapped",
            target_name=target_name,
            reason="direct TTS.cpp Style-Bert-VITS2 loader tensor name",
        )

    if source_name.endswith(".weight_g") or source_name.endswith(".weight_v"):
        if _has_complete_weight_norm_pair(source_name, source_names):
            return TensorMappingRecord(
                source_name=source_name,
                status="transform_source",
                target_name=_weight_norm_target_name(source_name),
                reason="weight_norm source pair materialized into target weight",
            )
        return TensorMappingRecord(
            source_name=source_name,
            status="requires_transform",
            target_name=_weight_norm_target_name(source_name),
            reason="weight_norm parameters must be materialized into weight first",
        )

    return TensorMappingRecord(
        source_name=source_name,
        status="unsupported",
        target_name=None,
        reason="no conservative TTS.cpp tensor mapping rule",
    )


def map_initializer_name(source_name: str) -> str | None:
    """Return the TTS.cpp GGUF tensor key for a safe direct ONNX initializer."""

    exact = {
        "emb_g.weight": "style_bert_vits2.speaker_embedding.weight",
        "style_vectors": "style_bert_vits2.style_vectors",
        "enc_p.emb.weight": "style_bert_vits2.text_encoder.token_embedding.weight",
        "enc_p.tone_emb.weight": "style_bert_vits2.text_encoder.tone_embedding.weight",
        "enc_p.language_emb.weight": "style_bert_vits2.text_encoder.language_embedding.weight",
        "/sdp/flows.0/Exp_output_0": "style_bert_vits2.sdp.flows.0.logs",
    }
    if source_name in exact:
        return exact[source_name]

    compact_encoder_name = _compact_encoder_tensor_name(
        _strip_prefix(source_name, "enc_p.encoder.")
    )
    if compact_encoder_name is not None:
        return f"style_bert_vits2.te.enc.{compact_encoder_name}"

    compact_flow_name = _compact_flow_tensor_name(_strip_prefix(source_name, "flow."))
    if compact_flow_name is not None:
        return f"style_bert_vits2.fl.{compact_flow_name}"

    prefix_rules = (
        ("enc_p.bert_proj.", "style_bert_vits2.text_encoder.bert_proj."),
        ("enc_p.ja_bert_proj.", "style_bert_vits2.text_encoder.ja_bert_proj."),
        ("enc_p.en_bert_proj.", "style_bert_vits2.text_encoder.en_bert_proj."),
        ("enc_p.style_proj.", "style_bert_vits2.text_encoder.style_proj."),
        ("enc_p.proj.", "style_bert_vits2.text_encoder.proj."),
        ("dp.", "style_bert_vits2.duration_predictor."),
        ("sdp.", "style_bert_vits2.sdp."),
        ("dec.", "style_bert_vits2.decoder."),
    )
    for source_prefix, target_prefix in prefix_rules:
        if source_name.startswith(source_prefix):
            leaf = source_name.removeprefix(source_prefix)
            if leaf.endswith(".weight_g") or leaf.endswith(".weight_v"):
                return None
            return target_prefix + leaf
    return None


def build_mapping_report_from_model(model_path: str | Path) -> TensorMappingReport:
    """Build a mapping report by reading ONNX initializers without keeping bytes."""

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        pack = write_initializer_tensor_pack(
            model_path=model_path,
            output_path=Path(tmp_dir) / "initializers.bin",
        )
    return build_tts_cpp_mapping_report(
        pack.records,
        target_name_overrides=build_graph_initializer_target_overrides(model_path),
    )


def build_graph_initializer_target_overrides(model_path: str | Path) -> dict[str, str]:
    """Map anonymous ONNX initializer names by inspecting their consumer nodes."""

    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    overrides: dict[str, str] = {}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        initializer_name = node.input[1]
        if initializer_name not in initializer_names:
            continue
        target_name = _map_anonymous_matmul_consumer(node.name)
        if target_name is not None:
            overrides[initializer_name] = target_name
    return overrides


def _weight_norm_target_name(source_name: str) -> str | None:
    if source_name.endswith(".weight_g") or source_name.endswith(".weight_v"):
        return map_initializer_name(source_name.rsplit(".", 1)[0] + ".weight")
    return None


def _has_complete_weight_norm_pair(
    source_name: str,
    source_names: frozenset[str],
) -> bool:
    if not (source_name.endswith(".weight_g") or source_name.endswith(".weight_v")):
        return False
    base_name = source_name.rsplit(".", 1)[0]
    return (
        f"{base_name}.weight_g" in source_names
        and f"{base_name}.weight_v" in source_names
        and map_initializer_name(f"{base_name}.weight") is not None
    )


def _is_onnx_graph_constant_name(source_name: str) -> bool:
    if source_name == "/sdp/flows.0/Exp_output_0":
        return False
    if source_name.startswith("/") or "_output_" in source_name:
        return True
    if source_name.startswith("_v_"):
        return True
    if source_name.startswith("onnx::") and not source_name.startswith("onnx::MatMul_"):
        return True
    return False


def _map_anonymous_matmul_consumer(node_name: str) -> str | None:
    if node_name == "/enc_p/style_proj/MatMul":
        return "style_bert_vits2.text_encoder.style_proj.weight"
    if node_name == "/enc_p/encoder/spk_emb_linear/MatMul":
        return "style_bert_vits2.te.enc.spk.w"

    flow_prefix = "/flow/flows."
    flow_suffix = "/enc/spk_emb_linear/MatMul"
    if node_name.startswith(flow_prefix) and node_name.endswith(flow_suffix):
        raw_index_text = node_name.removeprefix(flow_prefix).removesuffix(flow_suffix)
        try:
            raw_index = int(raw_index_text)
        except ValueError:
            return None
        if raw_index % 2:
            return None
        return f"style_bert_vits2.fl.{raw_index // 2}.enc.spk.w"
    return None


def _strip_prefix(value: str, prefix: str) -> str | None:
    if value.startswith(prefix):
        return value.removeprefix(prefix)
    return None


def _compact_encoder_tensor_name(name: str | None) -> str | None:
    """Compact encoder tensor names the same way TTS.cpp's GGUF encoder does."""

    if name is None:
        return None

    parts = name.split(".")
    if parts[0] == "spk_emb_linear" and len(parts) == 2:
        leaf = _compact_weight_leaf(parts[1])
        return None if leaf is None else f"spk.{leaf}"

    if parts[0] == "attn_layers" and len(parts) >= 3:
        layer = parts[1]
        if parts[2] == "emb_rel_k" and len(parts) == 3:
            return f"al.{layer}.rk"
        if parts[2] == "emb_rel_v" and len(parts) == 3:
            return f"al.{layer}.rv"
        if len(parts) == 4 and parts[2] in {
            "conv_q",
            "conv_k",
            "conv_v",
            "conv_o",
        }:
            projection = parts[2].removeprefix("conv_")
            leaf = _compact_weight_leaf(parts[3])
            return None if leaf is None else f"al.{layer}.{projection}.{leaf}"

    if parts[0] == "ffn_layers" and len(parts) == 4:
        layer = parts[1]
        if parts[2] not in {"conv_1", "conv_2"}:
            return None
        conv = "c1" if parts[2] == "conv_1" else "c2"
        leaf = _compact_weight_leaf(parts[3])
        return None if leaf is None else f"ffn.{layer}.{conv}.{leaf}"

    if parts[0] in {"norm_layers_1", "norm_layers_2"} and len(parts) == 3:
        norm = "n1" if parts[0] == "norm_layers_1" else "n2"
        if parts[2] == "gamma":
            return f"{norm}.{parts[1]}.g"
        if parts[2] == "beta":
            return f"{norm}.{parts[1]}.b"

    return None


def _compact_flow_tensor_name(name: str | None) -> str | None:
    """Compact flow tensor names the same way TTS.cpp's GGUF encoder does."""

    if name is None:
        return None

    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "flows":
        return None

    try:
        raw_index = int(parts[1])
    except ValueError:
        return None
    if raw_index % 2:
        return None

    layer = raw_index // 2
    rest = parts[2:]
    if rest[0] in {"pre", "post"} and len(rest) == 2:
        leaf = _compact_weight_leaf(rest[1])
        return None if leaf is None else f"{layer}.{rest[0]}.{leaf}"

    if rest[0] == "enc":
        compact_encoder_name = _compact_encoder_tensor_name(".".join(rest[1:]))
        return (
            None
            if compact_encoder_name is None
            else f"{layer}.enc.{compact_encoder_name}"
        )

    return None


def _compact_weight_leaf(leaf: str) -> str | None:
    if leaf == "weight":
        return "w"
    if leaf == "bias":
        return "b"
    return None
