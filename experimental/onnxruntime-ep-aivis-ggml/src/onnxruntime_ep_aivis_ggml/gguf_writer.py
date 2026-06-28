"""TTS.cpp Style-Bert-VITS2 GGUF writer for mapped ONNX initializers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import TensorMappingReport

STYLE_BERT_VITS2_ARCHITECTURE = "style-bert-vits2"


@dataclass(frozen=True)
class GgufWriteResult:
    """Portable metadata for a written GGUF artifact."""

    filename: str
    size_bytes: int
    sha256: str
    tensor_count: int
    tensor_f32_count: int
    tensor_f16_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def write_tts_cpp_style_bert_vits2_gguf(
    *,
    model_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    style_vectors_path: str | Path,
    mapping_report: TensorMappingReport,
    readiness: dict[str, Any],
    store_f16_weights: bool = True,
) -> GgufWriteResult:
    """
    Write a TTS.cpp-compatible GGUF from mapped ONNX initializers.

    This is intentionally strict: the caller must prove readiness first so the
    Plugin EP never creates a partial model that might later be claimed by ORT.
    """

    blockers = tuple(readiness.get("blockers", ()))
    if not readiness.get("can_write_gguf", False) or blockers:
        raise ValueError(
            "GGUF writing requires a ready converter plan; blockers: "
            + ", ".join(blockers)
        )

    if mapping_report.unsupported_count or mapping_report.requires_transform_count:
        raise ValueError(
            "GGUF writing requires fully mapped materialized tensors; "
            f"unsupported={mapping_report.unsupported_count}, "
            f"requires_transform={mapping_report.requires_transform_count}"
        )

    try:
        import gguf
    except ModuleNotFoundError as ex:
        raise RuntimeError(
            "GGUF writing requires the optional 'gguf' Python package. "
            "Install the converter extra for onnxruntime-ep-aivis-ggml."
        ) from ex

    import numpy as np
    import onnx
    from onnx import numpy_helper

    source_path = Path(model_path)
    gguf_path = Path(output_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    source_model = onnx.load(str(source_path), load_external_data=True)
    source_arrays = {
        initializer.name: np.ascontiguousarray(
            numpy_helper.to_array(initializer).astype(np.float32)
        )
        for initializer in source_model.graph.initializer
    }
    mapped_targets = {
        record.source_name: record.target_name
        for record in mapping_report.records
        if record.status == "mapped" and record.target_name is not None
    }
    weight_norm_targets = _weight_norm_targets(mapping_report)

    writer = gguf.GGUFWriter(path=None, arch=STYLE_BERT_VITS2_ARCHITECTURE)
    tensor_count = 0
    tensor_f32_count = 0
    tensor_f16_count = 0
    written_targets: set[str] = set()
    try:
        for source_name, target_name in mapped_targets.items():
            array = source_arrays.get(source_name)
            if array is None:
                continue
            array = prepare_mapped_tensor_array(
                source_name=source_name,
                target_name=target_name,
                array=array,
            )
            stored_dtype = _add_tensor(
                gguf,
                writer,
                target_name,
                array,
                store_f16_weights=store_f16_weights,
            )
            tensor_count += 1
            tensor_f16_count += 1 if stored_dtype == "F16" else 0
            tensor_f32_count += 1 if stored_dtype == "F32" else 0
            written_targets.add(target_name)

        for base_name, target_name in weight_norm_targets.items():
            if target_name in written_targets:
                continue
            weight_g = source_arrays[f"{base_name}.weight_g"]
            weight_v = source_arrays[f"{base_name}.weight_v"]
            array = materialize_weight_norm(weight_g=weight_g, weight_v=weight_v)
            stored_dtype = _add_tensor(
                gguf,
                writer,
                target_name,
                array,
                store_f16_weights=store_f16_weights,
            )
            tensor_count += 1
            tensor_f16_count += 1 if stored_dtype == "F16" else 0
            tensor_f32_count += 1 if stored_dtype == "F32" else 0
            written_targets.add(target_name)

        style_vectors = np.ascontiguousarray(
            np.load(style_vectors_path).astype(np.float32)
        )
        if "style_bert_vits2.style_vectors" not in written_targets:
            stored_dtype = _add_tensor(
                gguf,
                writer,
                "style_bert_vits2.style_vectors",
                style_vectors,
                store_f16_weights=store_f16_weights,
            )
            tensor_count += 1
            tensor_f16_count += 1 if stored_dtype == "F16" else 0
            tensor_f32_count += 1 if stored_dtype == "F32" else 0

        _add_style_bert_vits2_metadata(
            gguf,
            writer,
            config,
            has_f16_tensors=tensor_f16_count > 0,
        )
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        writer.write_header_to_file(path=gguf_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
    finally:
        writer.close()

    return GgufWriteResult(
        filename=gguf_path.name,
        size_bytes=gguf_path.stat().st_size,
        sha256=_file_sha256(gguf_path),
        tensor_count=tensor_count,
        tensor_f32_count=tensor_f32_count,
        tensor_f16_count=tensor_f16_count,
    )


def materialize_weight_norm(
    *,
    weight_g: Any,
    weight_v: Any,
    dim: int = 0,
) -> Any:
    """Materialize PyTorch weight_norm parameters into the final weight."""

    import numpy as np

    v = np.ascontiguousarray(np.asarray(weight_v, dtype=np.float32))
    g = np.asarray(weight_g, dtype=np.float32)
    if dim < 0:
        dim += v.ndim
    if dim < 0 or dim >= v.ndim:
        raise ValueError(f"weight_norm dim {dim} is out of range for shape {v.shape}")

    if g.ndim == 1 and v.ndim > 1:
        reshape = [1] * v.ndim
        reshape[dim] = g.shape[0]
        g = g.reshape(reshape)

    norm_axes = tuple(axis for axis in range(v.ndim) if axis != dim)
    norm = np.sqrt(np.sum(v * v, axis=norm_axes, keepdims=True))
    return np.ascontiguousarray(v * (g / norm), dtype=np.float32)


def prepare_mapped_tensor_array(
    *,
    source_name: str,
    target_name: str,
    array: Any,
) -> Any:
    """Normalize mapped ONNX initializer layout to the TTS.cpp GGUF contract."""

    import numpy as np

    if _requires_matmul_weight_transpose(
        source_name=source_name,
        target_name=target_name,
    ):
        return np.ascontiguousarray(np.asarray(array, dtype=np.float32).T)
    if _requires_negative_log_reconstruction(
        source_name=source_name,
        target_name=target_name,
    ):
        return np.ascontiguousarray(-np.log(np.asarray(array, dtype=np.float32)))
    return array


def _requires_matmul_weight_transpose(*, source_name: str, target_name: str) -> bool:
    if not source_name.startswith("onnx::MatMul_"):
        return False
    if target_name == "style_bert_vits2.text_encoder.style_proj.weight":
        return True
    if target_name == "style_bert_vits2.te.enc.spk.w":
        return True
    return target_name.startswith("style_bert_vits2.fl.") and target_name.endswith(
        ".enc.spk.w"
    )


def _requires_negative_log_reconstruction(
    *,
    source_name: str,
    target_name: str,
) -> bool:
    return (
        source_name == "/sdp/flows.0/Exp_output_0"
        and target_name == "style_bert_vits2.sdp.flows.0.logs"
    )


def _weight_norm_targets(mapping_report: TensorMappingReport) -> dict[str, str]:
    targets: dict[str, str] = {}
    grouped_sources: dict[str, set[str]] = {}
    for record in mapping_report.records:
        if record.status != "transform_source" or record.target_name is None:
            continue
        base_name, leaf = record.source_name.rsplit(".", 1)
        grouped_sources.setdefault(base_name, set()).add(leaf)
        targets[base_name] = record.target_name

    return {
        base_name: targets[base_name]
        for base_name, leaves in grouped_sources.items()
        if leaves == {"weight_g", "weight_v"}
    }


def _store_as_f16(name: str, *, enabled: bool = True) -> bool:
    if not enabled:
        return False
    if not name.startswith("style_bert_vits2."):
        return False
    if "embedding" in name:
        return False
    if ".norm" in name or "norm_" in name:
        return False
    if name.startswith("style_bert_vits2.decoder.ups."):
        return False
    return name.endswith(".weight") or name.endswith(".w")


def _add_tensor(
    gguf: Any,
    writer: Any,
    name: str,
    array: Any,
    *,
    store_f16_weights: bool,
) -> str:
    import numpy as np

    quant_types = getattr(gguf, "GGMLQuantizationType", None)
    if _store_as_f16(name, enabled=store_f16_weights):
        raw_dtype = getattr(quant_types, "F16", None)
        tensor = np.ascontiguousarray(np.asarray(array, dtype=np.float16))
        stored_dtype = "F16"
    else:
        raw_dtype = getattr(quant_types, "F32", None)
        tensor = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        stored_dtype = "F32"

    if raw_dtype is None:
        writer.add_tensor(name, tensor)
    else:
        writer.add_tensor(name, tensor, raw_dtype=raw_dtype)
    return stored_dtype


def _add_style_bert_vits2_metadata(
    gguf: Any, writer: Any, config: dict[str, Any], *, has_f16_tensors: bool
) -> None:
    arch = STYLE_BERT_VITS2_ARCHITECTURE
    model_config = dict(config.get("model", {}))
    data_config = dict(config.get("data", {}))

    gguf_type = getattr(getattr(gguf, "GGUFType", None), "MODEL", None)
    if gguf_type is not None:
        writer.add_type(gguf_type)
    llama_file_type = getattr(gguf, "LlamaFileType", None)
    file_type = getattr(
        llama_file_type,
        "MOSTLY_F16" if has_f16_tensors else "ALL_F32",
        None,
    )
    if file_type is not None:
        writer.add_file_type(file_type)
    quant_version = getattr(gguf, "GGML_QUANT_VERSION", None)
    if quant_version is not None:
        writer.add_quantization_version(quant_version)

    upsample_rates = _int_list(model_config.get("upsample_rates"))
    upsample_kernels = _int_list(model_config.get("upsample_kernel_sizes"))
    resblock_kernels = _int_list(model_config.get("resblock_kernel_sizes"))
    resblock_dilations = _int_matrix(model_config.get("resblock_dilation_sizes"))
    num_upsamples = max(len(upsample_rates), len(upsample_kernels), 1)
    num_kernels = max(len(resblock_kernels), 1)

    writer.add_uint32(f"{arch}.decoder_only", 1)
    writer.add_uint32(
        f"{arch}.sample_rate", _int_at(data_config, "sampling_rate", 44100)
    )
    writer.add_uint32(
        f"{arch}.inter_channels", _int_at(model_config, "inter_channels", 192)
    )
    writer.add_uint32(
        f"{arch}.hidden_channels", _int_at(model_config, "hidden_channels", 192)
    )
    writer.add_uint32(
        f"{arch}.filter_channels", _int_at(model_config, "filter_channels", 768)
    )
    writer.add_uint32(f"{arch}.n_heads", _int_at(model_config, "n_heads", 2))
    writer.add_uint32(
        f"{arch}.text_encoder.n_layers", _int_at(model_config, "n_layers", 6)
    )
    writer.add_uint32(
        f"{arch}.text_encoder.kernel", _int_at(model_config, "kernel_size", 3)
    )
    writer.add_uint32(f"{arch}.text_encoder.window_size", 4)
    writer.add_uint32(
        f"{arch}.text_encoder.cond_layer_idx",
        _int_at(model_config, "cond_layer_idx", 2),
    )
    writer.add_uint32(
        f"{arch}.gin_channels", _int_at(model_config, "gin_channels", 512)
    )
    writer.add_uint32(
        f"{arch}.jp_extra",
        1 if str(config.get("version", "")).endswith("JP-Extra") else 0,
    )
    writer.add_uint32(
        f"{arch}.duration_predictor.filter_channels",
        _int_at(model_config, "filter_channels_dp", 256),
    )
    writer.add_uint32(
        f"{arch}.duration_predictor.kernel",
        _int_at(
            model_config, "kernel_size_dp", _int_at(model_config, "kernel_size", 3)
        ),
    )
    writer.add_uint32(
        f"{arch}.duration_predictor.padding",
        _int_at(model_config, "kernel_size_dp", _int_at(model_config, "kernel_size", 3))
        // 2,
    )
    writer.add_uint32(f"{arch}.sdp.n_flows", _int_at(model_config, "n_flows", 4))
    writer.add_uint32(f"{arch}.sdp.n_layers", 3)
    writer.add_uint32(
        f"{arch}.sdp.kernel",
        _int_at(
            model_config, "kernel_size_dp", _int_at(model_config, "kernel_size", 3)
        ),
    )
    writer.add_uint32(f"{arch}.sdp.num_bins", 10)
    writer.add_uint32(f"{arch}.flow.use_transformer", 1)
    writer.add_uint32(f"{arch}.flow.n_flows", _int_at(model_config, "n_flow_layer", 4))
    writer.add_uint32(
        f"{arch}.flow.n_layers", _int_at(model_config, "n_layers_flow", 6)
    )
    writer.add_uint32(
        f"{arch}.flow.kernel", _int_at(model_config, "kernel_size_flow", 5)
    )
    writer.add_uint32(f"{arch}.flow.window_size", 4)
    writer.add_uint32(f"{arch}.flow.cond_layer_idx", 2)
    writer.add_uint32(
        f"{arch}.upsample_initial_channel",
        _int_at(model_config, "upsample_initial_channel", 512),
    )
    writer.add_uint32(f"{arch}.decoder.num_upsamples", num_upsamples)
    writer.add_uint32(f"{arch}.decoder.num_kernels", num_kernels)
    writer.add_uint32(f"{arch}.decoder.resblock", _int_at(model_config, "resblock", 1))
    writer.add_uint32(f"{arch}.decoder.conv_pre.padding", 3)
    writer.add_uint32(f"{arch}.decoder.conv_pre.kernel", 7)
    writer.add_uint32(f"{arch}.decoder.conv_post.padding", 3)
    writer.add_uint32(f"{arch}.decoder.conv_post.kernel", 7)

    for index in range(num_upsamples):
        stride = _list_value(upsample_rates, index, 1)
        kernel = _list_value(upsample_kernels, index, 1)
        writer.add_uint32(f"{arch}.decoder.ups.{index}.stride", stride)
        writer.add_uint32(
            f"{arch}.decoder.ups.{index}.padding", max((kernel - stride) // 2, 0)
        )
        writer.add_uint32(f"{arch}.decoder.ups.{index}.kernel", kernel)

    for block_index in range(num_upsamples * num_kernels):
        kernel = _list_value(resblock_kernels, block_index % num_kernels, 3)
        dilations = _matrix_row(
            resblock_dilations, block_index % num_kernels, (1, 3, 5)
        )
        for conv_index in range(3):
            dilation = _list_value(dilations, conv_index, 1)
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.padding",
                max((kernel * dilation - dilation) // 2, 0),
            )
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.dilation",
                dilation,
            )
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.kernel",
                kernel,
            )
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.padding",
                max((kernel - 1) // 2, 0),
            )
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.dilation",
                1,
            )
            writer.add_uint32(
                f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.kernel",
                kernel,
            )


def _int_at(values: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(values.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _int_list(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _int_matrix(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_int_list(item) for item in value)


def _list_value(values: tuple[int, ...], index: int, fallback: int) -> int:
    if 0 <= index < len(values):
        return values[index]
    return fallback


def _matrix_row(
    values: tuple[tuple[int, ...], ...],
    index: int,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    if 0 <= index < len(values) and values[index]:
        return values[index]
    return fallback


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
