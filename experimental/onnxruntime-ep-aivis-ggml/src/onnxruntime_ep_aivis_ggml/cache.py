"""GGML artifact cache planning for the Aivis ONNX Runtime Plugin EP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from onnxruntime_ep_aivis_ggml.gguf_writer import (
    GgufWriteResult,
    write_tts_cpp_style_bert_vits2_gguf,
)
from onnxruntime_ep_aivis_ggml.initializer_pack import (
    InitializerTensorPack,
    write_initializer_tensor_pack,
)
from onnxruntime_ep_aivis_ggml.signature import (
    OnnxGraphSignature,
    SignatureMatch,
    build_signature_contract,
    load_onnx_graph_signature,
    match_supported_style_bert_vits2_synthesis,
    signature_structural_sha256,
)
from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (
    TensorMappingReport,
    build_graph_initializer_target_overrides,
    build_tts_cpp_mapping_report,
)

CACHE_MANIFEST_VERSION = "aivis-ggml-onnx-cache-v1"
EP_CONTEXT_LITE_VERSION = "aivis-ggml-ep-context-lite-v1"
OFFICIAL_EP_CONTEXT_VERSION = "aivis-ggml-official-ep-context-v1"
RUNTIME_REGISTRY_CONTRACT = "aivis-ggml-runtime-registry-v1"
TTS_CPP_RUNTIME_CONTRACT = "tts-style-bert-vits2-c-api-v1"
COMPATIBILITY_MATRIX_VERSION = "aivis-ggml-compatibility-matrix-v1"
COMPILED_MODEL_COMPATIBILITY_VERSION = "aivis-ggml-compiled-model-compatibility-v1"
SIGNATURE_CONTRACT_VERSION = "aivis-ggml-signature-contract-v1"
PROVIDER_NAME = "AivisGgmlExecutionProvider"
PROVIDER_VERSION = "0.1.0"
DEFAULT_CONVERTER_VERSION = (
    "tts-cpp-style-bert-vits2-converter-f16-no-embed-norm-no-ups-v1"
)
F32_CONVERTER_VERSION = "tts-cpp-style-bert-vits2-converter-f32-v1"
TESTED_ORT_RUNTIME_VERSION = "1.26.0"
ORT_PLUGIN_EP_API_VERSION = 26
EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION = 1
EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION = 1
SUPPORTED_OFFICIAL_EP_CONTEXT_GRAPH_KINDS = ("synthesis", "jp-bert")
SUPPORTED_BACKENDS = ("vulkan", "metal", "cpu")
SUPPORTED_PRECISIONS = ("accurate", "fast")
OFFICIAL_EP_CONTEXT_ARTIFACT_KEYS = (
    "cache_manifest_path",
    "gguf_path",
    "jp_bert_gguf_path",
)

REQUIRED_TTS_CPP_SYMBOLS = (
    "tts_style_bert_vits2_last_error",
    "tts_style_bert_vits2_load_model",
    "tts_style_bert_vits2_free_model",
    "tts_style_bert_vits2_jp_bert_load_model",
    "tts_style_bert_vits2_jp_bert_free_model",
    "tts_style_bert_vits2_synthesize_front",
    "tts_style_bert_vits2_synthesize_front_with_style_vec",
    "tts_style_bert_vits2_jp_bert_encode_features",
)
OPTIONAL_TTS_CPP_VERSION_SYMBOLS = (
    "tts_style_bert_vits2_runtime_abi_version",
    "tts_style_bert_vits2_gguf_schema_version",
)


@dataclass(frozen=True)
class GgmlCachePlan:
    """Resolved cache locations and manifest payload for a supported ONNX graph."""

    cache_key: str
    artifact_dir: Path
    manifest_path: Path
    gguf_path: Path
    manifest: dict[str, Any]
    tensor_pack_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        data = {
            "cache_key": self.cache_key,
            "artifact_dir": str(self.artifact_dir),
            "manifest_path": str(self.manifest_path),
            "gguf_path": str(self.gguf_path),
            "manifest": self.manifest,
        }
        if self.tensor_pack_path is not None:
            data["tensor_pack_path"] = str(self.tensor_pack_path)
        return data


def prepare_ggml_cache(
    *,
    model_path: str | Path,
    cache_dir: str | Path,
    config_path: str | Path | None = None,
    style_vectors_path: str | Path | None = None,
    backend: str = "vulkan",
    precision: str = "accurate",
    converter_version: str = DEFAULT_CONVERTER_VERSION,
    allow_unsupported: bool = False,
    write_tensor_pack: bool = False,
    write_gguf: bool = False,
    fail_on_unsupported_mapping: bool = False,
) -> GgmlCachePlan:
    """Create a deterministic cache manifest for a future ONNX-to-GGML compile."""

    if write_gguf and not write_tensor_pack:
        raise ValueError("GGUF writing requires write_tensor_pack=True.")
    if write_gguf and config_path is None:
        raise ValueError("GGUF writing requires config_path.")
    if write_gguf and style_vectors_path is None:
        raise ValueError("GGUF writing requires style_vectors_path.")
    if write_gguf and converter_version in {"", "unimplemented"}:
        raise ValueError("GGUF writing requires a real converter_version.")

    source_path = Path(model_path)
    root_cache_dir = Path(cache_dir)
    signature = load_onnx_graph_signature(source_path)
    match = match_supported_style_bert_vits2_synthesis(signature)
    if not match.supported and not allow_unsupported:
        reasons = "; ".join(match.reasons)
        raise ValueError(f"Unsupported Style-Bert-VITS2 synthesis graph: {reasons}")

    source_sha256 = _file_sha256(source_path)
    cache_key = build_cache_key(
        source_sha256=source_sha256,
        signature=signature,
        converter_version=converter_version,
    )
    artifact_dir = root_cache_dir / cache_key
    manifest_path = artifact_dir / "manifest.json"
    gguf_path = artifact_dir / "model.gguf"
    tensor_pack_path = artifact_dir / "initializers.bin"
    initializer_pack: InitializerTensorPack | None = None
    mapping_report: TensorMappingReport | None = None
    gguf_write_result: GgufWriteResult | None = None
    external_sources: dict[str, dict[str, Any]] = {}

    if config_path is not None:
        external_sources["style_bert_vits2_config"] = _source_file_manifest(
            Path(config_path)
        )
    if style_vectors_path is not None:
        external_sources["style_vectors"] = _source_file_manifest(
            Path(style_vectors_path)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if write_tensor_pack:
        initializer_pack = write_initializer_tensor_pack(
            model_path=source_path,
            output_path=tensor_pack_path,
        )
        available_external_artifacts = (
            ("style_vectors.npy",) if style_vectors_path is not None else ()
        )
        mapping_report = build_tts_cpp_mapping_report(
            initializer_pack.records,
            available_external_artifacts=available_external_artifacts,
            target_name_overrides=build_graph_initializer_target_overrides(source_path),
        )
        if fail_on_unsupported_mapping and (
            mapping_report.unsupported_count > 0
            or mapping_report.requires_transform_count > 0
        ):
            raise ValueError(
                "Initializer tensor mapping is incomplete: "
                f"unsupported={mapping_report.unsupported_count}, "
                f"requires_transform={mapping_report.requires_transform_count}"
            )

    manifest = build_cache_manifest(
        source_path=source_path,
        source_sha256=source_sha256,
        signature=signature,
        match=match,
        cache_key=cache_key,
        backend=backend,
        precision=precision,
        converter_version=converter_version,
        initializer_pack=initializer_pack,
        mapping_report=mapping_report,
        external_sources=external_sources,
        gguf_write_result=None,
    )

    if write_gguf:
        assert mapping_report is not None
        assert config_path is not None
        assert style_vectors_path is not None
        gguf_write_result = write_tts_cpp_style_bert_vits2_gguf(
            model_path=source_path,
            output_path=gguf_path,
            config_path=config_path,
            style_vectors_path=style_vectors_path,
            mapping_report=mapping_report,
            readiness=manifest["converter"]["readiness"],
            store_f16_weights=_converter_stores_f16_weights(converter_version),
        )
        manifest = build_cache_manifest(
            source_path=source_path,
            source_sha256=source_sha256,
            signature=signature,
            match=match,
            cache_key=cache_key,
            backend=backend,
            precision=precision,
            converter_version=converter_version,
            initializer_pack=initializer_pack,
            mapping_report=mapping_report,
            external_sources=external_sources,
            gguf_write_result=gguf_write_result,
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return GgmlCachePlan(
        cache_key=cache_key,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        gguf_path=gguf_path,
        manifest=manifest,
        tensor_pack_path=tensor_pack_path if write_tensor_pack else None,
    )


def build_cache_key(
    *,
    source_sha256: str,
    signature: OnnxGraphSignature,
    converter_version: str,
) -> str:
    """Build a stable cache key for the source model and converter contract."""

    payload = {
        "version": CACHE_MANIFEST_VERSION,
        "source_sha256": source_sha256,
        "converter_version": converter_version,
        "graph_name": signature.graph_name,
        "opsets": signature.opsets,
        "op_sequence_sha256": signature.op_sequence_sha256,
        "initializer_names_sha256": signature.initializer_names_sha256,
        "signature_structural_sha256": signature_structural_sha256(signature),
        "metadata_model_architecture": signature.metadata_model_architecture,
        "metadata_model_format": signature.metadata_model_format,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()


def _converter_stores_f16_weights(converter_version: str) -> bool:
    return "f16" in converter_version.lower()


def build_cache_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    signature: OnnxGraphSignature,
    match: SignatureMatch,
    cache_key: str,
    backend: str,
    precision: str,
    converter_version: str,
    initializer_pack: InitializerTensorPack | None = None,
    mapping_report: TensorMappingReport | None = None,
    external_sources: dict[str, dict[str, Any]] | None = None,
    gguf_write_result: GgufWriteResult | None = None,
) -> dict[str, Any]:
    """Build a portable cache manifest without embedding local absolute paths."""

    external_sources = external_sources or {}
    artifacts = {
        "manifest": "manifest.json",
        "gguf": "model.gguf",
    }
    if initializer_pack is not None:
        artifacts["initializer_tensor_pack"] = "initializers.bin"
    runtime_contract = build_runtime_contract()
    ep_context = build_ep_context_lite(
        cache_key=cache_key,
        backend=backend,
        precision=precision,
        artifacts=artifacts,
    )

    converter_readiness = build_converter_readiness(
        signature=signature,
        initializer_pack=initializer_pack,
        mapping_report=mapping_report,
        external_sources=external_sources,
    )
    manifest = {
        "version": CACHE_MANIFEST_VERSION,
        "status": "ready" if gguf_write_result is not None else "planned",
        "cache_key": cache_key,
        "source": {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "graph_signature": signature.to_dict(),
        "signature_contract": build_signature_contract(signature),
        "match": match.to_dict(),
        "runtime_contract": runtime_contract,
        "compatibility_matrix": build_compatibility_matrix(),
        "ep_context": ep_context,
        "converter": {
            "name": "onnxruntime-ep-aivis-ggml",
            "version": converter_version,
            "state": "written" if gguf_write_result is not None else "not_implemented",
            "readiness": converter_readiness,
        },
        "provider_options": {
            "backend": backend,
            "precision": precision,
        },
        "artifacts": artifacts,
    }
    if external_sources:
        manifest["external_sources"] = external_sources
    if initializer_pack is not None:
        manifest["initializer_tensor_pack"] = initializer_pack.to_dict()
    if mapping_report is not None:
        manifest["tts_cpp_tensor_mapping"] = mapping_report.to_dict()
    if gguf_write_result is not None:
        manifest["gguf_write"] = gguf_write_result.to_dict()
    return manifest


def build_runtime_contract() -> dict[str, Any]:
    """Return the portable TTS.cpp runtime contract required by native claim."""

    return {
        "version": RUNTIME_REGISTRY_CONTRACT,
        "provider_name": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "tts_cpp_runtime_contract": TTS_CPP_RUNTIME_CONTRACT,
        "required_tts_cpp_symbols": REQUIRED_TTS_CPP_SYMBOLS,
        "optional_tts_cpp_version_symbols": OPTIONAL_TTS_CPP_VERSION_SYMBOLS,
        "expected_optional_versions": {
            "runtime_abi": EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
            "gguf_schema": EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
        },
    }


def build_compatibility_matrix() -> dict[str, Any]:
    """Return the explicit deployment compatibility matrix for this artifact."""

    return {
        "version": COMPATIBILITY_MATRIX_VERSION,
        "provider": {
            "name": PROVIDER_NAME,
            "version": PROVIDER_VERSION,
        },
        "onnxruntime": {
            "tested_runtime_version": TESTED_ORT_RUNTIME_VERSION,
            "plugin_ep_api_version": ORT_PLUGIN_EP_API_VERSION,
            "requires_model_editor_api": True,
        },
        "runtime_contract": {
            "registry": RUNTIME_REGISTRY_CONTRACT,
            "tts_cpp_c_api": TTS_CPP_RUNTIME_CONTRACT,
            "expected_optional_versions": {
                "runtime_abi": EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
                "gguf_schema": EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
            },
        },
        "model_signature_contracts": {
            "synthesis": SIGNATURE_CONTRACT_VERSION,
            "jp_bert": SIGNATURE_CONTRACT_VERSION,
        },
        "ep_context": {
            "lite_manifest": EP_CONTEXT_LITE_VERSION,
            "official_node_generation": "supported",
            "official_node_inference": "lazy_artifact_restore_tts_library_required",
            "official_payload_version": OFFICIAL_EP_CONTEXT_VERSION,
            "official_payload_validator": "supported",
        },
        "compiled_model_compatibility": {
            "version": COMPILED_MODEL_COMPATIBILITY_VERSION,
            "native_factory_validation": "supported",
            "optimal_requires_exact_contract": True,
            "ort_api_mismatch": "prefer_recompilation",
        },
    }


def build_compiled_model_compatibility_info(
    *,
    graph_kind: str,
    backend: str,
    precision: str,
    device: str = "",
    ort_api_version: int = ORT_PLUGIN_EP_API_VERSION,
) -> dict[str, Any]:
    """Build the JSON contract used by ORT model package variant selection."""

    payload = {
        "version": COMPILED_MODEL_COMPATIBILITY_VERSION,
        "provider_name": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "ort_api_version": ort_api_version,
        "runtime_registry_contract": RUNTIME_REGISTRY_CONTRACT,
        "tts_cpp_runtime_contract": TTS_CPP_RUNTIME_CONTRACT,
        "tts_cpp_runtime_abi_version": EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
        "gguf_schema_version": EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
        "model_signature_contract": SIGNATURE_CONTRACT_VERSION,
        "official_ep_context_payload_version": OFFICIAL_EP_CONTEXT_VERSION,
        "graph_kind": graph_kind,
        "backend": backend,
        "device": device,
        "precision": precision,
    }
    compatibility = validate_compiled_model_compatibility_info(payload)
    if compatibility == "unsupported":
        raise ValueError("Invalid compiled model compatibility info.")
    return payload


def validate_compiled_model_compatibility_info(
    payload: dict[str, Any] | str,
) -> str:
    """
    Mirror native EP compiled model compatibility validation.

    Returns one of ``optimal``, ``prefer_recompilation``, ``unsupported``, or
    ``not_applicable``.
    """

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return "not_applicable"
        payload = parsed
    if not isinstance(payload, dict):
        return "not_applicable"

    if payload.get("provider_name") != PROVIDER_NAME:
        return "not_applicable"

    expected_strings = {
        "version": COMPILED_MODEL_COMPATIBILITY_VERSION,
        "provider_version": PROVIDER_VERSION,
        "runtime_registry_contract": RUNTIME_REGISTRY_CONTRACT,
        "tts_cpp_runtime_contract": TTS_CPP_RUNTIME_CONTRACT,
        "model_signature_contract": SIGNATURE_CONTRACT_VERSION,
        "official_ep_context_payload_version": OFFICIAL_EP_CONTEXT_VERSION,
    }
    for key, expected in expected_strings.items():
        if payload.get(key) != expected:
            return "unsupported"

    if payload.get("tts_cpp_runtime_abi_version") != (
        EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION
    ):
        return "unsupported"
    if payload.get("gguf_schema_version") != EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION:
        return "unsupported"
    if payload.get("graph_kind") not in SUPPORTED_OFFICIAL_EP_CONTEXT_GRAPH_KINDS:
        return "unsupported"
    if payload.get("backend") not in SUPPORTED_BACKENDS:
        return "unsupported"
    if payload.get("precision") not in SUPPORTED_PRECISIONS:
        return "unsupported"

    if payload.get("ort_api_version") != ORT_PLUGIN_EP_API_VERSION:
        return "prefer_recompilation"
    return "optimal"


def build_ep_context_lite(
    *,
    cache_key: str,
    backend: str,
    precision: str,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    """Describe the portable artifact layout before official ORT EPContext."""

    return {
        "version": EP_CONTEXT_LITE_VERSION,
        "provider_name": PROVIDER_NAME,
        "cache_key": cache_key,
        "provider_options": {
            "backend": backend,
            "precision": precision,
        },
        "artifacts": dict(artifacts),
        "portable": True,
        "official_ort_ep_context": {
            "enabled": False,
            "status": "manifest_only",
        },
    }


def build_official_ep_context_payload(
    *,
    graph_kind: str,
    graph_name: str,
    graph_index: int,
    backend: str,
    precision: str,
    cache_manifest_path: str | Path | None,
    gguf_path: str | Path | None,
    jp_bert_gguf_path: str | Path | None,
    device: str = "",
    n_threads: int = 0,
) -> dict[str, Any]:
    """Build the portable JSON payload stored by official ORT EPContext nodes."""

    payload = {
        "version": OFFICIAL_EP_CONTEXT_VERSION,
        "provider_name": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "runtime_registry_contract": RUNTIME_REGISTRY_CONTRACT,
        "tts_cpp_runtime_contract": TTS_CPP_RUNTIME_CONTRACT,
        "graph_kind": graph_kind,
        "graph_name": graph_name,
        "graph_index": graph_index,
        "backend": backend,
        "device": device,
        "precision": precision,
        "n_threads": n_threads,
        "artifacts": {
            "cache_manifest_path": _artifact_path_text(cache_manifest_path),
            "gguf_path": _artifact_path_text(gguf_path),
            "jp_bert_gguf_path": _artifact_path_text(jp_bert_gguf_path),
        },
    }
    errors = validate_official_ep_context_payload(payload, graph_kind=graph_kind)
    if errors:
        raise ValueError("Invalid Aivis GGML EPContext payload: " + ", ".join(errors))
    return payload


def validate_official_ep_context_payload(
    payload: dict[str, Any],
    *,
    graph_kind: str | None = None,
) -> tuple[str, ...]:
    """Validate the official ORT EPContext payload contract before deployment."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ("ep_context_payload_root_invalid",)

    _validate_expected_string(
        errors,
        payload,
        "version",
        OFFICIAL_EP_CONTEXT_VERSION,
        "ep_context_payload_version_mismatch",
    )
    _validate_expected_string(
        errors,
        payload,
        "provider_name",
        PROVIDER_NAME,
        "ep_context_payload_provider_mismatch",
    )
    _validate_expected_string(
        errors,
        payload,
        "provider_version",
        PROVIDER_VERSION,
        "ep_context_payload_provider_version_mismatch",
    )
    _validate_expected_string(
        errors,
        payload,
        "runtime_registry_contract",
        RUNTIME_REGISTRY_CONTRACT,
        "ep_context_payload_runtime_registry_contract_mismatch",
    )
    _validate_expected_string(
        errors,
        payload,
        "tts_cpp_runtime_contract",
        TTS_CPP_RUNTIME_CONTRACT,
        "ep_context_payload_tts_cpp_runtime_contract_mismatch",
    )

    payload_graph_kind = payload.get("graph_kind")
    if payload_graph_kind not in SUPPORTED_OFFICIAL_EP_CONTEXT_GRAPH_KINDS:
        errors.append("ep_context_payload_graph_kind_invalid")
    if graph_kind is not None and payload_graph_kind != graph_kind:
        errors.append("ep_context_payload_graph_kind_mismatch")

    graph_index = payload.get("graph_index")
    if not _is_non_negative_int(graph_index):
        errors.append("ep_context_payload_graph_index_invalid")

    backend = payload.get("backend")
    if backend not in SUPPORTED_BACKENDS:
        errors.append("ep_context_payload_backend_invalid")

    precision = payload.get("precision")
    if precision not in SUPPORTED_PRECISIONS:
        errors.append("ep_context_payload_precision_invalid")

    device = payload.get("device")
    if not isinstance(device, str):
        errors.append("ep_context_payload_device_invalid")

    n_threads = payload.get("n_threads")
    if not _is_non_negative_int(n_threads):
        errors.append("ep_context_payload_n_threads_invalid")

    if _contains_key(payload, "tts_cpp_library_path"):
        errors.append("ep_context_payload_tts_library_path_embedded")

    artifact_paths: dict[str, str] = {}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("ep_context_payload_artifacts_missing")
    else:
        for name in OFFICIAL_EP_CONTEXT_ARTIFACT_KEYS:
            value = artifacts.get(name)
            if value is None:
                errors.append(f"ep_context_payload_artifact_missing:{name}")
                continue
            if not isinstance(value, str):
                errors.append(f"ep_context_payload_artifact_path_invalid:{name}")
                continue
            artifact_paths[name] = value
            if value and _is_absolute_or_parent_relative(value):
                errors.append(f"ep_context_payload_artifact_path_not_portable:{name}")

    if payload_graph_kind == "synthesis" and not artifact_paths.get("gguf_path"):
        errors.append("ep_context_payload_artifact_required:synthesis:gguf_path")
    if payload_graph_kind == "jp-bert" and not artifact_paths.get("jp_bert_gguf_path"):
        errors.append("ep_context_payload_artifact_required:jp-bert:jp_bert_gguf_path")

    return tuple(errors)


def load_official_ep_context_payload(path: str | Path) -> dict[str, Any]:
    """Load an official ORT EPContext payload JSON object from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EPContext payload root must be a JSON object.")
    return payload


def validate_cache_manifest(
    manifest: dict[str, Any],
    *,
    require_ready: bool = False,
) -> tuple[str, ...]:
    """Validate the portable manifest contract before provider deployment."""

    errors: list[str] = []
    if manifest.get("version") != CACHE_MANIFEST_VERSION:
        errors.append("version_mismatch")

    status = manifest.get("status")
    if status not in {"planned", "ready"}:
        errors.append("status_invalid")
    if require_ready and status != "ready":
        errors.append("status_not_ready")

    signature_contract = manifest.get("signature_contract")
    if not isinstance(signature_contract, dict):
        errors.append("signature_contract_missing")
    else:
        if signature_contract.get("version") != "aivis-ggml-signature-contract-v1":
            errors.append("signature_contract_version_mismatch")
        structural_sha256 = signature_contract.get("structural_sha256")
        if not isinstance(structural_sha256, str) or len(structural_sha256) != 64:
            errors.append("signature_contract_structural_hash_invalid")

    runtime_contract = manifest.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        errors.append("runtime_contract_missing")
    else:
        if runtime_contract.get("version") != RUNTIME_REGISTRY_CONTRACT:
            errors.append("runtime_contract_version_mismatch")
        if runtime_contract.get("provider_name") != PROVIDER_NAME:
            errors.append("runtime_contract_provider_mismatch")
        required_symbols = runtime_contract.get("required_tts_cpp_symbols")
        if not isinstance(required_symbols, (list, tuple)):
            errors.append("runtime_contract_required_symbols_invalid")
        else:
            missing_symbols = set(REQUIRED_TTS_CPP_SYMBOLS) - set(required_symbols)
            if missing_symbols:
                errors.append(
                    "runtime_contract_required_symbols_missing:"
                    + ",".join(sorted(missing_symbols))
                )

    ep_context = manifest.get("ep_context")
    if not isinstance(ep_context, dict):
        errors.append("ep_context_missing")
    else:
        if ep_context.get("version") != EP_CONTEXT_LITE_VERSION:
            errors.append("ep_context_version_mismatch")
        if ep_context.get("provider_name") != PROVIDER_NAME:
            errors.append("ep_context_provider_mismatch")
        official = ep_context.get("official_ort_ep_context")
        if not isinstance(official, dict) or official.get("enabled") is not False:
            errors.append("ep_context_official_status_invalid")

    compatibility_matrix = manifest.get("compatibility_matrix")
    if not isinstance(compatibility_matrix, dict):
        errors.append("compatibility_matrix_missing")
    else:
        _validate_compatibility_matrix(errors, compatibility_matrix)

    provider_options = manifest.get("provider_options")
    if not isinstance(provider_options, dict):
        errors.append("provider_options_missing")
    else:
        if provider_options.get("backend") not in SUPPORTED_BACKENDS:
            errors.append("provider_options_backend_invalid")
        if provider_options.get("precision") not in SUPPORTED_PRECISIONS:
            errors.append("provider_options_precision_invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts_missing")
    else:
        for name, relative_path in artifacts.items():
            if not isinstance(relative_path, str) or relative_path == "":
                errors.append(f"artifact_path_invalid:{name}")
                continue
            if _is_absolute_or_parent_relative(relative_path):
                errors.append(f"artifact_path_not_portable:{name}")

    return tuple(errors)


def load_cache_manifest(path: str | Path) -> dict[str, Any]:
    """Load a cache manifest from disk."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Cache manifest root must be a JSON object.")
    return manifest


def _is_absolute_or_parent_relative(value: str) -> bool:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _artifact_path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_expected_string(
    errors: list[str],
    payload: dict[str, Any],
    key: str,
    expected: str,
    error: str,
) -> None:
    if payload.get(key) != expected:
        errors.append(error)


def _validate_expected_int(
    errors: list[str],
    payload: dict[str, Any],
    key: str,
    expected: int,
    error: str,
) -> None:
    if payload.get(key) != expected:
        errors.append(error)


def _validate_expected_bool(
    errors: list[str],
    payload: dict[str, Any],
    key: str,
    expected: bool,
    error: str,
) -> None:
    if payload.get(key) is not expected:
        errors.append(error)


def _validate_compatibility_matrix(
    errors: list[str],
    compatibility_matrix: dict[str, Any],
) -> None:
    _validate_expected_string(
        errors,
        compatibility_matrix,
        "version",
        COMPATIBILITY_MATRIX_VERSION,
        "compatibility_matrix_version_mismatch",
    )

    provider = compatibility_matrix.get("provider")
    if not isinstance(provider, dict):
        errors.append("compatibility_matrix_provider_missing")
    else:
        _validate_expected_string(
            errors,
            provider,
            "name",
            PROVIDER_NAME,
            "compatibility_matrix_provider_name_mismatch",
        )
        _validate_expected_string(
            errors,
            provider,
            "version",
            PROVIDER_VERSION,
            "compatibility_matrix_provider_version_mismatch",
        )

    onnxruntime = compatibility_matrix.get("onnxruntime")
    if not isinstance(onnxruntime, dict):
        errors.append("compatibility_matrix_onnxruntime_missing")
    else:
        _validate_expected_string(
            errors,
            onnxruntime,
            "tested_runtime_version",
            TESTED_ORT_RUNTIME_VERSION,
            "compatibility_matrix_ort_runtime_version_mismatch",
        )
        _validate_expected_int(
            errors,
            onnxruntime,
            "plugin_ep_api_version",
            ORT_PLUGIN_EP_API_VERSION,
            "compatibility_matrix_ort_api_version_mismatch",
        )
        _validate_expected_bool(
            errors,
            onnxruntime,
            "requires_model_editor_api",
            True,
            "compatibility_matrix_ort_model_editor_requirement_mismatch",
        )

    runtime_contract = compatibility_matrix.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        errors.append("compatibility_matrix_runtime_contract_missing")
    else:
        _validate_expected_string(
            errors,
            runtime_contract,
            "registry",
            RUNTIME_REGISTRY_CONTRACT,
            "compatibility_matrix_runtime_registry_mismatch",
        )
        _validate_expected_string(
            errors,
            runtime_contract,
            "tts_cpp_c_api",
            TTS_CPP_RUNTIME_CONTRACT,
            "compatibility_matrix_tts_cpp_c_api_mismatch",
        )
        expected_versions = runtime_contract.get("expected_optional_versions")
        if not isinstance(expected_versions, dict):
            errors.append("compatibility_matrix_tts_cpp_versions_missing")
        else:
            _validate_expected_int(
                errors,
                expected_versions,
                "runtime_abi",
                EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
                "compatibility_matrix_tts_cpp_runtime_abi_mismatch",
            )
            _validate_expected_int(
                errors,
                expected_versions,
                "gguf_schema",
                EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
                "compatibility_matrix_tts_cpp_gguf_schema_mismatch",
            )

    signature_contracts = compatibility_matrix.get("model_signature_contracts")
    if not isinstance(signature_contracts, dict):
        errors.append("compatibility_matrix_signature_contracts_missing")
    else:
        _validate_expected_string(
            errors,
            signature_contracts,
            "synthesis",
            SIGNATURE_CONTRACT_VERSION,
            "compatibility_matrix_synthesis_signature_contract_mismatch",
        )
        _validate_expected_string(
            errors,
            signature_contracts,
            "jp_bert",
            SIGNATURE_CONTRACT_VERSION,
            "compatibility_matrix_jp_bert_signature_contract_mismatch",
        )

    ep_context = compatibility_matrix.get("ep_context")
    if not isinstance(ep_context, dict):
        errors.append("compatibility_matrix_ep_context_missing")
    else:
        _validate_expected_string(
            errors,
            ep_context,
            "lite_manifest",
            EP_CONTEXT_LITE_VERSION,
            "compatibility_matrix_ep_context_lite_mismatch",
        )
        _validate_expected_string(
            errors,
            ep_context,
            "official_node_generation",
            "supported",
            "compatibility_matrix_ep_context_generation_mismatch",
        )
        _validate_expected_string(
            errors,
            ep_context,
            "official_node_inference",
            "lazy_artifact_restore_tts_library_required",
            "compatibility_matrix_ep_context_inference_mismatch",
        )
        _validate_expected_string(
            errors,
            ep_context,
            "official_payload_version",
            OFFICIAL_EP_CONTEXT_VERSION,
            "compatibility_matrix_ep_context_payload_version_mismatch",
        )
        _validate_expected_string(
            errors,
            ep_context,
            "official_payload_validator",
            "supported",
            "compatibility_matrix_ep_context_payload_validator_mismatch",
        )

    compiled_model_compatibility = compatibility_matrix.get(
        "compiled_model_compatibility"
    )
    if not isinstance(compiled_model_compatibility, dict):
        errors.append("compatibility_matrix_compiled_model_compatibility_missing")
    else:
        _validate_expected_string(
            errors,
            compiled_model_compatibility,
            "version",
            COMPILED_MODEL_COMPATIBILITY_VERSION,
            "compatibility_matrix_compiled_model_compatibility_version_mismatch",
        )
        _validate_expected_string(
            errors,
            compiled_model_compatibility,
            "native_factory_validation",
            "supported",
            "compatibility_matrix_compiled_model_native_validation_mismatch",
        )
        _validate_expected_bool(
            errors,
            compiled_model_compatibility,
            "optimal_requires_exact_contract",
            True,
            "compatibility_matrix_compiled_model_exact_contract_mismatch",
        )
        _validate_expected_string(
            errors,
            compiled_model_compatibility,
            "ort_api_mismatch",
            "prefer_recompilation",
            "compatibility_matrix_compiled_model_ort_api_policy_mismatch",
        )


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(child, key) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(child, key) for child in value)
    return False


def build_converter_readiness(
    *,
    signature: OnnxGraphSignature,
    initializer_pack: InitializerTensorPack | None,
    mapping_report: TensorMappingReport | None,
    external_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe whether the planned cache entry has enough inputs for GGUF writing."""

    blockers: list[str] = []
    if initializer_pack is None:
        blockers.append("initializer_tensor_pack_missing")
    elif initializer_pack.tensor_count != signature.initializer_count:
        blockers.append(
            "initializer_count_mismatch:"
            f"{initializer_pack.tensor_count}!={signature.initializer_count}"
        )

    if mapping_report is None:
        blockers.append("tts_cpp_tensor_mapping_missing")
    else:
        if mapping_report.unsupported_count:
            blockers.append(
                f"unsupported_initializer_mappings:{mapping_report.unsupported_count}"
            )
        if mapping_report.requires_transform_count:
            blockers.append(
                "requires_materialized_weight_norm:"
                f"{mapping_report.requires_transform_count}"
            )
        for artifact in mapping_report.required_external_artifacts:
            blockers.append(f"missing_external_artifact:{artifact}")

    if "style_bert_vits2_config" not in external_sources:
        blockers.append("missing_external_source:style_bert_vits2_config")

    return {
        "can_write_gguf": len(blockers) == 0,
        "blockers": tuple(blockers),
    }


def _source_file_manifest(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
