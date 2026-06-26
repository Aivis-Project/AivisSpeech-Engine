"""ONNX initializer tensor pack extraction for the Aivis GGML Plugin EP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InitializerTensorRecord:
    """A single ONNX initializer's location in the tensor pack."""

    name: str
    elem_type: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class InitializerTensorPack:
    """Metadata for a binary pack of ONNX initializer tensors."""

    tensor_count: int
    total_bytes: int
    sha256: str
    records: tuple[InitializerTensorRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "tensor_count": self.tensor_count,
            "total_bytes": self.total_bytes,
            "sha256": self.sha256,
            "records": tuple(record.to_dict() for record in self.records),
        }


def write_initializer_tensor_pack(
    *,
    model_path: str | Path,
    output_path: str | Path,
) -> InitializerTensorPack:
    """Write ONNX initializers to a deterministic binary tensor pack."""

    import numpy as np
    import onnx
    from onnx import numpy_helper

    source_path = Path(model_path)
    tensor_pack_path = Path(output_path)
    tensor_pack_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source_path), load_external_data=True)
    records: list[InitializerTensorRecord] = []
    offset = 0
    pack_digest = sha256()

    with tensor_pack_path.open("wb") as output_file:
        for initializer in model.graph.initializer:
            array = numpy_helper.to_array(initializer)
            if array.dtype == np.dtype("O"):
                raise ValueError(
                    f"Initializer {initializer.name!r} has unsupported object dtype."
                )

            contiguous = np.ascontiguousarray(array)
            tensor_bytes = contiguous.tobytes(order="C")
            tensor_digest = sha256(tensor_bytes).hexdigest()
            output_file.write(tensor_bytes)
            pack_digest.update(tensor_bytes)

            records.append(
                InitializerTensorRecord(
                    name=initializer.name,
                    elem_type=onnx.TensorProto.DataType.Name(initializer.data_type),
                    dtype=contiguous.dtype.str,
                    shape=tuple(int(dim) for dim in contiguous.shape),
                    offset_bytes=offset,
                    size_bytes=len(tensor_bytes),
                    sha256=tensor_digest,
                )
            )
            offset += len(tensor_bytes)

    return InitializerTensorPack(
        tensor_count=len(records),
        total_bytes=offset,
        sha256=pack_digest.hexdigest(),
        records=tuple(records),
    )
