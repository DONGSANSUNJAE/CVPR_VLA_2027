"""Load this experiment's CUDA libraries before the first JAX GPU operation.

The recovered environment reuses immutable uv cache files through symlinks.
JAX's plugin-relative library lookup cannot locate adjacent NVIDIA packages
when its binary resolves into a different wheel's cache directory. Absolute
RTLD_GLOBAL loads preserve the selected package versions without modifying
system libraries or using a broad LD_LIBRARY_PATH override.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


_LIBRARIES = (
    "cuda_runtime/lib/libcudart.so.12",
    "nvjitlink/lib/libnvJitLink.so.12",
    "cuda_cupti/lib/libcupti.so.12",
    "cublas/lib/libcublasLt.so.12",
    "cublas/lib/libcublas.so.12",
    "cusparse/lib/libcusparse.so.12",
    "cusolver/lib/libcusolver.so.11",
    "cufft/lib/libcufft.so.11",
    "cudnn/lib/libcudnn.so.9",
    "nccl/lib/libnccl.so.2",
)
_HANDLES: dict[str, ctypes.CDLL] = {}


def preload_cuda_libraries(run_root: Path | str | None = None) -> dict:
    """Idempotently load verified local libraries; never initialize a JAX backend.

    Call before device discovery, distributed initialization or any JAX array
    construction. CPU-only tests and data workers deliberately skip GPU libs.
    The NVIDIA driver is loaded by the normal system mechanism.
    """
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        return {"status": "skipped_cpu", "libraries": []}
    root = Path(run_root) if run_root is not None else Path(__file__).resolve().parents[1]
    packages = root / "runtime_pro/.venv/lib/python3.11/site-packages/nvidia"
    paths = [packages / relative for relative in _LIBRARIES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Experiment CUDA library missing: {path}")
    loaded = []
    for path in paths:
        filename = str(path.absolute())
        if filename not in _HANDLES:
            _HANDLES[filename] = ctypes.CDLL(filename, mode=ctypes.RTLD_GLOBAL)
        loaded.append({"path": filename, "resolved_path": str(path.resolve())})
    return {"status": "preloaded", "libraries": loaded}
