"""External-facing runtime API for a built KVarN Torch extension.

Copy this module with a KVarN artifact into an application repository, then
use ``KVarNStoreTileK.run`` with contiguous fp16 NPU tiles.
"""

import ctypes
import glob
import importlib
import os
import sys
from typing import Dict, Optional

import torch


def _load_extension(artifact_dir: str):
    """Load the generated PyBind module and register its CANN custom op."""
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty path")

    artifact_dir = os.path.abspath(artifact_dir)
    opp = os.path.join(artifact_dir, "vendors", "customize")
    if not os.path.isdir(opp):
        raise FileNotFoundError(
            "custom-op package not found: {}\n"
            "Build KVarN first so the artifact contains vendors/customize/.".format(opp)
        )
    if not glob.glob(os.path.join(artifact_dir, "custop_torchapi*.so")):
        raise FileNotFoundError(
            "custop_torchapi extension not found under {}\n"
            "Build the PyBind extension after building the CANN custom op.".format(artifact_dir)
        )

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise ImportError("KVarN runtime requires torch_npu on the target machine") from exc

    previous_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if previous_opp:
        opp_paths = previous_opp.split(os.pathsep)
        if opp not in opp_paths:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = opp + os.pathsep + previous_opp
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = opp

    opapi = os.path.join(opp, "op_api", "lib", "libcust_opapi.so")
    if not os.path.isfile(opapi):
        raise FileNotFoundError("custom-op API library not found: {}".format(opapi))
    ctypes.CDLL(opapi, mode=ctypes.RTLD_GLOBAL)

    loaded = sys.modules.get("custop_torchapi")
    if loaded is not None:
        loaded_dir = os.path.dirname(os.path.abspath(getattr(loaded, "__file__", "")))
        if loaded_dir != artifact_dir:
            raise RuntimeError(
                "custop_torchapi is already loaded from {}; cannot also load {} in one process".format(
                    loaded_dir, artifact_dir
                )
            )
        return loaded

    if artifact_dir not in sys.path:
        sys.path.insert(0, artifact_dir)
    return importlib.import_module("custop_torchapi")


def _packed_group(bits: int) -> int:
    if isinstance(bits, bool) or bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    return 32 if bits == 2 else 64


def _detect_d(k_tile_rotated: torch.Tensor) -> int:
    if not isinstance(k_tile_rotated, torch.Tensor):
        raise TypeError("k_tile_rotated must be a torch.Tensor")
    if k_tile_rotated.ndim != 3 or k_tile_rotated.shape[2] != 128:
        raise ValueError("k_tile_rotated must have shape [N, 128, 128] or [N, 256, 128]")
    d = int(k_tile_rotated.shape[1])
    if d not in (128, 256):
        raise ValueError("k_tile_rotated.shape[1] must be 128 or 256, got {}".format(d))
    return d


class KVarNStoreTileK:
    """Run the static D128/D256 KVarN custom ops from one artifact directory."""

    def __init__(self, artifact_dir: Optional[str] = None) -> None:
        if artifact_dir is None:
            artifact_dir = os.path.dirname(os.path.abspath(__file__))
        self._mod = _load_extension(artifact_dir)

    def run(self, k_tile_rotated: torch.Tensor, bits: int = 2,
            sinkhorn_iters: int = 16) -> Dict[str, torch.Tensor]:
        """Dispatch by D and return packed codes plus KVarN metadata tensors."""
        d = _detect_d(k_tile_rotated)
        packed_group = _packed_group(bits)
        if isinstance(sinkhorn_iters, bool) or not isinstance(sinkhorn_iters, int):
            raise TypeError("sinkhorn_iters must be an int")
        if sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be non-negative")
        if k_tile_rotated.shape[0] <= 0:
            raise ValueError("k_tile_rotated must contain at least one tile")
        if k_tile_rotated.dtype != torch.float16:
            raise TypeError("k_tile_rotated must have dtype torch.float16")
        if not k_tile_rotated.is_contiguous():
            raise ValueError("k_tile_rotated must be contiguous")
        if k_tile_rotated.device.type != "npu":
            raise ValueError("KVarN runtime requires an NPU tensor")

        tile_count = int(k_tile_rotated.shape[0])
        outputs = {
            "q_packed": torch.empty((tile_count, d, packed_group), dtype=torch.uint8,
                                    device=k_tile_rotated.device),
            "s_col_K": torch.empty((tile_count, d), dtype=torch.float16, device=k_tile_rotated.device),
            "zp_K": torch.empty((tile_count, d), dtype=torch.float16, device=k_tile_rotated.device),
            "s_row_K": torch.empty((tile_count, 128), dtype=torch.float16, device=k_tile_rotated.device),
        }
        op = self._mod.kvarn_store_tile_k_kernel if d == 128 else self._mod.kvarn_store_tile_kd256_kernel
        # The generated PyBind wrapper keeps the @kernel signature order:
        # all tensors first, followed by scalar Vars. Internally it translates
        # this to ACLNN's input/scalar/output execution order.
        op(k_tile_rotated, outputs["q_packed"], outputs["s_col_K"], outputs["zp_K"],
           outputs["s_row_K"], bits, sinkhorn_iters, tile_count, packed_group)
        return outputs
