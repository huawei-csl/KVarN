"""Standalone NPU smoke test for a built KVarN artifact.

The script imports ``api.py`` only from ``--artifact`` and has no dependency on
kernel source or the surrounding repository.  A small copy of the PyTorch
reference is kept here so the test checks numerical results as well as the
D128/D256 and q2/q4 public ABI, output metadata, and repeated-call
determinism. Run it on a target with CANN and ``torch_npu`` installed:

    python api_test.py --artifact /path/to/kvarn_artifact
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu  # noqa: F401


def _load_runtime(artifact_dir: str):
    api_path = os.path.join(artifact_dir, "api.py")
    if not os.path.isfile(api_path):
        raise FileNotFoundError("artifact API not found: {}".format(api_path))
    spec = importlib.util.spec_from_file_location("kvarn_artifact_api", api_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load artifact API: {}".format(api_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.KVarNStoreTileK(artifact_dir)


def _check_outputs(outputs, expected_shapes, device) -> None:
    expected_dtypes = {
        "q_packed": torch.uint8,
        "s_col_K": torch.float16,
        "zp_K": torch.float16,
        "s_row_K": torch.float16,
    }
    if set(outputs) != set(expected_shapes):
        raise AssertionError("unexpected output names: {}".format(sorted(outputs)))
    for name, tensor in outputs.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise AssertionError("{} shape {} != {}".format(name, tuple(tensor.shape), expected_shapes[name]))
        if tensor.dtype != expected_dtypes[name]:
            raise AssertionError("{} dtype {} != {}".format(name, tensor.dtype, expected_dtypes[name]))
        if tensor.device != device:
            raise AssertionError("{} device {} != {}".format(name, tensor.device, device))
        if name != "q_packed" and not torch.isfinite(tensor).all().item():
            raise AssertionError("{} contains non-finite values".format(name))


def _reference(k_tile_rotated: torch.Tensor, bits: int, sinkhorn_iters: int):
    """Independent CPU golden for the public KVarN transform.

    Keep this implementation in the standalone test rather than importing a
    kernel module: an external repository should be able to copy and run the
    test using only its built artifact.
    """
    tile = k_tile_rotated.float()
    tile_count = tile.shape[0]
    tile_d, tile_g = tile.shape[1:]
    log_s_col = torch.zeros((tile_count, 1, tile_g), dtype=torch.float32, device=tile.device)
    log_s_row = torch.zeros((tile_count, tile_d, 1), dtype=torch.float32, device=tile.device)

    def imbalance(cur):
        col_std = cur.std(dim=1)
        row_std = cur.std(dim=2)
        return (col_std.amax(dim=1) / col_std.amin(dim=1).clamp_min(1.0e-8)
                + row_std.amax(dim=1) / row_std.amin(dim=1).clamp_min(1.0e-8))

    cur = tile / log_s_col.exp() / log_s_row.exp()
    imb_best = imbalance(cur)
    sc_best = log_s_col.exp().clone()
    sr_best = log_s_row.exp().clone()

    for _ in range(sinkhorn_iters):
        col_std = cur.std(dim=1, keepdim=True).clamp(1.0e-3, 1.0e3)
        log_s_col = (log_s_col + col_std.log()).clamp(-0.3, 10.0)
        cur = tile / log_s_col.exp() / log_s_row.exp()
        row_std = cur.std(dim=2, keepdim=True).clamp(1.0e-3, 1.0e3)
        log_s_row = (log_s_row + row_std.log()).clamp(-0.3, 10.0)
        cur = tile / log_s_col.exp() / log_s_row.exp()
        imb = imbalance(cur)
        better = imb <= imb_best
        mask = better.view(tile_count, 1, 1).to(log_s_col.dtype)
        sc_best = mask * log_s_col.exp() + (1 - mask) * sc_best
        sr_best = mask * log_s_row.exp() + (1 - mask) * sr_best
        imb_best = torch.where(better, imb, imb_best)

    balanced = tile / sc_best / sr_best
    lo = balanced.amin(dim=2, keepdim=True)
    hi = balanced.amax(dim=2, keepdim=True)
    scale = ((hi - lo) / ((1 << bits) - 1)).clamp_min(1.0e-10)
    q = ((balanced - lo) / scale).round().clamp(0, (1 << bits) - 1).to(torch.uint8)
    pack = 8 // bits
    q_packed = torch.zeros((tile_count, tile_d, tile_g // pack), dtype=torch.uint8, device=tile.device)
    for pack_idx in range(pack):
        q_packed |= q[:, :, pack_idx::pack] << (pack_idx * bits)
    return (q_packed, (sr_best * scale).squeeze(2),
            (sr_best * lo).squeeze(2), sc_best.squeeze(1))


def _unpack_q_packed(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Expand packed carriers into one logical quantized value per column."""
    pack = 8 // bits
    mask = (1 << bits) - 1
    packed = packed.to(torch.int16)
    codes_by_slot = torch.stack([
        (packed >> (slot * bits)) & mask for slot in range(pack)
    ], dim=-1)
    # Each carrier contains consecutive logical columns in its low-to-high
    # bit slots. Flatten [carrier, slot] to recover columns 0..G-1.
    return codes_by_slot.reshape(*packed.shape[:-1], -1)


def _run_case(runtime, d: int, bits: int, tiles: int, sinkhorn_iters: int,
              max_q_errors: int) -> None:
    torch.manual_seed(10000 + d + bits)
    cpu_tiles = torch.randn((tiles, d, 128), dtype=torch.float16)
    k_tiles = cpu_tiles.to("npu")
    try:
        first = runtime.run(k_tiles, bits=bits, sinkhorn_iters=sinkhorn_iters)
        second = runtime.run(k_tiles, bits=bits, sinkhorn_iters=sinkhorn_iters)
    except TypeError as exc:
        raise AssertionError("PyBind ABI call failed for D={} bits={}: {}".format(d, bits, exc)) from exc
    expected_shapes = {
        "q_packed": (tiles, d, 32 if bits == 2 else 64),
        "s_col_K": (tiles, d),
        "zp_K": (tiles, d),
        "s_row_K": (tiles, 128),
    }
    _check_outputs(first, expected_shapes, k_tiles.device)
    _check_outputs(second, expected_shapes, k_tiles.device)
    for name in first:
        if not torch.equal(first[name], second[name]):
            raise AssertionError("{} is not deterministic for D={} bits={}".format(name, d, bits))

    expected = _reference(cpu_tiles, bits, sinkhorn_iters)
    q_code_mismatches = None
    for name, actual, golden in zip(("q_packed", "s_col_K", "zp_K", "s_row_K"),
                                    first.values(), expected):
        actual_cpu = actual.cpu()
        if name == "q_packed":
            golden_cpu = golden.cpu()
            actual_codes = _unpack_q_packed(actual_cpu, bits)
            golden_codes = _unpack_q_packed(golden_cpu, bits)
            q_code_mismatches = int((actual_codes != golden_codes).sum().item())
            if q_code_mismatches > max_q_errors:
                raise AssertionError(
                    "{} code mismatches for D={} bits={}: {} values (allowed {})".format(
                        name, d, bits, q_code_mismatches, max_q_errors))
        else:
            torch.testing.assert_close(actual_cpu.float(), golden.float(), rtol=0.02, atol=0.1,
                                       msg="{} mismatch for D={} bits={}".format(name, d, bits))
    q_code_total = tiles * d * 128
    print("PASS D={} bits={} q_shape={} code_mismatches={}/{} (allowed <= {})".format(
        d, bits, expected_shapes["q_packed"], q_code_mismatches, q_code_total,
        max_q_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone KVarN artifact API smoke test")
    parser.add_argument("--artifact", default=os.path.dirname(os.path.abspath(__file__)),
                        help="directory containing api.py, custop_torchapi*.so, and vendors/customize")
    parser.add_argument("--tiles", type=int, default=128, help="number of tiles in each test case")
    parser.add_argument("--sinkhorn-iters", type=int, default=16)
    parser.add_argument("--max-q-errors", type=int, default=16,
                        help="maximum differing logical quantized values per case (default: 16)")
    args = parser.parse_args()
    if args.tiles <= 0:
        raise ValueError("--tiles must be positive")
    if args.sinkhorn_iters < 0:
        raise ValueError("--sinkhorn-iters must be non-negative")
    if args.max_q_errors < 0:
        raise ValueError("--max-q-errors must be non-negative")

    runtime = _load_runtime(os.path.abspath(args.artifact))
    for d in (128, 256):
        for bits in (2, 4):
            _run_case(runtime, d, bits, args.tiles, args.sinkhorn_iters, args.max_q_errors)
    print("KVarN external API smoke completed")


if __name__ == "__main__":
    main()
