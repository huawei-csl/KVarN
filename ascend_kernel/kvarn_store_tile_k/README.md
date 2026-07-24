# KVarN store-tile artifact

This directory contains the runtime package for the A5 KVarN store-tile
transform. It supports fp16 input tiles with either of these shapes:

- `[N, 128, 128]`
- `[N, 256, 128]`

The API supports `bits=2` and `bits=4`. It returns four tensors:
`q_packed` (`uint8`), `s_col_K` (`float16`), `zp_K` (`float16`), and
`s_row_K` (`float16`).

## Runtime layout

This directory is expected to contain:

```text
README.md
api.py
api_test.py
custop_torchapi.cpp
setup.py
custop_torchapi*.so       # after the PyBind build
vendors/customize/        # after the custom-op package is installed
kvarn_custom_op/          # custom-op source project
```

The runtime machine needs a compatible CANN installation, `torch_npu`, and a
PyTorch version compatible with the PyBind extension. The custom-op package is
built for the Ascend SOC selected by the CANN environment used during the
build.

## Build or rebuild

Run these commands from this directory on a machine with CANN and
`torch_npu`:

```bash
set -e

cd kvarn_custom_op
bash build.sh
cd ..
for package in kvarn_custom_op/build_out/custom_*.run; do
    test -e "$package"
    bash "$package" --install-path="$(pwd)"
done
python setup.py build_ext --inplace
```

Installing the package into the directory containing this README creates
`vendors/customize/`. The PyBind step creates `custop_torchapi*.so` beside
`api.py`.

## Use the API

```python
from api import KVarNStoreTileK

api = KVarNStoreTileK()
outputs = api.run(k_tiles_npu, bits=2, sinkhorn_iters=16)
```

`k_tiles_npu` must be a contiguous fp16 NPU tensor with shape `[N,128,128]`
or `[N,256,128]`.

## Smoke test

```bash
python api_test.py
```

The test exercises both D values and both bit widths, checks the PyBind ABI,
compares outputs with its embedded PyTorch reference, and allows up to 16
different packed bytes per case by default. Use `--max-q-errors N` to change
that bound.
