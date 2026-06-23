[![Built on vLLM](https://img.shields.io/badge/Built%20on-vLLM%20v0.23.0-30a14e)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03458-b31b1b.svg)](https://arxiv.org/abs/2606.03458)
[![hf-space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Huawei%20CSL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/huawei-csl)
[![GitHub stars](https://img.shields.io/github/stars/huawei-csl/KVarN?label=Stars&logo=github&logoColor=white&style=flat-square)](https://github.com/huawei-csl/KVarN/stargazers)



<p align="center">
  <img src="imgs/logo_600.png" alt="KVarN" width="640">
</p>

> ⚡️ **Built for agentic and long-context workloads.**

> 💡 KVarN delivers **3-5x more KV-cache capacity** and **up to ~1.3x the throughput** of FP16, so you fit far longer contexts and serve more concurrent requests, with **FP16-level accuracy**.

> 🔌 **Calibration-free, plug-and-play with vLLM.** A native vLLM attention backend: add one flag, no model changes, no calibration.

> 🥊 **Up to ~2.4× TurboQuant throughput**, same capacity, **higher accuracy**.

---

## Why KVarN (Variance Normalized KV-Cache)?

> **kvarn** /kvɑːɳ/ &nbsp;·&nbsp; *noun* (Swedish)
>
> 1. A grinding apparatus used to reduce substances into smaller particles or
>    powder, especially grains, seeds, spices, coffee beans, KV-caches.

KV-cache quantization usually comes with a catch. As the
[vLLM TurboQuant blog](https://vllm.ai/blog/2026-05-11-turboquant) shows, existing
methods buy extra KV-cache capacity but **give up throughput** (TurboQuant reports
**40 to 52% lower throughput** for 2.3-3.7x capacity), and aggressive low-bit
quantization also tends to **cost accuracy**. Losing both speed *and* quality is
the main reason KV-cache quantization is rarely turned on in production.

**KVarN is built to keep both.** On Qwen3-32B (AIME25, 16K-context burst, TP=2) it
matches FP16 accuracy and **beats its throughput** while delivering ~4× the KV-cache capacity:

<p align="center">
  <img src="imgs/pareto_qwen3-32b.png" alt="KVarN vs FP16 vs TurboQuant: accuracy, throughput and capacity" width="660">
</p>

KVarN stays in the upper-right corner the blog's methods can't reach: **FP16-level
accuracy, FP16-or-better throughput, and several times the context.**

---

## Quickstart

KVarN ships as a vLLM fork. Install it like vLLM, then select the KVarN KV-cache dtype.

```bash
# 1. Clone
git clone https://github.com/huawei-csl/KVarN.git
cd KVarN

# 2. Install (uses the upstream precompiled wheel; KVarN kernels are Triton, JIT-compiled at runtime)
VLLM_USE_PRECOMPILED=1 pip install -e .
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    dtype="float16",                    # KVarN runs in float16
    kv_cache_dtype="kvarn_k4v2_g128",   # enable KVarN
    block_size=128,                     # KVarN tile size
)
print(llm.generate("Explain KV-cache quantization in one sentence.",
                    SamplingParams(max_tokens=64))[0].outputs[0].text)
```

Serving works the same way:

```bash
vllm serve Qwen/Qwen3-32B --dtype float16 --kv-cache-dtype kvarn_k4v2_g128 --block-size 128
```

> **Note:** KVarN runs in `float16` compute. One vLLM block is one KVarN tile, so
> the tile / page size equals `--block-size`. Both **128** (default) and **64** are
> supported, selected by the matching preset (`kvarn_k4v2_g128` / `kvarn_k4v2_g64`).
> 128 is the design point; 64 gives finer quantization granularity at the cost of a
> little KV capacity (more per-tile scale overhead per token), at essentially the
> same throughput.

> **Tip (capacity):** KVarN realizes its full KV-cache capacity when there is room
> to amortize a small fixed decode workspace. On multi-GPU or generous
> `--gpu-memory-utilization` setups this is automatic. On a tight single-GPU budget,
> vLLM's CUDA-graph memory profiler can over-reserve and shrink the KV pool; set
> `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (and/or raise
> `--gpu-memory-utilization`) to recover the full capacity.

### MLA models

KVarN also supports **Multi-head Latent Attention (MLA)**. To the best of our knowledge, **KVarN is the first vLLM-compatible sub-8-bit KV-cache quantization method to support MLA-based models**: it quantizes the compressed KV latent to int4. Use the **same** `--kv-cache-dtype` you use on
dense models; on an MLA model it automatically routes to the MLA latent path (no
code or env changes), and the fast path is on by default:

```bash
vllm serve zai-org/GLM-4.7-Flash \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --tensor-parallel-size 2
```

**GLM-4.7-Flash, KVarN vs bf16** (TP=2):

| Metric | bf16 | KVarN | KVarN / bf16 |
| --- | --- | --- | --- |
| Burst throughput @32K (tok/s) | 401 | 377 | **0.94×** |
| KV-cache capacity (tokens) | 313K | 865K | **2.77×** |
| AIME25 accuracy | 53.3% | **53.3%** | parity |

The win is **2.77× KV capacity at ~parity accuracy**. MLA's
latent is already tiny so KVarN is not a latency play there, but it lets you fit
far more concurrent context in the same memory. 

### Hybrid models (Mamba / linear-attention)

KVarN supports **hybrid models** that interleave linear-attention or Mamba layers
with standard full-attention layers, such as Qwen3.6-27B. KVarN compresses only the
full-attention layers (the layers that hold a KV cache); the linear-attention and
Mamba layers keep their own recurrent state and are left untouched. The fp16 decode
pool is sized from the full-attention layer count only, so a hybrid model loads with
the default flags, no manual pool tuning.

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128
```

The KV-cache capacity gain applies to the full-attention layers (where the cache
lives), so a hybrid model fits far more concurrent context in the same memory at near
parity accuracy.

### Speculative decoding (MTP, DFlash, and draft models)

KVarN is compatible with **speculative decoding**, including Multi-Token Prediction
(MTP). Pass `--speculative-config` exactly as you normally would, alongside the KVarN
`--kv-cache-dtype`:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The speculative verify step attends over the full cached context (KVarN reconstructs
it from the quantized cache), and a block is committed to the quantized cache only
once all of its tokens are accepted, so rejected draft tokens never corrupt history.

> **Note (weight quantization):** KVarN quantizes the KV cache independently of the
> model weights, so it composes with weight-quantized checkpoints (for example
> `compressed-tensors` / AWQ INT4) and MTP at the same time. Validated on Qwen3.6-27B
> in both `bfloat16` and AWQ INT4.

**DFlash.** KVarN also supports [DFlash](https://github.com/z-lab/dflash), a
parallel-drafting method whose drafter attends to the cached context with *non-causal*
(bidirectional) cross-attention. KVarN's backend advertises non-causal support, so the
drafter reads the quantized cache exactly as the target model does, with no extra flags
beyond the usual `--speculative-config`:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

Each drafting step issues a block of query tokens (the bonus token plus the speculative
tokens) that attend over the *entire* cached context rather than a causal prefix; KVarN
serves that block against the quantized cache and reconstructs the context the same way
it does for ordinary decode. As with MTP, a block is committed to the quantized cache
only once its tokens are accepted, so rejected drafts never corrupt history.

---

### Running on WSL2

KVarN works on WSL2, with two caveats. Both come from system layers (the Windows graphics virtualization model and a vLLM policy), not from any particular hardware.

**1. Keep VRAM allocation under the resident limit.** Under WSL2 the GPU runs through the Windows display driver model (WDDM), which paravirtualizes graphics memory. Allocating past the resident limit does not fail: the system silently spills pages to host RAM, and identical CUDA kernels then run at PCIe speed instead of VRAM speed. KVarN adds an fp16 tail pool on top of weights and KV cache, so a `--gpu-memory-utilization` that fits a plain model can cross the resident limit once KVarN is active. Lower it until the whole working set stays resident.

How far to lower it depends on total VRAM and the model footprint, so locate the edge by measurement. On a 24 GB GPU running Qwen2.5-7B-Instruct, `0.85` still spilled while `0.80` kept everything resident, recovering throughput from 692 to 3523 tok/s at 128 concurrent decodes with quality unchanged.

Diagnostic, the behavioral signature of a spill: `nvidia-smi` shows GPU utilization pinned near 100% while power draw sits near idle. The two signals diverge because the kernels are stalled on the PCIe path rather than doing resident work. Power draw is the reliable signal here; utilization is not. Read the divergence between the two, not a fixed wattage, since the idle baseline varies by device. To watch both signals live:

```
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader'
```

**2. UVA must be enabled.** vLLM disables UVA on WSL2 by policy, which prevents the engine from starting. UVA itself works fine under WSL2, so the fix is to make `is_uva_available()` in `vllm/utils/platform_utils.py` return `True` on WSL2. Re-apply after pulling.

Background and measurements: [issue #15](https://github.com/huawei-csl/KVarN/issues/15).

*Validated on WSL 2.7.3 (kernel 6.6.114), Windows 11 build 26100. The resident limit and silent spill are properties of the WDDM paravirtualization stack, not of any GPU model, and may shift across driver and Windows versions.*

## How does KVarN work?

<p align="center">
  <img src="imgs/kvarn_pipeline.gif" alt="KVarN pipeline: Cache, Rotated Cache, Normalized Cache, Quantized Cache" width="760">
</p>

KVarN quantizes the KV cache one fixed-size token tile at a time, walking each tile
through the four stages above:

1. **Cache**: the raw fp16 KV tile (channels × tokens), straight from attention.

2. **Rotated Cache**: a **Hadamard rotation** along the channel dimension mixes
   channels so that per-channel outliers are spread out, making the tile easier to
   quantize. The rotation is orthonormal, so attention scores are preserved.

3. **Normalized Cache**: **iterative variance normalization** (Sinkhorn-like)
   alternates column- and row-wise standard-deviation normalization in log space,
   equalizing variance across the tile and shrinking quantization error before any
   rounding happens.

4. **Quantized Cache**: **asymmetric round-to-nearest** at low bit-width, with the
   scales folded back in at read time (keys per channel, values per token).

The shipped preset spends **more bits on keys than values** (`kvarn_k4v2_g128`:
4-bit keys, 2-bit values). We chose to release this configuration because it meets
the strictest accuracy bar, matching FP16, that the most demanding production
deployments and vLLM require, while still delivering throughput above FP16.

---

## Citation

KVarN is the official vLLM implementation of our paper:

> 📄 *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation
> in Reasoning Tasks* ([arXiv:2606.03458](https://arxiv.org/abs/2606.03458))

If you use KVarN, please cite:

```bibtex
@misc{muller2026kvarn,
      title={KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks}, 
      author={Lorenz K. Muller and Philippe Bich and Chiara Boretti and Hyun-Min Chang and Jiawei Zhuang and Lukas Cavigelli},
      year={2026},
      eprint={2606.03458},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={http://arxiv.org/abs/2606.03458}
}
```

---

## License and attribution

KVarN is built on [vLLM](https://github.com/vllm-project/vllm) (v0.23.0) and is
released under the Apache 2.0 License. The original vLLM README is preserved as
[`README_vLLM.md`](README_vLLM.md).
