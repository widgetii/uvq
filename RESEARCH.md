# Research: Fixing RKNN Aggregation Net for Full NPU Inference

## Problem Statement

The UVQ 1.5 aggregation network produces incorrect results when executed on
the Rockchip RK3588 NPU via the RKNN framework. The model outputs saturated
values (~1.0 or ~5.0) instead of the correct score (~3.35 for the reference
input). This forces a hybrid pipeline where content and distortion nets run
on the NPU but the aggregation net falls back to ONNX Runtime on the CPU,
preventing full NPU utilization.

The bug reproduces across all tested combinations:
- NPU drivers: 0.8.2, 0.9.6
- Toolkit versions: 2.3.0, 2.3.2
- Runtime versions: 2.3.0, 2.3.2

This is not a version mismatch — it is a systematic failure in how RKNN
compiles or executes this specific model architecture.

## Aggregation Net Architecture

The model is small (0.2 MB ONNX, 14 nodes) with the following graph:

```
content (1,128,8,8) → AveragePool(4,4) → ┐
                                          ├→ Concat(dim=1) → (1,256,4,4)
distortion (1,128,24,24) → AveragePool(4,4) → ┘
    → Conv2d(256→256, 1×1) → LayerNorm([256,4,4]) → ReLU
    → MaxPool(4,4) → Flatten → Linear(256→1) → Tanh → *2+3 → score [1,5]
```

PyTorch source: `uvq1p5_pytorch/utils/aggregationnet.py`
ONNX export wrapper: `scripts/export_onnx.py` (class `AggregationNetCoreONNX`)
RKNN export: `scripts/export_rknn.py`

Key parameters:
- `nn.LayerNorm(normalized_shape=[256, 4, 4], eps=0.001)`
- `nn.Conv2d(256, 256, kernel_size=(1,1), bias=True)`
- `nn.MaxPool2d(kernel_size=(4,4))`
- `nn.Linear(in_features=256, out_features=1, bias=True)`
- Final output: `tanh(x) * 2 + 3` (scales [-1,1] to [1,5])

## RKNN Compiler Transformations (from verbose build log)

The RKNN compiler applies these fusion passes to the aggregation net:

```
convert_layernorm_to_exnorm:  LayerNorm → ExNorm (custom RKNN op)
convert_exnorm_to_exnorm_mul_add:  ExNorm → Norm + Mul + Add
replace_mul_add_by_bn:  final scale (*2) and shift (+3) → BatchNorm
convert_gemm_by_exmatmul:  Gemm (Linear) → ExMatMul
convert_exmatmul_to_conv:  ExMatMul → Conv2d
```

The content and distortion nets (EfficientNet-B0) use BatchNorm natively
and work correctly on the NPU. The aggregation net is the only model
that uses LayerNorm, and it is the only one that fails.

## Research Questions

An AI research agent should investigate the following areas, in priority
order. Each section describes what to look for, where to look, and what
a useful answer looks like.

### 1. Known RKNN LayerNorm Issues

**Goal**: Determine if this is a known bug with existing fixes or workarounds.

Search the rknn-toolkit2 GitHub repo for issues related to LayerNorm
producing wrong results:
- https://github.com/airockchip/rknn-toolkit2/issues
- Search terms: `LayerNorm`, `layer_norm`, `exnorm`, `wrong result`,
  `incorrect output`, `aggregation`
- Also check closed issues — a fix may exist in a newer version or branch

Check the RKNN documentation for any notes about LayerNorm limitations:
- https://github.com/airockchip/rknn-toolkit2/tree/master/doc
- The "supported ops" list may have caveats for LayerNorm

Check the Rockchip NPU community forums and Chinese developer communities
(CSDN, Zhihu) for similar reports — many RK3588 users work in Chinese:
- Search: `RKNN LayerNorm 结果不对` (wrong result)
- Search: `RK3588 NPU LayerNorm`

**Useful output**: Links to specific issues, confirmed bugs, or patches.
Note the toolkit/driver versions mentioned in any fixes.

### 2. RKNN Compiler Options to Disable Problematic Fusions

**Goal**: Find `rknn.config()` or `rknn.build()` parameters that can
disable the LayerNorm-to-ExNorm fusion or other aggressive optimizations.

The RKNN compiler applies many optimization passes (visible in verbose
build logs). Research whether any of these can be controlled:

- `rknn.config()` parameters beyond `target_platform` — are there
  optimization level flags, fusion disable flags, or op-specific overrides?
- `rknn.build()` parameters beyond `do_quantization` — check for
  `optimization_level`, `disable_fuse`, or similar
- Environment variables that control the RKNN compiler behavior
- The `op_target` mechanism — can specific ops be forced to run on CPU
  while the rest stays on NPU? The log shows `RKNNSetOpTargetPass` which
  suggests per-op target assignment exists.

Look at:
- rknn-toolkit2 Python API source code (the `.pyx` files are compiled, but
  docstrings and help() output may reveal parameters)
- https://github.com/airockchip/rknn-toolkit2/tree/master/doc
- RKNN Model Zoo examples that use advanced config options
- Any `rknn.config()` examples with `optimization_level` or similar params

**Useful output**: Specific API calls or config parameters that could
bypass the buggy fusion. Example: `rknn.config(disable_rules=["convert_layernorm_to_exnorm"])`.

### 3. ONNX Graph Modifications to Avoid Triggering the Bug

**Goal**: Restructure the ONNX graph so the RKNN compiler doesn't apply
the problematic LayerNorm → ExNorm → BatchNorm fusion chain.

Possible approaches to investigate:

a) **Replace LayerNorm with equivalent ops**: Decompose
   `LayerNormalization` into its constituent operations (ReduceMean,
   Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add) before feeding to
   RKNN. If RKNN doesn't recognize the pattern, it may execute the
   individual ops correctly. Tools: `onnxruntime.transformers` or
   manual ONNX graph surgery.

b) **Replace LayerNorm with GroupNorm or InstanceNorm**: These are
   semantically similar normalization ops that RKNN may handle differently.
   `LayerNorm([256,4,4])` on a `(1,256,4,4)` tensor normalizes over
   all 256×4×4 = 4096 elements. `GroupNorm(num_groups=1, num_channels=256)`
   is mathematically equivalent. Check if RKNN handles GroupNorm correctly.

c) **Replace LayerNorm with BatchNorm**: For batch_size=1, BatchNorm in
   eval mode with carefully set running_mean/running_var could approximate
   LayerNorm. However, this changes semantics (per-channel vs per-sample
   normalization) so needs careful analysis.

d) **Insert identity ops to break fusion patterns**: Adding a no-op
   (e.g., Mul by 1.0 or Add 0.0) between Conv2d output and LayerNorm
   input might prevent the compiler from recognizing the fusable pattern.

For each approach, verify:
1. The modified ONNX produces identical (or very close) results to the
   original when run via ONNX Runtime
2. The RKNN compiler accepts it without errors
3. The NPU produces correct results

**Useful output**: A concrete ONNX graph transformation recipe that works.
Include code for the transformation and test results.

### 4. RKNN Op Target Assignment (Hybrid at Op Level)

**Goal**: Instead of running the whole model on NPU or whole model on CPU,
run most ops on NPU but force LayerNorm to CPU within the same RKNN model.

The RKNN verbose build log shows a `RKNNSetOpTargetPass` phase. Research:

- Is there an API to set per-op targets (NPU vs CPU)?
- The `rknn.config()` function may accept an `op_target` parameter or
  similar mechanism
- Some RKNN documentation mentions "fallback to CPU" for unsupported ops —
  can this be forced for specific ops even if they are technically supported?
- Check RKNN Model Zoo for examples using mixed NPU/CPU execution

This would be the ideal solution: the aggregation net stays as a single
RKNN model, but LayerNorm executes on CPU while everything else uses NPU.

**Useful output**: API calls to force specific ops to CPU, or documentation
confirming this is/isn't possible.

### 5. Alternative: Merging All Three Models into One RKNN Model

**Goal**: Investigate whether merging content_net + distortion_net +
aggregation_net into a single ONNX model and converting as one unit
changes the RKNN compiler's behavior.

Currently we have three separate models. A single end-to-end model would:
- Allow the compiler to see the full graph and potentially optimize
  differently
- Eliminate inter-model data transfer overhead
- Potentially trigger different fusion rules

Concerns:
- The distortion net processes 9 patches — the merged model would need
  to handle this (either via a loop in pre-processing, or by accepting
  pre-tiled inputs)
- A larger model may hit RKNN memory limits
- The bug may persist regardless

**Useful output**: Whether single-model conversion changes the compiler's
treatment of the LayerNorm ops, and if so, whether the NPU output is correct.

### 6. INT8 Quantization as a Workaround

**Goal**: Test whether INT8 quantization of the aggregation net bypasses
the FP16 bug.

RKNN supports INT8 quantization via `rknn.build(do_quantization=True,
dataset=calibration.txt)`. Quantized models use a different execution
path on the NPU. Research:

- Does INT8 quantization change which NPU instructions are used?
- Are there reports of FP16 bugs being fixed by switching to INT8?
- What calibration data would be needed? (Feature tensors, not raw images)
- What accuracy loss is acceptable? The aggregation net has small dynamic
  range ([1, 5] output), so INT8 might work well.

**Useful output**: Whether INT8 execution produces correct results, and
what the accuracy trade-off is.

### 7. RKNN-Toolkit2 Source-Level Analysis

**Goal**: Understand the ExNorm implementation to determine if the bug is
in the compiler or the NPU runtime.

The RKNN toolkit is partially open-source. Investigate:

- The `rknn/api/` Python files (some are `.pyx` Cython) — look for
  ExNorm implementation or configuration
- The `librknnc` compiler library — any symbols or debug options related
  to normalization
- The `librknnrt` runtime library — any debug environment variables that
  dump intermediate tensor values
- Is there a way to dump the compiled NPU instructions to verify the
  LayerNorm transformation is mathematically correct?

Check if `RKNN_LOG_LEVEL` or similar environment variables exist for
runtime debugging.

**Useful output**: Understanding of whether the bug is in compilation
(wrong instruction generation) or execution (correct instructions, wrong
NPU behavior). This determines whether a software fix is possible.

## Files to Reference

| File | Purpose |
|---|---|
| `RKNN_PORTING_NOTES.md` | Full porting documentation with test results |
| `uvq1p5_pytorch/utils/aggregationnet.py` | PyTorch model source |
| `scripts/export_onnx.py` | ONNX export (class `AggregationNetCoreONNX`) |
| `scripts/export_rknn.py` | RKNN conversion script |
| `uvq1p5_rknn/utils/uvq1p5.py` | Current hybrid pipeline implementation |
| `uvq1p5_web/public/models/aggregation_net.onnx` | ONNX model file |
| `uvq1p5_rknn/models/aggregation_net.rknn` | Compiled RKNN model |
| `uvq1p5_rknn/models/reference.json` | Reference test data (seed=42) |
| `tests/test_rknn_cross_backend.py` | On-device test |

## How to Test a Fix

Any proposed fix must pass this validation:

1. **ONNX equivalence**: If the ONNX graph was modified, verify the
   modified model produces the same output (within FP32 tolerance) as the
   original when run via ONNX Runtime on x86_64.

2. **RKNN conversion**: The modified model must convert with
   `scripts/export_rknn.py` without errors.

3. **NPU accuracy**: On real hardware (Orange Pi 5 Plus), the RKNN model
   must produce a score within abs=0.05 of the PyTorch reference (3.3493
   for seed=42).

4. **Cross-backend test**: `tests/test_rknn_cross_backend.py` must pass.

Quick NPU test script (on the board):

```python
from rknnlite.api import RKNNLite
import numpy as np, json

with open("uvq1p5_rknn/models/reference.json") as f:
    ref = json.load(f)

np.random.seed(ref["numpy_seed"])
content_input = np.random.randn(*ref["content_input_shape"]).astype(np.float32)
patches_input = np.random.randn(*ref["patches_input_shape"]).astype(np.float32)

# ... run through pipeline ...
# Score should be within 0.05 of ref["score"] (3.3493)
```

## Success Criteria

The research is successful if it finds **any** of:

1. A way to make `aggregation_net.rknn` produce correct results on the NPU
   (eliminating the ONNX Runtime dependency entirely)
2. An RKNN compiler flag to disable the problematic fusion
3. An ONNX graph transformation that avoids triggering the bug
4. Confirmation that this is a known bug with a fix in a newer
   toolkit/driver version (with specific version numbers)

Even partial results are valuable — e.g., confirming that the bug is in
the compiler (not the NPU hardware) narrows the solution space.
