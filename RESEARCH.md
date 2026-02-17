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

---

## Research Findings

Research was conducted in February 2026 using web search, GitHub issue
analysis, and RKNN documentation review. Four parallel investigations
covered: known bugs, compiler options, ONNX graph workarounds, and
quantization/debugging tools.

### 1. Known RKNN LayerNorm Bugs — Confirmed Systemic Issue

**LayerNorm on the RK3588 NPU via RKNN is a well-documented, unresolved
problem.** Multiple GitHub issues confirm the `exNorm` operator (RKNN's
fused replacement for LayerNorm) produces incorrect results on real NPU
hardware.

| Issue | Repo | Model | Status | Key Finding |
|---|---|---|---|---|
| [#149](https://github.com/airockchip/rknn-toolkit2/issues/149) | airockchip | Stable Diffusion 1.5 | Closed (no fix) | exNorm cosine similarity dropped to **0.79**, Euclidean error **1036** |
| [#186](https://github.com/airockchip/rknn-toolkit2/issues/186) | airockchip | Minimal LayerNorm(256) | **OPEN** | Maintainer acknowledged "suboptimal LayerNorm support" |
| [#460](https://github.com/airockchip/rknn-toolkit2/issues/460) | airockchip | ViTTracker | **OPEN** | Toolkit 2.3.2, driver 0.9.6, RK3588 — same exNorm fusion bug |
| [#322](https://github.com/airockchip/rknn-toolkit2/issues/322) | airockchip | Depth Anything v2 | Closed (no fix) | exNorm errors on **hardware** but not **simulator** — proves NPU runtime bug |
| [#220](https://github.com/airockchip/rknn-toolkit2/issues/220) | airockchip | ConvNeXt | **OPEN** | Uses LayerNorm extensively, wrong outputs |
| [#162](https://github.com/rockchip-linux/rknn-toolkit2/issues/162) | rockchip-linux | ONNX LayerNorm | **OPEN** | **"dimension -2 must be aligned to 16"** — NPU falls back to CPU |

**Critical insight from issue #322**: The simulator produces correct results,
but real NPU hardware does not. This proves the bug is in the **NPU runtime
execution** of the `exNorm` operator, not in the compiler's graph
transformation logic. The compiler generates correct instructions, but the
NPU hardware executes them incorrectly.

**Critical insight from issue #162 (16-alignment constraint)**: The RKNN
compiler log shows `"dimension -2 of first input must be aligned to 16"` and
`"LayerNorm: Shape not support Target:NPU, turn to Target:CPU implement"`.
Our aggregation net's LayerNorm operates on tensor `(1, 256, 4, 4)` — the
spatial dimensions (4, 4) are **NOT 16-aligned**. In older toolkit versions
(issue #162, ~2023), this caused a CPU fallback. In newer versions (2.3.x),
the compiler appears to force NPU execution without the alignment check,
producing silently wrong results instead of falling back to CPU. This
16-alignment hardware constraint may be the root cause of the exNorm bug
for small spatial dimensions.

Despite RKNN changelogs mentioning "improved LayerNorm support" in versions
1.5.0, 1.6.0, 2.0.0, 2.3.0, and 2.3.2, none of these changelog entries
say "fixed incorrect LayerNorm results." The latest toolkit version remains
**2.3.2** (April 2025) with no version 2.4 released or announced.

This is also a broader industry problem — NVIDIA TensorRT has documented
FP16 LayerNorm overflow issues
([TensorRT#2564](https://github.com/NVIDIA/TensorRT/issues/2564)).

### 2. RKNN Compiler Options — Two Key Parameters Found

#### `disable_rules` — Targeted Fusion Control

The most promising parameter. Accepts a list of pass names and skips them
during compilation:

```python
rknn.config(
    target_platform="rk3588",
    disable_rules=[
        'convert_layernorm_to_exnorm',
        'convert_exnorm_to_exnorm_mul_add',
        'replace_mul_add_by_bn',
    ],
)
```

**Evidence this works**: The toolkit itself suggests this parameter in error
messages. When a fusion rule causes a `KeyError`, the toolkit prints:
> "You can add `disable_rules=['rule_name']` in `rknn.config()` to
> temporarily disable the error rule!"

Observed in [ultralytics#20655](https://github.com/ultralytics/ultralytics/issues/20655)
and [rknn_model_zoo#221](https://github.com/airockchip/rknn_model_zoo/issues/221).

Rule names come from verbose build logs. There is no public master list.

#### `optimization_level` — Global Optimization Control

```python
rknn.config(target_platform="rk3588", optimization_level=0)  # all off
```

- Level 0: All optimizations disabled
- Level 3: All optimizations enabled (default)
- Levels 1–2: Intermediate (only documented in official PDF)

Disables ALL fusions, not just LayerNorm. Performance degrades.

#### Per-Op CPU Fallback — Does NOT Exist

Despite `RKNNSetOpTargetPass` appearing in build logs, there is **no user-facing
API** to assign individual ops to CPU vs NPU. RKNN is a static compiler — if an
op is not supported, the entire conversion fails. There is no `op_target`
parameter or `cpu_fallback=True` flag.

### 3. ONNX Graph Workarounds — Three Methods to Decompose LayerNorm

Since the bug is triggered by the `LayerNormalization` ONNX node being
converted to `exNorm`, removing that node eliminates the problem entirely.

#### Method A: Export aggregation net at ONNX opset 16

At opset < 17, PyTorch auto-decomposes `nn.LayerNorm` into primitives
(`ReduceMean`, `Sub`, `Pow`, `Sqrt`, `Div`, `Mul`, `Add`). Currently
`scripts/export_onnx.py` uses `opset_version=17` (line 143).

Changing to `opset_version=16` for the aggregation net eliminates the
`LayerNormalization` node. The primitive ops are all supported by RKNN
across opset 12–19.

**Caveat**: RKNN recommends opset 19 and warns about lower opsets. Also,
this changes the opset for all three models unless export is split.

#### Method B: `onnx.inliner` post-processing (cleanest)

Decompose `LayerNormalization` using ONNX's own function definition while
keeping opset 17:

```python
import onnx
import onnx.inliner

model = onnx.load("aggregation_net.onnx")
decomposed = onnx.inliner.inline_selected_functions(
    model,
    function_ids=[("", "LayerNormalization")],
    exclude=False,
    inline_schema_functions=True,
)
onnx.save(decomposed, "aggregation_net_decomposed.onnx")
```

Requires `onnx >= 1.15`. RKNN toolkit 2.3.x pins `onnx==1.16.2`, so this
should work in that environment.

#### Method C: Manual graph surgery with onnx-graphsurgeon

Replace the `LayerNormalization` node with explicit ReduceMean → Sub →
Mul → ReduceMean → Add → Sqrt → Reciprocal → Mul → Mul → Add subgraph.
Most control but most code.

#### GroupNorm(1) — NOT viable

`GroupNorm(num_groups=1)` is mathematically equivalent to LayerNorm, but
RKNN lists `GroupNormalization` as **"Not supported"** in the OP support
docs. It would fail to convert or fall back to CPU.

#### RKNN pattern re-detection risk

Even with decomposed primitives, the RKNN compiler has a
`replace_torch_layernorm` pass that detects the ReduceMean+Sub+Pow+...
pattern and re-fuses it into LayerNorm → exNorm. This pass would need to
be disabled via `disable_rules=['replace_torch_layernorm']` to prevent
re-detection.

### 4. Quantization and Debugging Tools

#### INT8 uses different NPU hardware

INT8 and FP16 use different MAC units on the RK3588 NPU:
- **INT8**: 1024 ops/cycle (3 TOPS total)
- **FP16**: 512 ops/cycle (1.5 TOPS total)

INT8 quantization triggers a completely different compilation path and may
bypass the FP16 exNorm bug. However, with only 4096 elements in the
aggregation net's tensors, quantization to 256 discrete levels risks
significant precision loss.

#### Hybrid quantization — per-layer precision control

RKNN supports a 3-step hybrid quantization workflow that allows keeping
specific layers in FP32 while quantizing others to INT8:

```python
# Step 1: Generate config
rknn.hybrid_quantization_step1(dataset='calibration.txt')
# Step 2: Edit .quantization.cfg — set LayerNorm layers to float32
# Step 3: Build with modified config
rknn.hybrid_quantization_step2(
    model_input='model.model',
    data_input='model.data',
    model_quantization_cfg='model.quantization.cfg')
```

Also available in v2.3.2: `auto_precision=True` and
`mixed_precision_dot_node_list=['layer_name', ...]` in `rknn.config()`.

#### Debugging tools

| Tool | Usage | Notes |
|---|---|---|
| `accuracy_analysis()` | PC-side, compares per-layer reference vs RKNN | Best for pinpointing divergent layers |
| `NN_LAYER_DUMP=1` | On-device env var, dumps per-layer tensors | Very slow (>80s), undocumented output location |
| `RKNN_LOG_LEVEL=5` | On-device env var, verbose runtime logging | Shows per-layer MACs and bandwidth |
| `eval_perf()` | PC-side, per-layer timing | Requires `perf_debug=True` in `init_runtime()` |

`accuracy_analysis()` cannot be called on pre-built `.rknn` models — must
start from the original ONNX and go through the full conversion pipeline.

### 5. Cross-Reference with Independent Research (ChatGPT Deep Research)

An independent investigation using ChatGPT Deep Research was conducted in
parallel to validate findings. The cross-reference identified one significant
new finding and confirmed all major conclusions.

**New finding — 16-alignment constraint (issue #162)**: The ChatGPT research
surfaced [rockchip-linux/rknn-toolkit2#162](https://github.com/rockchip-linux/rknn-toolkit2/issues/162),
which we had missed. This issue reveals the NPU's exNorm implementation
requires `dimension -2` (height) to be a multiple of 16. Our tensor
`(1, 256, 4, 4)` violates this constraint. This is potentially the most
important finding — it suggests the exNorm hardware simply cannot handle
4x4 spatial dimensions correctly, regardless of toolkit or driver version.

**Confirmed by both investigations**:
- exNorm fusion chain is the root cause
- Simulator and hardware diverge (bug is in NPU execution)
- `disable_rules` can selectively disable fusion passes
- ONNX opset 16 decomposes LayerNorm into primitives
- `optimization_level=0` disables all fusions
- Per-op CPU fallback does NOT exist in RKNN

**Items found only by our research (not in ChatGPT report)**:
- `disable_rules` actual parameter name and evidence from toolkit error messages
- `replace_torch_layernorm` re-detection risk when using decomposed primitives
- `accuracy_analysis()` API for per-layer debugging
- `NN_LAYER_DUMP=1` environment variable
- Hybrid quantization 3-step workflow
- `auto_precision` and `mixed_precision_dot_node_list` (v2.3.2)
- Issues #186, #322, #359, #220 (4 additional issues beyond ChatGPT's 2)
- Issue #322's proof that simulator is correct but hardware is wrong

**Items found only by ChatGPT (not in our initial research)**:
- Issue #162 with the 16-alignment constraint (added above)
- Suggestion to file a bug report directly with Rockchip support

**Items rated low-confidence by ChatGPT, confirmed negative by our research**:
- GroupNorm(1) replacement → NOT viable (GroupNorm is "Not Supported" in RKNN)
- Per-op CPU/NPU target assignment → does NOT exist as a user-facing API
- Model merging → untested, low priority given the root cause is now understood

---

### 6. Root Cause Found: FP16 Overflow in Conv2d (Experimental)

**The actual root cause is NOT LayerNorm/exNorm — it is FP16 overflow in
the Conv2d(256→256, 1×1) layer that precedes LayerNorm.**

This was discovered by isolating individual operations on the NPU:

1. Exported JUST the Conv2d as a standalone RKNN model
2. Fed it the real intermediate tensors from the pipeline
3. Conv2d alone produces cosine similarity of **-0.01** (random output)

The FP16 overflow chain:

```
Input to Conv2d:   (1, 256, 4, 4)  std ≈ 283,  abs_max ≈ 2,767
Output from Conv2d: (1, 256, 4, 4)  std ≈ 2,093, abs_max ≈ 87,365
FP16 max value:     65,504
Values exceeding FP16 range: 2 out of 4,096 (plus many near the limit)
```

The NPU computes Conv2d in FP16 (half-precision float, max ≈ 65,504). The
Conv2d output contains values up to 87,365, which overflow FP16 and become
`inf` or garbage. This corrupts everything downstream — LayerNorm receives
garbage input, produces garbage output, and the final score saturates.

**Why this was misdiagnosed as a LayerNorm bug**: LayerNorm is the first
operation that produces a "visible" error because it tries to normalize
already-corrupted values. The exNorm fusion discussion in GitHub issues
is a red herring — the real problem is FP16 overflow in the preceding
Conv2d, which happens BEFORE LayerNorm/exNorm even runs.

**Why the content/distortion nets work**: EfficientNet-B0 uses BatchNorm
after every Conv2d, which constrains intermediate values to a narrow
range (mean ≈ 0, std ≈ 1). The aggregation net's Conv2d(256→256, 1×1)
has no such constraint — its input (from AdaptiveAvgPool2d) has std ≈ 283,
and the Conv2d amplifies this to std ≈ 2,093, exceeding FP16 range.

**This is the same class of bug as**
[TensorRT#2564](https://github.com/NVIDIA/TensorRT/issues/2564) and
[PyTorch#66707](https://github.com/pytorch/pytorch/issues/66707) — FP16
overflow in intermediate computations. The standard solution is mixed
precision: run overflow-prone layers in FP32.

## Assessment

**The root cause is FP16 overflow, not a LayerNorm/exNorm bug.** The
aggregation net's Conv2d(256→256, 1×1) produces values exceeding the
FP16 range (65,504), and the RKNN NPU computes in FP16 by default with
no automatic overflow protection. All previous experiments (opset 16
decomposition, disable_rules, optimization_level=0) failed because they
all still compute Conv2d in FP16.

This reframes the solution space:

1. **The current hybrid approach (ONNX Runtime on CPU) is the correct
   production solution.** CPU runs in FP32, avoiding the overflow entirely.
   The aggregation net is tiny (0.2 MB, 14 ops, <1ms on CPU). Full NPU
   execution is not worth the complexity of working around a fundamental
   precision limitation.

2. **Model-level fix (if retraining were possible)**: Add BatchNorm or
   weight scaling before the Conv2d to constrain intermediate values within
   FP16 range. This would require retraining the model.

3. **RKNN-level fix (if it exists)**: Force specific layers to run in FP32
   using `mixed_precision_dot_node_list` or similar. However, RKNN does
   not clearly support per-layer FP32 override for non-quantized models.

## Experiments Completed

### Experiment A: Decompose LayerNorm (opset 16) + `disable_rules`

**Result: FAIL** — NPU score 1.009766 (expected 3.349337)

Exported aggregation_net.onnx at opset 16 (24 primitive nodes, no
`LayerNormalization`), with `disable_rules=['replace_torch_layernorm']`.
Verified ONNX matches PyTorch (0.00 diff). Build log shows no `exNorm`.
NPU still outputs 1.009766. **Decomposing LayerNorm does not fix the
bug because the overflow happens in Conv2d BEFORE LayerNorm runs.**

### Experiment B: `optimization_level=0` — all fusions disabled

**Result: FAIL** — `optimization_level=0` does NOT prevent re-fusion.
Build log still shows `replace_torch_layernorm` running and creating
`exNorm`. The parameter only controls later-stage optimizations, not
the initial graph pattern matching.

### Experiment C: All 4 `disable_rules` combined

**Result: FAIL** — NPU score 1.009766

Used `disable_rules=['replace_torch_layernorm',
'convert_layernorm_to_exnorm', 'convert_exnorm_to_exnorm_mul_add',
'replace_mul_add_by_bn']`. Build log confirmed zero `exNorm` nodes.
NPU output was still wrong. **This proves the bug is not in exNorm
fusion — it's in the preceding Conv2d.**

### Experiment D: Isolated Conv2d on NPU

**Result: FAIL** — cosine similarity -0.01 (random output)

Exported only Conv2d(256→256, 1×1) as a standalone RKNN model and tested
with real pipeline inputs (std ≈ 283). NPU output mean: -13.6 vs
reference -142.3. Conv2d output abs_max is 87,365, exceeding FP16 max
(65,504). **This is the root cause: FP16 overflow.**

## Action Items

Revised based on the FP16 overflow root cause. The LayerNorm-related
experiments are now moot.

### Accept the hybrid pipeline as the production solution

**What**: Keep the current architecture — content and distortion nets on
NPU, aggregation net on CPU via ONNX Runtime.

**Why**: The FP16 overflow is a fundamental precision limitation, not a
software bug. The aggregation net's Conv2d produces values exceeding
FP16 range (87,365 > 65,504) with real pipeline inputs. No amount of
fusion control, graph surgery, or compiler flags can fix a hardware
precision limit.

**The ONNX Runtime CPU path is correct**: It runs in FP32 and produces
the right score (3.37 vs reference 3.35, diff 0.023). The aggregation
net is tiny (0.2 MB, 14 ops, <2 ms on CPU). The performance bottleneck
is the EfficientNet feature extractors (already on NPU).

### Revert export_onnx.py and export_rknn.py changes

**What**: Revert the opset_version parameter addition in `export_onnx.py`
and the `disable_rules` addition in `export_rknn.py`. These were
experimental changes that did not produce a working fix.

### Optional: Investigate RKNN FP32 mode or input scaling

If full NPU execution is desired in the future, these avenues remain:

1. **`float_dtype='float32'`** in `rknn.config()` — if supported, this
   would force the NPU to compute in FP32 instead of FP16. However,
   the RK3588 NPU may not support FP32 at all (only FP16 and INT8).

2. **Input pre-scaling**: Scale the aggregation net inputs down before
   Conv2d (divide by a constant, e.g., 256) and scale the output back
   up. This keeps intermediate values within FP16 range. Would require
   modifying the ONNX model or the model weights directly.

3. **Model retraining**: Add BatchNorm after Conv2d or use weight
   initialization that constrains the output range. Not practical for
   pre-trained weights.

### Optional: File a feature request with Rockchip

**What**: File an issue on airockchip/rknn-toolkit2 requesting automatic
FP16 overflow detection and mixed-precision fallback for layers that
produce out-of-range values. Include our Conv2d(256→256, 1×1) reproducer
with input std ≈ 283, output abs_max ≈ 87,365 > FP16 max 65,504.

Note: Many of the "LayerNorm bugs" reported in issues #149, #186, #322,
#460 may also be FP16 overflow in preceding layers, not actual exNorm
implementation bugs. This insight could help other RKNN users.

---

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

## Sources

### RKNN GitHub Issues
- [#149 — NPU LayerNorm severe precision loss (Stable Diffusion)](https://github.com/airockchip/rknn-toolkit2/issues/149)
- [#186 — LayerNorm results incorrect (OPEN)](https://github.com/airockchip/rknn-toolkit2/issues/186)
- [#460 — ViTTracker incorrect inference after operator fusion (OPEN)](https://github.com/airockchip/rknn-toolkit2/issues/460)
- [#322 — Depth Anything v2 simulator vs device mismatch](https://github.com/airockchip/rknn-toolkit2/issues/322)
- [#359 — unsupport cpu exNorm op](https://github.com/airockchip/rknn-toolkit2/issues/359)
- [#220 — ConvNeXt wrong output results (OPEN)](https://github.com/airockchip/rknn-toolkit2/issues/220)
- [#162 — LayerNorm 16-alignment constraint (rockchip-linux repo)](https://github.com/rockchip-linux/rknn-toolkit2/issues/162)

### RKNN Documentation
- [OP Support v2.3.2](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_OP_Support-2.3.2.md)
- [API Differences doc](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_API_Difference_With_Toolkit1-2.3.2.md)
- [CHANGELOG.md](https://github.com/airockchip/rknn-toolkit2/blob/master/CHANGELOG.md)
- [Official PDF docs](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)

### `disable_rules` Evidence
- [ultralytics#20655 — toolkit suggests disable_rules in error message](https://github.com/ultralytics/ultralytics/issues/20655)
- [rknn_model_zoo#221 — same disable_rules suggestion](https://github.com/airockchip/rknn_model_zoo/issues/221)

### ONNX Decomposition
- [onnx.inliner API](https://onnx.ai/onnx/api/inliner.html)
- [PR #6931 — inline_schema_functions support](https://github.com/onnx/onnx/pull/6931)
- [LayerNormalization function definition](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)

### Related Industry Issues
- [TensorRT#2564 — FP16 LayerNorm overflow](https://github.com/NVIDIA/TensorRT/issues/2564)
- [PyTorch#66707 — LayerNorm needs FP32 for FP16 inputs](https://github.com/pytorch/pytorch/issues/66707)
- [rknn_model_zoo#314 — Whisper INT8 quantization produces garbage](https://github.com/airockchip/rknn_model_zoo/issues/314)

### Architecture References
- [DeepWiki — RKNN Model Conversion Process](https://deepwiki.com/airockchip/rknn-toolkit2/3.1-model-conversion-process)
- [DeepWiki — RKNN System Architecture](https://deepwiki.com/airockchip/rknn-toolkit2/2-rknn-system-architecture)
- [RKNN ONNX Opset Compatibility Guide](https://zediot.com/blog/rknn-onnx-opset-compatibility/)
