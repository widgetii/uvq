# RKNN Porting Notes

Findings from porting UVQ 1.5 to the Rockchip RK3588S NPU (Orange Pi 5 Plus)
via the RKNN framework. Tested February 2026 on two boards with different
kernel/driver configurations.

## Architecture overview

The RKNN ecosystem is split into two separate, incompatible packages:

| Component | Package | Arch | Python API | Purpose |
|---|---|---|---|---|
| Conversion toolkit | `rknn-toolkit2` | x86_64 only | `rknn.api.RKNN` | ONNX → RKNN conversion |
| On-device runtime | `rknn-toolkit-lite2` | aarch64 only | `rknnlite.api.RKNNLite` | NPU inference |

Neither package is on PyPI. Both are distributed as wheels from the
[Rockchip GitHub repo](https://github.com/airockchip/rknn-toolkit2).

## Conversion pipeline

```
PC (x86_64):
  export_onnx.py   -> content_net.onnx, distortion_net.onnx, aggregation_net.onnx
  export_rknn.py   -> content_net.rknn, distortion_net.rknn, aggregation_net.rknn
                      + reference.json

Device (aarch64):
  rknn-toolkit-lite2 loads .rknn files for content + distortion on NPU
  onnxruntime loads aggregation_net.onnx on CPU (see "Aggregation net bug" below)
```

The workflow is: **PyTorch → ONNX → RKNN** (two-step, not direct). There is no
PyTorch-to-RKNN converter.

## rknn-toolkit2 installation (PC, x86_64)

Wheels are in `rknn-toolkit2/packages/x86_64/`. Available for Python 3.7–3.12.

### Dependency conflicts

rknn-toolkit2 has strict and partially outdated dependency requirements that
conflict with modern Python environments:

- **torch<=2.4.0** is required. If your project uses a newer torch, create a
  separate venv for the conversion step.
- **onnx**: rknn-toolkit2 2.3.x uses `onnx.mapping` internally, which was
  removed in onnx>=1.17. Pin to `onnx==1.16.2`.
- **setuptools**: rknn-toolkit2 uses `pkg_resources` at import time.
  setuptools>=81 removed `pkg_resources`. Pin to `setuptools<81`.

### Working installation recipe

```bash
python3.12 -m venv /tmp/rknn-venv
/tmp/rknn-venv/bin/pip install "setuptools<81"
/tmp/rknn-venv/bin/pip install /path/to/rknn_toolkit2-2.3.0-cp312-cp312-manylinux_2_17_x86_64.whl
/tmp/rknn-venv/bin/pip install "onnx==1.16.2"
/tmp/rknn-venv/bin/pip install "torch==2.4.0+cpu" "torchvision==0.19.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

The rknn-toolkit2 wheel pulls in CUDA torch by default (~5 GB). Use the CPU
index to avoid this if you only need the conversion step.

## ONNX → RKNN conversion issues

### Dynamic batch axes are rejected

RKNN does not support dynamic shapes at all. Our ONNX models use dynamic axes
(`"batch"` as a symbolic dim on axis 0), which is standard practice for
PyTorch ONNX export. The `load_onnx()` call fails with:

```
The input shape ['batch', 3, 256, 256] of 'input' is not support!
Please set the 'inputs' / 'input_size_list' parameters of 'rknn.load_onnx'
```

**Fix**: pass `inputs` (list of input tensor names) and `input_size_list`
(list of concrete shapes) to `rknn.load_onnx()`:

```python
rknn.load_onnx(
    model="content_net.onnx",
    inputs=["input"],
    input_size_list=[[1, 3, 256, 256]],
)
```

For multi-input models like the aggregation net, both parameters are required:

```python
rknn.load_onnx(
    model="aggregation_net.onnx",
    inputs=["content", "distortion"],
    input_size_list=[[1, 128, 8, 8], [1, 128, 24, 24]],
)
```

The `inputs` names must match the ONNX graph's input names exactly. Use
`onnx.load()` + inspecting `graph.input` to find them if unsure.

### No baked-in normalization

`rknn.config()` supports `mean_values` and `std_values` for baked-in input
normalization. Our models expect float32 inputs already normalized to [-1, 1]
by the video reader, so we skip these. RKNN warns about it:

```
load_onnx: The config.mean_values is None, zeros will be set for input 0!
load_onnx: The config.std_values is None, ones will be set for input 0!
```

These warnings are harmless — zeros/ones for mean/std is a no-op transform.

### Simulator NHWC mismatch

The built-in RKNN simulator (used when `init_runtime(target=None)` on x86_64)
defaults to NHWC data format. Our models are NCHW. The simulator verification
step fails with:

```
The input(ndarray) shape (1, 128, 8, 8) is wrong, expect 'nhwc' like (1, 8, 8, 128)!
```

This affects all three models in the simulator. The content_net and
distortion_net simulators sometimes pass because their 3-channel inputs are
ambiguous (the NHWC check doesn't fire for small channel dims). The
aggregation net with 128-channel feature maps always fails.

This is a **simulator-only issue**. On real hardware with rknn-toolkit-lite2,
the runtime accepts NCHW data and transposes it internally.

### "Unknown op target: 0" warning

During `rknn.build()`, the aggregation net produces:

```
RKNN: Unkown op target: 0
```

This appears to be a benign warning from the graph optimizer, not an error.
The model builds and exports successfully.

## rknn-toolkit-lite2 inference (device, aarch64)

### Aggregation net produces wrong results on NPU (critical bug)

**This is the most important finding and drove the final architecture.**

The `aggregation_net.rknn` model produces completely wrong scores when run
on the NPU with driver version 0.8.2 (BSP kernel 5.10.110). Content and
distortion nets work correctly, but the aggregation net outputs saturated
values (~1.0 or ~5.0) instead of the correct score.

Systematic testing on the Orange Pi 5 Plus (BSP 5.10, driver 0.8.2):

| Pipeline | Score | Correct? |
|---|---|---|
| PyTorch FP32 (PC) | 3.3493 | reference |
| ONNX Runtime CPU (board) | 3.3493 | yes |
| RKNN NPU content+distortion + ONNX agg | 3.3509 | yes (within FP16 tolerance) |
| RKNN NPU all three models | 1.0098 | **no** |
| RKNN NPU aggregation only (PyTorch features as input) | 1.0098 | **no** |

Key observations:
- The content and distortion net NPU outputs match PyTorch within FP16
  tolerance (mean differs by <1%)
- The aggregation net is broken regardless of input — even feeding
  exact PyTorch-computed features produces wrong output
- All `data_format` combinations (`nchw`, `nhwc`, default) give the
  same wrong result
- Re-exporting with toolkit 2.3.0 (matching the runtime version) did
  not help — the toolkit 2.3.2 exports also fail identically
- The runtime warns: `Current driver version: 0.8.2, recommend to
  upgrade to >= 0.8.8`

The likely cause is NPU driver 0.8.2 mishandling one of the operations
in the aggregation net. The aggregation net uses LayerNorm, concat,
Conv2d, MaxPool, Linear, and Tanh — a different mix from the EfficientNet-based
content/distortion nets which use BatchNorm, SiLU, DepthwiseSeparableConv,
and SE blocks. LayerNorm is the most probable culprit as it is less
commonly supported by NPU accelerators.

**Workaround: hybrid pipeline.** The production code runs content and
distortion nets on the NPU via RKNN-Lite, and the aggregation net on
CPU via ONNX Runtime. The aggregation net is tiny (0.2 MB, negligible
CPU time) so there is no measurable performance impact — the heavy
computation (two EfficientNet-B0 forward passes per frame) stays on the NPU.

### data_format must be explicit

`RKNNLite.inference()` defaults to `data_format=None`, which the runtime
interprets as NHWC. Since the ONNX models (and therefore the RKNN models)
use NCHW layout, **every** `inference()` call must pass `data_format='nchw'`
explicitly:

```python
output = rknn.inference(inputs=[nchw_array], data_format='nchw')
```

Without this, the runtime silently reinterprets the memory layout, producing
garbage output. There is no error or warning — the shapes may even look
correct, but the values will be wrong.

On real hardware (tested with librknnrt 2.3.0), passing `data_format='nchw'`
causes the runtime to print:

```
W The input[0] need NHWC data format, but NCHW set, the data format
  and data buffer will be changed to NHWC.
```

This warning is expected and correct — the runtime is transposing the input
from NCHW to NHWC (the NPU's native format) before execution. The output
is returned in the original NCHW layout. We verified that passing NCHW data
with `data_format='nchw'` and passing manually-transposed NHWC data without
the flag produce identical results.

### Batch size is always 1

RKNN models are compiled with a fixed batch dimension (batch=1 in our case).
There is no dynamic batching. To process multiple inputs (e.g., the 9
distortion patches), loop and call inference one at a time:

```python
for i in range(9):
    feat = distortion_net.inference(
        inputs=[patches[i:i+1]],
        data_format='nchw')[0]
```

### Session lifecycle

Each `RKNNLite` instance holds NPU resources. Always call `release()`
when done:

```python
rknn = RKNNLite(verbose=False)
rknn.load_rknn("model.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
# ... use rknn.inference() ...
rknn.release()
```

Failing to call `release()` can leak GPU memory and eventually prevent
new sessions from initializing.

### Content resize: cv2 vs torch.nn.functional.interpolate

The PyTorch and ONNX backends resize 1080p frames to 256x256 using
`torch.nn.functional.interpolate(mode='bilinear')`. On the device, we don't
have PyTorch, so we use `cv2.resize(INTER_LINEAR)` instead.

These produce slightly different results due to implementation differences in
bilinear interpolation (pixel coordinate mapping, edge handling). The
difference is small enough to stay within the abs=0.05 tolerance for the
FP16 model, but it is a source of systematic error in cross-backend
comparisons.

The cross-backend test sidesteps this by feeding pre-processed inputs
(already at 256x256 / 360x640) to both PyTorch and RKNN, so the resize
difference doesn't affect the reference comparison.

## Hardware: Orange Pi 5 Plus (RK3588S)

We tested on two boards with different OS/kernel configurations.

### Board 1: Mainline kernel (NPU unavailable)

- **OS**: Ubuntu 24.10 (Joshua Riek `ubuntu-rockchip` distribution)
- **Kernel**: 6.11.x (mainline from `ppa:jjriek/rockchip`)
- **NPU status**: not available

The mainline kernel does not include the Rockchip NPU driver or device tree
node:
- `/dev/rknpu*` does not exist
- `rknpu.ko` is not in `/lib/modules/$(uname -r)/`
- The DTB has no NPU node at address `0xfdab0000`
- `rknnlite.init_runtime()` fails with:
  `failed to open rknpu module, need to insmod rknpu dirver!`

The NPU driver and its device tree binding only exist in the Rockchip
**vendor BSP kernel** (typically 5.10.x or 6.1.x from Rockchip's fork).

**Options to get NPU working on mainline:**

1. **Switch to vendor BSP kernel** (recommended): Reflash with an image
   that uses the Rockchip BSP kernel (e.g., Armbian with
   `edge-rockchip-rkbsp`, or Orange Pi's official image).

2. **Build rknpu.ko out-of-tree**: The driver source is at `drivers/rknpu/`
   in Rockchip's kernel fork. Depends on Rockchip-specific DMA/IOMMU APIs
   and a device tree node that mainline doesn't provide. Very fragile.

3. **Device tree overlay**: Even if you could build the module, you'd need
   to add the NPU node via an overlay. Mainline DTS for RK3588 does not
   define the NPU hardware block at all.

**Additional issue**: Ubuntu 24.10 is EOL and `ports.ubuntu.com` repos are
gone. To install packages, switch apt sources:

```bash
sudo sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://old-releases.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list.d/ubuntu.sources
sudo apt-get update
```

### Board 2: BSP kernel (NPU functional)

- **OS**: Ubuntu 22.04.5 LTS (Orange Pi official image)
- **Kernel**: 5.10.110+ (Rockchip vendor BSP, Oct 2023)
- **NPU driver**: rknpu 0.8.2 (built into the kernel, not a loadable module)
- **NPU status**: functional (with caveats — see aggregation net bug above)

#### NPU device node: /dev/dri/renderD129, not /dev/rknpu

On BSP kernel 5.10, the NPU registers as a **DRM device**, not as
`/dev/rknpu*`. The NPU appears at:
- `/dev/dri/renderD129` (render node)
- `/dev/dri/card1` (card node)

The `librknnrt.so` runtime accesses the NPU via DRM ioctls through these
nodes. It does not look for `/dev/rknpu*`.

**Tests should NOT check for `/dev/rknpu*`** to determine NPU availability.
Use `init_runtime()` return code as the authoritative check. Our test uses
`pytest.skip()` if `init_runtime()` fails.

#### dmesg output on boot

```
RKNPU fdab0000.npu: Adding to iommu group 0
RKNPU fdab0000.npu: RKNPU: rknpu iommu is enabled, using iommu mode
RKNPU fdab0000.npu: can't request region for resource [mem 0xfdab0000-0xfdabffff]
RKNPU fdab0000.npu: can't request region for resource [mem 0xfdac0000-0xfdacffff]
RKNPU fdab0000.npu: can't request region for resource [mem 0xfdad0000-0xfdadffff]
[drm] Initialized rknpu 0.8.2 20220829 for fdab0000.npu on minor 1
RKNPU fdab0000.npu: leakage=11
RKNPU fdab0000.npu: pvtm=898
RKNPU fdab0000.npu: failed to find power_model node
RKNPU fdab0000.npu: RKNPU: failed to initialize power model
RKNPU fdab0000.npu: RKNPU: failed to get dynamic-coefficient
```

The "can't request region" and "failed to initialize power model" warnings
are non-fatal on BSP 5.10. The driver initializes successfully despite them
(`Initialized rknpu 0.8.2`).

#### librknnrt.so version must match the toolkit

rknn-toolkit-lite2 depends on `librknnrt.so` (the C runtime). It is NOT
bundled with the Python wheel — install it separately:

```bash
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
sudo ldconfig
```

**Version compatibility is critical.** The pre-installed `librknnrt.so` on
the Orange Pi BSP image was version 1.4.0 (September 2022). Models exported
with toolkit 2.3.x use RKNN model format version 6, which librknnrt 1.4.0
rejects:

```
Invalid RKNN model version 6
rknn_init, load model failed!
```

After updating to librknnrt 2.3.0 from the rknn-toolkit2 repo, models load
and run. However, the NPU **kernel driver** version (0.8.2, built into the
kernel) cannot be updated without rebuilding or replacing the kernel. The
runtime warns about this mismatch:

```
Current driver version: 0.8.2, recommend to upgrade the driver
to the new version: >= 0.8.8
```

Summary of the version stack on Board 2:

| Component | Version | Updateable? |
|---|---|---|
| NPU kernel driver (rknpu) | 0.8.2 | No (built into kernel 5.10.110) |
| C runtime (librknnrt.so) | 2.3.0 | Yes (copy to /usr/lib/) |
| Python runtime (rknn-toolkit-lite2) | 2.3.0 | Yes (pip install) |
| Conversion toolkit (rknn-toolkit2) | 2.3.0 | Yes (pip install, x86_64 only) |

## Testing without PyTorch on the device

The standard cross-backend test pattern (load both models, feed same input,
compare live) doesn't work on the target device because PyTorch is not
installed there (and installing it on an aarch64 board is heavyweight).

### Reference-based testing approach

1. During `export_rknn.py` (on the PC, where PyTorch is available), generate
   a `reference.json` containing:
   - A numpy random seed (42)
   - Input shapes for content and distortion patches
   - The expected PyTorch output score

2. On the device, the test regenerates the same deterministic inputs using
   `np.random.seed(42)` and `np.random.randn()` (numpy RNG is portable
   across architectures and Python versions), feeds them through the hybrid
   RKNN+ONNX pipeline, and compares against the saved score.

This avoids a PyTorch dependency on the device entirely.

### conftest.py compatibility

The shared `conftest.py` uses `try/except` around `import torch` with a
`HAS_TORCH` flag. All torch-dependent fixtures call `pytest.skip("PyTorch
not installed")` when torch is absent. This ensures the entire test suite
doesn't crash when running on the board.

### Tolerance

The cross-backend test uses `abs=0.05` tolerance. The content and distortion
nets in FP16 introduce small numerical differences (feature mean differs
by <1% from FP32 PyTorch), which compound through the aggregation net.
With the hybrid pipeline (NPU features + ONNX aggregation), the final
score difference is typically <0.01.

## Model sizes

| Model           | ONNX   | RKNN (FP16) |
|-----------------|--------|-------------|
| content_net     | 10.0MB | 10.0MB      |
| distortion_net  | 10.5MB | 10.5MB      |
| aggregation_net | 0.2MB  | 0.4MB       |
| **Total**       | 20.7MB | 20.9MB      |

FP16 (no quantization) produces RKNN files roughly the same size as ONNX.
The aggregation net is slightly larger due to RKNN graph metadata.
INT8 quantization is available via `--quantize --dataset calibration.txt`
but was not tested.

## Summary of workarounds

| Issue | Workaround |
|---|---|
| Mainline kernel has no NPU | Use vendor BSP kernel (5.10.x or 6.1.x) |
| ONNX dynamic batch axes rejected | Pass `inputs` + `input_size_list` to `load_onnx()` |
| Simulator rejects NCHW inputs | Skip simulator verification; works on real hardware |
| Default data_format is NHWC | Always pass `data_format='nchw'` to `inference()` |
| Aggregation net wrong on NPU | Run aggregation via ONNX Runtime on CPU |
| librknnrt.so version mismatch | Update librknnrt.so to match the toolkit version |
| No `/dev/rknpu*` on BSP 5.10 | NPU is at `/dev/dri/renderD129`; check `init_runtime()` not device nodes |
| No PyTorch on device for testing | Pre-computed reference.json with numpy-seeded inputs |
| Ubuntu 24.10 repos gone (EOL) | Switch apt sources to `old-releases.ubuntu.com` |
| rknn-toolkit2 needs old onnx | Pin `onnx==1.16.2` |
| rknn-toolkit2 needs pkg_resources | Pin `setuptools<81` |
