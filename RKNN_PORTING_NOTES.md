# RKNN Porting Notes

Findings from porting UVQ 1.5 to the Rockchip RK3588S NPU (Orange Pi 5 Plus)
via the RKNN framework. Tested February 2026 with rknn-toolkit2 2.3.2 and
rknn-toolkit-lite2 2.3.2.

## Conversion pipeline

The workflow is: **PyTorch -> ONNX -> RKNN** (two-step, not direct).

```
PC (x86_64):
  export_onnx.py   -> content_net.onnx, distortion_net.onnx, aggregation_net.onnx
  export_rknn.py   -> content_net.rknn, distortion_net.rknn, aggregation_net.rknn
                      + reference.json

Device (aarch64):
  rknn-toolkit-lite2 loads .rknn files, runs on NPU via /dev/rknpu
```

The conversion tool (`rknn-toolkit2`, the `rknn.api.RKNN` class) only runs on
x86_64. The on-device runtime (`rknn-toolkit-lite2`, the `rknnlite.api.RKNNLite`
class) only runs on aarch64. They are separate packages with different APIs.

## rknn-toolkit2 installation (PC, x86_64)

rknn-toolkit2 is not on PyPI. Install from the Rockchip GitHub repo:
https://github.com/airockchip/rknn-toolkit2

Wheels are in `rknn-toolkit2/packages/x86_64/`. Available for Python 3.7-3.12.

### Dependency conflicts

- **torch<=2.4.0** is required. If your project uses a newer torch, create a
  separate venv for the conversion step.
- **onnx**: rknn-toolkit2 2.3.2 uses `onnx.mapping` internally, which was
  removed in onnx>=1.17. Pin to `onnx==1.16.2`.
- **setuptools**: rknn-toolkit2 uses `pkg_resources` at import time.
  setuptools>=81 removed `pkg_resources`. Pin to `setuptools<81`.

Working recipe:

```bash
python3.12 -m venv /tmp/rknn-venv
/tmp/rknn-venv/bin/pip install "setuptools<81"
/tmp/rknn-venv/bin/pip install /path/to/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.whl
/tmp/rknn-venv/bin/pip install "onnx==1.16.2"
/tmp/rknn-venv/bin/pip install "torch==2.4.0+cpu" "torchvision==0.19.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

## ONNX -> RKNN conversion issues

### Dynamic batch axes are rejected

Our ONNX models use dynamic axes (`"batch"` as a symbolic dim on axis 0),
which is standard practice for PyTorch ONNX export. RKNN does not support
dynamic shapes at all. The `load_onnx()` call fails with:

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

The `inputs` names must match the ONNX graph's input names exactly.

### No baked-in normalization

`rknn.config()` supports `mean_values` and `std_values` for baked-in input
normalization. Our models expect float32 inputs already normalized to [-1, 1]
by the video reader, so we skip these. RKNN warns about it:

```
load_onnx: The config.mean_values is None, zeros will be set for input 0!
load_onnx: The config.std_values is None, ones will be set for input 0!
```

These warnings are harmless -- zeros/ones for mean/std is a no-op transform.

### Simulator NHWC mismatch

The built-in RKNN simulator (used when `init_runtime(target=None)`) defaults to
NHWC data format. Our models are NCHW. The simulator verification step for the
aggregation net fails with:

```
The input(ndarray) shape (1, 128, 8, 8) is wrong, expect 'nhwc' like (1, 8, 8, 128)!
```

The content_net and distortion_net simulators happen to work because their
inputs are image-like (the 3-channel dim is small enough that the NHWC check
doesn't fire, or the default `data_format='nhwc'` warning is non-fatal).
The aggregation net has 128-channel feature map inputs where the dimension
mismatch is unambiguous.

This is a **simulator-only issue**. On real hardware with rknn-toolkit-lite2,
passing `data_format='nchw'` to `inference()` works correctly.

### "Unknown op target: 0" warning

During `rknn.build()`, the aggregation net produces:

```
RKNN: Unkown op target: 0
```

This appears to be a benign warning from the graph optimizer, not an error.
The model builds and exports successfully.

## rknn-toolkit-lite2 inference (device, aarch64)

### data_format must be explicit

`RKNNLite.inference()` defaults to `data_format=None`, which the runtime
interprets as NHWC. Since the ONNX models (and therefore the RKNN models)
use NCHW layout, **every** `inference()` call must pass `data_format='nchw'`
explicitly:

```python
output = rknn.inference(inputs=[nchw_array], data_format='nchw')
```

Without this, the runtime silently reinterprets the memory layout, producing
garbage output. There is no error or warning -- the shapes may even look
correct, but the values will be wrong.

### Batch size is always 1

The RKNN models are compiled with a fixed batch=1. To process multiple inputs
(e.g., the 9 distortion patches), loop and call inference one at a time:

```python
for i in range(9):
    feat = distortion_net.inference(
        inputs=[patches[i:i+1]],
        data_format='nchw')[0]
```

### Session lifecycle

Each `RKNNLite` instance holds GPU/NPU resources. Always call `release()`
when done. The pattern:

```python
rknn = RKNNLite(verbose=False)
rknn.load_rknn("model.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
# ... use rknn.inference() ...
rknn.release()
```

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

### NPU driver is missing on mainline kernels

This is the biggest blocker. The board ships with Ubuntu 24.10 using the
Joshua Riek `ubuntu-rockchip` distribution (kernel 6.11.x from the
`ppa:jjriek/rockchip` PPA). This is a **mainline** kernel -- it does not
include the Rockchip NPU driver or device tree node.

Specifically:
- `/dev/rknpu*` does not exist
- `rknpu.ko` is not in `/lib/modules/$(uname -r)/`
- The device tree blob (`rk3588-orangepi-5-plus.dtb`) has no NPU node at
  address `0xfdab0000`
- `rknnlite.init_runtime()` fails with:
  `failed to open rknpu module, need to insmod rknpu dirver!`

The NPU driver (`rknpu.ko`) and its device tree binding only exist in the
**Rockchip vendor BSP kernel** (typically kernel 5.10.x or 6.1.x from
Rockchip's fork).

### Options to get NPU working

1. **Switch to vendor BSP kernel**: Reflash with an image that uses the
   Rockchip BSP kernel (e.g., Armbian with the `edge-rockchip-rkbsp` kernel,
   or Orange Pi's official image). This is the most reliable path.

2. **Build rknpu.ko out-of-tree**: The driver source is at
   `drivers/rknpu/` in Rockchip's kernel fork. In theory it can be compiled
   against mainline headers, but it depends on Rockchip-specific DMA/IOMMU
   APIs and a device tree node that mainline doesn't provide. Very fragile.

3. **Device tree overlay**: Even if you could build the module, you'd need to
   add the NPU device tree node via an overlay. The mainline DTS for RK3588
   does not define the NPU hardware block at all.

### Ubuntu 24.10 is EOL

The `ports.ubuntu.com` repos for Oracular (24.10) are gone. To install
packages, switch apt sources to `old-releases.ubuntu.com`:

```bash
sudo sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://old-releases.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list.d/ubuntu.sources
sudo apt-get update
```

### librknnrt.so installation

rknn-toolkit-lite2 depends on `librknnrt.so` (the C runtime). It is not
bundled with the Python wheel. Install it from the rknn-toolkit2 repo:

```bash
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
sudo ldconfig
```

Without this, `from rknnlite.api import RKNNLite` imports fine but
`init_runtime()` will fail at the shared library level.

## Testing without PyTorch on the device

The standard test pattern (load PyTorch model + RKNN model, feed same input,
compare scores) doesn't work on the target device because PyTorch is not
installed there (and installing it on an aarch64 board is heavyweight).

Our approach:
1. During `export_rknn.py` (on the PC, where PyTorch is available), generate
   a `reference.json` containing:
   - A numpy random seed (42)
   - Input shapes for content and distortion patches
   - The expected PyTorch output score
2. On the device, the test regenerates the same deterministic inputs using
   `np.random.seed(42)` (numpy RNG is portable across architectures), feeds
   them through RKNN, and compares against the saved score.

This avoids a PyTorch dependency on the device entirely. The `conftest.py`
also has a `try/except` around `import torch` so that the entire test suite
doesn't crash when torch is absent.

## Model sizes

| Model           | ONNX   | RKNN (FP16) |
|-----------------|--------|-------------|
| content_net     | 10.0MB | 10.0MB      |
| distortion_net  | 10.5MB | 10.5MB      |
| aggregation_net | 0.2MB  | 0.2MB       |
| **Total**       | 20.7MB | 20.7MB      |

FP16 (no quantization) produces RKNN files roughly the same size as ONNX.
INT8 quantization is available via `--quantize --dataset calibration.txt`
but was not tested.
