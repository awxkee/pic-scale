# pic-scale

[![PyPI](https://img.shields.io/pypi/v/pic-scale)](https://pypi.org/project/pic-scale/)
[![CI](https://github.com/awxkee/pic-scale/actions/workflows/publish_pypi.yml/badge.svg)]([https://github.com/awxkee/pic-scale/actions](https://github.com/awxkee/pic-scale/actions/workflows/publish_pypi.yml))
[![Python](https://img.shields.io/pypi/pyversions/pic-scale)](https://pypi.org/project/pic-scale/)

High-performance image resampling for Python — a drop-in replacement for
`PIL.Image.resize` backed by a SIMD-optimized Rust engine.

## Installation

```bash
pip install pic-scale
```

Pre-built wheels for Linux (x86-64, aarch64), macOS (Intel + Apple Silicon),
and Windows (x86-64). No Rust toolchain needed.

## Quick start

```python
from PIL import Image
from pic_scale import resize, Resampling

img = Image.open("photo.jpg")
out = resize(img, (800, 600), Resampling.LANCZOS)
out.save("small.jpg")
```

That's the entire API change from Pillow — swap `img.resize(...)` for
`resize(img, ...)`.

## Resampling filters

| Filter | `Resampling` constant | Notes |
|---|---|---|
| Nearest neighbour | `NEAREST` | Fastest, blocky |
| Bilinear | `BILINEAR` | Fast, smooth |
| Bicubic | `BICUBIC` | Keys cubic a = −0.5 |
| Lanczos / sinc | `LANCZOS` | Window 3, Pillow default for quality |
| Lanczos 2 | `LANCZOS2` | Faster, slightly softer |
| Lanczos 4 | `LANCZOS4` | Slower, very sharp |
| Box / area | `BOX` | Best for downscaling |
| Hamming | `HAMMING` | |
| Mitchell-Netravali | `MITCHELL` | B=1/3 C=1/3, balanced |
| Catmull-Rom | `CATMULL_ROM` | Sharper than Mitchell |
| Gaussian | `GAUSSIAN` | |
| Hann | `HANN` | |

## Supported Pillow modes

| Mode | Description |
|---|---|
| `L` | 8-bit grayscale |
| `LA` | 8-bit grayscale + alpha |
| `RGB` | 8-bit colour |
| `RGBA` | 8-bit colour + alpha |
| `I;16` | 16-bit grayscale |
| `F` | 32-bit float grayscale |

Other Pillow modes (`CMYK`, `YCbCr`, `P`, …) must be converted first:

```python
out = resize(img.convert("RGB"), (800, 600))
```

## Alpha premultiplication

For `LA` and `RGBA` images, `premultiply_alpha=True` (the default)
pre-multiplies RGB by alpha before resampling and un-multiplies afterward.
This prevents dark fringing around transparent edges — the same technique
Photoshop and modern compositors use.

```python
# Correct — no dark halos around transparency
out = resize(img, (400, 400), Resampling.LANCZOS, premultiply_alpha=True)

# Disable if your pipeline handles premultiplication separately
out = resize(img, (400, 400), Resampling.LANCZOS, premultiply_alpha=False)
```

## Multithreading

Pass `workers` to use multiple threads. `0` lets pic-scale choose
automatically based on the image size and core count.

```python
out = resize(img, (3840, 2160), Resampling.LANCZOS, workers=0)
```

## Plan — reuse across many frames

When resizing a stream of images with the same source and target dimensions,
create a `Plan` once and reuse it. Filter weights are computed once at
construction; subsequent calls skip that work entirely.

```python
from pic_scale import Plan, Resampling

plan = Plan(
    src_size=(1920, 1080),
    dst_size=(960,  540),
    resampling=Resampling.LANCZOS,
    mode="RGB",
)

for frame in video_frames:
    small = plan.resize(frame)
    encoder.write(small)
```

The plan validates that each incoming image has the correct mode and raises
`ValueError` immediately if it doesn't, so mode mismatches are caught early.

## API reference

### `resize(image, size, resampling, premultiply_alpha, workers)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `PIL.Image.Image` | — | Source image |
| `size` | `(int, int)` | — | Target `(width, height)` |
| `resampling` | `Resampling` | `LANCZOS` | Filter |
| `premultiply_alpha` | `bool` | `True` | Premultiply alpha for `LA`/`RGBA` |
| `workers` | `int` | `1` | Thread count; `0` = adaptive |

Returns a new `PIL.Image.Image` in the same mode as the input.

### `Plan(src_size, dst_size, resampling, mode, premultiply_alpha, workers)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `src_size` | `(int, int)` | — | Source `(width, height)` |
| `dst_size` | `(int, int)` | — | Target `(width, height)` |
| `resampling` | `Resampling` | — | Filter |
| `mode` | `str` | — | Pillow mode string |
| `premultiply_alpha` | `bool` | `True` | Premultiply alpha |
| `workers` | `int` | `1` | Thread count; `0` = adaptive |

#### `Plan.resize(image) → PIL.Image.Image`

Resize `image` using the pre-computed plan. The image's mode must match the
mode the plan was created with.

## Comparison with Pillow

```python
# Pillow
out = img.resize((800, 600), Image.Resampling.LANCZOS)

# pic-scale — same result, faster
from pic_scale import resize, Resampling
out = resize(img, (800, 600), Resampling.LANCZOS)
```

pic-scale does not replace Pillow — it only replaces the resize step. All
other Pillow operations (open, save, colour conversion, drawing, …) continue
to use Pillow as normal.

## License

BSD 3-Clause