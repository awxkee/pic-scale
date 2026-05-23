"""tests/test_pic_scale.py — validate against Pillow's own resize output."""
import pytest
from PIL import Image
import numpy as np
import pic_scale
from pic_scale import resize, Resampling, Plan

# ─── helpers ──────────────────────────────────────────────────────────────────

def solid(mode: str, w: int, h: int, color) -> Image.Image:
    if mode == "L":
        arr = np.full((h, w), color, dtype=np.uint8)
    elif mode == "LA":
        arr = np.full((h, w, 2), color, dtype=np.uint8)
    elif mode == "RGB":
        arr = np.full((h, w, 3), color, dtype=np.uint8)
    elif mode == "RGBA":
        arr = np.full((h, w, 4), color, dtype=np.uint8)
    elif mode == "F":
        arr = np.full((h, w), color, dtype=np.float32)
    elif mode == "I;16":
        arr = np.full((h, w), color, dtype=np.uint16)
    else:
        raise ValueError(f"Unsupported mode {mode}")
    return Image.fromarray(arr, mode=mode)

def noise(mode: str, w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(42)
    if mode == "F":
        arr = rng.random((h, w), dtype=np.float32)
        return Image.fromarray(arr, mode="F")
    if mode == "I;16":
        arr = (rng.integers(0, 65535, (h, w), dtype=np.uint16))
        return Image.fromarray(arr, mode="I;16")
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    arr = rng.integers(0, 255, (h, w, channels), dtype=np.uint8).squeeze()
    return Image.fromarray(arr, mode=mode)

# ─── solid-colour preservation ────────────────────────────────────────────────

@pytest.mark.parametrize("mode,color", [
    ("L",    128),
    ("RGB",  (100, 150, 200)),
    ("RGBA", (100, 150, 200, 255)),
])
def test_solid_preserved(mode, color):
    """Resizing a solid colour image should keep all pixels identical."""
    img = solid(mode, 256, 256, color)
    out = resize(img, (128, 128), Resampling.LANCZOS)
    assert out.mode == mode
    arr = np.array(out)

    # Check only the interior — skip 8px border where edge ringing can occur
    border = 8
    interior = arr[border:-border, border:-border]

    channels = len(color) if isinstance(color, tuple) else 1
    if channels == 1:
        diff = np.abs(interior.astype(int) - color).max()
        assert diff < 3, f"L interior max diff {diff}, values: {np.unique(interior)}"
    else:
        expected = np.array(color, dtype=int)
        for c in range(channels):
            diff = np.abs(interior[:, :, c].astype(int) - expected[c]).max()
            assert diff < 3, (
                f"Channel {c} max diff {diff} >= 3 "
                f"(expected ~{expected[c]}, unique values: {np.unique(interior[:,:,c])})"
            )

# ─── size correctness ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("src,dst", [
    ((256, 256), (128, 128)),
    ((100, 200), (50,  100)),
    ((64,   64), (128, 128)),  # upscale
    ((300, 200), (100,  80)),
])
def test_output_size(mode, src, dst):
    img = noise(mode, *src)
    out = resize(img, dst, Resampling.BILINEAR)
    assert out.size == dst
    assert out.mode == mode

# ─── filter smoke tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("f", [
    Resampling.NEAREST,
    Resampling.BILINEAR,
    Resampling.BICUBIC,
    Resampling.LANCZOS,
    Resampling.BOX,
    Resampling.HAMMING,
    Resampling.MITCHELL,
    Resampling.CATMULL_ROM,
    Resampling.LANCZOS2,
    Resampling.LANCZOS4,
])
def test_filters_run(f):
    img = noise("RGB", 128, 128)
    out = resize(img, (64, 64), f)
    assert out.size == (64, 64)

# ─── float / 16-bit modes ─────────────────────────────────────────────────────

def test_mode_f():
    img = noise("F", 64, 64)
    out = resize(img, (32, 32), Resampling.BILINEAR)
    assert out.mode == "F"
    assert out.size == (32, 32)

def test_mode_i16():
    img = noise("I;16", 64, 64)
    out = resize(img, (32, 32), Resampling.BILINEAR)
    assert out.mode == "I;16"
    assert out.size == (32, 32)

# ─── alpha premultiplication ──────────────────────────────────────────────────

def test_premultiply_alpha_no_dark_fringe():
    """An RGBA image with a white object on transparent bg should not darken
    when downscaled with premultiply_alpha=True."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    # White square in the centre
    for x in range(24, 40):
        for y in range(24, 40):
            img.putpixel((x, y), (255, 255, 255, 255))
    out_pre   = resize(img, (32, 32), Resampling.BILINEAR, premultiply_alpha=True)
    out_nopre = resize(img, (32, 32), Resampling.BILINEAR, premultiply_alpha=False)
    arr_pre   = np.array(out_pre,   dtype=float)
    arr_nopre = np.array(out_nopre, dtype=float)
    # Premultiplied result should not have darker-than-correct RGB on opaque pixels
    opaque = arr_pre[:, :, 3] > 200
    if opaque.any():
        assert arr_pre[opaque, :3].mean() >= arr_nopre[opaque, :3].mean() - 10

# ─── Plan class ───────────────────────────────────────────────────────────────

def test_plan_resize():
    img = noise("RGB", 128, 128)
    plan = Plan(img.size, (64, 64), Resampling.LANCZOS, "RGB")
    out = plan.resize(img)
    assert out.size == (64, 64)
    assert out.mode == "RGB"

def test_plan_repr():
    plan = Plan((128, 128), (64, 64), Resampling.LANCZOS, "RGB")
    assert "128" in repr(plan)
    assert "64"  in repr(plan)

def test_plan_wrong_mode_raises():
    plan = Plan((128, 128), (64, 64), Resampling.LANCZOS, "RGB")
    rgba_img = noise("RGBA", 128, 128)
    with pytest.raises(Exception, match="mode"):
        plan.resize(rgba_img)

def test_plan_reuse():
    """Same plan should produce identical output when called twice."""
    img = noise("RGB", 128, 128)
    plan = Plan(img.size, (64, 64), Resampling.LANCZOS, "RGB")
    out1 = np.array(plan.resize(img))
    out2 = np.array(plan.resize(img))
    np.testing.assert_array_equal(out1, out2)

# ─── threading ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("workers", [1, 2, 0])
def test_workers(workers):
    img = noise("RGB", 256, 256)
    out = resize(img, (128, 128), Resampling.LANCZOS, workers=workers)
    assert out.size == (128, 128)

# ─── error handling ───────────────────────────────────────────────────────────

def test_unsupported_mode_raises():
    img = Image.new("CMYK", (64, 64), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="Unsupported"):
        resize(img, (32, 32))

def test_zero_size_raises():
    img = noise("RGB", 64, 64)
    with pytest.raises(ValueError):
        resize(img, (0, 32))
