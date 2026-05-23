"""
pic-scale — High-performance image resampling for Python
=========================================================

Drop-in replacement for Pillow's ``Image.resize``.

Quick start
-----------
::

    from PIL import Image
    from pic_scale import resize, Resampling

    img = Image.open("photo.jpg")

    # One-off resize (Pillow drop-in)
    small = resize(img, (800, 600), Resampling.LANCZOS)

    # Pre-planned — weights computed once, reused across many frames
    plan = Plan(img.size, (800, 600), Resampling.LANCZOS, "RGB")
    for frame in frames:
        out = plan.resize(frame)

Supported modes
---------------
``L``, ``LA``, ``RGB``, ``RGBA``, ``I;16``, ``F``

Convert other Pillow modes first::

    img.convert("RGB")
"""

from __future__ import annotations

from pic_scale._pic_scale import (  # noqa: F401
    Plan,
    Resampling,
    resize,
)

try:
    from pic_scale._pic_scale import __version__
except ImportError:
    __version__ = "0.0.0+dev"

__all__ = ["resize", "Resampling", "Plan"]
