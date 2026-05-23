/*
 * Copyright (c) Radzivon Bartoshyk 05/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// src/lib.rs — PyO3 bindings for pic-scale
//
// API design:
//   from pic_scale import Resampling, resize, Plan
//
//   # Pillow-style drop-in
//   out = resize(image, (new_w, new_h), Resampling.LANCZOS)
//
//   # Pre-planned (reuse filter weights across frames)
//   plan = Plan(image.size, (new_w, new_h), Resampling.LANCZOS)
//   out  = plan.resize(image)
//
// Supported Pillow modes: L, LA, RGB, RGBA, I;16, F
// All other modes should be converted by the caller before passing in.
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn ps_err(e: impl std::fmt::Debug) -> PyErr {
    PyRuntimeError::new_err(format!("pic-scale error: {e:?}"))
}

/// Resampling filter — mirrors ``PIL.Image.Resampling``.
///
/// All filters available in Pillow are present, plus additional high-quality
/// options from pic-scale (Mitchell, Catmull-Rom, Hann, …).
#[allow(non_camel_case_types)]
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum Resampling {
    /// Nearest-neighbour — fastest, blocky.
    NEAREST = 0,
    /// Lanczos / sinc (Pillow default for high quality). Window = 3.
    LANCZOS = 1,
    /// Bilinear interpolation.
    BILINEAR = 2,
    /// Bicubic interpolation (Keys cubic, a = −0.5).
    BICUBIC = 3,
    /// Box / area average — best for downscaling.
    BOX = 4,
    /// Hamming window sinc.
    HAMMING = 5,
    /// Mitchell-Netravali cubic (B=1/3, C=1/3) — good balance of blur/ringing.
    MITCHELL = 6,
    /// Catmull-Rom cubic — sharper than Mitchell.
    CATMULL_ROM = 7,
    /// Lanczos with window = 2 (faster, slightly softer).
    LANCZOS2 = 8,
    /// Lanczos with window = 4 (slower, very sharp).
    LANCZOS4 = 9,
    /// Gaussian blur kernel.
    GAUSSIAN = 10,
    /// Hann window sinc.
    HANN = 11,
}

impl Resampling {
    fn to_pic_scale(self) -> ResamplingFunction {
        match self {
            Resampling::NEAREST => ResamplingFunction::Nearest,
            Resampling::LANCZOS => ResamplingFunction::Lanczos3,
            Resampling::BILINEAR => ResamplingFunction::Bilinear,
            Resampling::BICUBIC => ResamplingFunction::Bicubic,
            Resampling::BOX => ResamplingFunction::Box,
            Resampling::HAMMING => ResamplingFunction::Hamming,
            Resampling::MITCHELL => ResamplingFunction::MitchellNetravalli,
            Resampling::CATMULL_ROM => ResamplingFunction::CatmullRom,
            Resampling::LANCZOS2 => ResamplingFunction::Lanczos2,
            Resampling::LANCZOS4 => ResamplingFunction::Lanczos4,
            Resampling::GAUSSIAN => ResamplingFunction::Gaussian,
            Resampling::HANN => ResamplingFunction::Hann,
        }
    }
}

/// Parse a Pillow image's `mode` string and return `(channels, is_float, is_u16)`.
fn parse_mode(mode: &str) -> PyResult<(usize, bool, bool)> {
    match mode {
        "L" => Ok((1, false, false)),
        "LA" => Ok((2, false, false)),
        "RGB" => Ok((3, false, false)),
        "RGBA" => Ok((4, false, false)),
        "I;16" | "I;16B" => Ok((1, false, true)),
        "F" => Ok((1, true, false)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported Pillow mode '{other}'. \
             Supported modes: L, LA, RGB, RGBA, I;16, F. \
             Convert with image.convert('RGB') etc. before resizing."
        ))),
    }
}

/// Extract raw bytes from a Pillow Image object via `image.tobytes()`.
fn image_to_bytes<'py>(image: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
    let bytes_obj = image.call_method0("tobytes")?;
    let bytes: &Bound<'py, PyBytes> = bytes_obj.cast()?;
    Ok(bytes.as_bytes().to_vec())
}

/// Read (width, height) from a Pillow Image's `.size` tuple.
fn image_size(image: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let size = image.getattr("size")?;
    let w: usize = size.get_item(0)?.extract()?;
    let h: usize = size.get_item(1)?.extract()?;
    Ok((w, h))
}

/// Read mode string from a Pillow Image.
fn image_mode(image: &Bound<'_, PyAny>) -> PyResult<String> {
    image.getattr("mode")?.extract()
}

/// Reconstruct a Pillow Image from raw bytes using `PIL.Image.frombytes`.
fn bytes_to_image<'py>(
    py: Python<'py>,
    mode: &str,
    width: usize,
    height: usize,
    data: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    let pil = py.import("PIL.Image")?;
    let size = (width, height);
    let bytes = PyBytes::new(py, data);
    pil.call_method1("frombytes", (mode, size, bytes))
}

#[allow(clippy::too_many_arguments)]
fn do_resize(
    raw: &[u8],
    mode: &str,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    filter: ResamplingFunction,
    premultiply_alpha: bool,
    threading: ThreadingPolicy,
    quality: WorkloadStrategy,
) -> PyResult<Vec<u8>> {
    let src_size = ImageSize::new(src_w, src_h);
    let dst_size = ImageSize::new(dst_w, dst_h);

    let scaler = Scaler::new(filter)
        .set_threading_policy(threading)
        .set_workload_strategy(quality);

    let (channels, is_float, is_u16) = parse_mode(mode)?;

    // ── u8 modes ─────────────────────────────────────────────────────────────
    if !is_float && !is_u16 {
        return match channels {
            1 => {
                let store = ImageStore::<u8, 1>::from_slice(raw, src_w, src_h).map_err(ps_err)?;
                let mut dst = ImageStoreMut::<u8, 1>::alloc(dst_w, dst_h);
                let plan = scaler
                    .plan_planar_resampling(src_size, dst_size)
                    .map_err(ps_err)?;
                plan.resample(&store, &mut dst).map_err(ps_err)?;
                Ok(dst.buffer.borrow().to_vec())
            }
            2 => {
                let store = ImageStore::<u8, 2>::from_slice(raw, src_w, src_h).map_err(ps_err)?;
                let mut dst = ImageStoreMut::<u8, 2>::alloc(dst_w, dst_h);
                let plan = scaler
                    .plan_gray_alpha_resampling(src_size, dst_size, premultiply_alpha)
                    .map_err(ps_err)?;
                plan.resample(&store, &mut dst).map_err(ps_err)?;
                Ok(dst.buffer.borrow().to_vec())
            }
            3 => {
                let store = ImageStore::<u8, 3>::from_slice(raw, src_w, src_h).map_err(ps_err)?;
                let mut dst = ImageStoreMut::<u8, 3>::alloc(dst_w, dst_h);
                let plan = scaler
                    .plan_rgb_resampling(src_size, dst_size)
                    .map_err(ps_err)?;
                plan.resample(&store, &mut dst).map_err(ps_err)?;
                Ok(dst.buffer.borrow().to_vec())
            }
            4 => {
                let store = ImageStore::<u8, 4>::from_slice(raw, src_w, src_h).map_err(ps_err)?;
                let mut dst = ImageStoreMut::<u8, 4>::alloc(dst_w, dst_h);
                let plan = scaler
                    .plan_rgba_resampling(src_size, dst_size, premultiply_alpha)
                    .map_err(ps_err)?;
                plan.resample(&store, &mut dst).map_err(ps_err)?;
                Ok(dst.buffer.borrow().to_vec())
            }
            _ => unreachable!(),
        };
    }

    // ── u16 mode (I;16) ──────────────────────────────────────────────────────
    if is_u16 {
        // Pillow stores I;16 as little-endian u16 bytes
        let pixels: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let store = ImageStore::<u16, 1>::from_slice(&pixels, src_w, src_h).map_err(ps_err)?;
        let mut dst = ImageStoreMut::<u16, 1>::alloc_with_depth(dst_w, dst_h, 16);
        let plan = scaler
            .plan_planar_resampling16(src_size, dst_size, 16)
            .map_err(ps_err)?;
        plan.resample(&store, &mut dst).map_err(ps_err)?;
        let out_u16 = dst.buffer.borrow();
        let out_bytes: Vec<u8> = out_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        return Ok(out_bytes);
    }

    // ── f32 mode (F) ─────────────────────────────────────────────────────────
    // Pillow stores F as little-endian f32 bytes
    let pixels: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let store = ImageStore::<f32, 1>::from_slice(&pixels, src_w, src_h).map_err(ps_err)?;
    let mut dst = ImageStoreMut::<f32, 1>::alloc(dst_w, dst_h);
    let plan = scaler
        .plan_planar_resampling_f32(src_size, dst_size)
        .map_err(ps_err)?;
    plan.resample(&store, &mut dst).map_err(ps_err)?;
    let out_f32 = dst.buffer.borrow();
    let out_bytes: Vec<u8> = out_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();
    Ok(out_bytes)
}

/// Pre-planned resampler — computes filter weights once, reuses across frames.
///
/// Parameters
/// ----------
/// src_size : tuple[int, int]
///     Source image ``(width, height)``.
/// dst_size : tuple[int, int]
///     Target ``(width, height)``.
/// resampling : Resampling
///     Filter to use.
/// mode : str
///     Pillow image mode (``"L"``, ``"LA"``, ``"RGB"``, ``"RGBA"``, ``"I;16"``, ``"F"``).
/// premultiply_alpha : bool, optional
///     Pre-multiply alpha before resampling (only affects ``LA`` and ``RGBA``).
///     Default ``True``.
/// workers : int, optional
///     Number of threads. ``0`` = adaptive. Default ``1``.
///
/// Example
/// -------
/// ::
///
///     plan = Plan((1920, 1080), (960, 540), Resampling.LANCZOS, "RGB")
///     for frame in video_frames:
///         small = plan.resize(frame)
#[pyclass]
pub struct Plan {
    mode: String,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    filter: ResamplingFunction,
    premultiply_alpha: bool,
    threading: ThreadingPolicy,
    quality: WorkloadStrategy,
}

#[pymethods]
impl Plan {
    #[new]
    #[pyo3(signature = (
        src_size,
        dst_size,
        resampling,
        mode,
        premultiply_alpha = true,
        workers = 1,
    ))]
    fn new(
        src_size: (usize, usize),
        dst_size: (usize, usize),
        resampling: Resampling,
        mode: &str,
        premultiply_alpha: bool,
        workers: usize,
    ) -> PyResult<Self> {
        parse_mode(mode)?; // validate early
        let threading = match workers {
            0 => ThreadingPolicy::Adaptive,
            1 => ThreadingPolicy::Single,
            n => ThreadingPolicy::Fixed(n),
        };
        Ok(Plan {
            mode: mode.to_string(),
            src_w: src_size.0,
            src_h: src_size.1,
            dst_w: dst_size.0,
            dst_h: dst_size.1,
            filter: resampling.to_pic_scale(),
            premultiply_alpha,
            threading,
            quality: WorkloadStrategy::PreferSpeed,
        })
    }

    /// Resize a Pillow Image using the pre-computed plan.
    ///
    /// The image mode must match the mode this plan was created with.
    /// Returns a new Pillow Image.
    fn resize<'py>(
        &self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mode = image_mode(image)?;
        if mode != self.mode {
            return Err(PyValueError::new_err(format!(
                "Plan was created for mode '{}' but image has mode '{mode}'",
                self.mode
            )));
        }
        let raw = image_to_bytes(image)?;
        let out = py.detach(|| {
            do_resize(
                &raw,
                &self.mode,
                self.src_w,
                self.src_h,
                self.dst_w,
                self.dst_h,
                self.filter,
                self.premultiply_alpha,
                self.threading,
                self.quality,
            )
        })?;
        bytes_to_image(py, &self.mode, self.dst_w, self.dst_h, &out)
    }

    #[getter]
    fn src_size(&self) -> (usize, usize) {
        (self.src_w, self.src_h)
    }

    #[getter]
    fn dst_size(&self) -> (usize, usize) {
        (self.dst_w, self.dst_h)
    }

    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }

    fn __repr__(&self) -> String {
        format!(
            "pic_scale.Plan(src_size=({}, {}), dst_size=({}, {}), mode='{}')",
            self.src_w, self.src_h, self.dst_w, self.dst_h, self.mode
        )
    }
}

/// Resize a Pillow Image using pic-scale's high-performance SIMD engine.
///
/// This is a drop-in replacement for ``PIL.Image.Image.resize``.
///
/// Parameters
/// ----------
/// image : PIL.Image.Image
///     Source image. Supported modes: ``L``, ``LA``, ``RGB``, ``RGBA``,
///     ``I;16``, ``F``. Convert other modes first (e.g. ``image.convert("RGB")``).
/// size : tuple[int, int]
///     Target ``(width, height)``.
/// resampling : Resampling, optional
///     Filter. Default ``Resampling.LANCZOS``.
/// premultiply_alpha : bool, optional
///     Pre-multiply alpha before resampling for ``LA`` / ``RGBA`` images.
///     Prevents dark fringing around transparent edges. Default ``True``.
/// workers : int, optional
///     Thread count. ``0`` = adaptive (uses all cores). Default ``1``.
///
/// Returns
/// -------
/// PIL.Image.Image
///     Resized image in the same mode as the input.
///
/// Example
/// -------
/// ::
///
///     from PIL import Image
///     from pic_scale import resize, Resampling
///
///     img = Image.open("photo.jpg")
///     small = resize(img, (800, 600), Resampling.LANCZOS)
///     small.save("small.jpg")
#[pyfunction]
#[pyo3(signature = (
    image,
    size,
    resampling = Resampling::LANCZOS,
    premultiply_alpha = true,
    workers = 1,
))]
fn resize<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    size: (usize, usize),
    resampling: Resampling,
    premultiply_alpha: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mode = image_mode(image)?;
    let (src_w, src_h) = image_size(image)?;
    let (dst_w, dst_h) = size;

    if dst_w == 0 || dst_h == 0 {
        return Err(PyValueError::new_err(
            "Target size must be > 0 in both dimensions",
        ));
    }

    let threading = match workers {
        0 => ThreadingPolicy::Adaptive,
        1 => ThreadingPolicy::Single,
        n => ThreadingPolicy::Fixed(n),
    };

    let raw = image_to_bytes(image)?;
    let out = py.detach(|| {
        do_resize(
            &raw,
            &mode,
            src_w,
            src_h,
            dst_w,
            dst_h,
            resampling.to_pic_scale(),
            premultiply_alpha,
            threading,
            WorkloadStrategy::PreferSpeed,
        )
    })?;
    bytes_to_image(py, &mode, dst_w, dst_h, &out)
}

#[pymodule]
fn _pic_scale(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Resampling>()?;
    m.add_class::<Plan>()?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
