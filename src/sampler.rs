/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::jinc;

#[inline(always)]
pub fn bc_spline(d: f32, b: f32, c: f32) -> f32 {
    let mut x = d;
    if x < 0.0f32 {
        x = -x;
    }
    let dp = x * x;
    let tp = dp * x;
    if x < 1f32 {
        return ((12f32 - 9f32 * b - 6f32 * c) * tp
            + (-18f32 + 12f32 * b + 6f32 * c) * dp
            + (6f32 - 2f32 * b))
            * (1f32 / 6f32);
    } else if x < 2f32 {
        return ((-b - 6f32 * c) * tp
            + (6f32 * b + 30f32 * c) * dp
            + (-12f32 * b - 48f32 * c) * x
            + (8f32 * b + 24f32 * c))
            * (1f32 / 6f32);
    }
    return 0f32;
}

#[inline(always)]
pub fn cubic_spline(d: f32) -> f32 {
    let mut x = d;
    if x < 0f32 {
        x = -x;
    }
    if x < 1f32 {
        return (4f32 + x * x * (3f32 * x - 6f32)) * (1f32 / 6f32);
    } else if x < 2f32 {
        return (8f32 + x * (-12f32 + x * (6f32 - x))) * (1f32 / 6f32);
    }
    return 0f32;
}

#[inline(always)]
pub fn bicubic_spline(d: f32) -> f32 {
    let x = d;
    let a = -0.5;
    let modulo = x.abs();
    if modulo >= 2f32 {
        return 0f32;
    }
    let floatd = modulo * modulo;
    let triplet = floatd * modulo;
    if modulo <= 1f32 {
        return (a + 2f32) * triplet - (a + 3f32) * floatd + 1f32;
    }
    return a * triplet - 5f32 * a * floatd + 8f32 * a * modulo - 4f32 * a;
}

#[inline(always)]
pub fn hermite_spline(x: f32) -> f32 {
    return bc_spline(x, 0f32, 0f32);
}

#[inline(always)]
pub fn b_spline(x: f32) -> f32 {
    return bc_spline(x, 1f32, 0f32);
}

#[inline(always)]
pub fn mitchell_netravalli(x: f32) -> f32 {
    return bc_spline(x, 1f32 / 3f32, 1f32 / 3f32);
}

#[inline(always)]
pub fn catmull_rom(x: f32) -> f32 {
    return bc_spline(x, 0f32, 0.5f32);
}

#[inline(always)]
pub fn robidoux(x: f32) -> f32 {
    return bc_spline(
        x,
        12f32 / (19f32 + 9f32 * std::f32::consts::SQRT_2),
        13f32 / (58f32 + 216f32 * std::f32::consts::SQRT_2),
    );
}

#[inline(always)]
pub fn robidoux_sharp(x: f32) -> f32 {
    return bc_spline(
        x,
        6f32 / (13f32 + 7f32 * std::f32::consts::SQRT_2),
        7f32 / (2f32 + 12f32 * std::f32::consts::SQRT_2),
    );
}

#[allow(dead_code)]
pub fn sinc(x: f32) -> f32 {
    return if x == 0.0 { 1f32 } else { x.sin() / x };
}

#[inline(always)]
pub fn bartlett(x: f32) -> f32 {
    if x >= 0f32 && x <= 1f32 {
        return 2f32 * x;
    }
    return 2f32 - 2f32 * x;
}

#[inline(always)]
pub fn lanczos_jinc(x: f32, a: f32) -> f32 {
    let scale_a: f32 = 1f32 / a;
    if x == 0f32 || x > 16.247661874700962f32 {
        return 0f32;
    }
    if x.abs() < a {
        let d = std::f32::consts::PI * x;
        return (jinc(d as f64) * jinc((d * scale_a) as f64)) as f32;
    }
    return 0f32;
}

#[inline(always)]
pub fn lanczos3_jinc(x: f32) -> f32 {
    const A: f32 = 3f32;
    lanczos_jinc(x, A)
}

#[inline(always)]
pub fn lanczos2_jinc(x: f32) -> f32 {
    const A: f32 = 2f32;
    lanczos_jinc(x, A)
}

#[inline(always)]
pub fn lanczos4_jinc(x: f32) -> f32 {
    const A: f32 = 4f32;
    lanczos_jinc(x, A)
}

#[inline(always)]
pub fn lanczos_sinc(x: f32, a: f32) -> f32 {
    let scale_a: f32 = 1f32 / a;
    if x.abs() < a {
        let d = std::f32::consts::PI * x;
        return sinc(d) * sinc(d * scale_a);
    }
    return 0f32;
}

#[inline(always)]
pub fn lanczos3(x: f32) -> f32 {
    const A: f32 = 3f32;
    lanczos_sinc(x, A)
}

#[inline(always)]
pub fn lanczos4(x: f32) -> f32 {
    const A: f32 = 4f32;
    lanczos_sinc(x, A)
}

#[inline(always)]
pub fn lanczos2(x: f32) -> f32 {
    const A: f32 = 2f32;
    lanczos_sinc(x, A)
}

#[inline(always)]
pub fn bilinear(x: f32) -> f32 {
    let x = x.abs();
    return if x < 1f32 { 1.0f32 - x } else { 0f32 };
}

#[inline(always)]
pub fn hann(x: f32) -> f32 {
    const LENGTH: f32 = 2.0f32;
    const SIZE: f32 = LENGTH * 2f32;
    const SIZE_SCALE: f32 = 1f32 / SIZE;
    const PART: f32 = std::f32::consts::PI / SIZE;
    if x.abs() > LENGTH {
        return 0f32;
    }
    let r = (x * PART).cos();
    let r = r * r;
    return r * SIZE_SCALE;
}

#[inline(always)]
fn hamming(x: f32) -> f32 {
    let x = x.abs();
    if x == 0f32 {
        1f32
    } else if x >= 1f32 {
        0f32
    } else {
        let x = x * std::f32::consts::PI;
        0.54f32 + 0.46f32 * x.cos()
    }
}

#[inline(always)]
fn welch(x: f32) -> f32 {
    if x == 0f32 {
        1f32
    } else if x >= 1f32 {
        0.0f32
    } else {
        1f32 - x * x
    }
}

#[inline(always)]
pub fn sphinx(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        return 1f32;
    }
    let x = x * std::f32::consts::PI;
    return 3.0 * (x.sin() - x * x.cos()) / (x * x * x);
}

#[inline(always)]
fn hanning(x: f32) -> f32 {
    let x = x.abs();
    if x == 0.0f32 {
        1.0f32
    } else if x >= 1.0f32 {
        0.0f32
    } else {
        let x = x * std::f32::consts::PI;
        0.5f32 + 0.5f32 * x.cos()
    }
}

#[inline(always)]
pub(crate) fn blackman_window(x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    0.42f32 - 0.49656062f32 * (2f32 * pi * x).cos() + 0.07684867f32 * (4f32 * pi * x).cos()
}

#[inline(always)]
pub(crate) fn blackman(x: f32) -> f32 {
    let x = x.abs();
    if x < 2.0f32 {
        sinc(x) * blackman_window(x / 2f32)
    } else {
        0f32
    }
}

#[inline(always)]
pub(crate) fn gaussian(x: f32) -> f32 {
    let sigma: f32 = 0.35f32;
    let pi = std::f32::consts::PI;
    let mut den = 2f32 * sigma * sigma;
    den *= den;
    return (1f32 / ((2f32 * pi).sqrt() * sigma)) * (-x / den).exp();
}

#[inline(always)]
pub(crate) fn quadric(x: f32) -> f32 {
    let x = x.abs();
    if x < 0.5f32 {
        return 0.75f32 - x * x;
    } else if x < 1.5f32 {
        let t = x - 1.5f32;
        return 0.5f32 * t * t;
    }
    return 0f32;
}

#[inline(always)]
pub(crate) fn spline16(x: f32) -> f32 {
    return if x < 1.0 {
        ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0
    } else {
        ((-1.0 / 3.0 * (x - 1f32) + 4.0 / 5.0) * (x - 1f32) - 7.0 / 15.0) * (x - 1f32)
    };
}

#[inline(always)]
pub(crate) fn spline36(x: f32) -> f32 {
    return if x < 1.0 {
        ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    } else if x < 2.0 {
        ((-6.0 / 11.0 * (x - 1f32) + 270.0 / 209.0) * (x - 1f32) - 156.0 / 209.0) * (x - 1f32)
    } else {
        ((1.0 / 11.0 * (x - 2f32) - 45.0 / 209.0) * (x - 2f32) + 26.0 / 209.0) * (x - 2f32)
    };
}

#[inline(always)]
pub(crate) fn spline64(x: f32) -> f32 {
    return if x < 1.0 {
        ((49.0 / 41.0 * x - 6387.0 / 2911.0) * x - 3.0 / 2911.0) * x + 1.0
    } else if x < 2.0 {
        ((-24.0 / 41.0 * (x - 1f32) + 4032.0 / 2911.0) * (x - 1f32) - 2328.0 / 2911.0) * (x - 1f32)
    } else if x < 3.0 {
        ((6.0 / 41.0 * (x - 2f32) - 1008.0 / 2911.0) * (x - 2f32) + 582.0 / 2911.0) * (x - 2f32)
    } else {
        ((-1.0 / 41.0 * (x - 3f32) + 168.0 / 2911.0) * (x - 3f32) - 97.0 / 2911.0) * (x - 3f32)
    };
}

#[inline(always)]
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let mut s = 1.0;
    let y = x * x / 4.0;
    let mut t = y;
    let mut i = 2;
    while t > 1e-12 {
        s += t;
        t *= y / (i * i) as f64;
        i += 1;
    }
    return s;
}

#[inline(always)]
pub(crate) fn bartlett_hann(x: f32) -> f32 {
    let x = x.abs();
    if x > 2f32 {
        return 0f32;
    }
    const L: f32 = 2.0f32;
    let fac = (x / (L - 1.0f32) - 0.5f32).abs();
    let w = 0.62f32 - 0.4832 * fac + 0.38f32 * (2f32 * std::f32::consts::PI * fac).cos();
    return w;
}

#[inline(always)]
pub(crate) fn kaiser(x: f32) -> f32 {
    if x > 1f32 {
        return 0f32;
    }
    let i0a = 1.0f64 / bessel_i0(6.33f64);
    return (bessel_i0(6.33f64 * (1.0 - x as f64 * x as f64).sqrt()) * i0a) as f32;
}

#[inline(always)]
pub(crate) fn box_weight(_: f32) -> f32 {
    1f32
}

#[inline(always)]
pub(crate) fn bohman(x: f32) -> f32 {
    if x < -1f32 || x > 1f32 {
        return 0f32;
    }
    let dx = std::f32::consts::PI * x.abs();
    return (1.0 - x.abs()) * dx.cos() + (1.0f32 / std::f32::consts::PI) * dx.sin();
}

#[derive(Debug, Copy, Clone, Default, Ord, PartialOrd, Eq, PartialEq)]
pub enum ResamplingFunction {
    Bilinear,
    Nearest,
    Cubic,
    #[default]
    MitchellNetravalli,
    CatmullRom,
    Hermite,
    BSpline,
    Hann,
    Bicubic,
    Hamming,
    Hanning,
    EwaHanning,
    Blackman,
    EwaBlackman,
    Welch,
    Quadric,
    EwaQuadric,
    Gaussian,
    Sphinx,
    Bartlett,
    Robidoux,
    EwaRobidoux,
    RobidouxSharp,
    EwaRobidouxSharp,
    Spline16,
    Spline36,
    Spline64,
    Kaiser,
    BartlettHann,
    Box,
    Bohman,
    Lanczos2,
    Lanczos3,
    Lanczos4,
    Lanczos2Jinc,
    Lanczos3Jinc,
    Lanczos4Jinc,
    EwaLanczos3Jinc,
}

impl From<u32> for ResamplingFunction {
    fn from(value: u32) -> Self {
        match value {
            0 => ResamplingFunction::Bilinear,
            1 => ResamplingFunction::Nearest,
            2 => ResamplingFunction::Cubic,
            3 => ResamplingFunction::MitchellNetravalli,
            4 => ResamplingFunction::CatmullRom,
            5 => ResamplingFunction::Hermite,
            6 => ResamplingFunction::BSpline,
            7 => ResamplingFunction::Hann,
            8 => ResamplingFunction::Bicubic,
            9 => ResamplingFunction::Hamming,
            10 => ResamplingFunction::Hanning,
            11 => ResamplingFunction::Blackman,
            12 => ResamplingFunction::Welch,
            13 => ResamplingFunction::Quadric,
            14 => ResamplingFunction::Gaussian,
            15 => ResamplingFunction::Sphinx,
            16 => ResamplingFunction::Bartlett,
            17 => ResamplingFunction::Robidoux,
            18 => ResamplingFunction::RobidouxSharp,
            19 => ResamplingFunction::Spline16,
            20 => ResamplingFunction::Spline36,
            21 => ResamplingFunction::Spline64,
            22 => ResamplingFunction::Kaiser,
            23 => ResamplingFunction::BartlettHann,
            24 => ResamplingFunction::Box,
            25 => ResamplingFunction::Bohman,
            26 => ResamplingFunction::Lanczos2,
            27 => ResamplingFunction::Lanczos3,
            28 => ResamplingFunction::Lanczos4,
            29 => ResamplingFunction::Lanczos2Jinc,
            30 => ResamplingFunction::Lanczos3Jinc,
            31 => ResamplingFunction::Lanczos4Jinc,
            32 => ResamplingFunction::EwaHanning,
            33 => ResamplingFunction::EwaRobidoux,
            34 => ResamplingFunction::EwaBlackman,
            35 => ResamplingFunction::EwaQuadric,
            36 => ResamplingFunction::EwaRobidouxSharp,
            37 => ResamplingFunction::EwaLanczos3Jinc,
            _ => ResamplingFunction::Bilinear,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResamplingFilter {
    pub function: fn(f32) -> f32,
    pub min_kernel_size: f32,
    pub is_ewa: bool,
}

impl ResamplingFilter {
    fn new(func: fn(f32) -> f32, min_kernel_size: f32, is_ewa: bool) -> ResamplingFilter {
        ResamplingFilter {
            function: func,
            min_kernel_size,
            is_ewa,
        }
    }
}

impl ResamplingFunction {
    pub fn get_resampling_filter(&self) -> ResamplingFilter {
        return match self {
            ResamplingFunction::Bilinear => ResamplingFilter::new(bilinear, 2f32, false),
            ResamplingFunction::Nearest => {
                // Just a stab for nearest
                ResamplingFilter::new(bilinear, 1f32, false)
            }
            ResamplingFunction::Cubic => ResamplingFilter::new(cubic_spline, 2f32, false),
            ResamplingFunction::MitchellNetravalli => {
                ResamplingFilter::new(mitchell_netravalli, 2f32, false)
            }
            ResamplingFunction::Lanczos3 => ResamplingFilter::new(lanczos3, 3f32, false),
            ResamplingFunction::CatmullRom => ResamplingFilter::new(catmull_rom, 2f32, false),
            ResamplingFunction::Hermite => ResamplingFilter::new(hermite_spline, 2f32, false),
            ResamplingFunction::BSpline => ResamplingFilter::new(b_spline, 2f32, false),
            ResamplingFunction::Hann => ResamplingFilter::new(hann, 3f32, false),
            ResamplingFunction::Bicubic => ResamplingFilter::new(bicubic_spline, 3f32, false),
            ResamplingFunction::Lanczos4 => ResamplingFilter::new(lanczos4, 4f32, false),
            ResamplingFunction::Lanczos2 => ResamplingFilter::new(lanczos2, 2f32, false),
            ResamplingFunction::Hamming => ResamplingFilter::new(hamming, 1f32, false),
            ResamplingFunction::Hanning => ResamplingFilter::new(hanning, 1f32, false),
            ResamplingFunction::EwaHanning => ResamplingFilter::new(hanning, 1f32, true),
            ResamplingFunction::Welch => ResamplingFilter::new(welch, 1f32, false),
            ResamplingFunction::Quadric => ResamplingFilter::new(quadric, 1.5f32, false),
            ResamplingFunction::EwaQuadric => ResamplingFilter::new(quadric, 1.5f32, true),
            ResamplingFunction::Gaussian => ResamplingFilter::new(gaussian, 2f32, false),
            ResamplingFunction::Sphinx => ResamplingFilter::new(sphinx, 2f32, false),
            ResamplingFunction::Bartlett => ResamplingFilter::new(bartlett, 1f32, false),
            ResamplingFunction::Robidoux => ResamplingFilter::new(robidoux, 2f32, false),
            ResamplingFunction::EwaRobidoux => ResamplingFilter::new(robidoux, 2f32, true),
            ResamplingFunction::RobidouxSharp => ResamplingFilter::new(robidoux_sharp, 2f32, false),
            ResamplingFunction::EwaRobidouxSharp => ResamplingFilter::new(robidoux_sharp, 2f32, true),
            ResamplingFunction::Spline16 => ResamplingFilter::new(spline16, 2f32, false),
            ResamplingFunction::Spline36 => ResamplingFilter::new(spline36, 2f32, false),
            ResamplingFunction::Spline64 => ResamplingFilter::new(spline64, 2f32, false),
            ResamplingFunction::Kaiser => ResamplingFilter::new(kaiser, 2f32, false),
            ResamplingFunction::BartlettHann => ResamplingFilter::new(bartlett_hann, 2f32, false),
            ResamplingFunction::Box => ResamplingFilter::new(box_weight, 0.5f32, false),
            ResamplingFunction::Bohman => ResamplingFilter::new(bohman, 2f32, false),
            ResamplingFunction::Lanczos2Jinc => ResamplingFilter::new(lanczos2_jinc, 2f32, false),
            ResamplingFunction::Lanczos3Jinc => ResamplingFilter::new(lanczos3_jinc, 3f32, false),
            ResamplingFunction::EwaLanczos3Jinc => ResamplingFilter::new(lanczos3_jinc, 3f32, true),
            ResamplingFunction::Lanczos4Jinc => ResamplingFilter::new(lanczos4_jinc, 4f32, false),
            ResamplingFunction::Blackman => ResamplingFilter::new(blackman, 2f32, false),
            ResamplingFunction::EwaBlackman => ResamplingFilter::new(blackman, 2f32, true),
        };
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use test::Bencher;

    use crate::sinc;

    use super::*;

    #[bench]
    fn bench_sinc(b: &mut Bencher) {
        b.iter(|| {
            for i in 1..100000 {
                let rebased = i as f32 / 100000f32;
                _ = sinc(rebased);
            }
        });
    }
}
