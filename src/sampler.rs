use crate::{jinc};

#[inline(always)]
pub fn bc_spline(d: f32, b: f32, c: f32) -> f32 {
    let mut x = d;
    if x < 0.0f32 {
        x = -x;
    }
    let dp = x * x;
    let tp = dp * x;
    if x < 1f32 {
        return ((12f32 - 9f32 * b - 6f32 * c) * tp + (-18f32 + 12f32 * b + 6f32 * c) * dp + (6f32 - 2f32 * b)) *
            (1f32 / 6f32);
    } else if x < 2f32 {
        return ((-b - 6f32 * c) * tp + (6f32 * b + 30f32 * c) * dp + (-12f32 * b - 48f32 * c) * x +
            (8f32 * b + 24f32 * c)) * (1f32 / 6f32);
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

// Only for testing, fast_sinc using Taylor expansion are in use
#[allow(dead_code)]
pub fn sinc(x: f32) -> f32 {
    return if x == 0.0 {
        1f32
    } else {
        x.sin() / x
    };
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
    return if x < 1f32 {
        1.0f32 - x
    } else {
        0f32
    };
}

#[inline(always)]
pub fn hann(x: f32) -> f32 {
    const LENGTH: f32 = 3.0f32;
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
    if x == 0.0f32 {
        1.0f32
    } else if x >= 1.0f32 {
        0.0f32
    } else {
        let x = x * std::f32::consts::PI;
        (0.54f32 + 0.46f32 * x.cos()) * x.sin() / x
    }
}

#[derive(Copy, Clone, Default, Ord, PartialOrd, Eq, PartialEq)]
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
    Lanczos2,
    Lanczos3,
    Lanczos4,
    Lanczos2Jinc,
    Lanczos3Jinc,
    Lanczos4Jinc,
}

#[derive(Copy, Clone)]
pub struct ResamplingFilter {
    pub function: fn(f32) -> f32,
    pub min_kernel_size: u32,
}

impl ResamplingFilter {
    fn new(func: fn(f32) -> f32, min_kernel_size: u32) -> ResamplingFilter {
        ResamplingFilter { function: func, min_kernel_size }
    }
}

impl ResamplingFunction {
    pub fn get_resampling_filter(&self) -> ResamplingFilter {
        return match self {
            ResamplingFunction::Bilinear => {
                ResamplingFilter::new(bilinear, 2)
            }
            ResamplingFunction::Nearest => {
                // Just a stab for nearest
                ResamplingFilter::new(bilinear, 1)
            }
            ResamplingFunction::Cubic => {
                ResamplingFilter::new(cubic_spline, 2)
            }
            ResamplingFunction::MitchellNetravalli => {
                ResamplingFilter::new(mitchell_netravalli, 2)
            }
            ResamplingFunction::Lanczos3 => {
                ResamplingFilter::new(lanczos3, 3)
            }
            ResamplingFunction::CatmullRom => {
                ResamplingFilter::new(catmull_rom, 2)
            }
            ResamplingFunction::Hermite => {
                ResamplingFilter::new(hermite_spline, 2)
            }
            ResamplingFunction::BSpline => {
                ResamplingFilter::new(b_spline, 2)
            }
            ResamplingFunction::Hann => {
                ResamplingFilter::new(hann, 3)
            }
            ResamplingFunction::Bicubic => {
                ResamplingFilter::new(bicubic_spline, 3)
            }
            ResamplingFunction::Lanczos4 => {
                ResamplingFilter::new(lanczos4, 4)
            }
            ResamplingFunction::Lanczos2 => {
                ResamplingFilter::new(lanczos2, 2)
            }
            ResamplingFunction::Hamming => {
                ResamplingFilter::new(hamming, 1)
            }
            ResamplingFunction::Lanczos2Jinc => {
                ResamplingFilter::new(lanczos2_jinc, 2)
            }
            ResamplingFunction::Lanczos3Jinc => {
                ResamplingFilter::new(lanczos3_jinc, 3)
            }
            ResamplingFunction::Lanczos4Jinc => {
                ResamplingFilter::new(lanczos4_jinc, 4)
            }
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