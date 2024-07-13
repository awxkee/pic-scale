/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
use crate::bartlett::{bartlett, bartlett_hann};
use crate::bc_spline::{
    b_spline, catmull_rom, hermite_spline, mitchell_netravalli, robidoux, robidoux_sharp,
};
use crate::bilinear::bilinear;
use crate::blackman::blackman;
use crate::bohman::bohman;
use crate::cubic::{bicubic_spline, cubic_spline};
use crate::gaussian::gaussian;
use crate::hann::{hamming, hann, hanning};
use crate::jinc_f32;
use crate::kaiser::kaiser;
use crate::lagrange::{lagrange2, lagrange3};
use crate::lanczos::{
    lanczos2, lanczos2_jinc, lanczos3, lanczos3_jinc, lanczos4, lanczos4_jinc, lanczos6,
    lanczos6_jinc,
};
use crate::quadric::quadric;
use crate::sinc::sinc;
use crate::sphinx::sphinx;
use crate::spline_n::{spline16, spline36, spline64};
use crate::welch::welch;
use num_traits::AsPrimitive;

#[inline(always)]
pub(crate) fn box_weight<V: Copy + 'static>(_: V) -> V
where
    f32: AsPrimitive<V>,
{
    1f32.as_()
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
    Ginseng,
    EwaGinseng,
    EwaLanczosSharp,
    EwaLanczos4Sharpest,
    EwaLanczosSoft,
    HaasnSoft,
    Lagrange2,
    Lagrange3,
    Lanczos6,
    Lanczos6Jinc,
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
            38 => ResamplingFunction::Ginseng,
            39 => ResamplingFunction::EwaGinseng,
            40 => ResamplingFunction::EwaLanczosSharp,
            41 => ResamplingFunction::EwaLanczos4Sharpest,
            42 => ResamplingFunction::EwaLanczosSoft,
            43 => ResamplingFunction::HaasnSoft,
            44 => ResamplingFunction::Lagrange2,
            45 => ResamplingFunction::Lagrange3,
            46 => ResamplingFunction::Lanczos6,
            47 => ResamplingFunction::Lanczos6Jinc,
            _ => ResamplingFunction::Bilinear,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResamplingWindow {
    pub(crate) window: fn(f32) -> f32,
    pub(crate) window_size: f32,
    pub(crate) blur: f32,
    pub(crate) taper: f32,
}

impl ResamplingWindow {
    fn new(window: fn(f32) -> f32, window_size: f32, blur: f32, taper: f32) -> ResamplingWindow {
        ResamplingWindow {
            window,
            window_size,
            blur,
            taper,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResamplingFilter {
    pub kernel: fn(f32) -> f32,
    pub window: Option<ResamplingWindow>,
    pub min_kernel_size: f32,
    pub is_ewa: bool,
    pub is_resizable_kernel: bool,
}

impl ResamplingFilter {
    fn new(kernel: fn(f32) -> f32, min_kernel_size: f32, is_ewa: bool) -> ResamplingFilter {
        ResamplingFilter {
            kernel,
            window: None,
            min_kernel_size,
            is_ewa,
            is_resizable_kernel: true,
        }
    }

    fn new_with_window(
        kernel: fn(f32) -> f32,
        window: ResamplingWindow,
        min_kernel_size: f32,
        is_ewa: bool,
    ) -> ResamplingFilter {
        ResamplingFilter {
            kernel,
            window: Some(window),
            min_kernel_size,
            is_ewa,
            is_resizable_kernel: true,
        }
    }

    fn new_with_fixed_kernel(
        kernel: fn(f32) -> f32,
        min_kernel_size: f32,
        is_ewa: bool,
    ) -> ResamplingFilter {
        ResamplingFilter {
            kernel,
            window: None,
            min_kernel_size,
            is_ewa,
            is_resizable_kernel: false,
        }
    }
}

const JINC_R3: f32 = 3.2383154841662362f32;
const JINC_R4: f32 = 4.2410628637960699f32;

impl ResamplingFunction {
    pub fn get_resampling_filter(&self) -> ResamplingFilter {
        return match self {
            ResamplingFunction::Bilinear => ResamplingFilter::new(bilinear, 2f32, false),
            ResamplingFunction::Nearest => {
                // Just a stab for nearest
                ResamplingFilter::new(bilinear, 1f32, false)
            }
            ResamplingFunction::Cubic => ResamplingFilter::new(cubic_spline::<f32>, 2f32, false),
            ResamplingFunction::MitchellNetravalli => {
                ResamplingFilter::new(mitchell_netravalli::<f32>, 2f32, false)
            }
            ResamplingFunction::Lanczos3 => ResamplingFilter::new(lanczos3, 3f32, false),
            ResamplingFunction::CatmullRom => {
                ResamplingFilter::new(catmull_rom::<f32>, 2f32, false)
            }
            ResamplingFunction::Hermite => {
                ResamplingFilter::new(hermite_spline::<f32>, 2f32, false)
            }
            ResamplingFunction::BSpline => ResamplingFilter::new(b_spline::<f32>, 2f32, false),
            ResamplingFunction::Hann => ResamplingFilter::new(hann, 3f32, false),
            ResamplingFunction::Bicubic => {
                ResamplingFilter::new(bicubic_spline::<f32>, 3f32, false)
            }
            ResamplingFunction::Lanczos4 => ResamplingFilter::new(lanczos4, 4f32, false),
            ResamplingFunction::Lanczos2 => ResamplingFilter::new(lanczos2, 2f32, false),
            ResamplingFunction::Hamming => ResamplingFilter::new(hamming, 1f32, false),
            ResamplingFunction::Hanning => ResamplingFilter::new(hanning, 1f32, false),
            ResamplingFunction::EwaHanning => ResamplingFilter::new_with_window(
                jinc_f32,
                ResamplingWindow::new(hanning, 1f32, 0f32, 0f32),
                1f32,
                true,
            ),
            ResamplingFunction::Welch => ResamplingFilter::new(welch, 1f32, false),
            ResamplingFunction::Quadric => ResamplingFilter::new(quadric, 1.5f32, false),
            ResamplingFunction::EwaQuadric => ResamplingFilter::new(quadric, 1.5f32, true),
            ResamplingFunction::Gaussian => ResamplingFilter::new(gaussian, 2f32, false),
            ResamplingFunction::Sphinx => ResamplingFilter::new(sphinx, 2f32, false),
            ResamplingFunction::Bartlett => ResamplingFilter::new(bartlett, 1f32, false),
            ResamplingFunction::Robidoux => ResamplingFilter::new(robidoux::<f32>, 2f32, false),
            ResamplingFunction::EwaRobidoux => ResamplingFilter::new(robidoux::<f32>, 2f32, true),
            ResamplingFunction::RobidouxSharp => {
                ResamplingFilter::new(robidoux_sharp::<f32>, 2f32, false)
            }
            ResamplingFunction::EwaRobidouxSharp => {
                ResamplingFilter::new(robidoux_sharp::<f32>, 2f32, true)
            }
            ResamplingFunction::Spline16 => {
                ResamplingFilter::new_with_fixed_kernel(spline16, 2f32, false)
            }
            ResamplingFunction::Spline36 => {
                ResamplingFilter::new_with_fixed_kernel(spline36, 4f32, false)
            }
            ResamplingFunction::Spline64 => {
                ResamplingFilter::new_with_fixed_kernel(spline64, 6f32, false)
            }
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
            ResamplingFunction::Ginseng => ResamplingFilter::new_with_window(
                sinc,
                ResamplingWindow::new(jinc_f32, 3f32, 1f32, 0f32),
                3f32,
                false,
            ),
            ResamplingFunction::EwaGinseng => ResamplingFilter::new_with_window(
                sinc,
                ResamplingWindow::new(jinc_f32, JINC_R3, 1f32, 0f32),
                3f32,
                true,
            ),
            ResamplingFunction::EwaLanczosSharp => ResamplingFilter::new_with_window(
                jinc_f32,
                ResamplingWindow::new(jinc_f32, JINC_R3, 0.9812505837223707f32, 0f32),
                3f32,
                true,
            ),
            ResamplingFunction::EwaLanczos4Sharpest => ResamplingFilter::new_with_window(
                jinc_f32,
                ResamplingWindow::new(jinc_f32, JINC_R4, 0.8845120932605005f32, 0f32),
                4f32,
                true,
            ),
            ResamplingFunction::EwaLanczosSoft => ResamplingFilter::new_with_window(
                jinc_f32,
                ResamplingWindow::new(jinc_f32, JINC_R3, 1.0164667662867047f32, 0f32),
                3f32,
                true,
            ),
            ResamplingFunction::HaasnSoft => ResamplingFilter::new_with_window(
                jinc_f32,
                ResamplingWindow::new(hanning, 3f32, 1.11f32, 0f32),
                3f32,
                false,
            ),
            ResamplingFunction::Lagrange2 => ResamplingFilter::new(lagrange2, 2f32, false),
            ResamplingFunction::Lagrange3 => ResamplingFilter::new(lagrange3, 3f32, false),
            ResamplingFunction::Lanczos6Jinc => ResamplingFilter::new(lanczos6_jinc, 6f32, false),
            ResamplingFunction::Lanczos6 => ResamplingFilter::new(lanczos6, 6f32, false),
        };
    }
}
