mod runner;
mod runner_f32;
mod runner_u16;

use crate::runner::{Backend, RunResult, runner_rgba_fir, runner_rgba_ps, runner_rgba_pss};
use crate::runner_f32::{runner_rgba_fir_f32, runner_rgba_ps_f32, runner_rgba_pss_f32};
use crate::runner_u16::{runner_rgba_fir16, runner_rgba_ps16, runner_rgba_pss16};
use fast_image_resize::FilterType;
use image::{DynamicImage, ImageReader};
use pic_scale::{ImageSize, ResamplingFunction};
use plotters::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SeriesKey {
    backend: Backend,
    filter: String,
}

impl SeriesKey {
    fn label(&self) -> String {
        let b = match self.backend {
            Backend::Fir => "Fir",
            Backend::PicScale => "PicScale",
            Backend::PicScaleSafe => "PicScaleSafe",
            Backend::Accelerate => "Apple Accelerate",
        };
        if self.filter.is_empty() {
            b.to_owned()
        } else {
            format!("{b} – {}", self.filter)
        }
    }

    fn is_baseline(&self) -> bool {
        self.backend == Backend::Fir
    }
}

fn backend_hue(backend: &Backend) -> f64 {
    match backend {
        Backend::Fir => 0.0,           // grey / black family
        Backend::PicScale => 210.0,    // blue family
        Backend::PicScaleSafe => 25.0, // orange family,
        Backend::Accelerate => 75.0,
    }
}

/// Convert HSL (h 0-360, s 0-1, l 0-1) → RGBColor.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> RGBColor {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h2 = h / 60.0;
    let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h2 < 1.0 {
        (c, x, 0.0)
    } else if h2 < 2.0 {
        (x, c, 0.0)
    } else if h2 < 3.0 {
        (0.0, c, x)
    } else if h2 < 4.0 {
        (0.0, x, c)
    } else if h2 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    RGBColor(
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Assign a distinct colour to every `(backend, filter)` series.
/// Filters within the same backend are spread across lightness levels.
fn assign_colors(keys: &[SeriesKey]) -> Vec<(SeriesKey, RGBColor)> {
    // Group keys by backend to count siblings
    use std::collections::HashMap;
    let mut by_backend: HashMap<&Backend, Vec<usize>> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        by_backend.entry(&k.backend).or_default().push(i);
    }

    let mut result = vec![None; keys.len()];
    for (backend, indices) in &by_backend {
        let hue = backend_hue(backend);
        let count = indices.len();
        for (slot, &idx) in indices.iter().enumerate() {
            // Spread lightness: single entry → 0.35, multiple → 0.25..0.55
            let l = if count == 1 {
                0.35
            } else {
                0.25 + 0.30 * (slot as f64 / (count - 1) as f64)
            };
            // Fir is greyscale (s=0), others saturated
            let (s, actual_hue) = if matches!(backend, Backend::Fir) {
                (0.0, 0.0)
            } else {
                (0.75, hue)
            };
            result[idx] = Some((keys[idx].clone(), hsl_to_rgb(actual_hue, s, l)));
        }
    }
    result.into_iter().flatten().collect()
}
pub fn plot(
    title: String,
    results: &[RunResult],
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;

    let mut test_cases: Vec<String> = Vec::new();
    for r in results {
        if !test_cases.contains(&r.test_case) {
            test_cases.push(r.test_case.clone());
        }
    }
    let n = test_cases.len();
    if n == 0 {
        return Err("no results to plot".into());
    }

    let mut fir_baseline: HashMap<String, f64> = HashMap::new();
    for r in results {
        if r.backend == Backend::Fir {
            fir_baseline
                .entry(r.test_case.clone())
                .or_insert(r.point_time);
        }
    }
    if fir_baseline.is_empty() {
        return Err("no Fir results found – cannot establish baseline".into());
    }

    let mut series_keys: Vec<SeriesKey> = Vec::new();
    for r in results {
        let key = SeriesKey {
            backend: r.backend.clone(),
            filter: r.filter.clone(),
        };
        if !series_keys.contains(&key) {
            if r.backend == Backend::Fir {
                series_keys.push(key);
            } else {
                let fir_pos = series_keys.iter().position(|k| k.backend == Backend::Fir);
                match fir_pos {
                    Some(i) => series_keys.insert(i, key),
                    None => series_keys.push(key),
                }
            }
        }
    }

    let colored = assign_colors(&series_keys);

    let series: Vec<(SeriesKey, RGBColor, Vec<(f64, f64)>)> = colored
        .into_iter()
        .map(|(key, colour)| {
            let mut points: Vec<(f64, f64)> = results
                .iter()
                .filter(|r| r.backend == key.backend && r.filter == key.filter)
                .filter_map(|r| {
                    let xi = test_cases.iter().position(|tc| tc == &r.test_case)? as f64;
                    let base = *fir_baseline.get(&r.test_case)?;
                    Some((xi, base / r.point_time))
                })
                .collect();
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            (key, colour, points)
        })
        .collect();

    let y_max: f64 = series
        .iter()
        .flat_map(|(_, _, pts)| pts.iter().map(|&(_, y)| y))
        .fold(1.1_f64, f64::max);
    let y_max = (y_max * 1.15).ceil().max(2.0);

    // ── 7. Canvas ────────────────────────────────────────────────────────────
    let root = SVGBackend::new(output, (1100, 620)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{title} — speed relative to Fir (higher = faster)"),
            ("sans-serif", 15).into_font(),
        )
        .margin(20)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..(n as f64 - 1.0 + 0.2), 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_labels(n)
        .x_label_formatter(&|v| {
            let idx = *v as usize;
            if idx < test_cases.len() && (*v - idx as f64).abs() < 0.01 {
                test_cases[idx].clone()
            } else {
                String::new()
            }
        })
        .x_desc(format!("{title} (src → dst)"))
        .y_desc("Speed relative to Fir (1× = same; 2× = twice as fast)")
        .y_label_formatter(&|v| format!("{:.1}×", v))
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(0.0, 1.0), (n as f64 - 1.0, 1.0)],
        Into::<ShapeStyle>::into(BLACK.mix(0.20)).stroke_width(1),
    ))?;

    for (key, colour, points) in &series {
        if points.is_empty() {
            continue;
        }

        let label = key.label();
        let stroke_w = if key.is_baseline() { 3u32 } else { 2u32 };
        let style = Into::<ShapeStyle>::into(*colour).stroke_width(stroke_w);

        chart
            .draw_series(LineSeries::new(points.clone(), style))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 18, y)], colour.stroke_width(stroke_w))
            });

        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 4, colour.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(BLACK.mix(0.3))
        .label_font(("sans-serif", 11))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    Ok(())
}

fn run_rgb_plotter(paths: &[&DynamicImage], file_name: &str, target_sizes: &[&[ImageSize]]) {
    assert_eq!(paths.len(), target_sizes.len());
    let mut results: Vec<RunResult> = vec![];
    let output = PathBuf::from(file_name);
    const ITERATIONS: usize = 10;
    for (img, &target_sizes) in paths.iter().zip(target_sizes.iter()) {
        let src_size = ImageSize::new(img.width() as usize, img.height() as usize);
        for size in target_sizes {
            let formatted_size = format!(
                "{}x{}->{}x{}",
                src_size.width, src_size.height, size.width, size.height
            );
            results.push(runner_rgba_fir(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                FilterType::Bilinear,
            ));
            results.push(runner_rgba_pss(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                pic_scale_safe::ResamplingFunction::Bilinear,
            ));
            results.push(runner_rgba_ps(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                ResamplingFunction::Bilinear,
            ));

            results.push(runner_rgba_fir(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                FilterType::Lanczos3,
            ));
            results.push(runner_rgba_ps(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                ResamplingFunction::Lanczos3,
            ));
            results.push(runner_rgba_pss(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                pic_scale_safe::ResamplingFunction::Lanczos3,
            ));
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                use crate::runner::runner_rgba_accelerate;
                results.push(runner_rgba_accelerate(
                    format!("{formatted_size}").to_string(),
                    &img,
                    *size,
                    ITERATIONS,
                    "Lanczos3".to_string(),
                ));
            }
            println!("Done with plan {}, file {file_name}", formatted_size);
        }
    }

    let arch = std::env::consts::ARCH;

    match plot(
        format!("RGBA8 image resizing {arch}").to_string(),
        &results,
        &output,
    ) {
        Ok(()) => println!("✓ Chart written to {}", output.display()),
        Err(e) => eprintln!("Error generating chart: {e}"),
    }
}

fn run_rgb_plotter_f32(paths: &[&DynamicImage], file_name: &str, target_sizes: &[&[ImageSize]]) {
    assert_eq!(paths.len(), target_sizes.len());
    let mut results: Vec<RunResult> = vec![];
    let output = PathBuf::from(file_name);
    const ITERATIONS: usize = 10;
    for (img, &target_sizes) in paths.iter().zip(target_sizes.iter()) {
        let src_size = ImageSize::new(img.width() as usize, img.height() as usize);
        for size in target_sizes {
            let formatted_size = format!(
                "{}x{}->{}x{}",
                src_size.width, src_size.height, size.width, size.height
            );
            results.push(runner_rgba_fir_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                FilterType::Bilinear,
            ));
            results.push(runner_rgba_pss_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                pic_scale_safe::ResamplingFunction::Bilinear,
            ));
            results.push(runner_rgba_ps_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                ResamplingFunction::Bilinear,
            ));

            results.push(runner_rgba_fir_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                FilterType::Lanczos3,
            ));
            results.push(runner_rgba_ps_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                ResamplingFunction::Lanczos3,
            ));
            results.push(runner_rgba_pss_f32(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                pic_scale_safe::ResamplingFunction::Lanczos3,
            ));
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                use crate::runner_f32::runner_rgba_accelerate_f32;
                results.push(runner_rgba_accelerate_f32(
                    format!("{formatted_size}").to_string(),
                    &img,
                    *size,
                    ITERATIONS,
                    "Lanczos3".to_string(),
                ));
            }
            println!("Done with plan {}, file {file_name}", formatted_size);
        }
    }

    let arch = std::env::consts::ARCH;

    match plot(
        format!("RGBA F32 image resizing {arch}").to_string(),
        &results,
        &output,
    ) {
        Ok(()) => println!("✓ Chart written to {}", output.display()),
        Err(e) => eprintln!("Error generating chart: {e}"),
    }
}

fn run_rgb_plotter16(paths: &[&DynamicImage], file_name: &str, target_sizes: &[&[ImageSize]]) {
    assert_eq!(paths.len(), target_sizes.len());
    let mut results: Vec<RunResult> = vec![];
    let output = PathBuf::from(file_name);
    const ITERATIONS: usize = 10;
    for (img, &target_sizes) in paths.iter().zip(target_sizes.iter()) {
        let src_size = ImageSize::new(img.width() as usize, img.height() as usize);
        for size in target_sizes {
            let formatted_size = format!(
                "{}x{}->{}x{}",
                src_size.width, src_size.height, size.width, size.height
            );
            results.push(runner_rgba_fir16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                FilterType::Bilinear,
            ));
            results.push(runner_rgba_pss16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                pic_scale_safe::ResamplingFunction::Bilinear,
            ));
            results.push(runner_rgba_ps16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Bilinear".to_string(),
                ResamplingFunction::Bilinear,
            ));

            results.push(runner_rgba_fir16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                FilterType::Lanczos3,
            ));
            results.push(runner_rgba_ps16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                ResamplingFunction::Lanczos3,
            ));
            results.push(runner_rgba_pss16(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "Lanczos3".to_string(),
                pic_scale_safe::ResamplingFunction::Lanczos3,
            ));
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                use crate::runner_u16::runner_rgba_accelerate16;
                results.push(runner_rgba_accelerate16(
                    format!("{formatted_size}").to_string(),
                    &img,
                    *size,
                    ITERATIONS,
                    "Lanczos3".to_string(),
                ));
            }
            println!("Done with plan {}, file {file_name}", formatted_size);
        }
    }

    let arch = std::env::consts::ARCH;

    match plot(
        format!("RGBA16 image resizing {arch}").to_string(),
        &results,
        &output,
    ) {
        Ok(()) => println!("✓ Chart written to {}", output.display()),
        Err(e) => eprintln!("Error generating chart: {e}"),
    }
}

fn run_rgb_plotter_cubic(paths: &[&DynamicImage], file_name: &str, target_sizes: &[&[ImageSize]]) {
    assert_eq!(paths.len(), target_sizes.len());
    let mut results: Vec<RunResult> = vec![];
    let output = PathBuf::from(file_name);
    const ITERATIONS: usize = 10;
    for (img, &target_sizes) in paths.iter().zip(target_sizes.iter()) {
        let src_size = ImageSize::new(img.width() as usize, img.height() as usize);
        for size in target_sizes {
            let formatted_size = format!(
                "{}x{}->{}x{}",
                src_size.width, src_size.height, size.width, size.height
            );
            results.push(runner_rgba_fir(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "MitchellNetravalli".to_string(),
                FilterType::Mitchell,
            ));
            results.push(runner_rgba_pss(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "MitchellNetravalli".to_string(),
                pic_scale_safe::ResamplingFunction::MitchellNetravalli,
            ));
            results.push(runner_rgba_ps(
                format!("{formatted_size}").to_string(),
                &img,
                *size,
                ITERATIONS,
                "MitchellNetravalli".to_string(),
                ResamplingFunction::MitchellNetravalli,
            ));
            println!("Done with plan {}, file {file_name}", formatted_size);
        }
    }

    let arch = std::env::consts::ARCH;

    match plot(
        format!("RGBA8 image resizing {arch}").to_string(),
        &results,
        &output,
    ) {
        Ok(()) => println!("✓ Chart written to {}", output.display()),
        Err(e) => eprintln!("Error generating chart: {e}"),
    }
}

fn make_sizes(img: &DynamicImage) -> Vec<ImageSize> {
    let src_size = ImageSize::new(img.width() as usize, img.height() as usize);
    let mut target_sizes = vec![];
    for i in 0..5 {
        if i == 0 {
            target_sizes.push(ImageSize::new(src_size.width - 1, src_size.height - 1));
        } else {
            target_sizes.push(ImageSize::new(
                src_size.width / (i * 2),
                src_size.height / (i * 2),
            ));
        }
    }
    target_sizes
}

fn run_image(img_suffix: &str, path: &str, large_path: &str, fhd_path: &str) {
    let mut img_medium = ImageReader::open(path).unwrap().decode().unwrap();

    if !matches!(img_medium, DynamicImage::ImageRgba8(_)) {
        img_medium = DynamicImage::ImageRgba8(img_medium.to_rgba8());
    }

    let mut img_large = ImageReader::open(large_path).unwrap().decode().unwrap();

    if !matches!(img_large, DynamicImage::ImageRgba8(_)) {
        img_large = DynamicImage::ImageRgba8(img_large.to_rgba8());
    }

    let mut fhd_image = ImageReader::open(fhd_path).unwrap().decode().unwrap();

    if !matches!(fhd_image, DynamicImage::ImageRgba8(_)) {
        fhd_image = DynamicImage::ImageRgba8(fhd_image.to_rgba8());
    }

    let target_sizes = make_sizes(&img_medium);
    let target_sizes_large = make_sizes(&img_large);
    let target_sizes_fhd = make_sizes(&fhd_image);

    let arch = std::env::consts::ARCH;
    run_rgb_plotter(
        &[&img_large, &img_medium],
        format!("rgba8_big_{arch}.svg").as_str(),
        &[&target_sizes_large, &target_sizes],
    );
    run_rgb_plotter(
        &[&img_medium, &fhd_image],
        format!("rgba8_medium_{arch}.svg").as_str(),
        &[&target_sizes, &target_sizes_fhd],
    );
    run_rgb_plotter_cubic(
        &[&img_large, &img_medium],
        format!("rgba8_big_{arch}_cubic{img_suffix}.svg").as_str(),
        &[&target_sizes_large, &target_sizes],
    );
    run_rgb_plotter_cubic(
        &[&img_medium, &fhd_image],
        format!("rgba8_medium_{arch}_cubic{img_suffix}.svg").as_str(),
        &[&target_sizes, &target_sizes_fhd],
    );
}

fn run_image_f32(path: &str, large_path: &str, fhd_path: &str) {
    let mut img_medium = ImageReader::open(path).unwrap().decode().unwrap();

    if !matches!(img_medium, DynamicImage::ImageRgba32F(_)) {
        img_medium = DynamicImage::ImageRgba32F(img_medium.to_rgba32f());
    }

    let mut img_large = ImageReader::open(large_path).unwrap().decode().unwrap();

    if !matches!(img_large, DynamicImage::ImageRgba32F(_)) {
        img_large = DynamicImage::ImageRgba32F(img_large.to_rgba32f());
    }

    let mut fhd_image = ImageReader::open(fhd_path).unwrap().decode().unwrap();

    if !matches!(fhd_image, DynamicImage::ImageRgba32F(_)) {
        fhd_image = DynamicImage::ImageRgba32F(fhd_image.to_rgba32f());
    }

    let target_sizes = make_sizes(&img_medium);
    let target_sizes_large = make_sizes(&img_large);
    let target_sizes_fhd = make_sizes(&fhd_image);

    let arch = std::env::consts::ARCH;
    run_rgb_plotter_f32(
        &[&img_large, &img_medium],
        format!("rgba_f32_big_{arch}.svg").as_str(),
        &[&target_sizes_large, &target_sizes],
    );
    run_rgb_plotter_f32(
        &[&img_medium, &fhd_image],
        format!("rgba_f32_medium_{arch}.svg").as_str(),
        &[&target_sizes, &target_sizes_fhd],
    );
}

fn run_image16(path: &str, large_path: &str, fhd_path: &str) {
    let mut img_medium = ImageReader::open(path).unwrap().decode().unwrap();

    if !matches!(img_medium, DynamicImage::ImageRgba16(_)) {
        img_medium = DynamicImage::ImageRgba16(img_medium.to_rgba16());
    }

    let mut img_large = ImageReader::open(large_path).unwrap().decode().unwrap();

    if !matches!(img_large, DynamicImage::ImageRgba16(_)) {
        img_large = DynamicImage::ImageRgba16(img_large.to_rgba16());
    }

    let mut fhd_image = ImageReader::open(fhd_path).unwrap().decode().unwrap();

    if !matches!(fhd_image, DynamicImage::ImageRgba16(_)) {
        fhd_image = DynamicImage::ImageRgba16(fhd_image.to_rgba16());
    }

    let target_sizes = make_sizes(&img_medium);
    let target_sizes_large = make_sizes(&img_large);
    let target_sizes_fhd = make_sizes(&fhd_image);

    let arch = std::env::consts::ARCH;
    run_rgb_plotter16(
        &[&img_large, &img_medium],
        format!("rgba16_big_{arch}.svg").as_str(),
        &[&target_sizes_large, &target_sizes],
    );
    run_rgb_plotter16(
        &[&img_medium, &fhd_image],
        format!("rgba16_medium_{arch}.svg").as_str(),
        &[&target_sizes, &target_sizes_fhd],
    );
}

fn main() {
    run_image(
        "",
        "./assets/nasa-4928x3279-rgba.png",
        "./assets/winter_huge.jpg",
        "./assets/sample_fhd.jpg",
    );
    run_image_f32(
        "./assets/nasa-4928x3279-rgba.png",
        "./assets/winter_huge.jpg",
        "./assets/sample_fhd.jpg",
    );
    run_image16(
        "./assets/nasa-4928x3279-rgba.png",
        "./assets/winter_huge.jpg",
        "./assets/sample_fhd.jpg",
    );
}
