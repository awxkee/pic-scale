# Image scaling library in Rust

[![crates.io](https://img.shields.io/crates/v/pic-scale.svg)](https://crates.io/crates/pic-scale)
![Build](https://github.com/awxkee/pic-scale/actions/workflows/build_push.yml/badge.svg)

Rust image scale in different color spaces using SIMD and multithreading.

Supported NEON, SSE, AVX-2, AVX-512, AVX-VNNI, WASM.

### Colorspace

This library provides for you a convenience to scale in different color spaces.\
Prebuilt options for CIE L\*a\*b, CIE L\*u\*v, CIE L\*c\*h, Linear, Sigmoidal, Oklab, Jzazbz available. \
Those transformations also very efficients.
Prefer downscale in linear colorspace.\
Upscaling might be done in LAB/LUB and simoidized components and also efficient in sRGB.

Have good `f16` (the “binary16” type defined in IEEE 754-2008) support.

#### Example integration with `image` crate

```rust
let img = ImageReader::open("./assets/asset.png")
    .unwrap()
    .decode()
    .unwrap();
let dimensions = img.dimensions();
let mut bytes = Vec::from(img.as_bytes());

let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
scaler.set_threading_policy(ThreadingPolicy::Single);
// ImageStore::<u8, 4> - (u8, 4) represents RGBA, (u8, 3) - RGB etc
let store = ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(dimensions.0 as usize / 2, dimensions.1 as usize / 2);
let plan = scaler.plan_rgba_resampling(
    ImageSize::new(dimensions.0 as usize, dimensions.1 as usize), //source size
    ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2), // target size
    true, // premultiply alpha
).unwrap();
plan.resample(&store, &mut dst_store).unwrap();
```

### Fastest paths using SIMD

Despite all implementations are fast, not all the paths are implemented using SIMD, therefore some paths are slower.
Here is a table which shows what's implemented with SIMD.

`~` - Partially implemented

|                 | NEON | SSE | AVX2 | AVX-512    | WASM | 
|-----------------|------|-----|------|------------|------| 
| RGBA (8 bit)    | x    | x   | x    | x(avxvnni) | x    | 
| RGB (8 bit)     | x    | x   | x    | x(avxvnni) | x    | 
| Plane (8 bit)   | x    | x   | ~    | ~          | ~    | 
| RGBA (8+ bit)   | x    | x   | x    | x(avxvnni) | -    | 
| RGB (8+ bit)    | x    | ~   | x    | ~          | -    | 
| Plane (8+ bit)  | x    | ~   | x    | ~          | -    | 
| Plane (S8+ bit) | x    | -   | x    | -          | -    | 
| RGBA (f32)      | x    | x   | x    | -          | -    | 
| RGB (f32)       | x    | x   | x    | -          | -    | 
| Plane (f32)     | x    | x   | x    | -          | -    | 
| RGBA (f16)      | x    | x   | x    | -          | -    | 
| RGB (f16)       | x    | ~   | ~    | -          | -    | 
| Plane (f16)     | ~    | ~   | ~    | -          | -    |
| AR30/RA30       | x    | ~   | x    | -          | -    |

#### Features

Features: 
 -  To enable support of `f16` the feature `nightly_f16` should be activated and `nightly` compiler are required.
 -  `nightly_avx512` activates AVX-512 feature set and requires `nightly` compiler channel.

#### Target features with runtime dispatch

For x86 and aarch64 NEON runtime dispatch is used.

`neon` optional target features are available, enable it when compiling on supported platform to get all features.

`avx2`, `fma`, `sse4.1`, `f16c` will be detected automatically if enabled, it will automatically detect and use the best path if enabled.

`avx512` requires feature `avx512` compiler channel, runtime detection if it is available then will be used.

`avxvnni` requires feature `avx512`, will be detected automatically if available, no additional actions need, it will automatically detect and use the best path if enabled.
AVX-VNNI is helpful extension on modern Intel and AMD CPUs, consider turn it on to get maximum performance.

`fullfp16`, `fhm` NEON target detection performed in runtime, will be detected automatically if enabled, it will automatically detect and use the best path if enabled.

WASM `simd128` target feature activating is mandatory in build flags.

##### About f16

To get full support of *f16* `nightly_f16` feature should be used.
For NEON `f16` feature use runtime detection, if CPU supports this feature then the very fast path is available

### Build C bindings

See `picscale/include/picscale.h` for more info

```bash
cd picscale && RUSTFLAGS="-C strip=symbols" cargo +nightly build -Z build-std=std,panic_abort --release
```

#### Resampling filters

Over 30 resampling filters is supported.

```rust
Bilinear
Nearest
Cubic
MitchellNetravalli
CatmullRom
Hermite
BSpline
Hann
Bicubic
Hamming
Hanning
Blackman
```
And others

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
