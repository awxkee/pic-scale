# Image scaling library in Rust

[![crates.io](https://img.shields.io/crates/v/pic-scale.svg)](https://crates.io/crates/pic-scale)
![Build](https://github.com/awxkee/pic-scale/actions/workflows/build_push.yml/badge.svg)

Rust image scale in different color spaces using SIMD and multithreading.

Supported NEON, SSE, AVX-2, AVX-512, AVX-VNNI, WASM.

### Colorspace

This library provides for you some conveniences to scale in different color spaces.\
Prebuilt options for CIE L\*a\*b, CIE L\*u\*v, CIE L\*c\*h, Linear, Sigmoidal, Oklab, Jzazbz available. \
Those transformations also very efficients.
Prefer downscale in linear colorspace or XYZ.\
Up scaling might be done in LAB/LUB and simoidized components and also efficient in sRGB.

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
scaler.set_threading_policy(ThreadingPolicy::Adaptive);
// ImageStore::<u8, 4> - (u8, 4) represents RGBA, (u8, 3) - RGB etc
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(dimensions.0 as usize / 2, dimensions.1 as usize / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
let resized_image = resized.as_bytes();
```

### Fastest paths using SIMD

Despite all implementation are fast, not all the paths are implemented using SIMD, so some paths are slower

`~` - Partially implemented

|                | NEON | SSE | AVX2 | AVX-512    | WASM | 
|----------------|------|-----|------|------------|------| 
| RGBA (8 bit)   | x    | x   | x    | x(avxvnni) | x    | 
| RGB (8 bit)    | x    | x   | x    | x(avxvnni) | x    | 
| Plane (8 bit)  | x    | x   | ~    | ~          | ~    | 
| RGBA (8+ bit)  | x    | x   | x    | x(avxvnni) | -    | 
| RGB (8+ bit)   | x    | ~   | ~    | ~          | -    | 
| Plane (8+ bit) | x    | ~   | x    | ~          | -    | 
| RGBA (f32)     | x    | x   | x    | -          | -    | 
| RGB (f32)      | x    | x   | x    | -          | -    | 
| Plane (f32)    | x    | x   | x    | -          | -    | 
| RGBA (f16)     | x    | x   | x    | -          | -    | 
| RGB (f16)      | x    | ~   | ~    | -          | -    | 
| Plane (f16)    | ~    | ~   | ~    | -          | -    |
| AR30/RA30      | x    | x   | ~    | -          | -    |

#### Features

Features: 
 -  To enable support of `f16` the feature `nightly_f16` should be activated and `nightly` compiler are required.
 -  `nightly_avx512` activates AVX-512 feature set and requires `nightly` compiler channel 

#### Target features with runtime dispatch

For x86 and aarch64 NEON runtime dispatch is used.

`neon` optional target features are available, enable it when compiling on supported platform to get full features.

`avx2`, `fma`, `sse4.1`, `f16c` will be detected automatically if available, no additional actions need, and called the best path.

`avx512` requires feature `nightly_avx512` and requires `nightly` compiler channel, runtime detection if it is available then will be used.

`avxvnni` requires feature `nightly_avx512` and requires `nightly` compiler channel, runtime detection if it is available then will be used.
AVX-VNNI is helpful extension on modern Intel and AMD CPU's, consider turn it on to get maximum performance.

`fullfp16`, `fhm` NEON target detection performed in runtime, when available best the best paths for *f16* images are available on ARM.

WASM `simd128` target feature activating is mandatory in build flags.

##### About f16

To enable full support of *f16* `half` feature should be used, and `f16c` enabled when targeting x86 platforms.
For NEON `f16` feature use runtime detection, if CPU supports this feature then the very fast path is available

Even when `half` feature activated but platform do not support or features not enabled for `f16` speed will be slow

### Performance

NEON test made on Apple M3 Pro.
AVX2 test made on Premium Intel CPU Optimized 4 vCPU Digital Ocean instance.

Example comparison with `fast-image-resize` time for downscale RGB 4928x3279 image in 4 times.

| Lanczos3  |  AVX  | NEON  |
|-----------|:-----:|:-----:|
| pic-scale | 10.47 | 6.97  |
| fir       | 15.62 | 21.74 |

Example comparison time for downscale RGBA 4928x3279 image in 4 times with pre-multiplying alpha.

| Lanczos3  |  AVX  | NEON  |
|-----------|:-----:|:-----:|
| pic-scale | 43.72 | 13.56 |
| fir       | 62.31 | 33.32 |

Example comparison time for downscale RGBA 4928x3279 image in 4 times without pre-multiplying alpha.

| Lanczos3         |  AVX  | NEON  |
|------------------|:-----:|:-----:|
| pic-scale        | 11.13 | 7.76  |
| fir              | 20.17 | 25.92 |
| Apple Accelerate |   -   | 9.98  |

Example comparison time for downscale RGBA 4928x3279 10 bit image in 4 times with pre-multiplying alpha.

| Lanczos3         |  AVX   | NEON  |
|------------------|:------:|:-----:|
| pic-scale        | 85.34  | 22.68 |
| fir              | 146.23 | 53.95 |

RGBA 4928x3279 10 bit downscale 4 two times without pre-multiplying alpha

| Lanczos3         |  AVX  | NEON  |
|------------------|:-----:|:-----:|
| pic-scale        | 19.15 | 8.91  |
| fir              | 58.57 | 38.46 |
| Apple Accelerate |   -   | 27.63 |

Example comparison time for downscale RGB 4000x6000 10 bit image in 4 times using *NEON*.

| Lanczos3  |  AVX  | NEON  |
|-----------|:-----:|:-----:|
| pic-scale | 25.68 | 15.56 |
| fir       | 68.84 | 39.40 |

#### Example in sRGB

In common, you should not downsize an image in sRGB colorspace, however if speed is more preferable than more proper scale you may omit linearizing 

```rust
let mut scaler = Scaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in linear

At the moment only sRGB transfer function is supported. This is also good optimized path so it is reasonably fast.

```rust
let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in CIE L\*a\*b
```rust
let mut scaler = LabScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in CIE L\*u\*v
```rust
let mut scaler = LuvScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in CIE XYZ colorspace
```rust
let mut scaler = XYZScale::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
    let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in LCh colorspace
```rust
let mut scaler = LChScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

#### Example in Oklab colorspace
```rust
let mut scaler = OklabScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store = ImageStore::<u8, 4>::from_slice(&bytes, width, height).unwrap();
let mut dst_store = ImageStoreMut::<u8, 4>::alloc(width / 2, height / 2);
let resized = scaler.resize_rgba(
    &store,
    &mut dst_store,
    true
);
```

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
