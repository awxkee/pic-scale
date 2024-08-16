# Image scaling library in Rust

Rust image scale in different color spaces using SIMD and multithreading.

Supported NEON, SSE, AVX-2, RISC-V (Vector Extension).

### Colorspace

This library provides for you some conveniences to scale in different color spaces.\
Prebuilt options for CIE L\*a\*b, CIE L\*u\*v, CIE L\*c\*h, Linear, Sigmoidal, Oklab, Jzazbz available. \
Those transformations also very efficients.
Prefer downscale in linear colorspace or XYZ.\
Up scaling might be done in LAB/LUB and simoidized components and also efficient in sRGB.

Have good f16 (binary float16) support.

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
let resized = scaler.resize_rgba(
    ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
    store,
    true
);
let resized_image = resized.as_bytes();
```

### Fastest paths using SIMD

Despite all implementation are fast, not all the paths are implemented using SIMD, so some paths are slower

`~` - Partially implemented

|                | NEON | SSE | AVX | RISC-V | 
|----------------|------|-----|-----|--------| 
| RGBA (8 bit)   | x    | x   | ~   | x      | 
| RGB (8 bit)    | x    | x   | ~   | ~      | 
| Plane (8 bit)  | x    | x   | ~   | ~      | 
| RGBA (8+ bit)  | x    | x   | ~   | -      | 
| RGB (8+ bit)   | x    | x   | ~   | -      | 
| Plane (8+ bit) | ~    | ~   | ~   | -      | 
| RGBA (f32)     | x    | x   | x   | x      | 
| RGB (f32)      | x    | x   | ~   | ~      | 
| Plane (f32)    | x    | x   | ~   | ~      | 
| RGBA (f16)     | x    | x   | x   | x      | 
| RGB (f16)      | x    | ~   | ~   | ~      | 
| Plane (f16)    | ~    | ~   | ~   | ~      | 

#### Features

For RISC-V `riscv` feature should be implicitly enabled, nightly compiler channel is required

To enable support of `f16` the feature `half` should be activated.

#### Target features

`neon` optional target features are available, enable it when compiling on supported platform to get full features

`avx2`, `fma`, `sse4.1`, `f16c` will be detected automatically if available, and called the best path

`fullfp16` NEON target detection performed in runtime, when available best the best paths for *f16* images are available on ARM.

RISC-V `V (vector extension)` will be detected in runtime if available when feature `riscv` is enabled. *Nightly* rust channel is required

##### About f16

To enable full support of *f16* `half` feature should be used, and `f16c` enabled when targeting x86 platforms.
For NEON `f16` feature use runtime detection, if CPU supports this feature then the very fast path is available

Even when `half` feature activated but platform do not support or features not enabled for `f16` speed will be slow

For RISC-V `V` and `zfh` along `zvfh` is required on cpu support for fastest paths

### Performance

Example comparison with `fast-image-resize` time for downscale RGB 4928x3279 image in two times.

| Lanczos3  |  SSE  |  AVX  | NEON  |
|-----------|:-----:|:-----:|:-----:|
| pic-scale | 43.84 | 33.98 | 24.62 |
| fir sse   | 45.36 | 36.05 | 36.69 |

Example comparison time for downscale RGBA 4928x3279 image in two times with premultiplying alpha.

| Lanczos3  |  SSE  |  AVX  | NEON  |
|-----------|:-----:|:-----:|:-----:|
| pic-scale | 68.51 | 47.33 | 37.17 |
| fir sse   | 73.28 | 54.66 | 54.66 |

Example comparison time for downscale RGBA 4928x3279 image in two times without premultiplying alpha.

| Lanczos3  |  SSE  |  AVX  | NEON  |
|-----------|:-----:|:-----:|:-----:|
| pic-scale | 52.42 | 38.37 | 29.54 |
| fir sse   | 51.89 | 39.82 | 44.54 |

Example comparison time for downscale RGBA 4928x3279 10 bit image in two times with premultiplying alpha.

| Lanczos3  |  SSE   | NEON  |
|-----------|:------:|:-----:|
| pic-scale | 156.90 | 62.44 |
| fir sse   | 150.65 | 91.08 |

RGBA 4928x3279 10 bit downscale in two times without premultiplying alpha 

| Lanczos3  |  SSE   | NEON  |
|-----------|:------:|:-----:|
| pic-scale | 156.90 | 45.09 |
| fir sse   | 150.65 | 73.82 |

Example comparison time for downscale RGB 4000x6000 10 bit image in two times using *NEON*.

| Lanczos3  |  SSE   |  NEON  |
|-----------|:------:|:------:|
| pic-scale | 138.75 | 56.89  |
| fir sse   | 125.85 | 100.36 |

#### Example in sRGB

In common, you should not downsize an image in sRGB colorspace, however if speed is more preferable than more proper scale you may omit linearizing 

```rust
let mut scaler = Scaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in linear

At the moment only sRGB transfer function is supported. This is also good optimized path so it is reasonably fast.

```rust
let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in CIE L\*a\*b
```rust
let mut scaler = LabScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in CIE L\*u\*v
```rust
let mut scaler = LuvScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in CIE XYZ colorspace
```rust
let mut scaler = XYZScale::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in Sigmoidal colorspace
```rust
let mut scaler = SigmoidalScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in LCh colorspace
```rust
let mut scaler = LChScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
```

#### Example in Oklab colorspace
```rust
let mut scaler = OklabScaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height).unwrap();
let resized = scaler.resize_rgba(
    ImageSize::new(new_width, new_height),
    store,
    true
);
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