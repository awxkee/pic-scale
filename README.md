# Image scaling library in Rust

Rust image scale in different color spaces using SIMD and multithreading.

Supported only NEON and SSE.

### Colorspace

This library provides for you some conveniences to scale in different color spaces.
Prebuilt options for CIE L\*a\*b, CIE L\*u\*v, CIE L\*c\*h, Linear, Sigmoidal, Oklab, Jzazbz available. Those transformations also very efficients.
Whether downscaling is preferred in linear colorspace, LAB/LUV and sigmoidal also provides very good results.
Up scaling might be done in LAB/LUB and simoidized components and also efficient in sRGB.

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
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize);
let resized = scaler.resize_rgba(
    ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
    store,
    true
);
```

### Performance

Example comparison with `fast-image-resize` time for downscale RGB 4928x3279 image in two times for x86_64 SSE.

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  27.87   |
| fir sse   |  36.94   |

M3 Pro. NEON

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  27.58   |
| fir sse   |  36.69   |

Example comparison time for downscale RGBA 4928x3279 image in two times for x86_64 SSE with premultiplying alpha.

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  35.51   |
| fir sse   |  30.87   |

M3 Pro. NEON

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  25.46   |
| fir sse   |  31.43   |

Example comparison time for downscale RGBA 4928x3279 image in two times for x86_64 SSE without premultiplying alpha.

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  24.00   |
| fir sse   |  23.13   |

M3 Pro. NEON

|           | Lanczos3 |
|-----------|:--------:|
| pic-scale |  17.41   |
| fir sse   |  25.82   |

#### Example in sRGB

In common, you should not downsize an image in sRGB colorspace, however if speed is more preferable than more proper scale you may omit linearizing 

```rust
let mut scaler = Scaler::new(ResamplingFunction::Hermite);
scaler.set_threading_policy(ThreadingPolicy::Single);
let store =
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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
    ImageStore::<u8, 4>::from_slice(&mut bytes, width, height);
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