# @radzivon.bartoshyk/pic-scale

[![npm](https://img.shields.io/npm/v/@radzivon.bartoshyk/pic-scale)](https://www.npmjs.com/package/@radzivon.bartoshyk/pic-scale)

High-performance image resizing for Node.js and the browser, powered by the
[pic-scale](https://github.com/awxkee/pic-scale) Rust engine with SIMD
acceleration (AVX2+FMA on x86-64, NEON on AArch64).

Two delivery modes, one package:

| Mode             | Import                                     | When to use                  |
|------------------|--------------------------------------------|------------------------------|
| **Native addon** | `require('@radzivon.bartoshyk/pic-scale')` | Node.js — fastest, file I/O  |
| **WASM**         | `import ... from '.../pic-scale/wasm'`     | Browser, Deno, edge runtimes |

## Installation

```bash
npm install @radzivon.bartoshyk/pic-scale
```

---

## Native addon (Node.js)

All heavy operations (`resize`, `toBuffer`, `save`) run on Tokio's thread pool
and return Promises, so the event loop is never blocked.

```js
import { Image } from '@radzivon.bartoshyk/pic-scale'

// Open from file — metadata (ICC, EXIF orientation) extracted automatically
const img = await Image.open('photo.jpg')
console.log(img.width, img.height, img.channels)

// Resize
const small = await img.resize(800, 600, {
  filter:      'lanczos',   // default
  mode:        'cover',     // fill box, crop from centre
  autoOrient:  true,        // default — bake EXIF rotation into pixels
  withIcc:     true,        // default — copy ICC profile to output
  withExif:    true,        // default — copy EXIF, orientation reset to 1
  withXmp:     false,       // default
  workers:     0,           // 0 = adaptive (all cores)
})

// Save — format inferred from extension, ICC + EXIF injected automatically
await small.save('small.jpg', { quality: 85 })

// Or encode to a Buffer
const buf = await small.toBuffer('webp', { quality: 90 })

// Sync variants — blocks the event loop, useful in scripts or worker threads
const small2 = img.resizeSync(800, 600, { mode: 'fit', bgColor: [0, 0, 0, 255] })
const png    = small2.toBufferSync('png')
```

### Resize modes

| Mode           | Description                                                         |
|----------------|---------------------------------------------------------------------|
| `"fill"`       | Stretch to exact `(width, height)` — ignores aspect ratio (default) |
| `"cover"`      | Scale to fill the box, crop excess from the centre                  |
| `"fit"`        | Scale to fit inside the box, pad edges with `bgColor`               |
| `"fit_width"`  | Scale to match width, height adjusts proportionally                 |
| `"fit_height"` | Scale to match height, width adjusts proportionally                 |

### Resampling filters

| Filter        | Notes                                           |
|---------------|-------------------------------------------------|
| `nearest`     | Fastest, blocky                                 |
| `bilinear`    | Fast, smooth                                    |
| `bicubic`     | Keys cubic                                      |
| `lanczos`     | Window-3 sinc — best general quality (default)  |
| `lanczos2`    | Faster, slightly softer                         |
| `lanczos4`    | Slower, very sharp                              |
| `box`         | Area average — best for heavy downscaling       |
| `hamming`     |                                                 |
| `mitchell`    | Mitchell-Netravali — balanced sharpness/ringing |
| `catmull_rom` | Sharper than Mitchell                           |
| `gaussian`    |                                                 |
| `hann`        |                                                 |

### Metadata

Metadata is extracted automatically when you call `Image.open()` or
`Image.fromBuffer()` and travels with the image through resize operations.

- **Auto-orient** (`autoOrient: true`, default) — EXIF orientation is baked
  into pixels before resizing. The orientation tag is reset to 1 in the output
  so viewers don't rotate twice.
- **ICC profile** (`withIcc: true`, default) — colour profile is injected into
  the output JPEG, PNG, or WebP.
- **EXIF** (`withExif: true`, default) — EXIF block is injected into the output
  with the orientation tag reset to 1.
- **XMP** (`withXmp: false`, default) — opt in if you need creative metadata
  (title, copyright, keywords).

To strip all metadata:

```js
const clean = await img.resize(800, 600, {
  withIcc: false, withExif: false, withXmp: false, autoOrient: true,
})
```

---

## WASM (browser / Deno / edge)

WASM is single-threaded — all operations are synchronous. For non-blocking
behaviour in the browser, run inside a **Web Worker**:

```js
// worker.js
import init, { Image } from '@radzivon.bartoshyk/pic-scale/wasm'
await init()

self.onmessage = ({ data }) => {
  const img   = Image.fromBytes(data.bytes)
  const small = img.resize(data.width, data.height, 'lanczos', 'cover')
  const out   = small.toBytes('jpeg', 85)
  self.postMessage(out, [out.buffer])   // zero-copy transfer
}
```

```js
// main.js
const worker = new Worker('./worker.js', { type: 'module' })
const bytes  = new Uint8Array(await (await fetch('photo.jpg')).arrayBuffer())
worker.postMessage({ bytes, width: 800, height: 600 }, [bytes.buffer])
worker.onmessage = ({ data }) => {
  const blob = new Blob([data], { type: 'image/jpeg' })
  document.querySelector('img').src = URL.createObjectURL(blob)
}
```

### WASM API

```js
import init, { Image } from '@radzivon.bartoshyk/pic-scale/wasm'
await init()

const bytes = new Uint8Array(await file.arrayBuffer())
const img   = Image.fromBytes(bytes)         // decode, extract ICC/EXIF
console.log(img.width, img.height, img.channels)

const small = img.resize(
  800, 600,
  'lanczos',  // filter    (default 'lanczos')
  'cover',    // mode      (default 'fill')
  true,       // premultiplyAlpha (default true)
  1,          // workers   (default 1)
  null,       // bgColor   [r,g,b,a] for 'fit' mode
  true,       // autoOrient (default true)
)

const out = small.toBytes(
  'jpeg',     // format  (default 'png')
  85,         // quality (default 85)
  true,       // withIcc  (default true)
  true,       // withExif (default true)
  false,      // withXmp  (default false)
)
// out is a Uint8Array
```

---

## Supported formats

| Format                 | Native decode | Native encode | WASM decode | WASM encode |
|------------------------|---------------|---------------|-------------|-------------|
| JPEG                   | ✓            | ✓            | ✓          | ✓          |
| PNG                    | ✓            | ✓            | ✓          | ✓          |
| WebP                   | ✓            | ✓            | ✓          | ✓          |
| TIFF                   | ✓            | ✓            | ✓          | ✓          |
| BMP                    | ✓            | ✓            | ✓          | ✓          |
| ICO                    | ✓            | —             | ✓          | —           |
| QOI                    | ✓            | ✓            | ✓          | ✓          |
| **AVIF**               | ✓            | ✓            | —           | —           |
| **HEIC/HEIF**          | ✓            | ✓            | —           | —           |
| **JPEG XL (JXL)**      | ✓            | ✓            | ✓          | ✓          |

---

## Native API reference

### `Image.open(path): Promise<Image>`
Decode from a file path. ICC, EXIF, and XMP are extracted automatically.

### `Image.fromBuffer(data: Buffer | Uint8Array): Image`
Decode from memory (sync).

### `img.width / img.height / img.channels`
Image dimensions and channel count (read-only).

### `img.resize(width, height, opts?): Promise<Image>`
Async resize. Returns a new `Image` — the original is unchanged.

```ts
interface ResizeOptions {
  filter?:          string    // default 'lanczos'
  mode?:            string    // default 'fill'
  premultiplyAlpha?: boolean  // default true
  workers?:         number    // default 1, 0 = adaptive
  bgColor?:         number[]  // [r,g,b,a], default [0,0,0,0]
  withIcc?:         boolean   // default true
  withExif?:        boolean   // default true
  withXmp?:         boolean   // default false
  autoOrient?:      boolean   // default true
}
```

### `img.resizeSync(width, height, opts?): Image`
Sync resize — same options as `resize`.

### `img.toBuffer(format?, opts?): Promise<Buffer>`
Encode to a `Buffer`. Injects ICC + EXIF by default.

### `img.toBufferSync(format?, opts?): Buffer`
Sync encode.

### `img.save(path, opts?): Promise<void>`
Write to a file. Format inferred from extension. Injects ICC + EXIF.

```ts
interface EncodeOpts { quality?: number }  // default 85
```

---

## License

BSD 3-Clause