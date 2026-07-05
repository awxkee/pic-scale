/*
 * Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

import {createRequire} from 'module'

const require = createRequire(import.meta.url)

const {Image} = require('./pic_scale.node')

const img = await Image.open('../assets/digital_art_portrait2.jpg')
console.log(`Loaded: ${img.width}x${img.height}, ${img.channels}ch`)

// ── resize — cover mode, lanczos, auto-orient + keep ICC/EXIF ────────────────
const small = await img.resize(img.width / 2, img.height / 2, {
    filter: 'lanczos',
    mode: 'cover',       // crop from center to fill 320×240
    autoOrient: true,    // default — bake EXIF rotation into pixels
    withIcc: true,       // default — copy ICC profile to output
    withExif: true,      // default — copy EXIF (orientation reset to 1)
    withXmp: false,      // default — skip XMP
})
console.log(`Resized: ${small.width}x${small.height}`)

await small.save('out.avif', {quality: 60})
console.log('Saved out.jpg')

// ── fit mode — letterbox with black padding ───────────────────────────────────
const fitted = await img.resize(img.width / 2, img.height / 2, {
    mode: 'fit',
    bgColor: [0, 0, 0, 255],   // black bars
})
await fitted.save('out_fit.avif', {quality: 60})
console.log(`Saved out_fit.jpg (${fitted.width}x${fitted.height})`)

// ── sync variants ─────────────────────────────────────────────────────────────
const smallSync = img.resizeSync(160, 120, {mode: 'fill'})
const pngBuf = smallSync.toBufferSync('png')
console.log(`PNG buffer (sync): ${pngBuf.length} bytes`)
