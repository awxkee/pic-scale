#
# // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
# //
# // Redistribution and use in source and binary forms, with or without modification,
# // are permitted provided that the following conditions are met:
# //
# // 1.  Redistributions of source code must retain the above copyright notice, this
# // list of conditions and the following disclaimer.
# //
# // 2.  Redistributions in binary form must reproduce the above copyright notice,
# // this list of conditions and the following disclaimer in the documentation
# // and/or other materials provided with the distribution.
# //
# // 3.  Neither the name of the copyright holder nor the names of its
# // contributors may be used to endorse or promote products derived from
# // this software without specific prior written permission.
# //
# // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target bundler --out-dir pkg/bundler

RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target web --out-dir pkg/web

RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target nodejs --out-dir pkg/nodejs

# 3. Merge into dist/
VERSION=$(grep '^version' pic-scale-js/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
mkdir -p dist
cp -r pkg/bundler dist/bundler
cp -r pkg/web     dist/web
cp -r pkg/nodejs  dist/nodejs
rm -f dist/*/package.json dist/*/.gitignore
cp README.md dist/

mv dist/nodejs/pic_scale_js.js dist/nodejs/pic_scale_js.cjs

sed "s/\"version\": \"0.1.0\"/\"version\": \"$VERSION\"/" package.json > dist/package.json

npm publish dist/ --dry-run --access public

# 5. Actual publish
npm publish dist/ --access public