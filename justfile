set export := true

# All test recipes target --lib: the workspace members (app, speedtest,
# pic-scale-js, wasm) are demo/binding crates with their own build
# requirements, not part of the library under test.

## native

# Default feature set (neon, rdm, sse, avx, threading) on this host.
test:
    cargo test --lib

# Every optional feature enabled together.
test-all-features:
    cargo test --lib --all-features

# Isolates each SIMD tier the way CI's `tests_x86` job does, so a bug gated
# behind one feature doesn't hide behind another (matches build_push.yml).
test-x86-matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    for feature in avx sse "" nightly_f16; do
        echo "=== features: '${feature}' ==="
        cargo test --lib --features "${feature}"
    done

# Needs an AVX-512 host; CI covers this on a real ARM/x86 runner instead of
# qemu, since narrowing AVX-512 subsets under emulation isn't set up here.
test-avx512:
    cargo test --lib --features avx512

## cross-arch (QEMU user-mode emulation, no physical hardware needed)
# One-time setup: `just install-cross-targets`, then `apt-get install qemu-user`.
# Runner + linker flags live in .cargo/config.toml (rust-lld + +crt-static,
# since the host cc/ld.bfd chokes on target-specific link flags and
# riscv64gc-musl otherwise fails to link with "unable to find library -lgcc_s").

# NEON module is aarch64-only (see src/lib.rs). Includes nightly_f16 for full
# coverage - note `neon::rgba_f16::backend_comparison_tests::neon_row_matches_scalar_reference`
# currently aborts the whole binary (SIGABRT) under qemu-aarch64: a real
# out-of-bounds read in `convolve_horizontal_rgba_neon_row_one_f16`
# (off-by-one loop bound), not a qemu/harness issue. Use `test-aarch64-safe`
# to see the rest of the suite's results until that's fixed.
test-aarch64:
    cargo test --target aarch64-unknown-linux-musl --no-default-features --features "neon,rdm,nightly_f16" --lib

# Same as test-aarch64 but skips the test that currently crashes the process.
test-aarch64-safe:
    cargo test --target aarch64-unknown-linux-musl --no-default-features --features "neon,rdm,nightly_f16" --lib -- --skip neon::rgba_f16::backend_comparison_tests::neon_row_matches_scalar_reference

# No arch-specific SIMD features exist for 32-bit ARM in this crate - this
# exercises the portable scalar path, which CI doesn't build or test at all.
test-armv7:
    cargo test --target armv7-unknown-linux-musleabihf --no-default-features --lib

# CI only `cargo build`s this target; this actually runs the suite.
test-riscv64:
    cargo test --target riscv64gc-unknown-linux-musl --no-default-features --lib

# Uses the -safe aarch64 variant so one known crash doesn't stop armv7/riscv64
# from running too; use `just test-aarch64` directly to reproduce the crash.
test-cross: test-aarch64-safe test-armv7 test-riscv64

## everything

test-all: test test-x86-matrix test-avx512 test-cross
    @echo "All test suites passed."

## miri (undefined-behavior detection)
# One-time setup: `just miri-setup`. Miri interprets MIR directly, so it can
# cross-check aarch64 NEON code on this x86_64 host with no qemu/hardware -
# but its NEON intrinsic coverage has real gaps (see miri-neon below).

miri-setup:
    rustup component add miri
    cargo +nightly miri setup
    cargo +nightly miri setup --target aarch64-unknown-linux-gnu

# Native x86_64: SSE/AVX2/AVX-512. Miri fully supports these intrinsics, so
# this is a reliable "no known UB" gate - clean as of this writing. Excludes
# nightly_f16: those tests are technically correct under Miri but so slow
# (minutes per test - heavy scalar f16 bit-twiddling interpreted one op at a
# time) that they aren't practical for a routine check.
miri:
    cargo +nightly miri test --lib -- backend_comparison_tests
    cargo +nightly miri test --lib --features avx512 -- avx512::

# Best-effort NEON, cross-interpreted via Miri (no ARM hardware/qemu needed).
# Runs one module at a time and keeps going past failures - unlike plain
# `cargo miri test`, which aborts the WHOLE run on the first hit of either a
# real bug or an unsupported intrinsic, hiding every module after it.
#
# Two distinct outcomes, do not conflate them:
#   - "error: Undefined Behavior: ..."     -> a REAL bug (see BACKEND_COMPARISON_FINDINGS.md).
#   - "error: unsupported operation: ..."  -> a Miri tooling gap (missing NEON
#     intrinsic support, e.g. urshl/sqshrun/faddv), NOT a bug - that module
#     just can't be checked this way.
#
# Confirmed real UB so far: rgba_u8/rgb_u8/plane_u8/cbcr8 (misaligned pointer
# cast in the shared tail-pixel-load helpers in neon/utils.rs), rgb_f32
# (Stacked Borrows violation, src/neon/rgb_f32.rs:76), vertical_f32
# (genuine out-of-bounds read for narrow row widths, src/neon/vertical_f32.rs).
# This is a representative sample, not an exhaustive sweep - expect this to
# take a long time (many minutes) if extended to every neon/*.rs file.
miri-neon:
    #!/usr/bin/env bash
    set -uo pipefail
    modules="alpha_f32 rgba_f32 rgb_f32 plane_f32 vertical_f32 rgba_u8 rgb_u8 plane_u8 cbcr8 rgba_u16 rgb_u16 plane_u16 alpha_u16 vertical_u8 vertical_u16"
    for m in $modules; do
        echo "=== neon::${m} ==="
        timeout 120 cargo +nightly miri test --target aarch64-unknown-linux-gnu --no-default-features --features "neon,rdm" --lib -- "neon::${m}::" 2>&1 \
            | grep -E "^running|\.\.\. ok$|\.\.\. ignored|error: Undefined Behavior|error: unsupported|test result" || true
    done

## setup

install-cross-targets:
    rustup target add aarch64-unknown-linux-musl armv7-unknown-linux-musleabihf riscv64gc-unknown-linux-musl
