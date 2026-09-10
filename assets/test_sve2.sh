#!/usr/bin/env bash
# Run from an AArch64 Linux host, or set the cross linker and QEMU_LD_PREFIX.
set -euo pipefail

cd "$(dirname "$0")/.."
qemu="${QEMU_AARCH64:-qemu-aarch64}"
command -v "$qemu" >/dev/null

test_binary="$(
    cargo +nightly test -p pic-scale --target aarch64-unknown-linux-gnu \
        --features sve --lib --no-run --message-format=json-render-diagnostics |
        python3 -c '
import json, sys
artifacts = [json.loads(line) for line in sys.stdin]
paths = [a["executable"] for a in artifacts
         if a.get("reason") == "compiler-artifact" and a.get("executable")
         and a["target"]["name"] == "pic_scale" and a["profile"]["test"]]
assert len(paths) == 1, "Expected exactly one pic-scale unit-test binary"
print(paths[0])
'
)"

# A successful run with zero matching tests must not count as verification.
listing="$("$qemu" -cpu max "$test_binary" sve2:: --list)"
for name in sve2_row_one_matches_scalar_reference sve2_rows_4_matches_scalar_reference \
    sve2_vertical_matches_scalar_reference sve2_vertical_rounds_once; do
    if [[ "$listing" != *"$name: test"* ]]; then
        echo "Missing required test: $name" >&2
        exit 1
    fi
done

for ((bits = 128; bits <= 2048; bits += 128)); do
    echo "Testing SVE2 at $bits bits"
    PIC_SCALE_REQUIRE_SVE2=1 PIC_SCALE_SVE_VL_BITS="$bits" \
        "$qemu" -cpu "max,sve-default-vector-length=$((bits / 8))" \
        "$test_binary" sve2:: --test-threads=1 --nocapture
done
