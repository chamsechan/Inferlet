#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build_debug}"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory '$BUILD_DIR' not found."
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_PREFIX="/tmp/inferlet_install_test_$$"
CONSUMER_BUILD_DIR="/tmp/inferlet_consumer_build_$$"

cleanup() {
    rm -rf "$INSTALL_PREFIX" "$CONSUMER_BUILD_DIR"
}
trap cleanup EXIT

echo "=== Step 18 Packaging and Consumer Verification ==="
echo "1. Installing Inferlet to isolated prefix: $INSTALL_PREFIX"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

echo "2. Verifying installed artifacts layout..."
test -f "$INSTALL_PREFIX/include/inferlet.h"
test -f "$INSTALL_PREFIX/lib/libinferlet.a"
test -f "$INSTALL_PREFIX/lib/cmake/Inferlet/InferletConfig.cmake"
test -f "$INSTALL_PREFIX/lib/cmake/Inferlet/InferletConfigVersion.cmake"
test -f "$INSTALL_PREFIX/lib/cmake/Inferlet/InferletTargets.cmake"
test -f "$INSTALL_PREFIX/lib/pkgconfig/inferlet.pc"
test -f "$INSTALL_PREFIX/share/licenses/Inferlet/LICENSE.llama.cpp"
test -f "$INSTALL_PREFIX/share/doc/Inferlet/third_party.md"
echo "Artifacts layout verified successfully."

echo "3. Building and executing C++ consumer out-of-tree..."
mkdir -p "$CONSUMER_BUILD_DIR/cpp"
cmake -S "$ROOT_DIR/tests/consumer/cpp" \
      -B "$CONSUMER_BUILD_DIR/cpp" \
      -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"
cmake --build "$CONSUMER_BUILD_DIR/cpp"
"$CONSUMER_BUILD_DIR/cpp/consumer_cpp"
echo "C++ consumer verified successfully."

echo "4. Building and executing C consumer out-of-tree..."
mkdir -p "$CONSUMER_BUILD_DIR/c"
cmake -S "$ROOT_DIR/tests/consumer/c" \
      -B "$CONSUMER_BUILD_DIR/c" \
      -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"
cmake --build "$CONSUMER_BUILD_DIR/c"
"$CONSUMER_BUILD_DIR/c/consumer_c"
echo "C consumer verified successfully."

echo "5. Testing pkg-config compilation and execution..."
if command -v pkg-config >/dev/null 2>&1; then
    export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    pkg-config --validate inferlet
    CFLAGS=$(pkg-config --cflags inferlet)
    LIBS=$(pkg-config --libs --static inferlet)
    echo "pkg-config Cflags: $CFLAGS"
    echo "pkg-config Libs: $LIBS"

    gcc "$ROOT_DIR/tests/consumer/c/main.c" $CFLAGS $LIBS -o "$CONSUMER_BUILD_DIR/consumer_pc"
    "$CONSUMER_BUILD_DIR/consumer_pc"
    echo "pkg-config build verified successfully."
else
    echo "pkg-config not installed on system, skipping pkg-config direct test."
fi


echo "6. Checking symbol cleanliness with nm -u on consumer binary..."
CPP_EXE="$CONSUMER_BUILD_DIR/cpp/consumer_cpp"
C_EXE="$CONSUMER_BUILD_DIR/c/consumer_c"

for BIN in "$CPP_EXE" "$C_EXE"; do
    UNDEF_SYMS=$(nm -u "$BIN" | awk '{print $NF}')
    
    # Verify no Inferlet, llama, or ggml symbols remain undefined
    INTERNAL_LEAKS=$(echo "$UNDEF_SYMS" | grep -E "inferlet_|llama_|ggml_" || true)
    if [[ -n "$INTERNAL_LEAKS" ]]; then
        echo "ERROR: Found undefined internal symbols in $BIN:"
        echo "$INTERNAL_LEAKS"
        exit 1
    fi
done
echo "Symbol cleanliness verified: no internal symbols leaked as undefined."

echo "7. Verifying archive bundling in a path containing spaces..."
SPACE_DIR="$CONSUMER_BUILD_DIR/archive with spaces"
mkdir -p "$SPACE_DIR"
cp "$INSTALL_PREFIX/lib/libinferlet.a" "$SPACE_DIR/input.a"
cmake -DOUTPUT_ARCHIVE="$SPACE_DIR/libinferlet.a" \
      -DWRAPPER_ARCHIVE="$SPACE_DIR/input.a" \
      -DDEPENDENCY_ARCHIVES= \
      -P "$ROOT_DIR/cmake/bundle_fat_archive.cmake"
cmake -DARCHIVE_FILE="$SPACE_DIR/libinferlet.a" \
      -DEXPECTED_SYMBOLS="$ROOT_DIR/tests/abi/expected_symbols.txt" \
      -P "$ROOT_DIR/tests/abi/test_symbols.cmake"

echo "=== All Step 18 Consumer & Packaging Checks PASSED! ==="
