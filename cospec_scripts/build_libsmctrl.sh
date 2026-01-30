#!/bin/bash
# Build libsmctrl.so for SM partitioning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBSMCTRL_DIR="$SCRIPT_DIR/libsmctrl"

echo "Building libsmctrl.so..."
echo "Source: $LIBSMCTRL_DIR/src"

cd "$LIBSMCTRL_DIR" || exit 1

mkdir -p build

# Build libsmctrl.so
gcc -shared -fPIC -o build/libsmctrl.so src/libsmctrl.c -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64

if [ $? -eq 0 ]; then
    echo "Successfully built: $LIBSMCTRL_DIR/build/libsmctrl.so"
    cp "$LIBSMCTRL_DIR/build/libsmctrl.so" /usr/local/lib/
    ldconfig
    echo "Installed to /usr/local/lib/libsmctrl.so"
else
    echo "Build failed!"
    exit 1
fi
