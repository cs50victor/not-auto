#!/bin/bash

# Script to build gst-plugins-rs with LiveKit feature enabled on macos

set -e

echo "=== Building gst-plugins-rs with LiveKit support (RUN WITH SUDO) ==="

BUILD_DIR="/tmp/gst-plugins-rs-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone the repository
if [ ! -d "gst-plugins-rs" ]; then
    echo "Cloning gst-plugins-rs repository..."
    git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
    cd gst-plugins-rs
else
    cd gst-plugins-rs
    echo "Updating existing repository..."
    git fetch
    git pull
fi

echo "Building gst-plugin-webrtc with livekit feature..."
cargo build --release --package gst-plugin-webrtc --features livekit

BUILT_LIB="target/release/libgstrswebrtc.dylib"

if [ ! -f "$BUILT_LIB" ]; then
    echo "Error: Built library not found at $BUILT_LIB"
    exit 1
fi

GST_PLUGIN_DIR="/opt/homebrew/lib/gstreamer-1.0"

echo "Installing to $GST_PLUGIN_DIR..."
cp "$BUILT_LIB" "$GST_PLUGIN_DIR/"

echo "=== Installation complete! ==="
echo ""
echo "To verify, run:"
echo "gst-inspect-1.0 livekitwebrtcsink"
echo ""
echo "To use in your project, run:"
echo "GST_PLUGIN_PATH=/opt/homebrew/lib/gstreamer-1.0 cargo run"
