#!/usr/bin/env bash

set -ex
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_GENAI_VERSION=$(awk -F '=' '/onnxruntimeGenai/ {gsub(/ ?"/, "", $2); print $2}' "$WORK_DIR/../../../gradle/libs.versions.toml")

pushd .

mkdir -p "$WORK_DIR/build/download"
cd "$WORK_DIR/build/download"
#curl -sfL -O "${GITHUB_URL}/onnxruntime-genai-${ORT_GENAI_VERSION}-linux-x64-cuda.tar.gz"
curl -sfL -O "${GITHUB_URL}/onnxruntime-genai-${ORT_GENAI_VERSION}-linux-x64.tar.gz"
curl -sfL -O "${GITHUB_URL}/onnxruntime-genai-${ORT_GENAI_VERSION}-osx-arm64.tar.gz"
curl -sfL -O "${GITHUB_URL}/onnxruntime-genai-${ORT_GENAI_VERSION}-win-x64.zip"
curl -sfL -O "https://github.com/microsoft/onnxruntime-genai/archive/refs/tags/v${ORT_GENAI_VERSION}.zip"

find . -name "*.tar.gz" -print0 | xargs -0 tar xvfz
find . -name "*.zip" -print0 | xargs -0 unzip

popd
