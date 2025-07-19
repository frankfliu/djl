#!/usr/bin/env bash

set -ex

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_GENAI_VERSION=$(awk -F '=' '/onnxruntimeGenai/ {gsub(/ ?"/, "", $2); print $2}' "$WORK_DIR/../../../gradle/libs.versions.toml")

pushd .

mkdir -p "$PWD/build"
cd "$PWD/build"

if [[ ! -d "onnxruntime-genai" ]]; then
  # curl -sfL -O "https://github.com/microsoft/onnxruntime-genai/archive/refs/tags/v${ORT_GENAI_VERSION}.zip"
  # unzip "v${ORT_GENAI_VERSION}.zip"
  # git clone https://github.com/microsoft/onnxruntime-genai.git -b "v$ORT_GENAI_VERSION"
  git clone https://github.com/microsoft/onnxruntime-genai.git
fi

cd onnxruntime-genai

python3 -m pip install -U requests

python3 build.py --build_java --skip_tests --skip_wheel --skip_examples
rm src/java/build/libs/onnxruntime-genai-*-sources.jar
rm src/java/build/libs/onnxruntime-genai-*-javadoc.jar
unzip -d build "src/java/build/libs/onnxruntime-genai-*.jar"

cp -rf build/ai/onnxruntime/genai/native "$WORK_DIR"

if [[ "$1" == "CUDA" ]]; then
  git clean -dffx .
  python3 build.py --build_java --skip_tests --skip_wheel --skip_examples --use_cuda --cuda_home=/usr/local/cuda
  rm src/java/build/libs/onnxruntime-genai-*-sources.jar
  rm src/java/build/libs/onnxruntime-genai-*-javadoc.jar
  unzip -d build "src/java/build/libs/onnxruntime-genai-*.jar"

  cp -rf build/ai/onnxruntime/genai/native "$WORK_DIR"
fi

popd

find "$WORK_DIR/native" -type f
