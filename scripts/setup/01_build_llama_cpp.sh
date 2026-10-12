#!/usr/bin/env bash
# llama.cpp(llama-server) 를 CUDA, Turing(sm_75) 전용으로 빌드. 고정 커밋 사용.
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
LLAMA_COMMIT=781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
fi
cd "$LLAMA_DIR"
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j 8
./build/bin/llama-server --version
