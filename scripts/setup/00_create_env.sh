#!/usr/bin/env bash
# conda env "vseek" (Python 3.11 + CUDA 12.8 nvcc + gcc 13, conda-forge 만 사용, sudo 불필요)
# + Python 패키지 + 저자 코드(VideoSeek, 고정 커밋) 설치.
#   - Ubuntu 26.04 기본 gcc(15)는 CUDA 12.x nvcc 와 호환되지 않고, 드라이버 576.80 은 CUDA 12.9 까지 지원하므로
#     conda-forge 의 gcc 13 + CUDA 12.8 조합을 사용함.
set -euo pipefail
cd "$(dirname "$0")/../.."
AUTHOR_COMMIT=443f8710bd1f9023f4a52ac95cb83610ad29c5ac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^vseek " || conda create -y -n vseek --override-channels -c conda-forge \
  python=3.11 cuda-version=12.8 cuda-nvcc cuda-cudart-dev libcublas-dev cuda-cccl \
  gcc_linux-64=13 gxx_linux-64=13 cmake ninja make ffmpeg
conda activate vseek
pip install -r requirements.txt

mkdir -p external
if [ ! -d external/videoseek ]; then
  git clone https://github.com/jylins/videoseek external/videoseek
  git -C external/videoseek checkout "$AUTHOR_COMMIT"
fi
# openai-whisper(torch 동반)는 사용하지 않으므로 의존성 없이 설치
pip install -e external/videoseek --no-deps
python -c "import decord, litellm, videoseek, config; print('env OK')"
