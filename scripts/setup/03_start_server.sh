#!/usr/bin/env bash
# llama-server 실행 (모든 실험 결과에 사용한 설정).
#   - ctx 65536, KV cache q8_0, 단일 slot
#   - --reasoning-format deepseek : 추론 텍스트를 content 와 분리 (z_t / o_t 에는 최종 content 만 들어감)
#   - --reasoning-budget 4096     : thinking ON 호출의 추론 상한 (reasoning_effort=medium 대응, 사전 고정)
#   - sampling: Qwen3.5 thinking 권장값 (top_p 0.95, top_k 20, min_p 0, presence 1.5). temperature 는 요청값(저자 config 1.0)
#   - --image-max-tokens 1024     : 이미지당 최대 약 1280x720 상당 (1080p focus frame 은 서버에서 축소)
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
CTX=${CTX:-65536}
BUDGET=${BUDGET:-4096}
LOG="$VSX_WORK/logs/server_$(date +%m%d_%H%M)_ctx${CTX}_b${BUDGET}.log"
echo "log: $LOG"
"$LLAMA_DIR/build/bin/llama-server" \
  -m "$VSX_MODEL_GGUF" --mmproj "$VSX_MMPROJ" \
  --host 127.0.0.1 --port 8080 -a qwen3.5-4b \
  -c "$CTX" -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 \
  --jinja --reasoning-format deepseek \
  --reasoning-budget "$BUDGET" \
  --reasoning-budget-message "Considering the limited time, I have to give the solution based on the thinking directly now." \
  --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 \
  --image-max-tokens 1024 --metrics 2>&1 | tee "$LOG"
