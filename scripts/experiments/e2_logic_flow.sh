#!/usr/bin/env bash
# Exp 2. Logic-flow ablation: 45초 segment 순서를 섞은 영상에서 VideoSeek / single-pass 재실행 (logic 30문항)
#   원본 영상 결과는 Exp 1 을 재사용. 섞은 영상은 scripts/data/04_make_shuffled_videos.py 로 미리 생성.
source "$(dirname "$0")/_common.sh"
OUT=e2_logic_flow
VDIR="$VSX_WORK/videos_shuffled"
step "videoseek_shuffled"      python -m vsx.agent_runner --out $OUT/videoseek_shuffled --uids subset:logic --video_dir "$VDIR"
step "single_pass_64_shuffled" python -m vsx.single_pass --out $OUT/single_pass_64_shuffled --n_frames 64 --uids subset:logic --video_dir "$VDIR"
