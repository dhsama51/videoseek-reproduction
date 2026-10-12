#!/usr/bin/env bash
# Exp 1. Reproduction: VideoSeek vs. uniform single-pass (main 63문항, 자막 없음)
#   videoseek           : 저자 VideoSeekAgent (lite thinking 설정)
#   single_pass_64      : 영상 전체 uniform 64 frame, 호출 1회
#   single_pass_matched : uniform N frame, N = videoseek 의 평균 #frames (주 비교: videoseek − single_pass_matched)
source "$(dirname "$0")/_common.sh"
OUT=e1_reproduction
step "videoseek"      python -m vsx.agent_runner --out $OUT/videoseek --uids subset:main
step "single_pass_64" python -m vsx.single_pass --out $OUT/single_pass_64 --n_frames 64 --uids subset:main
N=$(python - <<'PY'
import glob, json, os
from vsx import common as C
xs = [json.load(open(p)).get("metrics", {}).get("n_frames") for p in glob.glob(os.path.join(C.result_dir("e1_reproduction/videoseek"), "*.json"))]
xs = [x for x in xs if x]
print(min(384, max(16, round(sum(xs) / len(xs)))))
PY
)
echo "$N" > results/$OUT/single_pass_matched_N.txt
step "single_pass_matched (N=$N)" python -m vsx.single_pass --out $OUT/single_pass_matched --n_frames "$N" --uids subset:main
