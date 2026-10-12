#!/usr/bin/env bash
# Exp 4. Policy-decoupled observation guardrail (Exp 1, 3 결과 필요)
#   judge  : 자막 줄과 viewer 출력(o_t)을 같은 모델 + 별도 prompt 로 INJECTION/CLEAN 판정
#   replay : 판정된 텍스트를 제거한 history 로 VideoSeek 답 단계만 재생 (필터 없는 재생과 비교)
#   spprep + single_pass_64_defended : 판정된 자막 줄을 제거하고 single-pass 재실행
source "$(dirname "$0")/_common.sh"
OUT=e4_guardrail
step "judge / replay / spprep" python -m vsx.guardrail judge replay spprep
step "single_pass_64_defended" python -m vsx.single_pass --out $OUT/single_pass_64_defended --n_frames 64 --uids subset:inject \
     --subtitle_dir results/$OUT/defended_subtitles
