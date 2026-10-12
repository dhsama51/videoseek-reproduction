#!/usr/bin/env bash
# Exp 3. Subtitle indirect prompt injection: 주입 문장 1줄짜리 자막으로 VideoSeek / single-pass 실행 (inject 30문항)
#   깨끗한 조건은 Exp 1 결과를 재사용. 자막은 scripts/data/03_make_injection_subtitles.py 로 생성(레포에 포함).
source "$(dirname "$0")/_common.sh"
OUT=e3_subtitle_injection
SUBS=data/injection_subtitles
step "videoseek_attack"      python -m vsx.agent_runner --out $OUT/videoseek_attack --uids subset:inject --subtitle_dir $SUBS
step "single_pass_64_attack" python -m vsx.single_pass --out $OUT/single_pass_64_attack --n_frames 64 --uids subset:inject --subtitle_dir $SUBS
