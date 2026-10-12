#!/usr/bin/env bash
# Exp 6 (exploratory). Forced-choice answer replay: Exp 1 VideoSeek 의 탐색 기록을 고정하고 답 단계만 재생
source "$(dirname "$0")/_common.sh"
step "answer replay" python -m vsx.answer_replay --source e1_reproduction/videoseek
