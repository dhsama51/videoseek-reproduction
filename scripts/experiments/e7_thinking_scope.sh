#!/usr/bin/env bash
# Exp 7 (exploratory). Thinking scope: 모든 호출에 reasoning_effort 대로 thinking 을 적용(faithful)한 VideoSeek (faithful 20문항)
#   lite 결과는 Exp 1 을 재사용. 문항 목록은 data/subsets/faithful.txt (고정).
source "$(dirname "$0")/_common.sh"
step "videoseek_faithful" env VSX_THINK_MODE=faithful python -m vsx.agent_runner --out e7_thinking_scope/videoseek_faithful --uids subset:faithful
