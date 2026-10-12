#!/usr/bin/env bash
# Exp 5. Context-anchored viewer: 정답 구간(oracle)·무관 구간(decoy)에서 viewer 단일 호출, overview storyline 유무 비교
source "$(dirname "$0")/_common.sh"
step "overview + oracle/decoy viewer" python -m vsx.anchored_viewer overview viewer
