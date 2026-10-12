# 실험 스크립트 공통: 환경 로드, 서버 확인, 단계 실행 시간 기록
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env.sh
step() { local s=$SECONDS; echo "== [$(date '+%m-%d %T')] $1"; shift; "$@"; echo "== done in $(( (SECONDS - s) / 60 )) min"; }
if ! curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo "llama-server is not running. Start it first: bash scripts/setup/03_start_server.sh" >&2; exit 1
fi
