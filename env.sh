# Source this file in every shell:  conda activate vseek && source env.sh
export VSX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VSX_WORK="${VSX_WORK:-$HOME/vseek_data}"          # large / regenerable files: videos, models, logs
export VSX_EXT="$VSX_ROOT/external/videoseek"             # author code (git-ignored, unmodified)
export LLAMA_DIR="$VSX_ROOT/external/llama.cpp"           # inference server (git-ignored)

# OpenAI-compatible local server; the author code reads the *name* OPENAI_API_BASE via os.getenv
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-local
export VSX_MODEL=openai/qwen3.5-4b

# Run settings used for all reported results (Exp 7 overrides VSX_THINK_MODE=faithful inline)
export VSX_THINK_MODE=lite        # lite: thinking only for reasoning/answer calls | faithful: follow reasoning_effort
export VSX_MAX_TOKENS=24576
export VSX_MEM_LIMIT_MB=5000      # per-question process memory guard
export VSX_TRACE=1                # per-call trace lines in per-question logs

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$VSX_ROOT:${PYTHONPATH:-}"
[ -f "$VSX_WORK/models/model_paths.env" ] && source "$VSX_WORK/models/model_paths.env"
mkdir -p "$VSX_WORK/logs"
