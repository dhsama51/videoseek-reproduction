"""Qwen3.5-4B GGUF(Q4_K_M) + mmproj(F16) 다운로드 → $VSX_WORK/models, 경로를 model_paths.env 로 기록.
Turing GPU 는 bf16 연산을 지원하지 않으므로 F16 projector 를 우선 사용."""
import os
import sys

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "unsloth/Qwen3.5-4B-GGUF"
out = os.path.join(os.environ["VSX_WORK"], "models")
os.makedirs(out, exist_ok=True)
files = list_repo_files(REPO)


def pick(cands):
    for c in cands:
        for f in files:
            if os.path.basename(f) == c:
                return f
    return None


model = pick(["Qwen3.5-4B-Q4_K_M.gguf"])
mmproj = pick(["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"])
if not model or not mmproj:
    sys.exit(f"file not found: model={model}, mmproj={mmproj}")
mp = hf_hub_download(REPO, model, local_dir=out)
pp = hf_hub_download(REPO, mmproj, local_dir=out)
with open(os.path.join(out, "model_paths.env"), "w") as f:
    f.write(f'export VSX_MODEL_GGUF="{mp}"\nexport VSX_MMPROJ="{pp}"\nexport VSX_MODEL_REPO="{REPO}"\n')
print("model :", mp, round(os.path.getsize(mp) / 2**30, 2), "GiB")
print("mmproj:", pp, round(os.path.getsize(pp) / 2**30, 2), "GiB")
