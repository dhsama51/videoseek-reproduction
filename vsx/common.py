"""공통 유틸: 경로, 문항/부분집합 로드, 저자 config 구성, 선택지 추출, decord 우회 reader."""
import json
import os
import re

ROOT = os.environ.get("VSX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("VSX_WORK", os.path.expanduser("~/vseek_data"))   # 영상·모델 등 대용량 (레포 밖)
DATA = os.path.join(ROOT, "data")                                        # 레포에 포함되는 작은 데이터
RESULTS = os.path.join(ROOT, "results")
QFILE = os.path.join(DATA, "questions.jsonl")

# README 예시(LVBench)와 같은 형식: 질문(선택지 포함) + 답 지시문
ANSWER_SUFFIX = "\nPlease directly answer with the best option's letter from the given choices directly (A, B, C, or D)."


# ---------------------------------------------------------------- paths
def rel(path):
    """레포 또는 작업 폴더 기준 상대 경로로 기록 (결과 파일에 절대 경로를 남기지 않음)."""
    if not path:
        return path
    p = os.path.abspath(path)
    for base, tag in ((ROOT, ""), (WORK, "$VSX_WORK/")):
        b = os.path.abspath(base) + os.sep
        if p.startswith(b):
            return tag + p[len(b):]
    return path


def resolve(path):
    """rel() 로 기록한 경로를 실제 경로로 복원."""
    if not path:
        return path
    if path.startswith("$VSX_WORK/"):
        return os.path.join(WORK, path[len("$VSX_WORK/"):])
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def result_dir(name):
    """name 예: 'e1_reproduction/videoseek'."""
    return os.path.join(RESULTS, name)


def video_path(row, video_dir=None):
    return os.path.join(video_dir or os.path.join(WORK, "videos"), f"{row['video_key']}.mp4")


def subtitle_path(row, subtitle_dir=None):
    if not subtitle_dir:
        return None
    p = os.path.join(resolve(subtitle_dir), f"{row['uid']}.srt")
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------- questions / subsets
def load_questions():
    return {r["uid"]: r for r in (json.loads(l) for l in open(QFILE, encoding="utf-8"))}


def subset_file(name):
    return os.path.join(DATA, "subsets", f"{name}.txt")


def read_subset(name):
    return [l.strip() for l in open(subset_file(name)) if l.strip()]


def select_uids(spec, qs=None):
    """subset:<name> | uid1,uid2,..."""
    if spec.startswith("subset:"):
        return read_subset(spec.split(":", 1)[1])
    return [u for u in spec.split(",") if u]


def user_query(row):
    return row["question"].rstrip() + ANSWER_SUFFIX


# ---------------------------------------------------------------- author config
def build_config():
    """저자 cli.build_agent_config 결과(general.yaml + prompts.yaml + init_config)와 동일하게 구성하고
    model_name / api_base(local server) / max_tokens 만 바꿈."""
    from config import general_config, prompts_config
    c = {}
    c.update(general_config)
    c.update(prompts_config)
    c["model_name"] = os.environ["VSX_MODEL"]
    c["api_base"] = os.getenv("OPENAI_API_BASE")  # init_config 와 동일 (env 이름 → 값)
    c["api_key"] = os.getenv("OPENAI_API_KEY")
    c["api_version"] = None
    c["max_tokens"] = int(os.environ.get("VSX_MAX_TOKENS", "24576"))
    return c


# ---------------------------------------------------------------- scoring
_PATTERNS = [
    re.compile(r"^\s*\**\(([A-D])\)"),
    re.compile(r"^\s*\**([A-D])(?:[\.\:\),\*]|\s*$)"),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\**\(?([A-D])\)?(?![A-Za-z])", re.I),
]


def extract_choice(text):
    """응답에서 선택지 문자 추출. 실패 시 None (오답으로 채점하고 extract_failed 로 표시)."""
    if not text:
        return None
    t = text.strip()
    for p in _PATTERNS:
        m = p.search(t)
        if m:
            return m.group(1).upper()
    paren = set(re.findall(r"\(([A-D])\)", t))
    if len(paren) == 1:
        return paren.pop()
    return None


# ---------------------------------------------------------------- io
def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- decord
class Batch:
    """decord get_batch 결과 대체 (저자 코드는 .asnumpy() 만 사용)."""

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


def safe_reader_class():
    """decord 메모리 누수 우회: get_batch 직후 seek(0).
    (get_batch 가 영상 중간에서 끝나면 내부 디코더가 멈추지 않고 frame 을 계속 쌓아 수 초 만에 수 GB 를 점유함)"""
    import decord

    class SafeVideoReader(decord.VideoReader):
        on_batch = None  # 호출별 frame 기록용 hook (agent_runner 에서 설정)

        def get_batch(self, indices):
            idx = [int(i) for i in indices]
            if SafeVideoReader.on_batch is not None:
                SafeVideoReader.on_batch(idx)
            arr = super().get_batch(indices).asnumpy()
            self.seek(0)
            return Batch(arr)
    return SafeVideoReader


def safe_reader(path):
    return safe_reader_class()(path)
