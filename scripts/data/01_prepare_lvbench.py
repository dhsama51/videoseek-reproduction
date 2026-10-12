"""
LVBench 데이터 준비 (CPU / 네트워크만 사용).

  meta     : THUDM/LVBench video_info.meta.jsonl 다운로드              → $VSX_WORK/meta/
  index    : lmms-lab/LVBench 미러의 video_chunks/*.zip 내부 목록을 HTTP range 로 조회 (zip 전체를 받지 않음)
  select   : 영상 단위 층화 추출(seed 고정) + 영상당 7문항 + 부분집합 rank 부여  → $VSX_WORK/meta/questions_draft.jsonl
  extract  : data/videos.json 의 영상 mp4 만 원격 zip 에서 추출          → $VSX_WORK/videos/<key>.mp4, data/video_probe.json
  finalize : 실제 영상 길이 기준 decoy 구간, 주입 시점 / 목표 오답 확정  → data/questions.jsonl

레포에 포함된 data/questions.jsonl, data/videos.json 을 그대로 쓰려면 meta, index, extract 만 실행하면 됨.
select / finalize 는 그 파일들을 만든 과정이며, 다시 실행하면 레포의 선택 결과를 덮어씀.

사용: python scripts/data/01_prepare_lvbench.py meta index extract
옵션: --n_videos 14 --per_video 7 --seed 2026 --full_chunk (range 요청이 실패하면 chunk 전체를 받아서 추출)
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter

WORK = os.environ["VSX_WORK"]
REPO_DATA = os.path.join(os.environ["VSX_ROOT"], "data")
META = os.path.join(WORK, "meta")
VID = os.path.join(WORK, "videos")
for d in (META, VID, REPO_DATA):
    os.makedirs(d, exist_ok=True)

MIRROR = "lmms-lab/LVBench"
CATS = ["entity recognition", "event understanding", "key information retrieval",
        "temporal grounding", "reasoning", "summarization"]
TEMPORAL_KW = re.compile(r"\b(first|last|before|after|then|begin|begins|beginning|end|ends|ending)\b", re.I)


# ---------------------------------------------------------------- utils
def parse_tref(s):
    if not s:
        return None
    m = re.findall(r"\d+(?::\d+){1,2}", str(s))
    if len(m) < 2:
        return None

    def sec(x):
        r = 0
        for p in x.split(":"):
            r = r * 60 + int(p)
        return r
    a, b = sec(m[0]), sec(m[1])
    return (min(a, b), max(a, b))


def norm_types(t):
    if isinstance(t, str):
        t = [t]
    return sorted({x.strip().lower() for x in (t or [])})


def load_meta():
    path = os.path.join(META, "video_info.meta.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def hf_headers():
    from huggingface_hub import get_token
    t = get_token()
    return {"Authorization": f"Bearer {t}"} if t else {}


def mirror_url(path):
    return f"https://huggingface.co/datasets/{MIRROR}/resolve/main/{path}"


# ---------------------------------------------------------------- stages
def stage_meta(_):
    from huggingface_hub import hf_hub_download
    out = os.path.join(META, "video_info.meta.jsonl")
    if os.path.exists(out):
        print("meta: exists")
        return
    p = hf_hub_download("THUDM/LVBench", "video_info.meta.jsonl", repo_type="dataset", local_dir=META)
    recs = load_meta()
    nq = sum(len(r["qa"]) for r in recs)
    print(f"meta: {len(recs)} videos, {nq} questions -> {p}")
    print("record keys:", list(recs[0].keys()), "| qa keys:", list(recs[0]["qa"][0].keys()))
    print("types:", Counter(t for r in recs for q in r["qa"] for t in norm_types(q.get("question_type"))))


def stage_index(args):
    out = os.path.join(META, "chunk_index.json")
    if os.path.exists(out):
        print("index: exists")
        return
    from huggingface_hub import list_repo_files
    from remotezip import RemoteZip
    files = [f for f in list_repo_files(MIRROR, repo_type="dataset") if f.endswith(".zip")]
    print("mirror zips:", files)
    index = {}
    for f in files:
        try:
            with RemoteZip(mirror_url(f), headers=hf_headers()) as z:
                for info in z.infolist():
                    stem = os.path.splitext(os.path.basename(info.filename))[0]
                    if info.filename.endswith(".mp4"):
                        index[stem] = {"zip": f, "member": info.filename, "size": info.file_size}
            print(f"  {f}: ok ({len(index)} total)")
        except Exception as e:
            print(f"  {f}: range listing FAILED ({e!r}) -> use --full_chunk")
    json.dump(index, open(out, "w"), indent=1)
    print(f"index: {len(index)} mp4 in mirror -> {out}")


def stage_select(args):
    out_v = os.path.join(REPO_DATA, "videos.json")
    if os.path.exists(out_v):
        print("select: data/videos.json exists (다시 선택하려면 지우고 실행)")
        return
    recs = load_meta()
    index = json.load(open(os.path.join(META, "chunk_index.json")))
    rng = random.Random(args.seed)

    # 전체 LVBench 의 task 유형 비율 (multi-label occurrence 기준) = 층화 목표
    occ = Counter(t for r in recs for q in r["qa"] for t in norm_types(q.get("question_type")))
    tot = sum(occ[c] for c in CATS)
    target = {c: occ[c] / tot for c in CATS}

    cand = []
    for r in recs:
        key = r["key"]
        qs = [q for q in r["qa"] if parse_tref(q.get("time_reference"))]
        if key in index and len(qs) >= args.per_video:
            dur = (r.get("video_info") or {}).get("duration_minutes")
            cand.append({"key": key, "duration_min": dur, "type": r.get("type"), "qa": qs})
    print(f"select: {len(cand)} candidate videos (mirror 존재 & clue 있는 문항 >= {args.per_video})")

    # 길이 중앙값으로 short/long 분할 → 섞은 뒤 번갈아 뽑음 (앞에서 k개를 잘라도 길이 균형 유지)
    durs = sorted(c["duration_min"] or 0 for c in cand)
    med = durs[len(durs) // 2]
    short = [c for c in cand if (c["duration_min"] or 0) < med]
    long_ = [c for c in cand if (c["duration_min"] or 0) >= med]
    rng.shuffle(short)
    rng.shuffle(long_)
    order = []
    for a, b in zip(short, long_):
        order += [a, b]
    order = order[: args.n_videos]

    counts = Counter()
    selected_q = []
    for vi, v in enumerate(order):
        pool = list(v["qa"])
        rng.shuffle(pool)
        for qi in range(args.per_video):
            best, best_d = None, 1e9
            for q in pool:
                c2 = counts + Counter(norm_types(q.get("question_type")))
                s = sum(c2[c] for c in CATS) or 1
                d = sum(abs(c2[c] / s - target[c]) for c in CATS)
                if d < best_d:
                    best, best_d = q, d
            pool.remove(best)
            counts += Counter(norm_types(best.get("question_type")))
            selected_q.append((vi, qi, v, best))

    rows = []
    for vi, qi, v, q in selected_q:
        cs, ce = parse_tref(q["time_reference"])
        stem = q["question"].split("\n")[0]
        types = norm_types(q.get("question_type"))
        rows.append({
            "uid": str(q.get("uid") if q.get("uid") is not None else f"{v['key']}_{q['question'][:20]}"), "video_key": v["key"], "video_order": vi, "q_order": qi,
            "question": q["question"], "answer": q["answer"].strip().upper(), "question_type": types,
            "time_reference": q["time_reference"], "clue_start": cs, "clue_end": ce,
            "elig_logic": (not ({"temporal grounding", "summarization"} & set(types))) and not TEMPORAL_KW.search(stem),
            "elig_inject": ce > cs,
        })
    # 부분집합 rank: 고정 seed 무작위 순위 → 실험별로 상위 M 개를 사용 (scripts/data/02_make_subsets.py)
    for key in ("logic", "inject", "oracle"):
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        for rank, i in enumerate(perm):
            rows[i][f"rank_{key}"] = rank
    json.dump([{k: v[k] for k in ("key", "duration_min", "type")} | {"video_order": i}
               for i, v in enumerate(order)], open(out_v, "w"), indent=1)
    with open(os.path.join(META, "questions_draft.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sel_occ = Counter(t for r in rows for t in r["question_type"])
    s = sum(sel_occ[c] for c in CATS)
    print("task 비율 (목표 -> 선택):")
    for c in CATS:
        print(f"  {c:28s} {target[c]:.3f} -> {sel_occ[c] / s:.3f}")
    print(f"videos: {[(v['key'], v['duration_min']) for v in order]}")
    print(f"elig_logic={sum(r['elig_logic'] for r in rows)}, elig_inject={sum(r['elig_inject'] for r in rows)} / {len(rows)}")


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,codec_name:format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    j = json.loads(r.stdout)
    st = j["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    return {"width": st["width"], "height": st["height"], "fps": float(num) / float(den),
            "codec": st["codec_name"], "duration": float(j["format"]["duration"])}


def stage_extract(args):
    index = json.load(open(os.path.join(META, "chunk_index.json")))
    videos = json.load(open(os.path.join(REPO_DATA, "videos.json")))
    for v in videos:
        key = v["key"]
        dst = os.path.join(VID, f"{key}.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  {key}: exists")
            continue
        info = index[key]
        tmp = os.path.join(VID, "_tmp")
        os.makedirs(tmp, exist_ok=True)
        if not args.full_chunk:
            from remotezip import RemoteZip
            with RemoteZip(mirror_url(info["zip"]), headers=hf_headers()) as z:
                z.extract(info["member"], tmp)
        else:
            from huggingface_hub import hf_hub_download
            import zipfile
            zp = hf_hub_download(MIRROR, info["zip"], repo_type="dataset", local_dir=os.path.join(WORK, "_chunks"))
            with zipfile.ZipFile(zp) as z:
                z.extract(info["member"], tmp)
        shutil.move(os.path.join(tmp, info["member"]), dst)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {key}: {os.path.getsize(dst) / 2**20:.0f} MiB")
    probe = {}
    for v in videos:
        p = os.path.join(VID, f"{v['key']}.mp4")
        probe[v["key"]] = ffprobe(p)
        print(f"  {v['key']}: {probe[v['key']]}")
    json.dump(probe, open(os.path.join(REPO_DATA, "video_probe.json"), "w"), indent=1)


def stage_finalize(args):
    probe = json.load(open(os.path.join(REPO_DATA, "video_probe.json")))
    rows = [json.loads(l) for l in open(os.path.join(META, "questions_draft.jsonl"))]
    rng = random.Random(args.seed + 1)
    out = []
    for r in rows:
        D = probe[r["video_key"]]["duration"]
        cs, ce = r["clue_start"], min(r["clue_end"], D)
        L = max(ce - cs, 1.0)
        # decoy: clue 와 겹치지 않는 같은 길이 구간
        choices = []
        if cs - L >= 0:
            choices.append((0.0, cs - L))
        if ce <= D - L:
            choices.append((ce, D - L))
        if choices:
            lo, hi = rng.choice(choices)
            ds = rng.uniform(lo, hi)
            r["decoy_start"], r["decoy_end"] = round(ds, 1), round(ds + L, 1)
        else:
            r["decoy_start"] = r["decoy_end"] = None
        r["elig_oracle"] = r["elig_inject"] and r["decoy_start"] is not None
        wrong = [c for c in "ABCD" if c != r["answer"]]
        r["inject_target"] = rng.choice(wrong)
        r["inject_time"] = round(rng.uniform(cs, max(cs, ce - 2.0)), 1)
        r["duration"] = D
        out.append(r)
    with open(os.path.join(REPO_DATA, "questions.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"finalize: {len(out)} questions, elig_oracle={sum(r['elig_oracle'] for r in out)}")
    print("clue length (s): min/median/max =",
          sorted(r["clue_end"] - r["clue_start"] for r in out)[0],
          sorted(r["clue_end"] - r["clue_start"] for r in out)[len(out) // 2],
          sorted(r["clue_end"] - r["clue_start"] for r in out)[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["meta", "index", "select", "extract", "finalize"])
    ap.add_argument("--n_videos", type=int, default=14)
    ap.add_argument("--per_video", type=int, default=7)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--full_chunk", action="store_true")
    a = ap.parse_args()
    stages = ["meta", "index", "select", "extract", "finalize"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"===== {s} =====")
        globals()[f"stage_{s}"](a)
