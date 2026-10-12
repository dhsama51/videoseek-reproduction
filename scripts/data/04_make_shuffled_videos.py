"""
Exp 2 영상 변환: 45초 segment 의 순서만 고정 seed 로 섞은 영상 생성 (segment 내부 순서 유지)
  - 영상: $VSX_WORK/videos_shuffled/<key>.mp4 (libx264 crf 18 veryfast, 해상도·fps 유지, 오디오 없음)
  - 매핑: data/shuffled_videos/<key>.mapping.json (원래 구간 → 새 위치), data/shuffled_videos/clues.json (변환된 clue)
  - segment 를 정확한 시점에서 재인코딩한 뒤 concat(copy) 하므로 새 영상 길이는 원본과 1초 이내로 일치
사용: python scripts/data/04_make_shuffled_videos.py [--threads 3]
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time

from vsx import common as C

SEG = 45.0
SEED = 2026
VID_OUT = os.path.join(C.WORK, "videos_shuffled")
MAP_OUT = os.path.join(C.DATA, "shuffled_videos")


def ffprobe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def map_interval(a, b, segments):
    """원래 구간 [a, b] → 새 영상에서의 구간 목록."""
    out = []
    for m in segments:
        lo, hi = max(a, m["orig_start"]), min(b, m["orig_end"])
        if hi > lo or (a == b and m["orig_start"] <= a < m["orig_end"]):
            off = m["new_start"] - m["orig_start"]
            out.append([round(lo + off, 2), round(max(hi, lo) + off, 2)])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    qs = C.load_questions()
    probe = json.load(open(os.path.join(C.DATA, "video_probe.json")))
    logic = C.read_subset("logic")
    keys = sorted({qs[u]["video_key"] for u in logic})
    os.makedirs(VID_OUT, exist_ok=True)
    os.makedirs(MAP_OUT, exist_ok=True)
    print(f"  videos to shuffle: {len(keys)}", flush=True)
    for k in keys:
        out = os.path.join(VID_OUT, f"{k}.mp4")
        mp = os.path.join(MAP_OUT, f"{k}.mapping.json")
        if os.path.exists(out) and os.path.exists(mp):
            print(f"  {k}: exists")
            continue
        t = time.time()
        src = C.video_path({"video_key": k})
        D = probe[k]["duration"]
        nseg = math.ceil(D / SEG)
        perm = list(range(nseg))
        random.Random(f"{SEED}-{k}").shuffle(perm)          # 새 위치 p 에 원래 segment perm[p]
        tmp = os.path.join(VID_OUT, f"_tmp_{k}")
        os.makedirs(tmp, exist_ok=True)
        lens = {}
        for i in range(nseg):
            seg = os.path.join(tmp, f"{i:04d}.mp4")
            if not os.path.exists(seg):
                subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-ss", f"{i * SEG}", "-i", src,
                                "-t", f"{SEG}", "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "18", "-pix_fmt", "yuv420p", "-threads", str(a.threads), seg + ".part.mp4"],
                               check=True)
                os.replace(seg + ".part.mp4", seg)
            lens[i] = ffprobe_dur(seg)
        with open(os.path.join(tmp, "list.txt"), "w") as f:
            for i in perm:
                f.write(f"file '{os.path.join(tmp, f'{i:04d}.mp4')}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
                        "-c", "copy", out + ".part.mp4"], check=True)
        os.replace(out + ".part.mp4", out)
        mapping, pos = [], 0.0
        for i in perm:
            mapping.append({"seg": i, "orig_start": i * SEG, "orig_end": min((i + 1) * SEG, D),
                            "new_start": round(pos, 3), "new_end": round(pos + lens[i], 3)})
            pos += lens[i]
        mapping.sort(key=lambda m: m["seg"])
        json.dump({"seg_len": SEG, "perm": perm, "segments": mapping, "orig_duration": D, "new_duration": ffprobe_dur(out)},
                  open(mp, "w"), indent=1)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {k}: {nseg} segments, {D:.1f}s -> {ffprobe_dur(out):.1f}s, {(time.time() - t) / 60:.1f} min", flush=True)
    clues = {}
    for u in logic:
        r = qs[u]
        segs = json.load(open(os.path.join(MAP_OUT, f"{r['video_key']}.mapping.json")))["segments"]
        clues[u] = {"orig": [r["clue_start"], r["clue_end"]], "new": map_interval(r["clue_start"], r["clue_end"], segs)}
    json.dump(clues, open(os.path.join(MAP_OUT, "clues.json"), "w"), indent=1)
    print(f"  clue mapping: data/shuffled_videos/clues.json ({len(clues)} questions)")


if __name__ == "__main__":
    main()
