"""
Exp 4 입력측 오탐률(FPR) 측정용 깨끗한 자막: main 영상들의 YouTube 영어 자막(수동/자동)을 받음 → $VSX_WORK/yt_subs/
  - 제3자 저작물이므로 레포에는 포함하지 않음 (결과 파일에는 판정 결과와 원문 일부만 남음)
  - yt-dlp 의 JavaScript 런타임 부재, 영상 삭제 등으로 일부 영상은 실패할 수 있음 (보고 결과: 9개 중 6개 성공)
"""
import os
import subprocess

from vsx import common as C

OUT = os.path.join(C.WORK, "yt_subs")


def main():
    qs = C.load_questions()
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({qs[u]["video_key"] for u in C.read_subset("main")})
    ok = 0
    for k in keys:
        if any(f.startswith(k) and f.endswith(".srt") for f in os.listdir(OUT)):
            ok += 1
            continue
        cmd = ["yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
               "--sub-langs", "en,en-US,en-GB,en-orig", "--convert-subs", "srt",
               "-o", os.path.join(OUT, "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={k}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        got = [f for f in os.listdir(OUT) if f.startswith(k) and f.endswith(".srt")]
        ok += bool(got)
        print(f"  {k}: {'OK ' + str(got) if got else 'FAIL ' + (p.stderr.strip().splitlines() or [''])[-1][:160]}", flush=True)
    print(f"  youtube subtitles: {ok}/{len(keys)} videos")


if __name__ == "__main__":
    main()
