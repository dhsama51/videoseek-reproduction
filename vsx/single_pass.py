"""
Uniform single-pass baseline (논문의 GPT-5 Base 대응. 논문은 'uniform 384 frames' 외 입력 형식을 명시하지 않아
저자 코드의 관례를 그대로 사용).

  - 영상 전체에서 N frame uniform sampling (overview tool 과 같은 np.linspace(0, total-1, N))
  - 짧은 변 256 bicubic resize, JPEG base64 (overview / skim 과 같은 전처리)
  - frame 마다 "{t:.1f}s" 텍스트 + 이미지 (skim / focus 형식)
  - 끝에 VideoSeek 첫 질문과 같은 "Video Subtitles:\\n...\\n\\nQuestion:\\n..." 블록 (자막이 없으면 "No subtitle found.")
  - 호출 1회, thinking ON (reasoning_effort=medium 대응)

사용 예:
  python -m vsx.single_pass --out e1_reproduction/single_pass_64 --n_frames 64 --uids subset:main
"""
import argparse
import base64
import io
import os
import time
import traceback

import numpy as np
from PIL import Image

from vsx import common as C
from vsx import llm_adapter as A


def encode_frames(vr, n):
    fps = vr.get_avg_fps()
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    frames = vr.get_batch(idx).asnumpy()
    _, h, w, _ = frames.shape
    s = 256 / min(h, w)
    th, tw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    out = []
    for f, i in zip(frames, idx):
        buf = io.BytesIO()
        Image.fromarray(f).resize((tw, th), Image.BICUBIC).save(buf, format="jpeg")
        out.append((round(i / fps, 1), base64.b64encode(buf.getvalue()).decode()))
    return idx.tolist(), out


def run_question(row, n, subtitle_dir, config, video_dir=None):
    from videoseek.utils import convert_to_free_form_text_representation, load_subtitles
    vr = C.safe_reader(C.video_path(row, video_dir))
    duration = len(vr) / vr.get_avg_fps()
    idx, frames = encode_frames(vr, n)
    subs_str = convert_to_free_form_text_representation(load_subtitles(C.subtitle_path(row, subtitle_dir)), content_type="subtitle")
    content = [{"type": "text", "text": f"Video Duration: {duration:.1f}s\nThe following {n} video frames are uniformly sampled from the entire video:\n"}]
    for t, b in frames:
        content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": f"Video Subtitles:\n{subs_str}\n\nQuestion:\n{C.user_query(row)}"})
    calls = []
    with A.sink(calls), A.kind("single_pass"):
        resp = A.completion(model=config["model_name"], api_base=config["api_base"], api_key=config["api_key"],
                            messages=[{"role": "user", "content": content}],
                            max_completion_tokens=config["max_tokens"], seed=config["seed"],
                            temperature=config["temperature"], reasoning_effort="medium", timeout=900)
    return resp.choices[0].message.content, idx, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/ 아래 출력 폴더, 예: e1_reproduction/single_pass_64")
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--uids", required=True, help="subset:<name> 또는 uid 목록(쉼표 구분)")
    ap.add_argument("--subtitle_dir", default=None)
    ap.add_argument("--video_dir", default=None)
    a = ap.parse_args()
    config = C.build_config()
    qs = C.load_questions()
    uids = C.select_uids(a.uids, qs)
    outdir = C.result_dir(a.out)
    print(f"[{a.out}] {len(uids)} questions, N={a.n_frames}", flush=True)
    done_t, t_all = [], time.time()
    for i, uid in enumerate(uids):
        out = os.path.join(outdir, f"{uid}.json")
        if os.path.exists(out):
            print(f"  [{i + 1}/{len(uids)}] {uid}: exists, skip", flush=True)
            continue
        row = qs[uid]
        t = time.time()
        raw, idx, calls, err = None, [], [], None
        try:
            raw, idx, calls = run_question(row, a.n_frames, a.subtitle_dir, config, a.video_dir)
        except Exception:
            err = traceback.format_exc()[-2000:]
        pred = C.extract_choice(raw)
        dt = time.time() - t
        c0 = calls[0] if calls else {}
        C.save_json(out, {
            "uid": uid, "cond": a.out, "video_key": row["video_key"], "question": row["question"],
            "question_type": row["question_type"], "gt": row["answer"], "clue": [row["clue_start"], row["clue_end"]],
            "final_answer_raw": raw, "pred": pred, "correct": pred == row["answer"], "extract_failed": pred is None,
            "error": err, "runtime_s": round(dt, 1), "frame_indices": idx,
            "metrics": {"n_frames": len(idx), "n_turns": 0, "tokens_prompt": c0.get("prompt_tokens"),
                        "tokens_completion": c0.get("completion_tokens"), "finish_reason": c0.get("finish_reason")},
            "llm_calls": calls})
        done_t.append(dt)
        avg = sum(done_t) / len(done_t)
        print(f"  [{i + 1}/{len(uids)}] {uid} pred={pred} gt={row['answer']} {'O' if pred == row['answer'] else 'X'} "
              f"{dt:.0f}s | ETA {(len(uids) - i - 1) * avg / 60:.0f}min{' ERR' if err else ''}", flush=True)
    print(f"[{a.out}] total {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
