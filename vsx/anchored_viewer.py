"""
Exp 5: context-anchored viewer (agent 없이 viewer 단일 호출로 perception 분리)
  anchor : 영상별 저자 overview tool 1회 실행 → storyline (질문을 받지 않으므로 영상 단위 재사용)
  viewer : 저자 focus tool 과 같은 입력 형식/샘플링(구간 안 min(frame수,16) 장 uniform, resize 없음, 같은 prompt 문구)
           독립  = focus prompt 그대로 (query = LVBench 질문+선택지+답 지시)
           anchor = 같은 prompt 에서 "Question:" 앞에 "Video Storyline (from overview):\n{storyline}\n\n" 삽입
  구간   : oracle = clue (main 중 clue>0 전체) / decoy = clue 와 겹치지 않는 같은 길이 (subset:oracle, 30문항)
  호출 종류는 focus 와 동일하게 취급(kind=focus, reasoning_effort=medium → 현재 THINK_MODE 정책 적용)
사용: python -m vsx.anchored_viewer all  (또는 overview|viewer)
"""
import argparse
import base64
import json
import os
import time
from io import BytesIO

import numpy as np
from PIL import Image

from vsx import common as C
from vsx import llm_adapter as A

OUT = C.result_dir("e5_anchored_viewer")
OV_DIR = os.path.join(OUT, "overview")
NO_REL = "No relevant content found."


def focus_content(vr, start_time, end_time, query, storyline=None):
    """저자 tools/focus.py 의 content 구성과 동일 (자막 없음). storyline 이 있으면 Question 앞에 삽입."""
    from config import general_config
    from videoseek.utils import convert_to_free_form_text_representation
    max_num_frames = general_config["frame_sampling_factor"] * general_config["focus_base"]
    subtitles_str = convert_to_free_form_text_representation([], content_type="subtitle")
    fps = vr.get_avg_fps()
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), len(vr) - 1)
    num_frames = min(int(end_frame - start_frame), max_num_frames)
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    ts = [round(i / fps, 1) for i in frame_indices]
    frames = vr.get_batch(frame_indices).asnumpy()
    content = [{"type": "text", "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    for frame, t in zip(frames, ts):
        buf = BytesIO()
        Image.fromarray(frame).save(buf, format="jpeg")
        content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"}})
    anchor = f"Video Storyline (from overview):\n{storyline}\n\n" if storyline else ""
    content.append({"type": "text", "text": (
        "Video Subtitles:\n"
        f"{subtitles_str}\n\n"
        f"{anchor}"
        f"Question:\n{query}\n\n"
        "Please answer the question based on the given video clip. "
        "If the clip is not related to the question, please return 'No relevant content found.'")})
    return content, [int(i) for i in frame_indices]


def stage_overview(_):
    from videoseek.tools.overview import execute_overview
    A.install()
    config = C.build_config()
    qs = C.load_questions()
    os.makedirs(OV_DIR, exist_ok=True)
    keys = sorted({qs[u]["video_key"] for u in C.read_subset("oracle_all")})
    for k in keys:
        p = os.path.join(OV_DIR, f"{k}.json")
        if os.path.exists(p) and json.load(open(p)).get("storyline"):
            print(f"  {k}: exists")
            continue
        story, err, calls, t = None, None, [], time.time()
        for attempt in range(2):
            with A.sink(calls):
                try:
                    vr = C.safe_reader(C.video_path({"video_key": k}))
                    story = execute_overview(config=config, parameters={"vr": vr, "subtitles": []})
                    break
                except Exception as e:
                    err = repr(e)[:500]
        C.save_json(p, {"video_key": k, "storyline": story, "error": None if story else err,
                        "latency": round(time.time() - t, 1),
                        "calls": [{kk: c.get(kk) for kk in ("kind", "thinking", "latency", "prompt_tokens", "completion_tokens", "finish_reason", "error")} for c in calls]})
        print(f"  {k}: {'OK ' + str(len(story)) + ' chars' if story else 'FAIL ' + str(err)} ({time.time() - t:.0f}s)", flush=True)


def viewer_call(vr, s, e, query, storyline, config):
    content, idx = focus_content(vr, s, e, query, storyline)
    calls = []
    with A.sink(calls), A.kind("focus"):
        try:
            resp = A.completion(model=config["model_name"], api_base=config["api_base"], api_key=config["api_key"],
                                messages=[{"role": "user", "content": content}], max_completion_tokens=config["max_tokens"],
                                seed=config["seed"], temperature=config["temperature"],
                                reasoning_effort=config["reasoning_effort"], timeout=900)
            raw = resp.choices[0].message.content
        except Exception:
            raw = None
    c = calls[-1] if calls else {}
    pred = C.extract_choice(raw)
    return {"raw": raw, "pred": pred, "abstain": bool(raw) and NO_REL.lower().rstrip(".") in raw.lower(),
            "n_frames": len(idx), "frame_indices": idx,
            "call": {k: c.get(k) for k in ("thinking", "latency", "prompt_tokens", "completion_tokens", "finish_reason", "error")}}


def stage_viewer(_):
    config = C.build_config()
    qs = C.load_questions()
    decoy_set = set(C.read_subset("oracle"))
    uids = C.read_subset("oracle_all")
    t0, done_n = time.time(), 0
    for i, uid in enumerate(uids):
        out = os.path.join(OUT, f"{uid}.json")
        if os.path.exists(out):
            continue
        r = qs[uid]
        ov = json.load(open(os.path.join(OV_DIR, f"{r['video_key']}.json")))
        story = ov.get("storyline")
        vr = C.safe_reader(C.video_path(r))
        q = C.user_query(r)
        res = {"uid": uid, "video_key": r["video_key"], "gt": r["answer"], "question_type": r["question_type"],
               "clue": [r["clue_start"], r["clue_end"]], "clue_len": r["clue_end"] - r["clue_start"],
               "decoy": [r["decoy_start"], r["decoy_end"]], "in_decoy_subset": uid in decoy_set,
               "anchor_available": bool(story)}
        res["oracle_indep"] = viewer_call(vr, r["clue_start"], r["clue_end"], q, None, config)
        if story:
            res["oracle_anchor"] = viewer_call(vr, r["clue_start"], r["clue_end"], q, story, config)
        if uid in decoy_set and r["decoy_start"] is not None:
            res["decoy_indep"] = viewer_call(vr, r["decoy_start"], r["decoy_end"], q, None, config)
            if story:
                res["decoy_anchor"] = viewer_call(vr, r["decoy_start"], r["decoy_end"], q, story, config)
        for k in ("oracle_indep", "oracle_anchor", "decoy_indep", "decoy_anchor"):
            if k in res:
                res[k]["correct"] = res[k]["pred"] == r["answer"]
        C.save_json(out, res)
        done_n += 1
        el = time.time() - t0
        summ = " ".join(f"{k}={res[k]['pred']}{'(abst)' if res[k]['abstain'] else ''}" for k in
                        ("oracle_indep", "oracle_anchor", "decoy_indep", "decoy_anchor") if k in res)
        print(f"  [{i + 1}/{len(uids)}] {uid} gt={r['answer']} {summ} | {el / 60:.1f} min, "
              f"ETA {(len(uids) - i - 1) * el / done_n / 60:.0f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["overview", "viewer", "all"])
    a = ap.parse_args()
    for s in (["overview", "viewer"] if "all" in a.stages else a.stages):
        t = time.time()
        print(f"===== anchored viewer {s} =====", flush=True)
        globals()[f"stage_{s}"](a)
        print(f"[anchored viewer {s}] {(time.time() - t) / 60:.1f} min", flush=True)
