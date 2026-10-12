"""
Exp 6 (탐색적): 답 강제 재생
  Exp 1 VideoSeek(lite) 각 문항의 '답 생성 직전 messages'(탐색 경로 고정)로 답 단계만 다시 실행
    null   : messages 그대로 (sampling 변동 측정)
    forced : 마지막 답 지시 뒤에 "확신이 없어도 A~D 중 하나만 출력" 문장 추가
  호출 종류/설정은 원래 답 호출과 동일 (kind=answer|forced_answer, reasoning_effort=medium, 같은 seed/temperature)
사용: python -m vsx.answer_replay [--source e1_reproduction/videoseek]
"""
import argparse
import json
import os
import time

from vsx import common as C
from vsx import llm_adapter as A

FORCE_SUFFIX = ("\n\nYou must choose exactly one of the given options (A, B, C, or D), even if you are uncertain "
                "or the evidence is incomplete. Respond with only the option letter.")


def call(messages, kind, config):
    calls = []
    with A.sink(calls), A.kind(kind):
        try:
            resp = A.completion(model=config["model_name"], api_base=config["api_base"], api_key=config["api_key"],
                                messages=messages, max_completion_tokens=config["max_tokens"], seed=config["seed"],
                                temperature=config["temperature"], reasoning_effort="medium", timeout=900)
            raw = resp.choices[0].message.content
        except Exception:
            raw = None
    c = calls[-1] if calls else {}
    return raw, {k: c.get(k) for k in ("latency", "prompt_tokens", "completion_tokens", "finish_reason", "error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="e1_reproduction/videoseek")
    a = ap.parse_args()
    config = C.build_config()
    outdir = C.result_dir("e6_answer_replay")
    uids = C.read_subset("main")
    t0, n = time.time(), 0
    for i, uid in enumerate(uids):
        out = os.path.join(outdir, f"{uid}.json")
        p = os.path.join(C.result_dir(a.source), f"{uid}.json")
        if os.path.exists(out) or not os.path.exists(p):
            continue
        r = json.load(open(p))
        ans = [c for c in r.get("llm_calls", []) if c["kind"] in ("answer", "forced_answer") and c.get("messages")]
        if not ans:
            C.save_json(out, {"uid": uid, "skip": "no answer-stage messages (timeout 등)", "orig_pred": r.get("pred"), "gt": r.get("gt")})
            continue
        msgs, kind = ans[-1]["messages"], ans[-1]["kind"]
        forced = json.loads(json.dumps(msgs))
        last = forced[-1]
        if isinstance(last.get("content"), str):
            last["content"] = last["content"] + FORCE_SUFFIX
        else:
            forced.append({"role": "user", "content": FORCE_SUFFIX.strip()})
        null_raw, null_c = call(msgs, kind, config)
        f_raw, f_c = call(forced, kind, config)
        res = {"uid": uid, "video_key": r.get("video_key"), "gt": r.get("gt"), "kind": kind,
               "orig_pred": r.get("pred"), "orig_raw": (r.get("final_answer_raw") or "")[:500],
               "null_raw": null_raw, "null_pred": C.extract_choice(null_raw), "null_call": null_c,
               "forced_raw": f_raw, "forced_pred": C.extract_choice(f_raw), "forced_call": f_c}
        C.save_json(out, res)
        n += 1
        el = time.time() - t0
        print(f"  [{i + 1}/{len(uids)}] {uid} gt={res['gt']} orig={res['orig_pred']} null={res['null_pred']} "
              f"forced={res['forced_pred']} | {el / 60:.1f} min", flush=True)
    print(f"[answer replay] done {n} new, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
