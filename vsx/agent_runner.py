"""
저자 VideoSeekAgent 를 수정 없이 실행하고 문항별 실행 기록을 남기는 runner.

  - 문항 1개 = 별도 프로세스 1개(timeout, 메모리 감시) = 결과 JSON 1개. 이미 있으면 건너뜀(중단 후 재개 가능)
  - 외부 hook 만 사용:
      (1) videoseek.utils.completion → vsx.llm_adapter.completion
      (2) agent 의 VideoReader → frame index 기록 + decord 누수 우회 reader
      (3) tool registry 함수 → tool 호출 단위 기록
  - 결과: results/<out>/<uid>.json, 프로세스 로그: results/<out>/logs/<uid>.log

사용 예:
  python -m vsx.agent_runner --out e1_reproduction/videoseek --uids subset:main
  python -m vsx.agent_runner --out e3_subtitle_injection/videoseek_attack --uids subset:inject \
         --subtitle_dir data/injection_subtitles
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from vsx import common as C

NO_CALL_MSG = "There is no function call in your response."


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 2**20


def start_memwatch(limit_mb):
    """메모리가 limit 를 넘으면 main thread 스택을 기록하고 종료(코드 99). OOM killer 가 서버를 위협하기 전에 차단."""
    main_id = threading.main_thread().ident

    def loop():
        last = 0
        while True:
            r = rss_mb()
            if r > last + 500:
                print(f"[mem] {time.strftime('%H:%M:%S')} rss={r}MB", flush=True)
                last = r
            if r > limit_mb:
                print(f"[mem] LIMIT rss={r}MB > {limit_mb}MB, main thread stack:", flush=True)
                traceback.print_stack(sys._current_frames()[main_id], file=sys.stdout)
                sys.stdout.flush()
                os._exit(99)
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True).start()


def run_one(uid, out_name, video_dir, subtitle_dir, out_path):
    start_memwatch(int(os.environ.get("VSX_MEM_LIMIT_MB", "5000")))
    from vsx import llm_adapter as A
    A.install()
    import videoseek.agent as AG
    from videoseek.tools import DEFAULT_TOOL_REGISTRY as REG

    cur = {"tool": None}
    tool_log, calls = [], []
    VR = C.safe_reader_class()

    def on_batch(idx):
        if cur["tool"] is not None:
            cur["tool"]["frame_indices"].extend(idx)
    VR.on_batch = on_batch
    AG.VideoReader = VR

    for name, fn in list(REG._tool_functions.items()):
        def wrapped(config, parameters, _fn=fn, _name=name):
            rec = {"tool": _name,
                   "params": {k: v for k, v in parameters.items() if k not in ("vr", "subtitles", "messages", "question")},
                   "frame_indices": [], "t_start": time.time(), "llm_call_start": len(calls)}
            cur["tool"] = rec
            print(f"[tool] {time.strftime('%H:%M:%S')} start {_name} {rec['params']} rss={rss_mb()}MB", flush=True)
            try:
                out = _fn(config=config, parameters=parameters)
                rec["outcome"] = out
                return out
            except Exception as e:
                rec["exception"] = repr(e)[:1500]
                raise
            finally:
                rec["latency"] = round(time.time() - rec["t_start"], 3)
                rec["llm_call_end"] = len(calls)
                cur["tool"] = None
                tool_log.append(rec)
                print(f"[tool] {time.strftime('%H:%M:%S')} end {_name} {rec['latency']}s "
                      f"frames={len(rec['frame_indices'])} rss={rss_mb()}MB", flush=True)
        REG._tool_functions[name] = wrapped

    row = C.load_questions()[uid]
    config = C.build_config()
    vpath = C.video_path(row, video_dir)
    spath = C.subtitle_path(row, subtitle_dir)
    t0 = time.time()
    agent, final, finish, err = None, None, None, None
    with A.sink(calls):
        try:
            agent = AG.VideoSeekAgent(config=config, video_path=vpath, subtitle_path=spath,
                                      output_dir=tempfile.gettempdir(), tools=config["tools"], verbose=False)
            traj = agent.run(C.user_query(row))
            final, finish = traj.final_answer, traj.finish_reason
        except Exception:
            err = traceback.format_exc()[-3000:]
    runtime = time.time() - t0

    fps = agent.vr.get_avg_fps() if agent is not None else None
    steps = [s.to_dict() for s in agent.trajectory_steps] if agent is not None else []
    messages = agent.messages if agent is not None else []
    for t in tool_log:
        t["n_frames"] = len(t["frame_indices"])
        t["timestamps"] = [round(i / fps, 2) for i in t["frame_indices"]] if fps else []
        t["llm_calls"] = list(range(t["llm_call_start"], t["llm_call_end"]))
    for c in calls:
        if c["kind"] not in ("answer", "forced_answer"):
            c.pop("messages", None)

    pred = C.extract_choice(final)
    all_idx = [i for t in tool_log for i in t["frame_indices"]]
    obs = [t.get("outcome") for t in tool_log if t["tool"] != "answer"]
    kinds = sorted({c["kind"] for c in calls})

    def tok(kind=None, field="completion_tokens"):
        return sum((c.get(field) or 0) for c in calls if kind is None or c["kind"] == kind)

    result = {
        "uid": uid, "cond": out_name, "video_key": row["video_key"],
        "video_path": C.rel(vpath), "subtitle_path": C.rel(spath),
        "question": row["question"], "question_type": row["question_type"], "gt": row["answer"],
        "clue": [row["clue_start"], row["clue_end"]], "duration": row.get("duration"),
        "final_answer_raw": final, "pred": pred, "correct": pred == row["answer"], "extract_failed": pred is None,
        "finish_reason": finish, "error": err, "runtime_s": round(runtime, 1),
        "settings": {"think_mode": A.THINK_MODE, "max_tokens": config["max_tokens"]},
        "metrics": {
            "n_frames": len(all_idx), "n_unique_frames": len(set(all_idx)),
            "n_turns": max((s["step_id"] for s in steps), default=0),
            "n_thinking_calls": sum(1 for c in calls if c["kind"] == "thinking"),
            "tool_counts": {k: sum(1 for t in tool_log if t["tool"] == k) for k in ("overview", "skim", "focus", "answer")},
            "first_tool": tool_log[0]["tool"] if tool_log else None,
            "action_parse_fail": sum(1 for m in messages if m.get("role") == "user" and str(m.get("content", "")).startswith(NO_CALL_MSG)),
            "tool_exec_failed": sum(1 for o in obs if o in (None, "Tool execution failed.")),
            "overview_fail": sum(1 for t in tool_log if t["tool"] == "overview" and ("exception" in t or t.get("outcome") is None)),
            "forced_answer": finish == "reach_max_steps",
            "llm_errors": sum(1 for c in calls if c.get("error")),
            "length_truncated": sum(1 for c in calls if c.get("finish_reason") == "length"),
            "args_repr_fixed": sum(c.get("n_args_repr_fixed", 0) for c in calls),
            "tokens_prompt": tok(field="prompt_tokens"), "tokens_completion": tok(),
            "tokens_by_kind": {k: [tok(k, "prompt_tokens"), tok(k)] for k in kinds},
            "latency_by_kind": {k: round(sum(c["latency"] for c in calls if c["kind"] == k), 1) for k in kinds},
        },
        "steps": steps, "tools": tool_log, "llm_calls": calls, "messages": messages,
    }
    C.save_json(out_path, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/ 아래 출력 폴더, 예: e1_reproduction/videoseek")
    ap.add_argument("--uids", required=True, help="subset:<name> 또는 uid 목록(쉼표 구분)")
    ap.add_argument("--video_dir", default=None, help="기본: $VSX_WORK/videos")
    ap.add_argument("--subtitle_dir", default=None, help="<uid>.srt 가 있는 폴더 (없으면 자막 없음)")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    outdir = C.result_dir(a.out)
    if a.one:
        run_one(a.one, a.out, a.video_dir, a.subtitle_dir, os.path.join(outdir, f"{a.one}.json"))
        return

    qs = C.load_questions()
    uids = C.select_uids(a.uids, qs)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    print(f"[{a.out}] {len(uids)} questions | think_mode={os.environ.get('VSX_THINK_MODE')} "
          f"max_tokens={os.environ.get('VSX_MAX_TOKENS')}", flush=True)
    done_t, t_all = [], time.time()
    for i, uid in enumerate(uids):
        out = os.path.join(outdir, f"{uid}.json")
        if os.path.exists(out):
            print(f"  [{i + 1}/{len(uids)}] {uid}: exists, skip", flush=True)
            continue
        cmd = [sys.executable, "-m", "vsx.agent_runner", "--out", a.out, "--uids", uid, "--one", uid]
        if a.video_dir:
            cmd += ["--video_dir", a.video_dir]
        if a.subtitle_dir:
            cmd += ["--subtitle_dir", a.subtitle_dir]
        t = time.time()
        with open(os.path.join(logdir, f"{uid}.log"), "w") as lf:
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
                status = "MEMLIMIT" if rc == 99 else f"rc={rc}"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
        dt = time.time() - t
        if not os.path.exists(out):
            row = qs[uid]
            C.save_json(out, {"uid": uid, "cond": a.out, "video_key": row["video_key"], "gt": row["answer"],
                              "question_type": row["question_type"], "pred": None, "correct": False,
                              "extract_failed": True, "error": status, "runtime_s": round(dt, 1), "metrics": {}})
        r = json.load(open(out))
        done_t.append(dt)
        avg = sum(done_t) / len(done_t)
        m = r.get("metrics", {})
        print(f"  [{i + 1}/{len(uids)}] {uid} {status} pred={r.get('pred')} gt={r.get('gt')} "
              f"{'O' if r.get('correct') else 'X'} frames={m.get('n_frames')} turns={m.get('n_turns')} "
              f"{dt:.0f}s | avg {avg:.0f}s, ETA {(len(uids) - i - 1) * avg / 60:.0f}min", flush=True)
    print(f"[{a.out}] total {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
