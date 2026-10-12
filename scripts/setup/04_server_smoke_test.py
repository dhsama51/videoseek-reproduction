"""
llama-server + adapter 호환성 점검 (서버 실행 후 실험 전에 1회).
  T1 텍스트 + thinking 분리, T2 이미지 16장(720p) 순서 인식, T3 overview 형식 JSON 출력(64 frame 설명),
  T4 tool_choice="required" tool call, T5 저자 agent 와 같은 형식의 history(Python repr arguments 포함)
사용: python scripts/setup/04_server_smoke_test.py   → $VSX_WORK/logs/server_smoke_test.json
"""
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback

import requests
from PIL import Image, ImageDraw, ImageFont

os.environ["VSX_THINK_MODE"] = "faithful"  # thinking ON/OFF 를 reasoning_effort 로 직접 지정해 둘 다 점검
from vsx import llm_adapter as A  # noqa: E402

MODEL = os.environ["VSX_MODEL"]
BASE = os.environ["OPENAI_API_BASE"]
OUT = os.path.join(os.environ["VSX_WORK"], "logs", "server_smoke_test.json")
MAXTOK = int(os.environ.get("VSX_MAX_TOKENS", "24576"))
random.seed(0)


def vram_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def call(k, messages, effort="medium", **kw):
    recs = []
    with A.sink(recs), A.kind(k):
        resp = A.completion(model=MODEL, api_base=BASE, api_key="sk-local", messages=messages,
                            max_completion_tokens=kw.pop("max_tokens", MAXTOK), seed=42,
                            temperature=1.0, reasoning_effort=effort, timeout=900, **kw)
    return resp, recs[-1]


def num_img(n, w, h):
    img = Image.new("RGB", (w, h), (random.randint(0, 80),) * 3)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default(size=int(h * 0.5))
    except TypeError:
        f = ImageFont.load_default()
    d.text((w * 0.3, h * 0.2), str(n), fill=(255, 255, 0), font=f)
    return img


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="jpeg")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


results = {}


def run(name, fn):
    t0 = time.time()
    try:
        ok, info = fn()
    except Exception as e:
        ok, info = False, {"exception": repr(e), "tb": traceback.format_exc()[-1500:]}
    info["wall"] = round(time.time() - t0, 1)
    info["vram_after_mb"] = vram_mb()
    results[name] = {"ok": ok, **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: " + json.dumps({k: v for k, v in info.items() if k not in ('tb', 'content')}, ensure_ascii=False)[:600])


def brief(rec):
    keys = ["latency", "prompt_tokens", "completion_tokens", "finish_reason", "reasoning_chars", "think_tag_leak", "thinking"]
    d = {k: rec.get(k) for k in keys}
    if rec.get("completion_tokens") and rec.get("latency"):
        d["tok_per_s_overall"] = round(rec["completion_tokens"] / rec["latency"], 1)
    return d


# ---------------- T1 text ----------------
def t1():
    _, r = call("thinking", [{"role": "user", "content": "What is 17*23? Answer with just the number."}])
    ok = "391" in (r.get("content") or "") and not r["think_tag_leak"] and r["reasoning_chars"] > 0
    return ok, {**brief(r), "content": r.get("content")}


# ---------------- T2 16 images (720p, focus-like) ----------------
def t2(thinking_effort):
    nums = [random.randint(10, 99) for _ in range(16)]
    content = [{"type": "text", "text": "Video clip (100.0s - 116.0s):\n"}]
    for i, n in enumerate(nums):
        content.append({"type": "text", "text": f"{100 + i:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": b64(num_img(n, 1280, 720))}})
    content.append({"type": "text", "text": "Question:\nWhat number is shown in each frame? Answer as a comma-separated list in frame order, numbers only."})
    _, r = call("focus", [{"role": "user", "content": content}], effort=thinking_effort)
    got = [int(x) for x in __import__("re").findall(r"\d+", (r.get("content") or ""))]
    acc = sum(1 for a, b in zip(got, nums) if a == b) / 16
    return acc >= 0.75, {**brief(r), "order_acc": acc, "tokens_per_image_est": round((r.get("prompt_tokens") or 0) / 16)}


# ---------------- T3 overview-like JSON (8 grids, 64 ts) ----------------
def t3(effort):
    ts = [round(i * 56.25, 1) for i in range(64)]
    content = [{"type": "text", "text": "The video segment is located at 0.0s - 3600.0s:\nThe video frames are uniformly sampled.\n"}]
    for g in range(8):
        cells = [num_img(random.randint(10, 99), 455, 256) for _ in range(8)]
        grid = Image.new("RGB", (455 * 4, 256 * 2))
        for j, c in enumerate(cells):
            grid.paste(c, ((j % 4) * 455, (j // 4) * 256))
        row = ts[g * 8:(g + 1) * 8]
        tm = f"[[{', '.join(f'{t:.1f}s' for t in row[:4])}],\n[{', '.join(f'{t:.1f}s' for t in row[4:])}]]"
        content.append({"type": "text", "text": f"[{row[0]:.1f}s - {row[-1]:.1f}s]:\nTimestamp Matrix:\n{tm}\n"})
        content.append({"type": "image_url", "image_url": {"url": b64(grid)}})
    content.append({"type": "text", "text": (
        "Video Subtitles:\nNo subtitle found.\n\n"
        "Please generate descriptions for each frame in the video. The descriptions should be concise and detailed (~50 words each).\n"
        f"Ensure every timestamp value exactly matches a timestamp from the provided timestamp matrices (same values and formatting): [{', '.join(f'{t:.1f}s' for t in ts)}].\n"
        "Return ONLY valid JSON. Use this exact schema:\n"
        "{\"frames\": [{\"timestamp\": \"1.0s\", \"description\": \"FRAME_DESCRIPTION_1\"}, ...]}\n")})
    _, r = call("overview", [{"role": "user", "content": content}], effort=effort,
                response_format={"type": "json_object"})
    js = json.loads(r.get("content") or "")
    n = len(js["frames"])
    return n >= 60, {**brief(r), "n_frames_described": n}


# ---------------- T4 tool call (tool_choice=required) ----------------
def t4(effort):
    from videoseek.tools import DEFAULT_TOOL_REGISTRY
    tools = DEFAULT_TOOL_REGISTRY.resolve_tools(["overview", "skim", "focus", "answer"])
    out = {}
    ok_all = True
    for name, thought in [
        ("overview", "I will use the `overview` tool to get a 64-frame summary of the entire video before narrowing down."),
        ("skim", "The overview suggests the scene happens between 1200s and 1500s. I will skim 1200-1500s with the query 'Where does the red car appear?'."),
    ]:
        _, r = call("action_parse", [{"role": "user", "content": f"Please call the appropriate tool(s) based on the following thought:\n{thought}"}],
                    effort=effort, tools=tools, tool_choice="required")
        tc = r.get("tool_calls") or []
        valid = bool(tc) and all(json.loads(t["arguments"] or "{}") is not None for t in tc)
        right = bool(tc) and tc[0]["name"] == name
        ok_all &= valid and right
        out[name] = {**brief(r), "tool_calls": tc}
    return ok_all, out


# ---------------- T5 multi-turn history with python-repr arguments ----------------
def t5(repr_args=True):
    params = {"query": "Where does the red car appear?", "start_time": 1200, "end_time": 1500}
    args = str(params) if repr_args else json.dumps(params)
    msgs = [
        {"role": "system", "content": "You are a video agent."},
        {"role": "user", "content": "Video Duration: 3600.0s\n\nVideo Subtitles:\nNo subtitle found.\n\nQuestion:\nWhat color is the car?\n(A) red\n(B) blue\n(C) green\n(D) black"},
        {"role": "user", "content": "Step [1 / 20]: Please follow the Thinking Policy to do **reasoning over the current state** and **plan the next action(s) to take**."},
        {"role": "assistant", "content": "I will skim 1200-1500s.",
         "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "skim", "arguments": args}}]},
        {"role": "tool", "tool_call_id": "call_abc123",
         "content": f"Observation from `{{'function': 'skim', 'parameters': {params}, 'id': 'call_abc123'}}`:\n1230.0s: a red car parks near a building."},
        {"role": "user", "content": "Step [2 / 20]: Please follow the Thinking Policy to do **reasoning over the current state** and **plan the next action(s) to take**."},
    ]
    _, r = call("thinking", msgs)
    return bool(r.get("content")) and not r["think_tag_leak"], {**brief(r), "content": (r.get("content") or "")[:300]}


if __name__ == "__main__":
    try:
        props = requests.get(BASE.replace("/v1", "") + "/props", timeout=10).json()
        results["server_props"] = {"n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
                                   "build": props.get("build_info"), "modalities": props.get("modalities")}
    except Exception as e:
        results["server_props"] = {"error": repr(e)}
    print("server:", results["server_props"], "| THINK_MODE =", A.THINK_MODE, "| vram", vram_mb(), "MB")
    run("T1_text_thinking", t1)
    run("T2_16img_720p_thinkON", lambda: t2("medium"))
    run("T2_16img_720p_thinkOFF", lambda: t2("low"))
    run("T3_overview_json_thinkON", lambda: t3("medium"))
    run("T3_overview_json_thinkOFF", lambda: t3("low"))
    run("T4_toolcall_required_thinkON", lambda: t4("medium"))
    run("T4_toolcall_required_thinkOFF", lambda: t4("low"))
    run("T5_history_pyrepr_args", lambda: t5(True))
    run("T5b_history_json_args", lambda: t5(False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("saved:", OUT)
