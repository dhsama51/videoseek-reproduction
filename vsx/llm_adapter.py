"""
저자 코드의 유일한 LLM 진입점인 `videoseek.utils.completion`(= litellm.completion 참조)을 감싸는 adapter.
agent / tool 로직은 수정하지 않음.

  1) reasoning_effort -> llama-server `chat_template_kwargs.enable_thinking`
       faithful : effort in {medium, high} -> ON, {low, minimal} -> OFF   (저자 코드에서 low 는 skim 뿐)
       lite     : thinking / answer / forced_answer / single_pass 호출만 ON
  2) llama-server 가 받지 않는 None 인자 제거, max_completion_tokens -> max_tokens
  3) [서버 호환] history 의 tool_calls.arguments 가 Python repr(str(dict))이면 JSON 으로 바꾼 사본을 전송
       (저자 agent 는 str(action.parameters) 를 기록하고, llama-server 는 JSON 이 아니면 500 을 반환)
  4) [서버 호환] local 서버 오류는 저자 retry 데코레이터(최대 8회 지수 backoff)의 키워드에 걸리지 않는
       메시지로 바꿔 올림 → 결정적 오류의 장시간 재시도 방지. 연결 오류만 10초 간격 3회 재시도
  5) 호출 종류 분류 + token / 지연 / finish_reason / 오류 기록 (sink)
"""
import ast
import contextlib
import contextvars
import copy
import json
import os
import time

import litellm

THINK_MODE = os.environ.get("VSX_THINK_MODE", "lite")
_kind_override = contextvars.ContextVar("vsx_kind", default=None)
_sink = contextvars.ContextVar("vsx_sink", default=None)  # list 또는 None

LITE_THINK_KINDS = {"thinking", "answer", "forced_answer", "single_pass"}
KEEP_MESSAGES_KINDS = {"answer", "forced_answer"}  # 답 단계 재생(Exp 4, 6)용


@contextlib.contextmanager
def kind(name):
    tok = _kind_override.set(name)
    try:
        yield
    finally:
        _kind_override.reset(tok)


@contextlib.contextmanager
def sink(records: list):
    tok = _sink.set(records)
    try:
        yield records
    finally:
        _sink.reset(tok)


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _n_images(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def classify(kw) -> str:
    """저자 코드의 호출 위치를 인자 모양으로 판별."""
    if kw.get("tools"):
        return "action_parse"
    if kw.get("response_format"):
        return "overview"
    if kw.get("reasoning_effort") == "low":
        return "skim"
    msgs = kw.get("messages") or []
    if len(msgs) == 1 and isinstance(msgs[0].get("content"), list):
        first = _text_of(msgs[0]["content"][:1])
        if first.startswith("Video clip"):
            return "focus"
        if first.startswith("Video segment"):
            return "skim"
        if first.startswith("The video segment is located"):
            return "overview"
    last = _text_of(msgs[-1].get("content")) if msgs else ""
    if last.startswith("You have reached the maximum number of steps"):
        return "forced_answer"
    if last.startswith("Question:\n") and last.rstrip().endswith("Please directly provide the final answer."):
        return "answer"
    return "thinking"


def thinking_for(k: str, effort) -> bool:
    if THINK_MODE == "lite":
        return k in LITE_THINK_KINDS
    return effort not in ("low", "minimal", "none")


def _strip_images(messages):
    out = []
    for m in messages:
        m2 = {kk: vv for kk, vv in m.items() if kk != "content"}
        c = m.get("content")
        if isinstance(c, list):
            m2["content"] = [p if p.get("type") != "image_url" else {"type": "image_url", "image_url": "<omitted>"} for p in c]
        else:
            m2["content"] = c
        out.append(m2)
    return out


class VSXLocalError(RuntimeError):
    pass


def _fix_tool_args(messages):
    """repr 형식 arguments 를 JSON 으로 바꾼 사본과 변환 개수 반환 (원본 agent.messages 는 건드리지 않음)."""
    need = False
    for m in messages:
        for tc in m.get("tool_calls") or []:
            a = (tc.get("function") or {}).get("arguments")
            if isinstance(a, str):
                try:
                    json.loads(a)
                except Exception:
                    need = True
    if not need:
        return messages, 0
    msgs = copy.deepcopy(messages)
    n = 0
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    json.loads(a)
                except Exception:
                    try:
                        fn["arguments"] = json.dumps(ast.literal_eval(a), ensure_ascii=False)
                        n += 1
                    except Exception:
                        fn["arguments"] = json.dumps({"raw": a}, ensure_ascii=False)
                        n += 1
    return msgs, n


def _is_connection_error(e):
    s = repr(e).lower()
    return "connection refused" in s or "connecterror" in s or "apiconnectionerror" in s


def completion(**kw):
    k = _kind_override.get() or classify(kw)
    effort = kw.pop("reasoning_effort", None)
    think = thinking_for(k, effort)

    if kw.get("tools") is None:
        kw.pop("tools", None)
        kw.pop("tool_choice", None)
    for key in ("response_format", "api_version"):
        if kw.get(key) is None:
            kw.pop(key, None)
    mct = kw.pop("max_completion_tokens", None)
    if mct is not None:
        kw["max_tokens"] = mct
    eb = dict(kw.pop("extra_body", None) or {})
    eb["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    kw["extra_body"] = eb

    msgs, n_fixed = _fix_tool_args(kw.get("messages") or [])
    kw["messages"] = msgs
    rec = {
        "kind": k, "effort": effort, "thinking": think, "t_start": time.time(),
        "n_images": _n_images(msgs), "n_messages": len(msgs),
        "tool_choice": kw.get("tool_choice"), "json_mode": "response_format" in kw,
        "max_tokens": kw.get("max_tokens"), "n_args_repr_fixed": n_fixed,
    }
    if k in KEEP_MESSAGES_KINDS:
        rec["messages"] = copy.deepcopy(_strip_images(msgs))

    if os.environ.get("VSX_TRACE"):
        print(f"[llm] {time.strftime('%H:%M:%S')} start {k} think={think} imgs={rec['n_images']}", flush=True)
    resp = None
    try:
        for attempt in range(4):
            try:
                resp = litellm.completion(**kw)
                break
            except Exception as e:
                if _is_connection_error(e) and attempt < 3:
                    rec["conn_retries"] = attempt + 1
                    time.sleep(10)
                    continue
                raise
        return resp
    except Exception as e:
        rec["error"] = repr(e)[:2000]
        # 저자 retry 키워드(rate limit/timed out/internal/...)에 걸리지 않게 바꿔서 올림 → 저자 코드는 즉시 None 처리
        raise VSXLocalError("vsx local server error; see call log") from None
    finally:
        rec["latency"] = round(time.time() - rec["t_start"], 3)
        if resp is not None:
            try:
                u = resp.usage
                rec["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                rec["completion_tokens"] = getattr(u, "completion_tokens", None)
                ch = resp.choices[0]
                rec["finish_reason"] = ch.finish_reason
                msg = ch.message
                content = msg.content or ""
                rc = getattr(msg, "reasoning_content", None)
                if rc is None:
                    psf = getattr(msg, "provider_specific_fields", None) or {}
                    rc = psf.get("reasoning_content")
                rec["content"] = content
                rec["reasoning_chars"] = len(rc or "")
                if k in ("thinking", "answer", "forced_answer", "single_pass", "guardrail"):
                    rec["reasoning"] = rc
                tcs = getattr(msg, "tool_calls", None) or []
                rec["tool_calls"] = [
                    {"name": t.function.name, "arguments": t.function.arguments} for t in tcs
                ]
                rec["think_tag_leak"] = "<think>" in content or "</think>" in content
            except Exception as e:
                rec["parse_error"] = repr(e)
        if os.environ.get("VSX_TRACE"):
            print(f"[llm] {time.strftime('%H:%M:%S')} end {k} {rec['latency']}s ptok={rec.get('prompt_tokens')} "
                  f"ctok={rec.get('completion_tokens')} finish={rec.get('finish_reason')} err={bool(rec.get('error'))}", flush=True)
        s = _sink.get()
        if s is not None:
            s.append(rec)


def install():
    """videoseek.utils 의 litellm 참조를 adapter 로 교체 (agent/tool 코드는 그대로)."""
    import videoseek.utils as U
    U.completion = completion
    return U
