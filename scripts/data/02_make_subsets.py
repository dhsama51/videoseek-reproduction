"""
실험별 문항 부분집합 생성 → data/subsets/*.txt
  main       : 앞 9개 영상 × 7문항 = 63문항 (Exp 1)
  logic      : main 중 시간 순서를 묻지 않는 문항 (Temporal Grounding·Summarization 제외, first/last/before/after/
               then/begin/end 류 키워드 제외), 사전 rank 순 상위 30 (Exp 2)
  inject     : main 중 clue 구간 길이 > 0, 사전 rank 순 상위 30 (Exp 3, 4)
  oracle     : main 중 clue > 0 이고 decoy 구간을 둘 수 있는 문항, 사전 rank 순 상위 30 (Exp 5 decoy)
  oracle_all : main 중 clue > 0 인 전체 (Exp 5 oracle)
rank 는 01_prepare_lvbench.py select 단계에서 고정 seed 로 부여됨.
data/subsets/faithful.txt (Exp 7) 은 이 스크립트가 만들지 않음: lite 설정 확정 전에 faithful 로 측정한 5문항 +
main 나머지에서 영상별 round-robin 으로 고른 15문항(seed 2027)이며, 레포에 고정 목록으로 포함됨.
"""
import os

from vsx import common as C

N_MAIN_VIDEOS, N_SUB = 9, 30


def main():
    qs = C.load_questions()
    rows = sorted(qs.values(), key=lambda r: (r["video_order"], r["q_order"]))
    main_rows = [r for r in rows if r["video_order"] < N_MAIN_VIDEOS]
    out = {"main": [r["uid"] for r in main_rows]}
    for key, elig in (("logic", "elig_logic"), ("inject", "elig_inject"), ("oracle", "elig_oracle")):
        cand = sorted((r for r in main_rows if r[elig]), key=lambda r: r[f"rank_{key}"])
        out[key] = [r["uid"] for r in cand[:N_SUB]]
        print(f"  {key}: eligible {len(cand)} / {len(main_rows)} -> {len(out[key])}")
    out["oracle_all"] = [r["uid"] for r in main_rows if r["clue_end"] > r["clue_start"]]
    os.makedirs(os.path.join(C.DATA, "subsets"), exist_ok=True)
    for k, v in out.items():
        with open(C.subset_file(k), "w") as f:
            f.write("\n".join(v) + "\n")
        print(f"  subsets/{k}.txt: {len(v)}")


if __name__ == "__main__":
    main()
