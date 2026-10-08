"""두 검색 평가 결과(eval/results/*.json)를 비교한다. 평균 표 + 질문별로 좋아진 것·나빠진 것.

평균 차이만 보면 54문항에서는 우연과 구분하기 어려우므로, 질문마다 정답 순위가
어떻게 바뀌었는지 센다. (순위 = 1/MRR. 상위 10개에 없으면 "없음")

실행: uv run python -m eval.compare_runs r0_dense r1_multi_query            두 결과 비교 (질문별 개선·악화 포함)
      uv run python -m eval.compare_runs r0_dense r1_multi_query r2_rerank  3개 이상이면 한 장짜리 요약 표(마크다운)
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")


def load(label: str) -> dict:
    return json.loads((RESULTS_DIR / f"{label}.json").read_text(encoding="utf-8"))


def _rank(detail: dict) -> int | None:
    return round(1 / detail["mrr"]) if detail["mrr"] else None


def _fmt_rank(rank: int | None) -> str:
    return f"{rank}위" if rank else "없음"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run_row(label: str) -> dict:
    """한 결과 파일의 전체 지표. Chunk@K는 gold_chunks가 있는 문항만으로 계산한다(결과 파일의 질문별 값에서)."""
    d = load(label)
    details = d["details"]
    chunk_rows = [x for x in details if "chunk@5" in x]
    row = {k: d["summary"]["전체"][k] for k in ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")}
    for k in (1, 3, 5):
        row[f"chunk@{k}"] = _mean([x[f"chunk@{k}"] for x in chunk_rows])
    row["n_chunk"] = len(chunk_rows)
    return row


def summary_table(labels: list[str]) -> str:
    """여러 결과를 한 표로. 전체 + 유형별 Hit@1·MRR. 문서에 그대로 붙일 수 있는 마크다운."""
    runs = {lab: load(lab) for lab in labels}
    rows = {lab: run_row(lab) for lab in labels}
    n_chunk = rows[labels[0]]["n_chunk"]
    out = ["| 구성 | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | " f"Chunk@1 | Chunk@5 (n={n_chunk}) |", "|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        r = rows[lab]
        out.append(f"| {lab} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['recall@5']:.2f} | {r['mrr']:.2f} | {r['chunk@1']:.2f} | {r['chunk@5']:.2f} |")
    types = [t for t in runs[labels[0]]["summary"] if t != "전체"]
    out += ["", "유형별 Hit@1 / MRR", "", "| 유형 | n | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for t in types:
        n = runs[labels[0]]["summary"][t]["n"]
        cells = " | ".join(f"{runs[l]['summary'][t]['hit@1']:.2f} / {runs[l]['summary'][t]['mrr']:.2f}" for l in labels)
        out.append(f"| {t} | {n} | {cells} |")
    return "\n".join(out)


def compare(base_label: str, new_label: str) -> None:
    base, new = load(base_label), load(new_label)
    print(f"[{base_label} → {new_label}] 질문 {base['n_questions']}개\n")

    header = f"{'유형':<10}{'n':>3}  " + "".join(f"{m:>16}" for m in METRICS)
    print(header)
    for t in base["summary"]:
        b, n = base["summary"][t], new["summary"][t]
        cells = "".join(f"{b[m]:>7.2f}→{n[m]:.2f}{n[m] - b[m]:+6.2f}".rjust(16) for m in METRICS)
        print(f"{t:<10}{b['n']:>3}  {cells}")

    base_by_id = {d["id"]: d for d in base["details"]}
    better, worse, same = [], [], 0
    for d in new["details"]:
        b = base_by_id[d["id"]]
        rb, rn = _rank(b), _rank(d)
        # 순위가 낮을수록(숫자가 작을수록) 좋다. 없음은 가장 나쁨.
        key_b, key_n = rb or 99, rn or 99
        line = f"  {d['id']} [{d['type']}] {d['question']}  ({_fmt_rank(rb)} → {_fmt_rank(rn)})"
        if key_n < key_b:
            better.append(line)
        elif key_n > key_b:
            worse.append(line)
        else:
            same += 1

    print(f"\n정답 순위가 좋아진 질문 {len(better)}개 / 나빠진 질문 {len(worse)}개 / 그대로 {same}개")
    for title, lines in (("좋아진 질문", better), ("나빠진 질문", worse)):
        if lines:
            print(f"\n{title}:")
            print("\n".join(lines))

    lost = [d for d in new["details"] if d["hit@5"] == 0 and base_by_id[d["id"]]["hit@5"] == 1]
    gained = [d for d in new["details"] if d["hit@5"] == 1 and base_by_id[d["id"]]["hit@5"] == 0]
    print(f"\nTop-5 안에 새로 들어온 질문 {len(gained)}개 / 밀려난 질문 {len(lost)}개")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        compare(args[0], args[1])
    elif len(args) >= 3:
        print(summary_table(args))
        for a, b in zip(args, args[1:]):
            print()
            compare(a, b)
    else:
        sys.exit("사용: python -m eval.compare_runs <기준 결과> <비교 결과> [<다음 결과> ...]")
