"""골든셋으로 검색 성능을 잰다. LLM은 쓰지 않는다. (질문 임베딩 호출만 문항 수만큼)

채점 (조문 단위)
    Hit@K    : 상위 K개 청크의 조문 중 gold_articles가 하나라도 있으면 1
    Recall@K : gold_articles 중 상위 K개에서 찾은 비율
    MRR      : 처음 적중한 청크 순위의 역수 (top_k 안에 없으면 0)
    Chunk@K  : gold_chunks가 있는 문항만. 상위 K개에 정답 청크가 하나라도 있으면 1
"법에 없음"(answerable=false)은 검색 점수에서 제외한다. 답변 단계에서 따로 본다.

실행: uv run python -m eval.evaluate r0         (R0: Dense)
      uv run python -m eval.evaluate r1         (R1: + Multi Query)
      uv run python -m eval.evaluate r2         (R2: + Reranker)
결과: eval/results/<label>.json (질문별 상세) + 터미널에 표
"""
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import openai

from eval.check_golden_set import load_golden_set
from rag.pipeline import search_r2
from rag.retriever import Hit, dense_search, multi_query_search

KS = (1, 3, 5)
TOP_K = 10
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def score_question(row: dict, hits: list[Hit]) -> dict:
    gold = set(row["gold_articles"])
    gold_chunks = set(row["gold_chunks"])
    ranks = [i for i, h in enumerate(hits, 1) if h.article in gold]
    result = {"mrr": 1 / ranks[0] if ranks else 0.0}
    for k in KS:
        found = {h.article for h in hits[:k]} & gold
        result[f"hit@{k}"] = float(bool(found))
        result[f"recall@{k}"] = len(found) / len(gold)
        if gold_chunks:
            result[f"chunk@{k}"] = float(bool({h.chunk_id for h in hits[:k]} & gold_chunks))
    return result


def _mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if key in r]
    return sum(values) / len(values) if values else None


def summarize(scored: list[dict]) -> OrderedDict:
    """유형별·전체 평균. scored의 각 항목은 {"type", **지표}."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for s in scored:
        groups.setdefault(s["type"], []).append(s)
    groups["전체"] = scored
    keys = ["mrr"] + [f"{m}@{k}" for m in ("hit", "recall") for k in KS]
    return OrderedDict((t, {"n": len(rs), **{key: _mean(rs, key) for key in keys}}) for t, rs in groups.items())


def _search_with_retry(search, question: str, attempts: int = 4) -> list[Hit]:
    """API 분당 한도(429)에 걸려 클라이언트 재시도까지 실패하면 1분 쉬고 다시 시도한다. 평가가 중간에 끊기지 않게 한다."""
    for attempt in range(1, attempts + 1):
        try:
            return search(question, TOP_K)
        except openai.RateLimitError:
            if attempt == attempts:
                raise
            print(f"  429 한도 초과, 60초 대기 후 재시도 ({attempt}/{attempts - 1})", flush=True)
            time.sleep(60)


def run(search=dense_search, label: str = "r0_dense") -> None:
    rows = [r for r in load_golden_set() if r["answerable"]]
    details, scored = [], []
    for row in rows:
        hits = _search_with_retry(search, row["question"])
        s = score_question(row, hits)
        scored.append({"type": row["type"], **s})
        details.append({
            "id": row["id"], "type": row["type"], "question": row["question"],
            "gold_articles": row["gold_articles"], "gold_chunks": row["gold_chunks"],
            "retrieved": [{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits[:5]],
            **s,
        })

    summary = summarize(scored)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{label}.json").write_text(
        json.dumps({"label": label, "n_questions": len(rows), "summary": summary, "details": details},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    header = f"{'유형':<10}{'n':>3}  {'Hit@1':>6}{'Hit@3':>6}{'Hit@5':>6}  {'Rec@3':>6}{'Rec@5':>6}  {'MRR':>6}"
    print(f"[{label}] 검색 평가 (근거 있는 질문 {len(rows)}개)\n{header}")
    for t, m in summary.items():
        print(f"{t:<10}{m['n']:>3}  {m['hit@1']:6.2f}{m['hit@3']:6.2f}{m['hit@5']:6.2f}  "
              f"{m['recall@3']:6.2f}{m['recall@5']:6.2f}  {m['mrr']:6.2f}")
    misses = [d for d in details if d["hit@5"] == 0]
    print(f"\nTop-5에 정답이 없는 질문 {len(misses)}개:")
    for d in misses:
        print(f"  {d['id']} [{d['type']}] {d['question']}\n      정답 {d['gold_articles']} / 검색 {[r['chunk_id'] for r in d['retrieved']]}")


STEPS = {"r0": (dense_search, "r0_dense"), "r1": (multi_query_search, "r1_multi_query"),
         "r2": (search_r2, "r2_rerank")}

if __name__ == "__main__":
    search, label = STEPS[sys.argv[1] if len(sys.argv) > 1 else "r0"]
    run(search, label)
