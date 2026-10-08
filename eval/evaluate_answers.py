"""답변 품질 평가. 골든셋 질문을 `/ask`와 같은 파이프라인(rag.pipeline)에 넣고, 답변을 채점한다.

코드로 확인하는 것 (LLM 없이)
    인용 일치  : 답변 본문에 정답 조문 번호가 하나라도 들어 있는가 (answerable 질문)
    거절 동작  : "확인되지 않습니다"로 답했는가. answerable=false면 거절이 정답, true면 거절은 오류
LLM 심판이 보는 것 (질문당 1회 호출, 정답 근거 문구와 검색된 본문을 함께 준다)
    정답성     : 정답 근거와 맞는 핵심 내용을 담았는가 (answerable=false면: 근거 없음을 밝히고 지어내지 않았는가)
    관련성     : 질문에 대한 답인가
    근거 충실성: 답변의 주장이 모두 검색된 본문에서 확인되는가
    환각       : 검색된 본문에 없는 사실을 말했는가 (1이 나쁨)

주의: 답변을 만든 모델과 채점하는 모델이 같은 계열이라 후하게 줄 수 있다. 사람이 샘플을 확인한다.
실행: uv run python -m eval.evaluate_answers [질문 수 제한]   (이어쓰기 가능. 이미 채점한 질문은 건너뜀)
결과: eval/results/answers_r2.json
"""
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import openai

from common.ai_model import get_llm_model
from eval.check_golden_set import load_golden_set
from rag.pipeline import NOT_FOUND, TOP_K, answer_question

RESULT = Path(__file__).resolve().parent / "results" / "answers_r2.json"
REFUSAL = re.compile(r"확인되지\s*않")

JUDGE_PROMPT = """당신은 법령 질의응답 시스템의 답변을 채점하는 심사자입니다.

[질문]
{question}

[정답 기준]
{gold}

[시스템이 검색해 답변에 쓴 법령 본문]
{context}

[시스템의 답변]
{answer}

아래 4가지를 채점해 JSON 객체 하나만 출력하세요. 다른 말은 쓰지 않습니다.
- correctness (0 또는 1): {correctness_rule}
- relevance (0 또는 1): 답변이 질문에 대한 답인가 (딴소리나 회피만 하면 0)
- groundedness (0 또는 1): 답변의 모든 주장이 위 [검색해 답변에 쓴 법령 본문]에서 확인되는가
- hallucination (0 또는 1): 위 본문에 없는 사실, 조문, 숫자를 말했으면 1 (없으면 0)
- reason: 한 문장

형식: {{"correctness": 1, "relevance": 1, "groundedness": 1, "hallucination": 0, "reason": "..."}}"""


def _gold_text(row: dict) -> tuple[str, str]:
    if row["answerable"]:
        gold = f"정답 조문: {', '.join(row['gold_articles'])}\n근거 문구: {row['evidence']}"
        rule = "답변이 위 정답 기준의 핵심 내용과 맞고 틀린 내용이 없는가"
    else:
        gold = "이 질문은 법 본문에 근거가 없다. 올바른 답변은 근거가 없음을 밝히는 것이다."
        rule = "근거가 없음을 밝히고, 없는 내용을 지어내 답하지 않았는가"
    return gold, rule


def _judge(row: dict, answer: str, context: str) -> dict:
    gold, rule = _gold_text(row)
    prompt = JUDGE_PROMPT.format(question=row["question"], gold=gold, correctness_rule=rule, context=context, answer=answer)
    reply = get_llm_model(max_tokens=400).invoke(prompt).content
    match = re.search(r"\{.*\}", reply, re.S)
    return json.loads(match.group()) if match else {}


def _evaluate_one(row: dict) -> dict:
    result = answer_question(row["question"])
    context = "\n\n".join(f"[{s.chunk_id}]\n{s.content}" for s in result.sources)
    judged = _judge(row, result.answer, context)
    refused = bool(REFUSAL.search(result.answer))
    cited = any(a in result.answer for a in row["gold_articles"]) if row["answerable"] else None
    return {
        "id": row["id"], "type": row["type"], "question": row["question"], "answerable": row["answerable"],
        "answer": result.answer, "sources": [s.chunk_id for s in result.sources[:TOP_K]],
        "refused": refused, "cited_gold": cited, **judged,
    }


def _with_retry(row: dict, attempts: int = 4) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            return _evaluate_one(row)
        except openai.RateLimitError:
            if attempt == attempts:
                raise
            print(f"  429 한도 초과, 60초 대기 ({attempt}/{attempts - 1})", flush=True)
            time.sleep(60)


def _mean(items: list[dict], key: str) -> float | None:
    values = [x[key] for x in items if x.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> OrderedDict:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in results:
        groups.setdefault(r["type"], []).append(r)
    groups["전체"] = results
    out = OrderedDict()
    for t, rs in groups.items():
        ans = [r for r in rs if r["answerable"]]
        out[t] = {
            "n": len(rs),
            "correctness": _mean(rs, "correctness"), "relevance": _mean(rs, "relevance"),
            "groundedness": _mean(rs, "groundedness"), "hallucination": _mean(rs, "hallucination"),
            "cited_gold": _mean([{"v": float(r["cited_gold"])} for r in ans if r["cited_gold"] is not None], "v"),
            "false_refusal": _mean([{"v": float(r["refused"])} for r in ans], "v"),
            "correct_refusal": _mean([{"v": float(r["refused"])} for r in rs if not r["answerable"]], "v"),
        }
    return out


def _fmt(v) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def main(limit: int | None = None) -> None:
    rows = load_golden_set()[:limit] if limit else load_golden_set()
    done = {r["id"]: r for r in json.loads(RESULT.read_text(encoding="utf-8"))["results"]} if RESULT.exists() else {}
    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        done[row["id"]] = _with_retry(row)
        RESULT.parent.mkdir(exist_ok=True)
        RESULT.write_text(json.dumps({"results": list(done.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(rows)}] {row['id']} 완료", flush=True)

    results = [done[r["id"]] for r in rows if r["id"] in done]
    summary = summarize(results)
    RESULT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n답변 평가 (질문 {len(results)}개)\n{'유형':<10}{'n':>3}  정답성 관련성 충실성 환각  조문인용 오거절 올바른거절")
    for t, m in summary.items():
        print(f"{t:<10}{m['n']:>3}  {_fmt(m['correctness'])} {_fmt(m['relevance'])} {_fmt(m['groundedness'])} {_fmt(m['hallucination'])}  "
              f"{_fmt(m['cited_gold'])} {_fmt(m['false_refusal'])} {_fmt(m['correct_refusal'])}")
    bad = [r for r in results if r.get("correctness") == 0 or r.get("hallucination") == 1]
    print(f"\n정답성 0 또는 환각 1인 질문 {len(bad)}개:")
    for r in bad:
        print(f"  {r['id']} [{r['type']}] {r['question']}\n      → {r.get('reason', '')}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
