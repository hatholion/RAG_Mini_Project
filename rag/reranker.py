"""검색된 후보를 LLM으로 다시 줄 세운다. (R2)

후보 전체를 한 번의 LLM 호출로 채점한다. (후보마다 부르면 호출이 후보 수만큼 늘어난다)
같은 질문·후보 조합은 캐시를 써서 매번 같은 순위를 얻는다.
이 파일은 검색 방식을 모른다. 후보를 받아 순서만 바꾼다. 연결은 pipeline.py에서 한다.
"""
import json
import re
from pathlib import Path

from common.ai_model import get_llm_model
from rag.retriever import Hit

RERANK_CACHE = Path(__file__).resolve().parent.parent / "eval" / "results" / "cache_rerank.json"
RERANK_MAX_TOKENS = 1024

RERANK_PROMPT = """당신은 대한민국 인공지능기본법 검색 결과의 관련도를 매기는 심사자입니다.
[질문]에 답하는 데 각 [후보]가 얼마나 직접적으로 도움이 되는지 0~10점으로 채점하세요.

채점 기준:
- 10점: 질문이 묻는 내용(정의, 의무, 절차, 처벌, 시행일 등)을 그대로 담은 조문
- 5~9점: 답의 일부를 담고 있거나 직접 관련된 조문
- 1~4점: 같은 단어만 나오거나 주변 내용인 조문
- 0점: 관련 없음
- 질문에 "제○조"가 있으면 그 번호의 조문이 가장 높은 점수를 받습니다.
- 정의를 묻는 질문에서는 용어를 정의한 조항이 그 용어를 단순히 사용한 조항보다 높습니다.

후보 {n}개 모두에 대해 JSON 배열만 출력하세요. 다른 말은 쓰지 않습니다.
형식: [{{"n": 1, "score": 7}}, {{"n": 2, "score": 0}}, ...]

[질문]
{question}

{candidates}"""


def _load_cache() -> dict[str, list[float]]:
    return json.loads(RERANK_CACHE.read_text(encoding="utf-8")) if RERANK_CACHE.exists() else {}


def _parse_scores(reply: str, n: int) -> list[float] | None:
    """LLM 답변에서 후보별 점수를 읽는다. 형식이 틀리면 None."""
    match = re.search(r"\[.*\]", reply, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group())
        scores = {int(item["n"]): float(item["score"]) for item in items}
    except (ValueError, KeyError, TypeError):
        return None
    return [scores.get(i, 0.0) for i in range(1, n + 1)] if scores else None


def rerank(question: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """후보 hits를 관련도 순으로 다시 정렬해 top_k개를 돌려준다. 점수가 같으면 원래 순서를 지킨다.
    채점에 실패하면 원래 순서를 그대로 쓴다."""
    key = question + "||" + ",".join(h.chunk_id for h in hits)
    cache = _load_cache()
    if key not in cache:
        candidates = "\n\n".join(f"[후보 {i}] {h.chunk_id}\n{h.content}" for i, h in enumerate(hits, 1))
        prompt = RERANK_PROMPT.format(n=len(hits), question=question, candidates=candidates)
        reply = get_llm_model(max_tokens=RERANK_MAX_TOKENS).invoke(prompt).content
        scores = _parse_scores(reply, len(hits))
        if scores is None:
            return hits[:top_k]  # 실패는 캐시하지 않아 다음에 다시 시도한다
        cache[key] = scores
        RERANK_CACHE.parent.mkdir(exist_ok=True)
        RERANK_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    order = sorted(range(len(hits)), key=lambda i: (-cache[key][i], i))
    return [Hit(hits[i].chunk_id, hits[i].article, cache[key][i], hits[i].content) for i in order[:top_k]]
