"""질문을 임베딩해서 Qdrant에서 비슷한 청크를 찾는다.

    R0  dense_search        질문 하나를 임베딩해 검색
    R1  multi_query_search  LLM이 질문을 법률 용어로 3개 바꿔 쓰고, 원 질문과 함께 검색해 RRF로 합친다

검색 개선은 이 파일에 한 가지씩 더하고, eval/evaluate.py로 같은 골든셋을 다시 잰다.
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from common.ai_model import get_embedding_model, get_llm_model
from common.qdrant import get_qdrant_client
from rag.vectorstore import COLLECTION_NAME, DENSE


@dataclass
class Hit:
    chunk_id: str
    article: str  # 조문번호. 부칙은 "부칙"
    score: float
    content: str


def dense_search(question: str, top_k: int = 10) -> list[Hit]:
    """질문 임베딩과 코사인 유사도가 높은 청크 top_k개를 점수 순으로 돌려준다."""
    vector = get_embedding_model().embed_query(question)
    response = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME, query=vector, using=DENSE, limit=top_k, with_payload=True
    )
    return [
        Hit(p.payload["chunk_id"], p.payload["article"], p.score, p.payload["content"])
        for p in response.points
    ]


# ── R1: Multi Query ──────────────────────────────────────────────────────

N_REWRITES = 3
RRF_K = 60  # RRF 점수 = Σ 1/(RRF_K + 순위). 관례값
REWRITE_CACHE = Path(__file__).resolve().parent.parent / "eval" / "results" / "cache_rewrites.json"

REWRITE_PROMPT = """당신은 대한민국 '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법'(인공지능기본법)을 검색하는 도우미입니다.
사용자의 질문을 법 조문에서 실제로 쓰이는 용어(예: 고영향 인공지능, 인공지능사업자, 이용자, 학습용데이터, 과태료, 신고, 시책, 의무)로 바꿔서,
같은 뜻을 서로 다른 표현으로 묻는 검색용 질문 {n}개를 만드세요.

규칙:
- 질문마다 한 줄로 쓰고, 번호나 기호는 붙이지 않습니다.
- 질문의 뜻을 바꾸거나 새로운 사실을 덧붙이지 않습니다.
- 질문에 조문 번호가 있으면 그대로 유지합니다.

사용자 질문: {question}"""


def _load_cache() -> dict[str, list[str]]:
    return json.loads(REWRITE_CACHE.read_text(encoding="utf-8")) if REWRITE_CACHE.exists() else {}


def rewrite_queries(question: str, n: int = N_REWRITES) -> list[str]:
    """질문을 법률 용어로 n개 바꿔 쓴다. 같은 질문은 캐시를 써서 매번 같은 결과를 얻는다."""
    cache = _load_cache()
    if question not in cache:
        reply = get_llm_model().invoke(REWRITE_PROMPT.format(n=n, question=question)).content
        lines = [line.strip(" -•\t0123456789.)") for line in reply.splitlines()]
        cache[question] = [line for line in lines if line][:n]
        REWRITE_CACHE.parent.mkdir(exist_ok=True)
        REWRITE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache[question]


def multi_query_search(question: str, top_k: int = 10) -> list[Hit]:
    """원 질문 + 바꿔 쓴 질문 3개를 각각 검색하고, 순위를 RRF로 합쳐 top_k개를 돌려준다."""
    queries = [question] + rewrite_queries(question)
    vectors = get_embedding_model().embed_documents(queries)  # 한 번의 호출로 모두 임베딩
    client = get_qdrant_client()

    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Hit] = {}
    for vector in vectors:
        points = client.query_points(
            collection_name=COLLECTION_NAME, query=vector, using=DENSE, limit=top_k, with_payload=True
        ).points
        for rank, p in enumerate(points, 1):
            cid = p.payload["chunk_id"]
            scores[cid] += 1 / (RRF_K + rank)
            by_id.setdefault(cid, Hit(cid, p.payload["article"], 0.0, p.payload["content"]))

    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [Hit(by_id[c].chunk_id, by_id[c].article, scores[c], by_id[c].content) for c in ranked]


if __name__ == "__main__":
    question = "고영향 인공지능이란 무엇인가요?"
    print("R0 dense")
    for hit in dense_search(question, top_k=5):
        print(f"  {hit.score:.3f}  {hit.chunk_id:<14} {hit.content[:40]!r}")
    print("R1 바꿔 쓴 질문:", rewrite_queries(question))
    for hit in multi_query_search(question, top_k=5):
        print(f"  {hit.score:.4f} {hit.chunk_id:<14} {hit.content[:40]!r}")
