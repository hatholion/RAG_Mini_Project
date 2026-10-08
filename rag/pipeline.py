"""질문 → 검색 → 근거 기반 답변. `/ask`가 부르는 유일한 진입점.

    answer_question(question)
        1. search()로 관련 청크를 찾는다 (retriever.py. 검색 방식은 인자로 바꿔 끼운다)
        2. 찾은 청크를 컨텍스트로 묶어 LLM에게 준다
        3. 답변과, 답변의 근거가 된 청크(sources)를 돌려준다

검색 방식을 바꾸는 법: SEARCH 한 줄만 바꾼다. (R2 Reranker는 retriever 결과를 reranker로 거르는 함수를 만들어 끼운다)
"""
from collections.abc import Callable
from dataclasses import dataclass

from common.ai_model import get_llm_model
from rag.reranker import rerank
from rag.retriever import Hit, multi_query_search

CANDIDATES = 20  # 재정렬 전 후보 수. 정답이 6위 밖에 있어도 끌어올릴 수 있게 TOP_K보다 넓게 잡는다


def search_r2(question: str, top_k: int) -> list[Hit]:
    """R2: Multi Query로 후보 20개를 모으고, LLM이 질문과의 관련도로 다시 줄 세운다."""
    return rerank(question, multi_query_search(question, CANDIDATES), top_k)


SEARCH: Callable[[str, int], list[Hit]] = search_r2  # R2 (Dense + Multi Query + Reranker). R0~R2 중 최고 (Hit@1 0.87, MRR 0.92)
TOP_K = 5  # LLM에게 주는 청크 수. 평가에서 Hit@5를 천장으로 본 것과 같은 값
ANSWER_MAX_TOKENS = 1024
NOT_FOUND = "제공된 법령 본문에서는 확인되지 않습니다."

ANSWER_PROMPT = """당신은 대한민국 '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법'(인공지능기본법)을 설명하는 도우미입니다.
법을 잘 모르는 사용자가 이해할 수 있게 쉬운 말로 답하세요.

규칙:
- 아래 [법령 본문]에 있는 내용만 근거로 답합니다. 본문에 없는 내용은 일반 지식으로 보충하거나 추측하지 않습니다.
- 답변 안에서 근거가 되는 조문을 "제34조 제1항"처럼 밝힙니다.
- 질문에 답할 근거가 [법령 본문]에 없으면 "{not_found}"라고만 답하고, 관련 있어 보이는 조문이 있으면 한 줄로 알려 줍니다.
- 시행령(대통령령)에 위임된 세부 사항은 이 본문에 없으므로 "대통령령으로 정한다"까지만 말합니다.

[법령 본문]
{context}

[질문]
{question}"""


@dataclass
class Source:
    chunk_id: str
    article: str  # 조문번호. 부칙은 "부칙"
    content: str  # 원문


@dataclass
class Answer:
    answer: str
    sources: list[Source]


def build_context(hits: list[Hit]) -> str:
    """청크마다 [출처 ID]를 붙여 구분한다. 원문(content)에 조문 제목이 이미 들어 있다."""
    return "\n\n".join(f"[{h.chunk_id}]\n{h.content}" for h in hits)


def answer_question(question: str, search: Callable[[str, int], list[Hit]] | None = None) -> Answer:
    hits = (search or SEARCH)(question, TOP_K)
    prompt = ANSWER_PROMPT.format(not_found=NOT_FOUND, context=build_context(hits), question=question)
    reply = get_llm_model(max_tokens=ANSWER_MAX_TOKENS).invoke(prompt).content
    return Answer(
        answer=reply.strip(),
        sources=[Source(h.chunk_id, h.article, h.content) for h in hits],
    )


if __name__ == "__main__":
    result = answer_question("고영향 인공지능이란 무엇인가요?")
    print(result.answer, "\n\n근거:")
    for s in result.sources:
        print(f"  [{s.chunk_id}] {s.content[:40]!r}")
