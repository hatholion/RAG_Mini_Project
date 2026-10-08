"""법령 구조 기반 청킹.

기본 단위는 조(條)이고, 검색이 흐려지는 경우에만 더 잘게 나눈다.

    정의 조항(제2조)   → 호 단위. 정의 12개가 한 벡터에 섞이지 않게 한다. 목은 상위 호와 함께 둔다.
    긴 조(MAX_CHARS↑)  → 항 단위. 항 경계는 지키고, 짧은 이웃 항은 한 청크로 묶는다.
    부칙               → 조 단위와 별개 청크. 시행일 질문에 답하기 위해 둔다.
    그 밖의 조         → 조 하나가 청크 하나

모든 청크의 임베딩 텍스트 앞에 `[법 > 장 > 절 > 조(제목) > 항/호]` 경로를 붙여 문맥을 보존한다.
"""
from dataclasses import dataclass

from rag.loader import Article, Item, Law, Paragraph, load_law

MAX_CHARS = 1000  # 이보다 긴 조는 항 단위로 분할한다. (조 길이 분포를 보고 정한 값)
DEFINITION_TITLE = "정의"  # 이 제목의 조는 길이와 무관하게 호 단위로 분할한다.

# chunk_type 값
ARTICLE, PARAGRAPHS, ITEM, ADDENDUM = "article", "paragraphs", "item", "addendum"


@dataclass
class Chunk:
    chunk_id: str  # 사람이 읽을 수 있는 ID. 예: "제2조#제4호", "제7조", "부칙#21311"
    text: str  # 임베딩 대상 (경로 prefix + 본문)
    content: str  # 원문. /ask 응답의 sources에 쓴다.
    metadata: dict  # Qdrant payload로 저장되는 필터·출처 정보


# ── 텍스트 조립 ──────────────────────────────────────────────────────────


def _item_text(item: Item) -> str:
    """호 본문 + 목(들여쓰기)."""
    return "\n".join([item.text] + [f"  {sub.text}" for sub in item.subitems])


def _paragraph_text(paragraph: Paragraph) -> str:
    """항 본문 + 호들. 항번호 없는 항(제2조)은 본문이 비어 있다."""
    lines = [paragraph.text] if paragraph.text else []
    return "\n".join(lines + [_item_text(item) for item in paragraph.items])


def _article_text(article: Article) -> str:
    parts = [article.header] + [_paragraph_text(p) for p in article.paragraphs]
    return "\n".join(part for part in parts if part)


def _build_chunk(
    law: Law,
    chunk_id: str,
    chunk_type: str,
    content: str,
    path: str,
    *,
    article: str = "",
    article_title: str = "",
    chapter: str = "",
    section: str = "",
    part: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"[{path}]\n{content}",
        content=content,
        metadata={
            "law_name": law.name,
            "chapter": chapter,
            "section": section,
            "article": article,  # 조문번호 필터용. 부칙은 "부칙"
            "article_title": article_title,
            "part": part,  # 항/호 범위. 조 단위 청크면 ""
            "chunk_type": chunk_type,
            "promulgation_date": law.promulgation_date,
            "effective_date": law.effective_date,
        },
    )


def _article_chunk(law: Law, a: Article, chunk_id: str, chunk_type: str, content: str, part: str = "") -> Chunk:
    """조에 속한 청크를 만든다. 경로 prefix와 조 메타데이터를 채운다."""
    where = " > ".join(x for x in (a.chapter, a.section) if x)
    path = f"{law.name} > {where} > {a.no}({a.title})" + (f" > {part}" if part else "")
    return _build_chunk(
        law, chunk_id, chunk_type, content, path,
        article=a.no, article_title=a.title, chapter=a.chapter, section=a.section, part=part,
    )


# ── 분할 규칙 ────────────────────────────────────────────────────────────


def _split_by_item(law: Law, a: Article) -> list[Chunk]:
    """정의 조항: 호 하나가 청크 하나. 도입 문장("용어의 뜻은 다음과 같다")을 앞에 붙인다."""
    chunks = []
    for paragraph in a.paragraphs:
        for item in paragraph.items:
            part = f"제{item.no.rstrip('.')}호"
            content = f"{a.header}\n{_item_text(item)}"
            chunks.append(_article_chunk(law, a, f"{a.no}#{part}", ITEM, content, part))
    return chunks


def _split_by_paragraph(law: Law, a: Article, max_chars: int) -> list[Chunk]:
    """긴 조: 항 경계를 지키며, max_chars를 넘지 않는 범위에서 이웃 항을 묶는다."""
    groups: list[list[Paragraph]] = [[]]
    size = len(a.header)
    for paragraph in a.paragraphs:
        length = len(_paragraph_text(paragraph))
        if groups[-1] and size + length > max_chars:
            groups.append([])
            size = len(a.header)
        groups[-1].append(paragraph)
        size += length

    chunks = []
    for group in groups:
        part = group[0].no if len(group) == 1 else f"{group[0].no}~{group[-1].no}"
        content = "\n".join([a.header] + [_paragraph_text(p) for p in group])
        chunks.append(_article_chunk(law, a, f"{a.no}#{part}", PARAGRAPHS, content, part))
    return chunks


def _chunk_addenda(law: Law) -> list[Chunk]:
    """부칙. 다른 법률을 고치는 내용(본문에 "생략"이 들어 있는 부칙)은 이 법과 무관하므로 제외한다."""
    chunks = []
    for addendum in law.addenda:
        if "생략" in addendum.text:
            continue
        content = f"{addendum.title}\n{addendum.text}"
        chunks.append(
            _build_chunk(
                law, f"부칙#{addendum.no}", ADDENDUM, content, f"{law.name} > {addendum.title}",
                article="부칙", article_title=addendum.title,
            )
        )
    return chunks


def chunk_law(law: Law, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for a in law.articles:
        text = _article_text(a)
        if a.title == DEFINITION_TITLE:
            chunks += _split_by_item(law, a)
        elif len(text) > max_chars and len(a.paragraphs) > 1:
            chunks += _split_by_paragraph(law, a, max_chars)
        else:
            chunks.append(_article_chunk(law, a, a.no, ARTICLE, text))
    return chunks + _chunk_addenda(law)


if __name__ == "__main__":
    import statistics

    law = load_law()
    chunks = chunk_law(law)
    lengths = [len(c.content) for c in chunks]
    print(f"조문 {len(law.articles)}개 → 청크 {len(chunks)}개")
    print(f"길이(자) 최소 {min(lengths)} / 중앙 {statistics.median(lengths):.0f} / 최대 {max(lengths)}")
    print("분할되었거나 긴 청크:")
    for c in chunks:
        if c.metadata["chunk_type"] != ARTICLE or len(c.content) > 600:
            print(f"  {c.chunk_id:<20} {c.metadata['chunk_type']:<11} {len(c.content)}자")
