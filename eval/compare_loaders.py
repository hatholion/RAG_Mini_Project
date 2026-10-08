"""로더 비교: 정답지(XML)와 다른 형식(HWPX, PDF 등)으로 읽은 결과가 얼마나 같은지 잰다.

세 단계로 비교한다.
    1. 구조 개수  : 조문·항·호·목·부칙이 몇 개씩 잡혔는가
    2. 조문 본문  : 조문마다 글자가 (공백·따옴표를 맞춘 뒤) 정확히 같은가
    3. 청크       : 같은 청킹 규칙을 적용했을 때 청크 ID와 내용이 같은가 (검색 결과에 직접 영향)

실행: uv run python -m eval.compare_loaders
다른 형식(PDF 등)을 비교하려면 compare_laws(load_law(), 내_로더_결과)를 호출하면 된다.
"""
import re

from rag.chunker import chunk_law
from rag.hwpx_loader import load_law_hwpx
from rag.law_parser import normalize
from rag.loader import Article, Law, load_law

MAX_SHOWN = 5  # 불일치 예시를 몇 개까지 보여 줄지


def flatten_article(article: Article) -> str:
    """조 하나의 모든 글자를 한 줄로 이어 비교용 문자열을 만든다."""
    parts = [article.header]
    for paragraph in article.paragraphs:
        parts.append(paragraph.text)
        for item in paragraph.items:
            parts.append(item.text)
            parts += [sub.text for sub in item.subitems]
    return re.sub(r"\s+", " ", normalize(" ".join(parts))).strip()


def _first_difference(a: str, b: str, width: int = 25) -> str:
    """두 문자열이 처음 달라지는 위치 주변을 보여 준다."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"…{a[max(0, i - width):i + width]}…  ≠  …{b[max(0, i - width):i + width]}…"


def _counts(law: Law) -> dict[str, int]:
    paragraphs = [p for a in law.articles for p in a.paragraphs]
    items = [i for p in paragraphs for i in p.items]
    return {
        "조문": len(law.articles),
        "항": len(paragraphs),
        "호": len(items),
        "목": sum(len(i.subitems) for i in items),
        "부칙": len(law.addenda),
    }


def compare_laws(reference: Law, candidate: Law) -> dict:
    ref_by_no = {a.no: a for a in reference.articles}
    cand_by_no = {a.no: a for a in candidate.articles}

    same_text, text_diffs = 0, []
    for no, ref in ref_by_no.items():
        cand = cand_by_no.get(no)
        if cand and flatten_article(ref) == flatten_article(cand):
            same_text += 1
        else:
            text_diffs.append((no, _first_difference(flatten_article(ref), flatten_article(cand)) if cand else "후보에 없음"))

    ref_chunks = {c.chunk_id: c for c in chunk_law(reference)}
    cand_chunks = {c.chunk_id: c for c in chunk_law(candidate)}
    same_chunks = sum(
        1 for cid, c in ref_chunks.items()
        if cid in cand_chunks and re.sub(r"\s+", " ", c.text) == re.sub(r"\s+", " ", cand_chunks[cid].text)
    )

    return {
        "counts": (_counts(reference), _counts(candidate)),
        "article_numbers_equal": list(ref_by_no) == list(cand_by_no),
        "titles_equal": sum(ref_by_no[n].title == cand_by_no[n].title for n in ref_by_no if n in cand_by_no),
        "section_equal": sum(
            (ref_by_no[n].chapter, ref_by_no[n].section) == (cand_by_no[n].chapter, cand_by_no[n].section)
            for n in ref_by_no if n in cand_by_no
        ),
        "text_same": same_text,
        "text_total": len(ref_by_no),
        "text_diffs": text_diffs,
        "chunk_ids_equal": list(ref_chunks) == list(cand_chunks),
        "chunk_same": same_chunks,
        "chunk_total": len(ref_chunks),
        "chunk_extra": sorted(set(cand_chunks) - set(ref_chunks)),
    }


def print_report(result: dict, candidate_name: str) -> None:
    ref_counts, cand_counts = result["counts"]
    print(f"\n[1] 구조 개수 (XML 정답지 vs {candidate_name})")
    for key in ref_counts:
        mark = "일치" if ref_counts[key] == cand_counts[key] else "불일치"
        print(f"  {key:<4} {ref_counts[key]:>4} vs {cand_counts[key]:>4}  {mark}")
    total = result["text_total"]
    print(f"\n[2] 조문 단위 비교 (정답지 조문 {total}개 기준)")
    print(f"  조문번호 순서 일치      : {result['article_numbers_equal']}")
    print(f"  조문 제목 일치          : {result['titles_equal']}/{total}")
    print(f"  장·절 소속 일치         : {result['section_equal']}/{total}")
    print(f"  본문 글자 완전 일치     : {result['text_same']}/{total}")
    for no, diff in result["text_diffs"][:MAX_SHOWN]:
        print(f"    - {no}: {diff}")
    print(f"\n[3] 청크 비교 (정답지 청크 {result['chunk_total']}개 기준)")
    print(f"  청크 ID 목록 동일       : {result['chunk_ids_equal']}")
    print(f"  청크 내용 완전 일치     : {result['chunk_same']}/{result['chunk_total']}")
    if result["chunk_extra"]:
        print(f"  정답지에 없는 청크      : {result['chunk_extra']}")


if __name__ == "__main__":
    print_report(compare_laws(load_law(), load_law_hwpx()), "HWPX")
