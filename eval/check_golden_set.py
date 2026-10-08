"""골든셋(eval/golden_set.jsonl)을 법 원문과 대조해 검사하고, 사람이 검토하기 쉬운 시트를 만든다.

검사하는 것
    - id가 겹치지 않는가
    - gold_articles의 조문이 실제로 있는가 (부칙은 "부칙")
    - gold_chunks의 청크 ID가 실제로 있는가
    - evidence(근거 문구)가 정답 조문의 원문에 그대로 들어 있는가  ← 정답이 맞다는 최소한의 증거
    - 답할 수 있는 질문(answerable)에는 정답이, 없는 질문에는 정답이 비어 있는가

실행: uv run python -m eval.check_golden_set
결과: 검사 결과를 출력하고 eval/golden_set_review.md(검토용 시트)를 다시 만든다.
"""
import json
import re
from collections import Counter
from pathlib import Path

from eval.compare_loaders import flatten_article
from rag.chunker import Chunk, chunk_law
from rag.hwpx_loader import load_law_hwpx
from rag.law_parser import normalize
from rag.loader import Law

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "golden_set.jsonl"
REVIEW_SHEET = EVAL_DIR / "golden_set_review.md"
ADDENDUM = "부칙"


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", normalize(text)).strip()


def _article_texts(law: Law) -> dict[str, str]:
    """조문번호 → 원문 전체 (비교용). 부칙은 "부칙" 하나로 합친다."""
    texts = {a.no: _squash(flatten_article(a)) for a in law.articles}
    texts[ADDENDUM] = _squash(" ".join(f"{ad.title} {ad.text}" for ad in law.addenda))
    return texts


def validate(rows: list[dict], law: Law, chunks: list[Chunk]) -> list[str]:
    """문제가 있는 항목을 사람이 읽을 수 있는 문장으로 돌려준다. 비어 있으면 통과."""
    texts = _article_texts(law)
    chunk_ids = {c.chunk_id for c in chunks}
    problems: list[str] = []

    for row_id, count in Counter(r["id"] for r in rows).items():
        if count > 1:
            problems.append(f"{row_id}: id가 {count}번 나온다")

    for r in rows:
        rid = r["id"]
        for article in r["gold_articles"]:
            if article not in texts:
                problems.append(f"{rid}: 정답 조문 '{article}'이 법에 없다")
        for chunk_id in r["gold_chunks"]:
            if chunk_id not in chunk_ids:
                problems.append(f"{rid}: 정답 청크 '{chunk_id}'가 없다")
        if r["answerable"] and not r["gold_articles"]:
            problems.append(f"{rid}: 답할 수 있는 질문인데 정답 조문이 비어 있다")
        if not r["answerable"] and r["gold_articles"]:
            problems.append(f"{rid}: 답할 수 없는 질문인데 정답 조문이 있다")
        if r["answerable"]:
            evidence = _squash(r["evidence"])
            if not any(evidence in texts.get(a, "") for a in r["gold_articles"]):
                problems.append(f"{rid}: 근거 문구가 정답 조문 원문에서 발견되지 않는다 → {r['evidence'][:30]}…")
    return problems


def _render(law: Law) -> dict[str, str]:
    """조문번호 → 사람이 읽기 좋은 원문 (머리글 1번, 항·호·목은 들여쓰기). 부칙은 "부칙" 하나로 합친다."""
    rendered = {}
    for a in law.articles:
        lines = [a.header]
        for p in a.paragraphs:
            if p.text:
                lines.append(p.text)
            for item in p.items:
                lines.append(f"  {item.text}")
                lines += [f"    {sub.text}" for sub in item.subitems]
        rendered[a.no] = "\n".join(lines)
    rendered[ADDENDUM] = "\n\n".join(f"{ad.title}\n{ad.text}" for ad in law.addenda)
    return rendered


def write_review_sheet(rows: list[dict], law: Law, path: Path = REVIEW_SHEET) -> None:
    """질문마다 제안 정답과 조문 원문을 나란히 보여 주는 검토용 마크다운을 만든다."""
    articles = _render(law)
    lines = [
        "# 골든셋 검토 시트",
        "",
        "> `eval/check_golden_set.py`가 자동으로 만든 파일이다. 직접 고치지 말고 `golden_set.jsonl`을 고친 뒤 다시 생성한다.",
        "> 질문마다 **제안 정답 조문**이 맞는지 원문(펼침)과 비교해 확인한다.",
        "",
        f"질문 {len(rows)}개: " + ", ".join(f"{t} {n}" for t, n in Counter(r['type'] for r in rows).items()),
        "",
    ]
    for r in rows:
        lines += [f"## {r['id']} · {r['type']}", "", f"**질문**: {r['question']}", ""]
        if r["answerable"]:
            chunk_note = f" (청크: {', '.join(r['gold_chunks'])})" if r["gold_chunks"] else ""
            lines += [
                f"- 제안 정답 조문: **{', '.join(r['gold_articles'])}**{chunk_note}",
                f"- 근거 문구: \"{r['evidence']}\"",
            ]
        else:
            lines += ["- 제안 정답: **없음** (법에 근거가 없으므로 \"확인되지 않는다\"고 답해야 하는 질문)"]
        if r["note"]:
            lines.append(f"- 비고: {r['note']}")
        lines += ["- [ ] 검토 완료 (정답이 맞다 / 고쳤다)", ""]
        for article in r["gold_articles"]:
            lines += [f"<details><summary>{article} 원문 보기</summary>", "", "```text", articles[article], "```", "", "</details>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    law = load_law_hwpx()
    chunks = chunk_law(law)
    rows = load_golden_set()

    problems = validate(rows, law, chunks)
    print(f"질문 {len(rows)}개:", dict(Counter(r["type"] for r in rows)))
    if problems:
        print(f"문제 {len(problems)}건:")
        for p in problems:
            print("  -", p)
    else:
        print("자동 검사 통과: 정답 조문·청크가 모두 존재하고, 근거 문구가 원문에서 발견됨")
    write_review_sheet(rows, law)
    print(f"검토용 시트 생성: {REVIEW_SHEET.relative_to(EVAL_DIR.parent)}")
