"""법제처 Open API의 법령 본문 XML을 읽어 장·조·항·호·목 트리(Law)로 변환한다.

XML 태그와 이 모듈의 데이터 구조는 다음과 같이 대응한다.

    <조문단위>  조문여부=전문   → 장·절 제목 행 (조문이 아님. 현재 위치로만 기억)
    <조문단위>  조문여부=조문   → Article  (조)
      <항>                      → Paragraph (항, ①②…)
        <호>                    → Item      (호, 1. 2. …)
          <목>                  → SubItem   (목, 가. 나. …)
    <부칙단위>                  → Addendum  (부칙)

주의할 XML 특성
    - 제1조처럼 항이 없는 조는 본문이 <조문내용>에 통째로 들어 있다.
    - 제2조(정의)는 항번호 없는 <항> 안에 <호>가 바로 들어 있다.
    - "제17조의2"는 <조문가지번호>로, "4의2호"는 호내용 문자열로 표현된다.
"""
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_MST = "282791"  # 인공지능기본법 현행 법령일련번호. 개정되면 번호가 바뀐다.
DEFAULT_XML = (
    Path(__file__).resolve().parent.parent / "data" / "ai_basic_law" / f"ai_basic_law_{DEFAULT_MST}.xml"
)

# <개정 2026.1.20>, <신설 2026.1.20> 같은 개정 표기. 검색에 도움이 안 되므로 제거한다.
_AMENDMENT_MARK = re.compile(r"<(?:개정|신설|삭제|본조신설|전문개정|제목개정)[^>]*>")
_CHAPTER_HEADING = re.compile(r"제\d+장")


# ── 데이터 구조 ──────────────────────────────────────────────────────────


@dataclass
class SubItem:  # 목
    no: str
    text: str


@dataclass
class Item:  # 호
    no: str
    text: str
    subitems: list[SubItem] = field(default_factory=list)


@dataclass
class Paragraph:  # 항. 제2조처럼 항번호가 없는 경우 no="", text=""
    no: str
    text: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Article:  # 조
    no: str  # "제17조의2"
    title: str
    header: str  # <조문내용>. 항이 없는 조(제1조)는 본문 전체가 여기에 들어 있다.
    chapter: str  # "제1장 총칙"
    section: str  # "제2절 …". 절이 없으면 ""
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Addendum:  # 부칙
    no: str  # 공포번호. 예: "21311"
    title: str  # "부칙 <제21311호,2026.1.20>"
    text: str


@dataclass
class Law:
    name: str
    promulgation_date: str  # 공포일자 (YYYYMMDD)
    effective_date: str  # 시행일자 (YYYYMMDD)
    articles: list[Article]
    addenda: list[Addendum] = field(default_factory=list)


# ── 다운로드 ─────────────────────────────────────────────────────────────


def download_law_xml(mst: str = DEFAULT_MST, out_path: Path = DEFAULT_XML) -> Path:
    """법제처 본문 API로 법령 XML을 내려받는다. .env의 LAW_OC(API 인증키)가 필요하다."""
    oc = os.getenv("LAW_OC")
    if not oc:
        raise RuntimeError(".env에 LAW_OC가 없습니다.")
    url = f"http://www.law.go.kr/DRF/lawService.do?OC={oc}&target=law&MST={mst}&type=XML"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        out_path.write_bytes(response.read())
    return out_path


# ── XML 파싱 ─────────────────────────────────────────────────────────────


def clean(text: str | None) -> str:
    """개정 표기를 지우고 줄 안의 공백을 정리한다 (줄바꿈은 유지)."""
    text = _AMENDMENT_MARK.sub("", text or "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _text(node: ET.Element, tag: str) -> str:
    return clean(node.findtext(tag))


def _article_no(unit: ET.Element) -> str:
    no = f"제{_text(unit, '조문번호')}조"
    branch = _text(unit, "조문가지번호")
    return f"{no}의{branch}" if branch else no


def _parse_item(node: ET.Element) -> Item:
    # 호번호는 가지번호("4의2")를 따로 담아 불완전하다. 번호가 이미 들어 있는 호내용을 쓴다.
    return Item(
        no=_text(node, "호번호"),
        text=_text(node, "호내용"),
        subitems=[SubItem(_text(m, "목번호"), _text(m, "목내용")) for m in node.findall("목")],
    )


def _parse_paragraph(node: ET.Element) -> Paragraph:
    return Paragraph(
        no=_text(node, "항번호"),
        text=_text(node, "항내용"),
        items=[_parse_item(h) for h in node.findall("호")],
    )


def _parse_articles(root: ET.Element) -> list[Article]:
    articles: list[Article] = []
    chapter = section = ""
    for unit in root.find("조문").findall("조문단위"):
        if _text(unit, "조문여부") == "전문":
            # 장·절 제목 행. 조문이 아니므로 현재 위치만 갱신한다.
            heading = _text(unit, "조문내용")
            if _CHAPTER_HEADING.match(heading):
                chapter, section = heading, ""
            else:
                section = heading
            continue
        articles.append(
            Article(
                no=_article_no(unit),
                title=_text(unit, "조문제목"),
                header=_text(unit, "조문내용"),
                chapter=chapter,
                section=section,
                paragraphs=[_parse_paragraph(p) for p in unit.findall("항")],
            )
        )
    return articles


def _parse_addenda(root: ET.Element) -> list[Addendum]:
    addenda = []
    for unit in root.find("부칙").findall("부칙단위"):
        # 부칙내용은 여러 CDATA 조각이 하나의 문자열로 합쳐져 있고, 첫 줄이 제목이다.
        lines = [ln.strip() for ln in clean(unit.findtext("부칙내용")).splitlines() if ln.strip()]
        addenda.append(Addendum(no=_text(unit, "부칙공포번호"), title=lines[0], text="\n".join(lines[1:])))
    return addenda


def load_law(xml_path: Path | str = DEFAULT_XML) -> Law:
    root = ET.parse(xml_path).getroot()
    info = root.find("기본정보")
    return Law(
        name=_text(info, "법령명_한글"),
        promulgation_date=_text(info, "공포일자"),
        effective_date=_text(info, "시행일자"),
        articles=_parse_articles(root),
        addenda=_parse_addenda(root),
    )


if __name__ == "__main__":
    law = load_law()
    print(f"{law.name} (공포 {law.promulgation_date}) / 조문 {len(law.articles)}개, 부칙 {len(law.addenda)}개")
    for article in law.articles[:5]:
        print(f"  {article.chapter} | {article.no} {article.title} (항 {len(article.paragraphs)}개)")
