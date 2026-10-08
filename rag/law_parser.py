"""텍스트 줄 목록에서 장·절·조·항·호·목 구조를 찾아 Law 트리로 만든다.

입력이 HWPX든 PDF든 "한 줄 = 한 문단"인 줄 목록으로 바꿔 주기만 하면 이 파서를 그대로 쓸 수 있다.
XML 로더(rag.loader.load_law)와 같은 Law 구조를 돌려주므로, 이후의 청킹·적재는 바뀌지 않는다.

줄을 구분하는 패턴
    장        제1장 총칙
    절        제2절 인공지능산업 …
    조        제17조의2(제목) …      ← 같은 줄에 첫 항(①)이나 본문이 이어 붙기도 한다
    항        ① …
    호        1. …   /  4의2. …
    목        가. …
    부칙      부칙 <법률 제21311호, 2026. 1. 20.>
    (버림)    [제목개정 2026. 1. 20.] 같은 개정 이력 줄

패턴에 맞지 않는 줄(문단이 줄바꿈된 경우)은 바로 앞 요소의 이어지는 글로 붙인다.
"""
import logging
import re

from rag.loader import Addendum, Article, Item, Law, Paragraph, SubItem, clean

logger = logging.getLogger(__name__)

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿"

_CHAPTER = re.compile(r"^제\d+장\s")
_SECTION = re.compile(r"^제\d+절\s")
_ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\([^)]+\))\s*(.*)$")  # (제17조의2(제목), 같은 줄의 나머지)
_PARAGRAPH = re.compile(rf"^([{_CIRCLED}])")
_ITEM = re.compile(r"^(\d+(?:의\d+)?)\.\s")
_SUBITEM = re.compile(r"^([가-힣])\.\s")
_ADDENDUM = re.compile(r"^부칙\s*<")
_REVISION_NOTE = re.compile(r"^\[[가-힣 ]+\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\]$")  # "[제목개정 2026. 1. 20.]" 조문 아래의 개정 이력 줄

# 머리말: "[시행 2026. 7. 21.] [법률 제21311호, 2026. 1. 20., 일부개정]"
_EFFECTIVE = re.compile(r"\[시행\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\]")
_PROMULGATED = re.compile(r"\[법률\s*제\d+호,\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.")
_ADDENDUM_TITLE = re.compile(r"<(?:법률\s*)?제(\d+)호,\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.>")
_NICKNAME = re.compile(r"\s*\(\s*약칭.*\)\s*$")  # "법령명( 약칭: …)" 의 약칭 부분

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})


def normalize(text: str) -> str:
    """개정 표기를 지우고 따옴표·공백을 통일한다. (XML 로더의 결과와 같은 모양으로 맞추기 위함)"""
    return clean(text.translate(_QUOTES))


def _date(match: re.Match | None) -> str:
    """정규식의 (연, 월, 일) 그룹을 YYYYMMDD로. 없으면 ""."""
    if not match:
        return ""
    year, month, day = match.groups()[-3:]
    return f"{year}{int(month):02d}{int(day):02d}"


def _addendum_title(line: str) -> tuple[str, str]:
    """부칙 제목 줄 → (공포번호, XML 로더와 같은 형식의 제목). 예: ("21311", "부칙 <제21311호,2026.1.20>")"""
    match = _ADDENDUM_TITLE.search(line)
    if not match:
        return "", line
    no, year, month, day = match.groups()
    return no, f"부칙 <제{no}호,{year}.{month}.{day}>"


class _Builder:
    """줄을 한 줄씩 받아 트리를 만든다. 지금 열려 있는 조·항·호·목을 기억한다."""

    def __init__(self) -> None:
        self.articles: list[Article] = []
        self.addenda: list[Addendum] = []
        self.chapter = self.section = ""
        self._addendum_lines: list[str] = []

    # 열려 있는 요소 (가장 최근에 시작된 것)
    @property
    def _article(self) -> Article | None:
        return self.articles[-1] if self.articles else None

    @property
    def _paragraph(self) -> Paragraph | None:
        return self._article.paragraphs[-1] if self._article and self._article.paragraphs else None

    @property
    def _item(self) -> Item | None:
        return self._paragraph.items[-1] if self._paragraph and self._paragraph.items else None

    def feed(self, line: str) -> None:
        text = normalize(line)
        if _REVISION_NOTE.match(text):
            return  # 개정 이력 줄은 본문이 아니므로 버린다
        if _ADDENDUM.match(text):
            self._start_addendum(text)
        elif self.addenda:  # 부칙이 시작된 뒤의 줄은 모두 부칙 본문
            self._addendum_lines.append(text)
        elif _CHAPTER.match(text):
            self.chapter, self.section = text, ""
        elif _SECTION.match(text):
            self.section = text
        elif match := _ARTICLE.match(text):
            self._start_article(*match.groups())
        elif not self._article:
            logger.debug("조문 앞의 줄을 건너뜀: %s", text[:40])
        elif _PARAGRAPH.match(text):
            self._article.paragraphs.append(Paragraph(no=text[0], text=text))
        elif match := _ITEM.match(text):
            self._add_item(f"{match.group(1)}.", text)
        elif (match := _SUBITEM.match(text)) and self._item:
            self._item.subitems.append(SubItem(no=f"{match.group(1)}.", text=text))
        else:
            self._append_continuation(text)

    def _start_article(self, head: str, rest: str) -> None:
        no = re.match(r"제\d+조(?:의\d+)?", head).group()
        title = head[head.index("(") + 1 : -1]
        article = Article(no=no, title=title, header=head, chapter=self.chapter, section=self.section)
        self.articles.append(article)
        if _PARAGRAPH.match(rest):
            # "제3조(제목) ① …": 머리글은 제목까지만, 첫 항은 따로 만든다.
            article.paragraphs.append(Paragraph(no=rest[0], text=rest))
        elif rest:
            article.header = f"{head} {rest}"  # 제1조처럼 본문이 머리글과 한 줄인 경우

    def _add_item(self, no: str, text: str) -> None:
        if not self._article.paragraphs:  # 제2조(정의)처럼 항 없이 호가 바로 나오는 경우
            self._article.paragraphs.append(Paragraph(no="", text=""))
        self._article.paragraphs[-1].items.append(Item(no=no, text=text))

    def _append_continuation(self, text: str) -> None:
        logger.warning("패턴에 맞지 않아 앞 요소에 이어 붙임: %s", text[:50])
        if self._item and self._item.subitems:
            target = self._item.subitems[-1]
        else:
            target = self._item or self._paragraph
        if target:
            target.text += f" {text}"
        else:
            self._article.header += f" {text}"

    def _start_addendum(self, title_line: str) -> None:
        self._flush_addendum()
        self.addenda.append(Addendum(no="", title=title_line, text=""))

    def _flush_addendum(self) -> None:
        if self.addenda and self._addendum_lines:
            last = self.addenda[-1]
            last.text = "\n".join(self._addendum_lines)
        self._addendum_lines = []

    def finish(self) -> None:
        self._flush_addendum()
        for addendum in self.addenda:  # 제목 줄을 XML 로더와 같은 형식으로 통일
            addendum.no, addendum.title = _addendum_title(addendum.title)


def parse_law_lines(lines: list[str]) -> Law:
    """줄 목록(제목·버전 머리말 + 본문 + 부칙) → Law."""
    builder = _Builder()
    for line in lines:
        builder.feed(line)
    builder.finish()

    preface = " ".join(lines[:5])  # 제목과 "[시행 …] [법률 제…호 …]" 줄은 맨 앞에 있다
    return Law(
        name=_NICKNAME.sub("", normalize(lines[0])),
        promulgation_date=_date(_PROMULGATED.search(preface)),
        effective_date=_date(_EFFECTIVE.search(preface)),
        articles=builder.articles,
        addenda=builder.addenda,
    )
