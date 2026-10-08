"""HWPX(한글 문서) 파일에서 본문 문단을 줄 단위 텍스트로 꺼낸다.

HWPX는 zip 파일이고, 본문은 `Contents/section0.xml`에 문단(<hp:p>) 단위로 들어 있다.
    <hp:p>               문단 하나 = 텍스트 한 줄
      <hp:run>           글자 모양이 같은 조각
        <hp:t>           실제 글자
        <hp:ctrl>        머리글·바닥글·표 등 (안에 또 <hp:p>가 있으므로 읽지 않는다)

여기서는 줄 목록까지만 만든다. 줄에서 장·조·항·호·목 구조를 찾는 일은
형식과 무관하게 `rag.law_parser`가 담당한다. (PDF 로더도 같은 파서를 쓸 수 있다)
"""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from rag.law_parser import parse_law_lines
from rag.loader import Law

DEFAULT_HWPX = Path(__file__).resolve().parent.parent / "data" / "ai_basic_law" / "ai_basic_law.hwpx"

_SECTION_FILE = "Contents/section0.xml"
_NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}


def read_hwpx_lines(path: Path | str = DEFAULT_HWPX) -> list[str]:
    """본문 최상위 문단을 한 줄씩 돌려준다. 빈 줄은 건너뛴다."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(_SECTION_FILE))

    lines = []
    for paragraph in root.findall("hp:p", _NS):  # 직계 문단만: 표·머리글 안의 문단은 제외
        # 글자(t)도 run 바로 아래 것만 읽는다. ctrl 안에 든 머리글·바닥글 글자는 제외된다.
        text = "".join(
            "".join(t.itertext()) for run in paragraph.findall("hp:run", _NS) for t in run.findall("hp:t", _NS)
        ).strip()
        if text:
            lines.append(text)
    return lines


def load_law_hwpx(path: Path | str = DEFAULT_HWPX) -> Law:
    return parse_law_lines(read_hwpx_lines(path))


if __name__ == "__main__":
    lines = read_hwpx_lines()
    print(f"문단 {len(lines)}개. 앞 12줄:")
    for line in lines[:12]:
        print(f"  | {line[:90]}")
