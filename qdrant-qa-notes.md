# Qdrant / Docker Compose Q&A 정리

> **이 문서의 성격**: 프로젝트를 진행하며 나온 질문과 답을 순서대로 모은 **참고용 Q&A 노트**입니다. 과제의 설계 결정과 근거는 [rag-strategy-notes.md](rag-strategy-notes.md)에 정리되어 있고, 이 문서는 환경 설정(Docker·Qdrant·uv) 이해에 쓰세요.
>
> | 번호 | 주제 | 지금도 유효한가 |
> |---|---|---|
> | 1~5 | Docker Compose, Qdrant 개념, 온프레미스, API 키 | 유효 |
> | 6~7 | LangChain 연결, `pyproject.toml`의 패키지 설정 | 유효 (`rag` 패키지가 추가됨) |
> | 8~10 | 문서 파서·입력 형식·설계 초안 | **일부 변경됨.** 각 절 맨 위의 "현재 상태" 참고 |

## 1. docker-compose.yaml 주석
- [docker-compose.yaml](docker-compose.yaml)에 한국어 설명 주석 추가 (설정 값은 변경 없음)
- 포트: `6333` = REST API + 웹 대시보드, `6334` = gRPC
- `restart: unless-stopped` = 직접 stop하지 않는 한 자동 재시작
- 볼륨 `qdrant_storage` = 컨테이너를 삭제해도 데이터 유지

## 2. docker compose 명령은 파일 위치에서 실행해야 하나?
- `up`, `down`, `ps`, `logs`는 현재 디렉터리의 compose 파일을 찾음 → `docker-qdrant/`에서 실행
- 다른 위치에서는 `docker compose -f <경로>/docker-compose.yaml up -d`
- 컨테이너가 떠 있으면 파이썬 코드는 어느 폴더에서든 `localhost:6333`으로 접속 가능
- Qdrant는 벡터 DB이지 모델(LLM/임베딩)이 아님. 모델은 별도
- `docker compose down -v`는 볼륨(데이터)까지 삭제하므로 주의

## 3. Pinecone vs Qdrant
| | Pinecone | Qdrant (현재 설정) |
|---|---|---|
| 방식 | 완전 관리형 클라우드(SaaS) | 오픈소스, 직접 실행 |
| 실행 위치 | Pinecone 서버 | 내 PC의 Docker 컨테이너 |
| 접속 | API 키 + 인터넷 | `localhost:6333`, 키/인터넷 불필요 |
| 데이터 | Pinecone 클라우드 | 내 PC의 Docker 볼륨 |
| 비용 | 플랜별 과금 | 무료 (내 PC 자원 사용) |

- Qdrant에도 클라우드(Qdrant Cloud)가 있음. 접속 주소/키만 바꾸면 같은 코드 사용 가능

## 4. 온프레미스?
- Qdrant는 오픈소스라 사내 서버에 설치해 온프레미스 운영 가능
- 지금은 내 PC에서 띄운 "로컬 개발 환경". 같은 compose를 사내 서버에서 돌리면 온프레미스
- 배포 방식: 셀프호스팅 / Qdrant Cloud / Hybrid Cloud
- 운영 시 챙길 것: API 키·TLS, 볼륨 백업, 버전 태그 고정, 대규모 시 클러스터

## 5. 클라우드 안 쓰면 API 키 필요 없나?
- 기본 설정은 인증 없음 → 로컬 개발은 키 없이 사용 가능
- 서버/외부 접근 환경에서는 키 설정 권장:
  ```yaml
  environment:
    QDRANT__SERVICE__API_KEY: "내가정한키"
  ```
  ```python
  QdrantClient(host="localhost", port=6333, api_key="내가정한키")
  ```
- OpenAI 임베딩/LLM을 쓰면 OpenAI 키는 별도로 필요 (로컬 모델이면 불필요)

## 6. LangChain 연결과 경로
- [pyproject.toml](../pyproject.toml)에 `langchain-qdrant`, `langchain-openai` 이미 포함
  - `qdrant-client`: 기본 클라이언트
  - `langchain-qdrant`: LangChain VectorStore 어댑터
  - `langchain-openai`: LangChain용 OpenAI 임베딩/LLM
- 설치: 프로젝트 루트에서 `uv sync`
- 경로 주의점
  - `docker compose` → `docker-qdrant/`에서 실행
  - `uv sync`/`uv add` → `pyproject.toml`이 있는 루트에서 실행
  - `packages = ["common", "app", "rag"]` → `from common...`, `from rag...` 임포트는 루트 기준
  - `.env`(OpenAI 키) 위치도 `python-dotenv`가 찾을 수 있는 곳에 두기

## 7. pyproject.toml의 build-system 설정은 import에 관여하나?
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["common", "app", "rag"]
```
- 관여함. 단, import를 직접 처리하는 게 아니라 **설치 단계**에서 영향을 줌
- `[build-system]`: 프로젝트를 패키지로 빌드·설치할 때 `hatchling` 사용. `uv sync` 시 프로젝트 자체가 `.venv`에 설치됨
- `packages = ["common", "app", "rag"]`: 설치 시 `common/`, `app/`, `rag/` 폴더를 패키지로 포함 (`rag`는 이후 추가함)
- 효과
  - `uv sync` 후 `common`, `app`, `rag`가 `.venv`에 등록됨 (보통 editable 모드라 코드 수정이 바로 반영)
  - `notebook/` 등 어느 위치에서도 `from common.xxx import ...`, `from rag.xxx import ...` 가능
  - 설정이 없으면 루트에서만 import되고, `notebook/` 안에서는 `ModuleNotFoundError` 가능
- 주의
  - 폴더에 `__init__.py`가 있으면 안전
  - 새 패키지 폴더(`src/` 등)를 만들면 `packages` 목록에 추가
  - 노트북 커널이 이 프로젝트의 `.venv`(ipykernel)를 사용해야 함

## 8. 법률 문서 구조 파싱: Docling보다 나은 게 있나?
> **현재 상태**: 실제로 법제처 Open API의 XML이 가장 정확하다는 것을 확인했고(46개 조문이 태그로 구조화돼 있음), 로더는 XML 기준으로 구현했다. 입력 형식 선택의 전체 근거는 [rag-strategy-notes.md](rag-strategy-notes.md)의 "결정 1"을 참고.

- 항상 "더 낫다"고 할 수는 없고 문서 종류에 따라 다름
- Docling: PDF/DOCX → 레이아웃 구조(제목/표/문단) 마크다운·JSON 변환 범용 파서. 제N조/①항/1호/가목 같은 법률 의미 구조는 모름, HWP 기본 미지원
- 대안
  1. 국가법령정보센터 Open API(law.go.kr): 이미 조·항·호 단위로 구조화된 XML/JSON → 법령 원문이면 가장 정확
  2. 정규식·규칙 기반 청킹: `제\d+조`, `①`, `1.` 패턴으로 조문 단위 분할 (payload `law_name/article/title/content`와 잘 맞음)
  3. 한국어/스캔/표 많은 PDF: Upstage Document Parse, LlamaParse, Azure Document Intelligence, MinerU, PyMuPDF4LLM 비교
- 결론: 조문 더미데이터를 조 단위로 저장하는 구조면 파서 교체보다 규칙 기반 조문 청킹이 효과적. 실제 PDF/HWP 투입 시 샘플로 직접 비교

## 9. 인공지능 기본법 QA RAG 설계 (hwpx/doc/pdf/html 입력)
> **현재 상태**: 초기 설계 초안이다. 과제 요구사항(Docling, 골든셋 평가)을 확인한 뒤 **청킹 방침이 바뀌었다**: 항 단위가 아니라 "조 단위 + 정의 조항·긴 조만 분할"이고, Parent-Child는 쓰지 않는다. 확정된 내용은 [rag-strategy-notes.md](rag-strategy-notes.md)를 기준으로 한다. 아래는 당시 생각의 기록으로 남긴다.

핵심: 법령 논리 구조(장 → 조 → 항 → 호 → 목)를 청킹 단위로 사용

### 파이프라인
```text
hwpx/doc/pdf/html → 포맷별 로더 → 공통 중간 표현
→ 법령 구조 파서(장/조/항/호/목 트리) → 구조 기반 청킹(+메타데이터, 상위 문맥)
→ 임베딩(dense+sparse) → Qdrant
→ 질의 → 조문번호 필터/하이브리드 검색 → 리랭커 → 부모 조문 확장 → LLM(인용 필수)
```

### 로더
| 포맷 | 방법 |
|---|---|
| HTML | BeautifulSoup (법제처 페이지는 구조가 태그/클래스에 살아 있음) |
| hwpx | zip 해제 후 `Contents/section*.xml`의 `<hp:p>`, `<hp:t>` 직접 파싱 |
| doc | LibreOffice로 docx 변환(`soffice --headless --convert-to docx`) → python-docx |
| pdf | 텍스트 PDF: PyMuPDF4LLM, 복잡한 표/레이아웃: Docling, 스캔본: OCR |
- 같은 법령이 여러 포맷이면 HTML/Open API(XML)를 정본으로, PDF는 대조용
- 정규화: 머리글/바닥글/쪽번호 제거, 공백·줄바꿈 정리, 전각 숫자·원문자 통일

### 청킹 규칙
- 구조 인식 정규식 예: 장 `^제(\d+)장`, 조 `^제(\d+)조(?:의(\d+))?\(([^)]+)\)`, 항 `^[①-⑳]`, 호 `^\d+\.\s`, 목 `^[가-힣]\.\s`
1. 기본 단위는 조
2. 조가 길면(약 800토큰 초과) 항 단위로 분할
3. 짧은 항/호는 같은 조 안에서 합치고, 호/목은 상위 문장과 항상 함께 둠
4. 조 제목/번호/장 제목을 청크 앞에 붙여 임베딩 (예: `[법명 > 제3장 > 제33조(…) > 제1항]`)
5. 정의 조항(제2조)은 호 단위로 분할
6. 별표/서식/표, 부칙은 별도 청크
- Parent-Child(small-to-big): 항·호 단위로 검색 → `parent_id`로 조 전체를 LLM에 전달
- 메타데이터(payload): `law_name`, `law_type`(법률/시행령/시행규칙/고시/가이드라인), `chapter`, `article_no`, `article_title`, `paragraph`, `item`, `chunk_type`, `parent_id`, `promulgation_date`, `effective_date`, `version`, `source_file`, `refs`, `text`
  - `effective_date`/`version`: 개정 시 현행만 검색하도록 필터
  - `refs`: "제○조에 따른" 참조 추출 → 참조 조문 함께 로드

### 임베딩·저장
- dense + sparse 하이브리드 권장 (bge-m3, OpenAI text-embedding-3-large 등). 현재 text-embedding-3-small로 시작 후 평가해서 교체 검토
- Qdrant named vectors / sparse vectors / RRF 융합 쿼리 지원
- `article_no`, `law_type`, `effective_date` 등에 payload index 생성

### 검색·생성·평가
- 검색: 조문번호 직접 질의는 정규식 → 메타데이터 필터 / 쿼리 재작성(구어체→법률 용어) / 하이브리드 top-20 → 리랭커(bge-reranker-v2-m3, Cohere Rerank) top-5 → 부모 조문·refs 확장
- 생성: 근거(「법」 제○조 제○항) 명시, 근거 없으면 "확인되지 않는다", 법률 자문이 아니라는 고지, 온도 0 부근
- 평가: 질문-정답조문 골든셋 30~50개, Recall@k/MRR, 근거 인용 정확도·환각(Ragas), 청킹·모델 변경 시 같은 골든셋으로 비교
- 적용 순서: ① 조문 트리 파서(`common/`) → ② Qdrant 적재 + 조문번호 필터/벡터 검색 → ③ FastAPI(`app/`) QA 엔드포인트 → ④ 골든셋 평가 후 하이브리드/리랭커/다른 포맷 로더 추가

## 10. 법률 API 발급이 오래 걸리면 hwpx로 가야 하나?
> **현재 상태**: Open API 승인이 나서 XML을 내려받았고 hwpx 경로는 쓰지 않았다. 입력 형식은 팀에서 **PDF + Docling을 메인**으로 하고 XML은 검증용 정답지로 쓰기로 정했다. 근거는 [rag-strategy-notes.md](rag-strategy-notes.md)의 "결정 1" 참고.

- 꼭 그렇진 않지만, hwpx로 먼저 시작하는 것은 좋은 선택
- API 없이도 법제처(law.go.kr) 사이트에서 본문을 직접 저장/다운로드 가능 (제공 형식은 사이트에서 확인). API는 개정 자동 반영이 필요할 때 나중에 추가
- hwpx 장점: zip 안의 XML이라 PDF보다 텍스트·문단 구조가 깨끗, OCR 불필요, 법령 hwpx는 "제1조(목적)", "①", "1."이 글자로 들어 있는 경우가 많아 정규식 파서가 잘 맞음
- 주의: 자동 번호 매기기면 텍스트에 번호가 없을 수 있어 복원 필요 / 표·별표는 XML 구조가 달라 별도 처리 / 머리글·바닥글·각주가 본문에 섞일 수 있음
- 권장 진행
  1. hwpx 한 개에서 텍스트를 추출해 실제 결과 확인
  2. 로더와 구조 파서 분리: 로더는 포맷별로 텍스트 줄 목록만 출력, 구조 파서·청커는 공통
  3. 나중에 API(XML)/HTML 로더를 추가해도 파서·청커 재사용
- 다음 단계: hwpx 샘플 파일 경로 확보 → `common/`에 hwpx 로더 + 조문 파서·청커 구현
