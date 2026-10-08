from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import answer_question

app = FastAPI(title="AI 기본법 QA")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="AI 기본법에 대한 질문")


class SourceOut(BaseModel):
    chunk_id: str
    article: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/")
def root():
    return {"message": "RAG API"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = answer_question(request.question)
    except Exception as e:  # Qdrant·LLM·임베딩 API 장애를 500 대신 503으로 알린다
        raise HTTPException(status_code=503, detail=f"답변을 만들지 못했습니다: {type(e).__name__}") from e
    return AskResponse(
        answer=result.answer,
        sources=[SourceOut(chunk_id=s.chunk_id, article=s.article, content=s.content) for s in result.sources],
    )
