"""청크를 임베딩해서 Qdrant 컬렉션에 저장한다.

저장되는 것
    vector   : prefix가 붙은 청크 텍스트(Chunk.text)의 임베딩. 이름은 "dense".
    payload  : 원문(content), 조문번호(article) 등 메타데이터. 검색 결과를 보여 주거나 필터링할 때 쓴다.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client
from rag.chunker import Chunk, chunk_law
from rag.loader import load_law

COLLECTION_NAME = "ai_basic_law"  # 노트북 실습용 더미 컬렉션(law_articles)과 분리한다.
DENSE = "dense"  # named vector 이름. 나중에 sparse 벡터를 추가해도 기존 구조를 바꾸지 않아도 된다.
FILTER_FIELDS = ("article", "chunk_type")  # 조문번호·청크 종류로 필터링하므로 인덱스를 만든다.


def point_id(chunk_id: str) -> str:
    """Qdrant는 "제2조#제4호" 같은 문자열 ID를 받지 못한다.
    chunk_id에서 항상 같은 UUID를 만들어(uuid5) 다시 적재해도 같은 포인트를 덮어쓰게 한다."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai_basic_law/{chunk_id}"))


def _create_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config={DENSE: VectorParams(size=vector_size, distance=Distance.COSINE)},
    )
    for field in FILTER_FIELDS:
        client.create_payload_index(collection, field, PayloadSchemaType.KEYWORD)


def _to_points(chunks: list[Chunk], vectors: list[list[float]]) -> list[PointStruct]:
    return [
        PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE: vector},
            payload={"chunk_id": chunk.chunk_id, "content": chunk.content, "text": chunk.text, **chunk.metadata},
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def build_index(
    chunks: list[Chunk],
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    recreate: bool = True,
) -> int:
    """청크를 임베딩해 컬렉션에 저장하고, 저장한 개수를 돌려준다.

    recreate=True이면 이 컬렉션만 지우고 새로 만든다. 청킹 규칙을 바꿨을 때
    예전 청크가 남아 검색 결과에 섞이는 것을 막기 위해서다. (임베딩 호출 비용은 청크 수만큼)
    """
    client = client or get_qdrant_client()
    vectors = get_embedding_model().embed_documents([chunk.text for chunk in chunks])

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        _create_collection(client, collection, vector_size=len(vectors[0]))

    client.upsert(collection_name=collection, points=_to_points(chunks, vectors))
    return len(chunks)


if __name__ == "__main__":
    # 입력은 법제처 Open API에서 받은 XML. HWPX·PDF 로더는 같은 청크가 나오는지 비교하는 용도다.
    count = build_index(chunk_law(load_law()))
    print(f"'{COLLECTION_NAME}'에 {count}개 청크 저장 완료")
