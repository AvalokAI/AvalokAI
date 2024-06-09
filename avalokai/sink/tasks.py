import torch
from celery.utils.log import get_task_logger

from .celery import app
from .vectordb import ChromaVectorDB, VectorDBData

logger = get_task_logger(__name__)


@app.task(name="avalokai.sink.tasks.insert_embeddings")
def insert_embeddings(
    embeddings: torch.Tensor,
    metadata: list[dict],
    ids: list[str],
    contents: list[str],
    embedding_size: int,
    dbname: str,
):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.tolist()
    elif not isinstance(embeddings, list):
        raise ValueError(f"unsupported embedding type {type(embeddings)}")

    vectors: list[VectorDBData] = VectorDBData.get_data(
        embeddings, metadata, ids, contents
    )

    db = ChromaVectorDB(embedding_size, dbname, create=False)
    db.insert_multiple(vectors)
