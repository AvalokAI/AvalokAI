import time

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
    embedding_size: int,
    dbname: str,
):
    start = time.time()
    vectors: list[VectorDBData] = VectorDBData.get_data(
        embeddings.tolist(), metadata, ids
    )

    logger.info(f"convert to vector data {time.time()-start}")

    start = time.time()
    db = ChromaVectorDB(embedding_size, dbname, create=False)
    db.insert_multiple(vectors)
    logger.info(f"insert in db {time.time()-start}")
