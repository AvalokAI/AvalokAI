import os

import chromadb
from langchain_core.documents.base import Document
from pinecone import Pinecone, ServerlessSpec

from data.data import VectorDBData


class VectorDB:
    def __init__(self, dimension) -> None:
        self.dimension = dimension

    def close(self):
        raise NotImplementedError("")

    def insert_multiple(self, vectors: list):
        raise NotImplementedError("")

    def check_format(self, vectors: list[dict]):
        raise NotImplementedError("")

    def retrive_chunks(self, query_embedding: list, top_k: int):
        raise NotImplementedError("")

    def get_insertable_format(self, data: list[VectorDBData]):
        raise NotImplementedError("")


class ChromaVectorDB(VectorDB):
    def __init__(self, dimension, name: str) -> None:
        super().__init__(dimension)
        self.client = chromadb.HttpClient(host="localhost", port=8000)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def get_insertable_format(self, data: list[VectorDBData]):
        embeddings = [sample.embedding for sample in data]
        metadatas = [sample.metadata for sample in data]
        ids = [sample.id for sample in data]

        return embeddings, metadatas, ids

    def insert_multiple(self, data: list[VectorDBData]):
        embeddings, metadatas, ids = self.get_insertable_format(data)
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )


class PineConeVectorDB:
    def __init__(self, dimension) -> None:
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.dimension = dimension
        self.index_name = "court-listener-index"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(self.index_name)

    def insert_multiple(self, vectors):
        self.check_format(vectors)
        self.index.upsert(vectors=vectors)

    def check_format(self, vectors: list[dict]):
        assert isinstance(vectors, list)
        for vector in vectors:
            assert len(vector.keys()) == 2
            assert "id" in vector
            assert "values" in vector

    def retrive_chunks(self, query_embedding: list, top_k: int):
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
        )

        return response["matches"]

    def get_insertable_format(
        self, embeddings: list, splits: list[Document], document_id: str
    ):
        vectors = []
        for i, (embedding, split) in enumerate(zip(embeddings, splits)):
            metadata = split.metadata
            vector = {"id": f"{document_id}-{i}", "values": embedding}
            vectors.append(vector)
        return vectors
