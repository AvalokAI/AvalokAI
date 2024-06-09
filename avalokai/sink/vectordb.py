import chromadb


class VectorDBData:
    def __init__(self, id: str, embedding: list, metadata: dict, content: str) -> None:
        self.id = id
        self.embedding = embedding
        self.metadata = metadata
        self.content = content

    def get_data(
        embeddings: list, metadatas: list[dict], ids: list[str], contents: list[str]
    ):
        vectors: list[VectorDBData] = []
        for embedding, metadata, id, content in zip(
            embeddings, metadatas, ids, contents
        ):
            vector = VectorDBData(id, embedding, metadata, content)
            vectors.append(vector)
        return vectors


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
    def __init__(self, dimension: int, name: str, create: bool) -> None:
        super().__init__(dimension)
        self.client = chromadb.HttpClient(host="localhost", port=8000)
        if create:
            self.collection = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self.client.get_collection(name=name)

    def get_insertable_format(self, data: list[VectorDBData]):
        embeddings = [sample.embedding for sample in data]
        metadatas = [sample.metadata for sample in data]
        ids = [sample.id for sample in data]
        contents = [sample.content for sample in data]

        return embeddings, metadatas, ids, contents

    def insert_multiple(self, data: list[VectorDBData]):
        embeddings, metadatas, ids, contents = self.get_insertable_format(data)
        self.collection.add(
            embeddings=embeddings, metadatas=metadatas, ids=ids, documents=contents
        )

    def retrive_chunks(self, query_embedding: list, top_k: int):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "documents"],
        )

        matches = [
            {"id": id, "score": distance, "content": content}
            for id, distance, content in zip(
                results["ids"][0], results["distances"][0], results["documents"][0]
            )
        ]

        return matches
