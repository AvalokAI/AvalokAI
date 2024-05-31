from langchain_core.documents.base import Document


class RawData:
    def __init__(self, id: str, content: str, metadata: dict = None) -> None:
        assert isinstance(id, str)
        assert isinstance(content, str)
        if metadata is not None:
            assert isinstance(metadata, dict)
        self.id = id
        self.content = content
        self.metadata = metadata

    def get_langchain_document(self):
        return Document(self.content, metadata=self.metadata)


class VectorDBData:
    def __init__(self, id: str, embedding: list, metadata: dict) -> None:
        self.id = id
        self.embedding = embedding
        self.metadata = metadata

    def get_data(embeddings: list, splits: list[Document], id: str):
        vectors: list[VectorDBData] = []
        for i, (embedding, split) in enumerate(zip(embeddings, splits)):
            vector = VectorDBData(f"{id}-{i}", embedding, split.metadata)
            vectors.append(vector)
        return vectors
