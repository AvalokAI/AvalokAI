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
