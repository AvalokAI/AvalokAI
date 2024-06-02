from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def get_chunks(self, docs: list[Document]):
        splits = self.text_splitter.split_documents(docs)

        return splits
