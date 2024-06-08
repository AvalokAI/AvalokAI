from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..configs.config import Config


class Chunk:
    def __init__(self, config: Config) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        assert self.chunk_size > self.chunk_overlap

    def get_chunks(self, docs: list[Document]):
        splits = self.text_splitter.split_documents(docs)

        return splits
