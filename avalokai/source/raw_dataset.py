from langchain_core.documents.base import Document

from ..embed.chunk import Chunk
from .base_dataset import BaseDataset


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


def get_raw_dataset(datas: list[RawData], chunker: Chunk):

    def raw_transform(sample: RawData):
        document = sample.get_langchain_document()
        splits = chunker.get_chunks([document])

        for i, split in enumerate(splits):
            item = {
                "id": f"{sample.id}-{i}",
                "content": split.page_content,
                "metadata": document.metadata,
            }

            yield item

    dataset = BaseDataset(datas, raw_transform)

    return dataset
