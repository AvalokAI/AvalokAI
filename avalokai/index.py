from tqdm import tqdm

from .data.data import RawData, VectorDBData
from .data.dataloader import get_data_loader
from .embed.chunk import Chunk
from .embed.embed import Embed
from .vectordb.vectordb import ChromaVectorDB


class Indexer:
    def __init__(self, name: str) -> None:
        self.chunker = Chunk()
        self.embedder = Embed()
        self.db = ChromaVectorDB(self.embedder.get_embedding_size(), name)

    def index_single_document(self, data: RawData):
        document = data.get_langchain_document()
        splits = self.chunker.get_chunks([document])
        contents = [split.page_content for split in splits]
        metadatas = [split.metadata for split in splits]
        embeddings = self.embedder.embed_multiple_documents(contents)

        vectors: list[VectorDBData] = VectorDBData.get_data(
            embeddings, metadatas, [data.id] * len(splits)
        )
        self.db.insert_multiple(vectors)

    def index_multiple_documents(self, datas: list[RawData], batch_size=10):
        if batch_size is None:
            batch_size = 10
        dataloader = get_data_loader(datas, self.chunker, batch_size)
        for batch in tqdm(dataloader):
            embeddings = self.embedder.embed_multiple_documents(batch["content"])
            vectors: list[VectorDBData] = VectorDBData.get_data(
                embeddings, batch["metadata"], batch["id"]
            )
            self.db.insert_multiple(vectors)
