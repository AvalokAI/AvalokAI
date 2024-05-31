from tqdm import tqdm

from .data.data import RawData, VectorDBData
from .embed.chunk import Chunk
from .embed.embed import Embed
from .vectordb.vectordb import ChromaVectorDB


class Indexer:
    def __init__(self) -> None:
        self.chunker = Chunk()
        self.embedder = Embed()
        self.db = ChromaVectorDB(self.embedder.get_embedding_size())

    def index_single_document(self, data: RawData):
        document = data.get_langchain_document()
        splits = self.chunker.get_chunks([document])
        embeddings = self.embedder.embed_multiple_documents(splits)
        vectors: list[VectorDBData] = VectorDBData.get_data(embeddings, splits, data.id)
        self.db.insert_multiple(vectors)
