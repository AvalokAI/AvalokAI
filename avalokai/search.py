from .embed.chunk import Chunk
from .embed.embed import Embed
from .vectordb.vectordb import ChromaVectorDB


class Searcher:
    def __init__(self, name: str) -> None:
        self.chunker = Chunk()
        self.embedder = Embed()
        self.db = ChromaVectorDB(self.embedder.get_embedding_size(), name)

    def search(self, query, top_k=10):
        embedding = self.embedder.embed_single_text(query)
        matches = self.db.retrive_chunks(embedding, top_k)
        document_ids = []
        for i, match in enumerate(matches):
            document_id, chunk_id = tuple(map(int, match["id"].split("-")))
            match["id"] = document_id
            match["chunk_id"] = chunk_id
        return matches
