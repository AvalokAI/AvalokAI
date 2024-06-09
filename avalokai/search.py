import torch

from .configs import Config
from .embed import Chunk, get_embedder
from .sink.vectordb import ChromaVectorDB


class Searcher:
    def __init__(self, dbname: str, config_file: str) -> None:
        self.config = Config(config_file)
        self.chunker = Chunk(self.config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedder = get_embedder(self.config, self.device)
        self.db = ChromaVectorDB(self.config.embedding_size, dbname, create=True)

    def search(self, query, top_k=10):
        embedding = self.embedder.embed_single_text(query)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            raise ValueError(f"unsupported embedding type {type(embedding)}")

        matches = self.db.retrive_chunks(embedding, top_k)

        for match in matches:
            document_id, chunk_id = tuple(map(int, match["id"].rsplit("-", 1)))
            match["id"] = document_id
            match["chunk_id"] = chunk_id
        return matches
