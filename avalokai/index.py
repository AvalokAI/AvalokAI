import pathlib

import torch
from tqdm import tqdm

from .configs import Config
from .embed import Chunk, get_embedder
from .sink.tasks import insert_embeddings
from .sink.vectordb import ChromaVectorDB
from .source import (
    BaseDataset,
    RawData,
    get_data_loader,
    get_raw_dataset,
    get_s3_dataset,
)


class Indexer:
    def __init__(self, dbname: str) -> None:
        repo_path = pathlib.Path(__file__).parent.resolve()
        self.config = Config(repo_path.joinpath("configs", "config.yaml"))
        self.chunker = Chunk(self.config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedder = get_embedder(self.config, self.device)
        self.db = ChromaVectorDB(self.config.embedding_size, dbname, create=True)
        self.dbname = dbname

    def _index(self, dataset: BaseDataset):
        dataloader = get_data_loader(dataset, self.config.batch_size, 4)

        for batch in tqdm(dataloader):
            embeddings = self.embedder.embed_multiple_documents(batch["content"])

            insert_embeddings.delay(
                embeddings,
                batch["metadata"],
                batch["id"],
                batch["content"],
                self.config.embedding_size,
                self.dbname,
            )

    def index_raw_data(self, datas: list[RawData]):
        dataset = get_raw_dataset(datas, self.chunker)
        self._index(dataset)

    def index_s3_data(self, url: str, region: str):
        dataset = get_s3_dataset(url, region, self.chunker)
        self._index(dataset)
