import pathlib
import time

import torch
from tqdm import tqdm

from .configs.config import Config
from .embed import get_embedder
from .embed.chunk import Chunk
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

    # def index_single_document(self, data: RawData):
    #     document = data.get_langchain_document()
    #     splits = self.chunker.get_chunks([document])
    #     contents = [split.page_content for split in splits]
    #     metadatas = [split.metadata for split in splits]
    #     embeddings = self.embedder.embed_multiple_documents(contents)

    #     vectors: list[VectorDBData] = VectorDBData.get_data(
    #         embeddings, metadatas, [data.id] * len(splits)
    #     )
    #     self.db.insert_multiple(vectors)

    def _index(self, dataset: BaseDataset):
        dataloader = get_data_loader(dataset, self.config.batch_size, 4)
        start = time.time()
        for batch in tqdm(dataloader):
            print("---------------------------------------------------")
            print(f"Data load {time.time()-start}")

            embeddings = self.embedder.embed_multiple_documents(batch["content"])

            start = time.time()
            insert_embeddings.delay(
                embeddings,
                batch["metadata"],
                batch["id"],
                self.config.embedding_size,
                self.dbname,
            )
            print(f"Insert to celery {time.time()-start}")

            start = time.time()

    def index_raw_data(self, datas: list[RawData]):
        dataset = get_raw_dataset(datas, self.chunker)
        self._index(dataset)

    def index_s3_data(self, url: str, region: str):
        dataset = get_s3_dataset(url, region, self.chunker)
        for sample in dataset:
            import pdb

            pdb.set_trace()
        self._index(dataset)
