import pathlib

from tqdm import tqdm
from transformers import AutoTokenizer

from .configs.config import Config
from .data.data import RawData, VectorDBData
from .data.dataloader import get_data_loader
from .embed.chunk import Chunk
from .embed.embed import Embed
from .vectordb.vectordb import ChromaVectorDB


class Indexer:
    def __init__(self, dbname: str) -> None:
        repo_path = pathlib.Path(__file__).parent.resolve()
        self.config = Config(repo_path.joinpath("configs", "config.yaml"))
        self.chunker = Chunk(self.config.chunk_size, self.config.chunk_overlap)
        self.embedder = Embed(self.config.model_name, self.config.model_type)
        self.db = ChromaVectorDB(self.config.embedding_size, dbname)

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

    def index_multiple_documents(self, datas: list[RawData]):
        dataloader = get_data_loader(datas, self.chunker, self.config)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        for batch in tqdm(dataloader):
            tokenized_text = tokenizer(
                batch["content"],
                max_length=self.config.max_seq_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # final_data["tokenized_text"] = tokenized_text
            embeddings = self.embedder.embed_multiple_documents(tokenized_text)
            vectors: list[VectorDBData] = VectorDBData.get_data(
                embeddings, batch["metadata"], batch["id"]
            )
            self.db.insert_multiple(vectors)
