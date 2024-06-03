import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from ..configs.config import Config, ModelType
from ..embed.chunk import Chunk
from .data import RawData


class RawDataDataset(Dataset):
    def __init__(self, datas: list[RawData], chunker: Chunk):
        self.datas = datas
        self.chunker = chunker

    def __getitem__(self, index: int):
        data = self.datas[index]
        document = data.get_langchain_document()
        splits = self.chunker.get_chunks(document.page_content)

        items = {"id": [], "content": [], "metadata": []}
        for i, split in enumerate(splits):
            items["id"].append(f"{data.id}-{i}")
            items["content"].append(split)
            items["metadata"].append(document.metadata)

        return items

    def __len__(self):
        return len(self.datas)


def collate_fn(samples):
    final_data = {"id": [], "content": [], "metadata": []}
    for sample in samples:
        for key in final_data.keys():
            final_data[key].extend(sample[key])

    return final_data


def get_data_loader(datas: list[RawData], chunker: Chunk, config: Config):

    dataset = RawDataDataset(datas, chunker)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False,
    )
    return dataloader
