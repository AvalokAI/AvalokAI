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
        splits = self.chunker.get_chunks([document])

        items = {"id": [], "content": [], "metadata": []}
        for i, split in enumerate(splits):
            items["id"].append(f"{data.id}-{i}")
            items["content"].append(split.page_content)
            items["metadata"].append(split.metadata)

        return items

    def __len__(self):
        return len(self.datas)


def get_data_loader(datas: list[RawData], chunker: Chunk, config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def collate_fn(samples):
        final_data = {"id": [], "content": [], "metadata": []}
        for sample in samples:
            for key in final_data.keys():
                final_data[key].extend(sample[key])
        tokenized_text = tokenizer(
            final_data["content"],
            max_length=config.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        final_data["tokenized_text"] = tokenized_text

        return final_data

    dataset = RawDataDataset(datas, chunker)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=False,
    )
    return dataloader
