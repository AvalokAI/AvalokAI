from torch.utils.data import DataLoader, Dataset

from ..embed.chunk import Chunk
from .data import RawData


class RawDataDataset(Dataset):
    def __init__(self, datas: list[RawData], chunker: Chunk) -> None:
        self.datas = datas
        self.chunker = chunker

    def __getitem__(self, index: int):
        data = self.datas[index]
        document = data.get_langchain_document()
        splits = self.chunker.get_chunks([document])

        items = [
            {
                "id": f"{data.id}-{i}",
                "content": split.page_content,
                "metadata": split.metadata,
            }
            for i, split in enumerate(splits)
        ]
        return items

    def __len__(self):
        return len(self.datas)


def collate_fn(samples):
    final_data = {"id": [], "content": [], "metadata": []}
    for sample in samples:
        for item in sample:
            for key in final_data.keys():
                final_data[key].append(item[key])
    return final_data


def get_data_loader(datas: list[RawData], chunker: Chunk, batch_size: int):
    dataset = RawDataDataset(datas, chunker)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4
    )
    return dataloader
