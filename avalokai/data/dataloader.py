from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..configs.config import Config
from ..embed.chunk import Chunk
from .data import RawData


class RawDataDataset(IterableDataset):
    def __init__(self, datas: list[RawData], chunker: Chunk):
        self.datas = datas
        self.chunker = chunker

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            num_worker = 1
            worker_id = 0
        else:
            num_worker = worker_info.num_workers
            worker_id = worker_info.id
        per_worker = len(data) // num_worker
        datas = self.datas[worker_id * per_worker : (worker_id + 1) * per_worker]

        for data in datas:
            document = data.get_langchain_document()
            # splits = self.chunker.get_chunks(document.page_content)
            splits = self.chunker.get_chunks([document])

            for i, split in enumerate(splits):
                item = {
                    "id": f"{data.id}-{i}",
                    "content": split.page_content,
                    "metadata": document.metadata,
                }

                yield item


def collate_fn(samples):
    final_data = {"id": [], "content": [], "metadata": []}
    for sample in samples:
        for key in final_data.keys():
            final_data[key].append(sample[key])

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
