from torch.utils.data import DataLoader

from .base_dataset import BaseDataset


def collate_fn(samples):
    final_data = {"id": [], "content": [], "metadata": []}
    for sample in samples:
        for key in final_data.keys():
            final_data[key].append(sample[key])

    return final_data


def get_data_loader(dataset: BaseDataset, batch_size: int, num_workers: int):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloader
