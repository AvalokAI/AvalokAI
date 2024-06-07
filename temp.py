import math

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class RawDataDataset(IterableDataset):
    def __init__(self):
        self.datas = [i for i in range(19)]

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            num_worker = 1
            worker_id = 0
        else:
            num_worker = worker_info.num_workers
            worker_id = worker_info.id
        # per_worker = math.ceil(len(self.datas) / num_worker)
        # datas = self.datas[worker_id * per_worker : (worker_id + 1) * per_worker]

        for i, data in enumerate(self.datas):
            if i % num_worker == worker_id:
                yield data


def collate_fn(samples):
    return np.array(samples)


def get_data_loader():

    dataset = RawDataDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False,
    )
    return dataloader


def main():
    for data in get_data_loader():
        print(data)


if __name__ == "__main__":
    main()
