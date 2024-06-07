from torch.utils.data import IterableDataset, get_worker_info


class BaseDataset(IterableDataset):
    def __init__(self, iterable, transform) -> None:
        super().__init__()
        self.iterable = iterable
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            num_worker = 1
            worker_id = 0
        else:
            num_worker = worker_info.num_workers
            worker_id = worker_info.id

        for i, sample in enumerate(self.iterable):
            if i % num_worker == worker_id:
                for split in self.transform(sample):
                    yield split
