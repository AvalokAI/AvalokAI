from s3torchconnector import S3IterableDataset

from .base_dataset import BaseDataset


def s3_transform(sample):
    pass


def get_s3_dataset(url: str, region: str):
    s3_iterable = S3IterableDataset.from_prefix(url, region=region)
    dataset = BaseDataset(s3_iterable, s3_transform)

    return dataset
