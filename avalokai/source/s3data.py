from s3torchconnector import S3ClientConfig, S3IterableDataset, S3Reader

from ..embed.chunk import Chunk
from .base_dataset import BaseDataset


def s3_transform(sample: S3Reader):
    bucket = sample.bucket
    filename = sample.key
    data = sample.read().decode("utf-8")
    yield bucket, filename, data


def get_s3_dataset(url: str, region: str, chunker: Chunk):

    s3_iterable = S3IterableDataset.from_prefix(url, region=region)
    dataset = BaseDataset(s3_iterable, s3_transform)

    return dataset
