import os

from langchain_core.documents.base import Document
from s3torchconnector import S3IterableDataset, S3Reader

from ..embed.chunk import Chunk
from .base_dataset import BaseDataset


def get_s3_dataset(url: str, region: str, chunker: Chunk):

    def s3_transform(sample: S3Reader):
        bucket = sample.bucket
        filename = sample.key
        data = sample.read().decode("utf-8")
        document = Document(page_content=data)
        splits = chunker.get_chunks([document])

        id = os.path.join("s3:/", bucket, filename)

        for i, split in enumerate(splits):
            item = {
                "id": f"{id}-{i}",
                "content": split.page_content,
                "metadata": None,
            }

            yield item

    s3_iterable = S3IterableDataset.from_prefix(url, region=region)
    dataset = BaseDataset(s3_iterable, s3_transform)

    return dataset
