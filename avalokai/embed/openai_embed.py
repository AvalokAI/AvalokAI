import os

from openai import OpenAI

from ..configs import Config
from .embed import Embed


class OpenAIEmbed(Embed):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = config.model_name
        self.embedding_size = config.embedding_size

    def embed_single_text(self, text: str):
        response = self.client.embeddings.create(
            input=[text], model=self.model_name, dimensions=self.embedding_size
        )

        embedding = response.data[0].embedding
        return embedding

    def embed_multiple_documents(self, content: list[str]):

        responses = self.client.embeddings.create(
            input=content, model=self.model_name, dimensions=self.embedding_size
        )

        embeddings = [response.embedding for response in responses.data]

        return embeddings
