import os

import google.generativeai as genai

from ..configs import Config
from .embed import Embed


class GeminiEmbed(Embed):
    def __init__(self, config: Config) -> None:
        super().__init__()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        supported_models = [
            m.name
            for m in genai.list_models()
            if "embedContent" in m.supported_generation_methods
        ]

        assert config.model_name in supported_models
        self.model_name = config.model_name

    def embed_single_text(self, text: str):
        embedding = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document",
            title=None,
        )
        return embedding["embedding"]

    def embed_multiple_documents(self, content: list[str]):

        embeddings = []
        for sample in content:
            embedding = genai.embed_content(
                model=self.model_name,
                content=sample,
                task_type="retrieval_document",
                title=None,
            )
            embeddings.append(embedding["embedding"])

        return embeddings
