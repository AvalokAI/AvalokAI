from enum import Enum

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from ..configs.config import ModelType


class Embed:
    def __init__(self, model_name: str, model_type: ModelType, device: str) -> None:

        if model_type == ModelType.HUGGING_FACE:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        elif model_type == ModelType.SENTENCE_TRANSFORMER:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            raise ValueError("model type not correct")

        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_single_text(self, text: str):
        sentences = [text]
        embeddings = self.model.encode(sentences)
        return embeddings[0].tolist()

    # def embed_multiple_documents(self, sentences: list[str]):
    #     embeddings = self.model.encode(sentences)
    #     return embeddings.tolist()

    def embed_multiple_documents(self, tokenized_text: torch.tensor):
        import time

        start = time.time()
        outputs = self.model(**tokenized_text)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        print(f"Embed {time.time()-start}")

        start = time.time()
        list_embed = embeddings.cpu().numpy()
        print(f"To list {time.time()-start}")

        return list_embed

    def print_config(self):
        print(self.model.get_max_seq_length())
        print(self.model[0].auto_model.config.max_position_embeddings)
