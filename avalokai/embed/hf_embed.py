import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..configs import Config, ModelType
from .embed import Embed


class HFEmbed(Embed):
    def __init__(self, config: Config, device: str) -> None:
        super().__init__()
        assert config.model_type == ModelType.HUGGING_FACE

        self.model: nn.Module = AutoModel.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model.to(device)
        self.max_seq_len = config.max_seq_len
        self.device = device

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_single_text(self, text: str):
        raise NotImplementedError("")

    def embed_multiple_documents(self, content: list[str]):
        import time

        start = time.time()
        tokenized_text = self.tokenizer(
            content,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        print(f"Tokenize {time.time()-start}")

        start = time.time()
        tokenized_text = tokenized_text.to(self.device)
        print(f"To gpu {time.time()-start}")

        start = time.time()
        outputs = self.model(**tokenized_text)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        print(f"Embed {time.time()-start}")

        start = time.time()
        embeddings = embeddings.cpu()
        print(f"To cpu {time.time()-start}")

        return embeddings
