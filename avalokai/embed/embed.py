import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from ..configs.config import Config, ModelType


class Embed:
    def __init__(self, config: Config, device: str) -> None:

        self.model_type = config.model_type
        if self.model_type == ModelType.HUGGING_FACE:
            self.model = AutoModel.from_pretrained(
                config.model_name, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model.to(device)
            self.max_seq_len = config.max_seq_len
            self.device = device
        elif self.model_type == ModelType.SENTENCE_TRANSFORMER:
            self.model = SentenceTransformer(
                config.model_name, trust_remote_code=True, device=device
            )
            self.model.max_seq_length = config.max_seq_len

        else:
            raise ValueError("model type not correct")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_single_text(self, text: str):
        sentences = [text]
        embeddings = self.model.encode(sentences)
        return embeddings[0].tolist()

    def embed_huggingface(self, content: list[str]):
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

    def embed_sentencetransformer(self, content: list[str]):
        import time

        start = time.time()

        embeddings = self.model.encode(content, convert_to_tensor=True)
        print(embeddings.device)
        print(f"Embed {time.time()-start}")

        start = time.time()
        embeddings = embeddings.cpu()
        print(f"To cpu {time.time()-start}")

        return embeddings

    def embed_multiple_documents(self, content: list[str]):
        if self.model_type == ModelType.HUGGING_FACE:
            return self.embed_huggingface(content)

        if self.model_type == ModelType.SENTENCE_TRANSFORMER:
            return self.embed_sentencetransformer(content)

    def print_config(self):
        print(self.model.get_max_seq_length())
        print(self.model[0].auto_model.config.max_position_embeddings)
