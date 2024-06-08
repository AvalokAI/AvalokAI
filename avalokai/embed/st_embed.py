from sentence_transformers import SentenceTransformer

from ..configs import Config, ModelType
from .embed import Embed


class STEmbed(Embed):
    def __init__(self, config: Config, device: str) -> None:
        super().__init__()
        assert config.model_type == ModelType.SENTENCE_TRANSFORMER

        self.model = SentenceTransformer(
            config.model_name, trust_remote_code=True, device=device
        )
        self.model.max_seq_length = config.max_seq_len

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_single_text(self, text: str):
        sentences = [text]
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings[0]

    def embed_multiple_documents(self, content: list[str]):
        import time

        start = time.time()

        embeddings = self.model.encode(content, convert_to_tensor=True)
        print(embeddings.device)
        print(f"Embed {time.time()-start}")

        start = time.time()
        embeddings = embeddings.cpu()
        print(f"To cpu {time.time()-start}")

        return embeddings

    def print_config(self):
        print(self.model.get_max_seq_length())
        print(self.model[0].auto_model.config.max_position_embeddings)
