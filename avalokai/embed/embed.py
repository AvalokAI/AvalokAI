from enum import Enum

from sentence_transformers import SentenceTransformer


class Embed:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        # model_name = "nlpaueb/legal-bert-base-uncased"
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def get_embedding_size(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_single_text(self, text: str):
        sentences = [text]
        embeddings = self.model.encode(sentences)
        return embeddings[0].tolist()

    def embed_multiple_documents(self, sentences: list[str]):
        embeddings = self.model.encode(sentences)
        return embeddings.tolist()

    def print_config(self):
        print(self.model.get_max_seq_length())
        print(self.model[0].auto_model.config.max_position_embeddings)
