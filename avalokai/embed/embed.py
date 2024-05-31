from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer


class Embed:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        # model_name = "nlpaueb/legal-bert-base-uncased"
        self.model = SentenceTransformer(model_name)

    def get_embedding_size(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_single_text(self, text: str):
        sentences = [text]
        embeddings = self.model.encode(sentences)
        return embeddings[0].tolist()

    def embed_multiple_documents(self, sentences: list[str]):
        embeddings = self.model.encode(sentences)
        return embeddings.tolist()
