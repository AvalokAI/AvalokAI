class Embed:
    def __init__(self) -> None:
        pass

    def embed_single_text(self, text: str):
        raise NotImplementedError("")

    def embed_multiple_documents(self, content: list[str]):
        raise NotImplementedError("")
