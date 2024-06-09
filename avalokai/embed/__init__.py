from ..configs import Config, ModelType
from .chunk import Chunk
from .embed import Embed
from .gemini_embed import GeminiEmbed
from .hf_embed import HFEmbed
from .openai_embed import OpenAIEmbed
from .st_embed import STEmbed


def get_embedder(config: Config, device: str):
    if config.model_type == ModelType.HUGGING_FACE:
        return HFEmbed(config, device)

    if config.model_type == ModelType.SENTENCE_TRANSFORMER:
        return STEmbed(config, device)

    if config.model_type == ModelType.GEMINI:
        return GeminiEmbed(config)

    if config.model_type == ModelType.OPENAI:
        return OpenAIEmbed(config)

    return Embed()
