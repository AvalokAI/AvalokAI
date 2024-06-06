from ..configs.config import Config, ModelType
from .embed import Embed
from .gemini import GeminiEmbed
from .hf_embed import HFEmbed
from .st_embed import STEmbed


def get_embedder(config: Config, device: str):
    if config.model_type == ModelType.HUGGING_FACE:
        return HFEmbed(config, device)

    if config.model_type == ModelType.SENTENCE_TRANSFORMER:
        return STEmbed(config, device)

    if config.model_type == ModelType.GEMINI:
        return GeminiEmbed(config)

    return Embed()
