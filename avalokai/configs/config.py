import yaml


class Config:
    def __init__(self, filepath) -> None:
        with open(filepath) as stream:
            config = yaml.safe_load(stream)

        main_config = config["main_config"]
        self.batch_size = main_config["batch_size"]
        self.model_name = main_config["model_name"]

        model_config = config[self.model_name]
        self.context_length = model_config["context_length"]
        self.chunk_size = model_config["chunk_size"]
        self.chunk_overlap = model_config["chunk_overlap"]

        for key in ["context_length", "chunk_size", "chunk_overlap"]:
            if key in main_config:
                setattr(self, key, main_config[key])
