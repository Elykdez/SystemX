"""Pinned configuration for the current decision-model adapter."""

MODEL_NAME = "StartLux-Decision-v1.1-4B-Instant"
MODEL_ID = "startlux-models/" + MODEL_NAME
MODEL_REVISION = "416088f77cfce09633906a3675c7cd2e5dabcf10"
MODEL_OWNER = "StartLuxLabs"
MODEL_DESCRIPTION = "Pinned local " + MODEL_NAME
WEIGHT_FILES = {"model-00001-of-00002.safetensors": 5356493792,
                "model-00002-of-00002.safetensors": 3963334448}
FILES = ["config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json",
         "model.safetensors.index.json", "chat_template.jinja", "preprocessor_config.json",
         "video_preprocessor_config.json", "vocab.json", "merges.txt", "LICENSE", "NOTICE", *WEIGHT_FILES]
