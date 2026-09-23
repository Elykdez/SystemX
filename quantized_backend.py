"""Int8 loading for the pinned upstream Transformers reference evaluator."""

import torch
import transformers
from startlux_decide.backends.transformers import TransformersBackend
from startlux_decide.backends import bidirectional


class Int8Backend(TransformersBackend):
    def _load(self):
        config = transformers.AutoConfig.from_pretrained(self.path)
        cls = getattr(transformers, config.architectures[0])
        self.device = torch.device(self.device_name)
        # Outlier selection across a batch breaks padding invariance. Quantize
        # all activations instead; keep the upstream attention checks enabled.
        quantization = transformers.BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=0.0)
        self.model = cls.from_pretrained(
            self.path, dtype=getattr(torch, self.dtype_name), device_map={"": self.device},
            quantization_config=quantization).eval()
        inner = self.model.model
        self.body = getattr(inner, "language_model", inner)
        self.head = self.model.get_output_embeddings().weight
        # SystemX's decision API is text-only; image requests use the vision service.
        self.reads_images = False
        self.check = bidirectional.enable(self.body)

    def describe(self):
        return {**super().describe(), "quantization": "int8", "outlier_threshold": 0.0}
