"""Pinned Von OptionMarker loader; keeps inference and weights upstream."""

import json
import logging
import os
from pathlib import Path

MODEL_ID = "wfzyx/von-1.0"
MODEL_REVISION = "aa2fdc9630ecdadef32c56073553b3a69bed38bf"
MODEL_NAME = "von-option-marker"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
         "marker_calibration.json", "option_marker.pt"]


def configure_cache():
    os.environ["HF_HOME"] = os.environ.get("VON_MODEL_CACHE", str(Path(__file__).parent / ".cache" / "huggingface"))


def download_model():
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=FILES)


class VonBackend:
    def __init__(self, device="cuda", max_tokens=2048):
        self.device = device
        self.max_tokens = max_tokens
        self.backend = None

    def load(self):
        import torch
        from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast
        from von.backends.option_marker_backend import OptionMarkerBackend
        from von.models.option_marker import OptionMarkerModel, OptionMarkerScorer

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch or explicitly use --device cpu.")
        checkpoint = download_model()
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        # ModernBERT's optional torch.compile path needs Triton on Windows.
        config.reference_compile = False
        config._attn_implementation = "sdpa"
        # option_marker.pt contains the complete encoder and scorer. Loading the
        # separate 1.58 GB Berta weights first would be redundant.
        model = OptionMarkerModel.__new__(OptionMarkerModel)
        torch.nn.Module.__init__(model)
        model.encoder = AutoModel.from_config(config)
        model.hidden_size = config.hidden_size
        model.scorer = OptionMarkerScorer(hidden_size=config.hidden_size)
        # The pinned tokenizer was saved by Transformers 5 as TokenizersBackend.
        # Its tokenizers JSON is compatible with the stable Transformers 4 runtime.
        tokenizer_config = json.loads((Path(checkpoint) / "tokenizer_config.json").read_text())
        model.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(checkpoint) / "tokenizer.json"),
            **{name: tokenizer_config[name] for name in (
                "cls_token", "mask_token", "pad_token", "sep_token", "unk_token", "model_input_names")},
            model_max_length=self.max_tokens,
        )
        model.mask_token_id = model.tokenizer.mask_token_id
        weights = torch.load(Path(checkpoint) / "option_marker.pt", map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(weights, strict=True)
        del weights
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(device=self.device, dtype=dtype).eval()
        backend = OptionMarkerBackend(checkpoint_dir=checkpoint, device=self.device)
        backend._model = model
        calibration = json.loads((Path(checkpoint) / "marker_calibration.json").read_text())
        backend._default_temp = float(calibration["temperature"])
        self.backend = backend
        self.evaluate("A customer reports a duplicate charge on an invoice.", {
            "route": {"type": "choice", "instructions": "Choose the support team.",
                      "criteria": {"billing": "Payment and invoice issues.", "technical": "Software faults."}}})
        if self.device == "cuda":
            logging.getLogger("von-server").info(
                "CUDA after warmup: allocated %.1f MiB, reserved %.1f MiB, peak allocated %.1f MiB",
                torch.cuda.memory_allocated() / 1048576,
                torch.cuda.memory_reserved() / 1048576,
                torch.cuda.max_memory_allocated() / 1048576,
            )

    def evaluate(self, state, questions):
        from von.backends.option_marker_backend import _format_state

        model = self.backend._model
        state_text = _format_state(state)
        content = json.dumps([state, questions], ensure_ascii=False)
        if any(token in content for token in model.tokenizer.all_special_tokens):
            raise ValueError("Input contains reserved model tokens.")
        input_tokens = 0
        for question in questions.values():
            criteria = question.get("criteria")
            if question["type"] == "choice":
                options = [description or key for key, description in criteria.items()]
            elif question["type"] == "score":
                options = criteria
            else:
                criteria = criteria or {}
                options = [criteria.get("true") or "Yes, condition holds true.",
                           criteria.get("false") or "No, condition is false."]
            packed = model.pack_sequence(state_text, question["instructions"], options)
            count = len(model.tokenizer.encode(packed))
            if count > self.max_tokens:
                raise ValueError(f"Question needs {count} tokens; limit is {self.max_tokens}. Shorten state/options.")
            input_tokens += count
        if input_tokens > self.max_tokens * 4:
            raise ValueError("Combined question token budget exceeded.")
        response = self.backend.evaluate(state, questions, model=MODEL_NAME).model_dump()
        response["usage"]["input_tokens"] = input_tokens
        return response


if __name__ == "__main__":
    configure_cache()
    print(download_model())
