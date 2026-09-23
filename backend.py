"""Pinned System One Instant loader using upstream decision and attention code."""
import asyncio
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path

from model_config import FILES, MODEL_ID, MODEL_NAME, MODEL_REVISION, WEIGHT_FILES


def configure_cache():
    os.environ["HF_HOME"] = os.environ.get("SYSTEMX_MODEL_CACHE", str(Path(__file__).parent / ".cache" / "huggingface"))


def download_model():
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=FILES)


class ModelBackend:
    def __init__(self, device="cuda", max_tokens=2048):
        self.device = device
        self.max_tokens = max_tokens
        self.quantization = os.environ.get("SYSTEMX_QUANTIZATION", "int8" if device == "cuda" else "none")
        if self.quantization not in ("none", "int8"):
            raise ValueError("SYSTEMX_QUANTIZATION must be none or int8.")
        if device != "cuda" and self.quantization != "none":
            raise ValueError("Int8 inference requires CUDA; use --quantization none for CPU.")
        self.backend = self.decider = None

    def load(self):
        import torch
        from startlux_decide.backends.transformers import TransformersBackend
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer
        from startlux_decide.pipeline import Decider, default_controls

        if os.name == "nt":
            from startlux_decide.backends import bidirectional
            # Windows wheels lack fused GQA kernels. Keep bidirectional semantics,
            # allowing SDPA's math fallback; requests remain bounded to 2048 tokens.
            bidirectional._kernel_guard = lambda device: nullcontext()

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch or explicitly use --device cpu.")
        checkpoint = download_model()
        folder = load_model_folder(checkpoint)
        if folder.kind != "instant" or not folder.bidirectional or folder.auto_think is not None:
            raise RuntimeError("Expected a bidirectional Instant checkpoint without AutoThink.")
        tokenizer = load_tokenizer(checkpoint)
        engine = TransformersBackend
        if self.quantization == "int8":
            from quantized_backend import Int8Backend
            engine = Int8Backend
        dtype = "float16" if self.quantization == "int8" else ("bfloat16" if self.device == "cuda" else "float32")
        self.backend = engine(checkpoint, tokenizer, bidirectional=True, device=self.device, dtype=dtype)
        asyncio.run(self.backend.start())
        if not self.backend.check or self.backend.check.get("late_token_reach", 0) <= 0:
            raise RuntimeError("System One bidirectional attention self-check did not pass.")
        self.decider = Decider(folder, self.backend,
                               defaults=default_controls(folder, think_mode="never", allow_none=False),
                               context_length=self.max_tokens, max_images=0)
        self.evaluate("A customer reports a duplicate charge on an invoice.", {
            "route": {"type": "choice", "instructions": "Choose the support team.",
                      "criteria": {"billing": "Payment and invoice issues.", "technical": "Software faults."}}})
        logging.getLogger("systemx-server").info("System One Instant loaded: %s", self.backend.describe())
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def evaluate(self, state, questions):
        from startlux_decide.prompt import from_systemone, messages, plain_prompt, count_tokens

        content = json.dumps([state, questions], ensure_ascii=False)
        if any(token in content for token in self.backend.tokenizer.all_special_tokens):
            raise ValueError("Input contains reserved model tokens.")
        total = 0
        for question in questions.values():
            prompt = plain_prompt(self.backend.tokenizer, messages(from_systemone(state, question)))
            count = count_tokens(self.backend.tokenizer, prompt)
            if count > self.max_tokens:
                raise ValueError(f"Question needs {count} tokens; limit is {self.max_tokens}. Shorten state/options.")
            total += count
        if total > self.max_tokens * 4:
            raise ValueError("Combined question token budget exceeded.")
        result = asyncio.run(self.decider.decide(state, questions))
        if result.usage["output_tokens"] != 0:
            raise RuntimeError("Instant unexpectedly generated tokens.")
        return {"answers": result.answers, "usage": result.usage, "model": MODEL_NAME}

    def close(self):
        if self.backend is not None:
            asyncio.run(self.backend.close())


if __name__ == "__main__":
    configure_cache()
    print(download_model())
