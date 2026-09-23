from types import SimpleNamespace

import torch
import pytest

pytest.importorskip('bitsandbytes', reason='Install requirements-cuda.txt for int8 loader tests')

from quantized_backend import Int8Backend


def test_int8_loader_preserves_head_and_attention_checks(monkeypatch):
    import quantized_backend as module
    calls = {}
    body, head = object(), object()
    model = SimpleNamespace(model=SimpleNamespace(language_model=body),
                            get_output_embeddings=lambda: SimpleNamespace(weight=head))
    model.eval = lambda: model

    def load(path, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(module.transformers.AutoConfig, 'from_pretrained',
                        lambda _: SimpleNamespace(architectures=['Qwen3_5ForConditionalGeneration']))
    monkeypatch.setattr(module.transformers.Qwen3_5ForConditionalGeneration, 'from_pretrained', load)
    checked = []
    monkeypatch.setattr(module.bidirectional, 'enable', lambda b: checked.append(b) or {'late_token_reach': 1})
    backend = Int8Backend('checkpoint', None, bidirectional=True, device='cuda', dtype='float16')
    try:
        backend._load()
        assert calls['quantization_config'].load_in_8bit
        assert calls['quantization_config'].llm_int8_threshold == 0.0
        assert calls['device_map'] == {'': torch.device('cuda')}
        assert calls['dtype'] == torch.float16
        assert backend.head is head and backend.body is body
        assert checked == [body] and backend.check['late_token_reach'] > 0
        assert not backend.reads_images
    finally:
        backend._executor.shutdown()
