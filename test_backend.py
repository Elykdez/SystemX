"""Instant adapter contract without downloading weights or using a GPU."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from backend import MODEL_NAME, ModelBackend


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.delenv('SYSTEMX_QUANTIZATION', raising=False)
    state = SimpleNamespace(kind="instant", bidirectional=True, auto_think=None,
                            reach=1, tokens=10, output_tokens=0, calls=0)

    class Engine:
        def __init__(self, checkpoint, tokenizer, **kwargs):
            self.tokenizer = tokenizer
            self.check = {"late_token_reach": state.reach}
            self.options = kwargs
            self.closed = False
            state.engine = self

        async def start(self):
            pass

        async def close(self):
            self.closed = True

        def describe(self):
            return self.options

    class Decider:
        def __init__(self, folder, engine, **kwargs):
            state.controls = kwargs

        async def decide(self, context, questions):
            state.calls += 1
            return SimpleNamespace(answers={key: {"choice": "a"} for key in questions},
                                   usage={"input_tokens": state.tokens, "output_tokens": state.output_tokens})

    modules = {
        "startlux_decide.backends": {"bidirectional": SimpleNamespace(_kernel_guard=None)},
        "startlux_decide.backends.transformers": {"TransformersBackend": Engine},
        "startlux_decide.modelcfg": {"load_model_folder": lambda _: state,
                                    "load_tokenizer": lambda _: SimpleNamespace(all_special_tokens=["<reserved>"])},
        "startlux_decide.pipeline": {"Decider": Decider, "default_controls": lambda folder, **kwargs: kwargs},
        "startlux_decide.prompt": {"from_systemone": lambda state, question: question,
                                  "messages": lambda question: question,
                                  "plain_prompt": lambda tokenizer, messages: messages,
                                  "count_tokens": lambda tokenizer, prompt: state.tokens},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("backend.download_model", lambda: "test-checkpoint")
    return state


def test_quantization_defaults_and_validation(monkeypatch):
    monkeypatch.delenv('SYSTEMX_QUANTIZATION', raising=False)
    assert ModelBackend('cuda').quantization == 'int8'
    assert ModelBackend('cpu').quantization == 'none'
    monkeypatch.setenv('SYSTEMX_QUANTIZATION', 'int8')
    with pytest.raises(ValueError, match='requires CUDA'):
        ModelBackend('cpu')
    monkeypatch.setenv('SYSTEMX_QUANTIZATION', 'nf4')
    with pytest.raises(ValueError, match='none or int8'):
        ModelBackend('cuda')


def test_instant_controls_and_close(upstream):
    backend = ModelBackend("cpu")
    backend.load()
    assert upstream.engine.options == {"bidirectional": True, "device": "cpu", "dtype": "float32"}
    assert upstream.controls["defaults"] == {"think_mode": "never", "allow_none": False}
    assert upstream.controls["max_images"] == 0
    if os.name == "nt":
        with sys.modules["startlux_decide.backends"].bidirectional._kernel_guard("cuda"):
            pass
    result = backend.evaluate("state", {"q": {}})
    assert result["model"] == MODEL_NAME and result["usage"]["output_tokens"] == 0
    backend.close()
    assert upstream.engine.closed


@pytest.mark.parametrize("field,value", [("kind", "auto_think"), ("bidirectional", False),
                                         ("auto_think", {})])
def test_rejects_non_instant_checkpoint(upstream, field, value):
    setattr(upstream, field, value)
    with pytest.raises(RuntimeError, match="without AutoThink"):
        ModelBackend("cpu").load()
    assert upstream.calls == 0


def test_rejects_failed_attention_check(upstream):
    upstream.reach = 0
    with pytest.raises(RuntimeError, match="self-check"):
        ModelBackend("cpu").load()
    assert upstream.calls == 0


@pytest.mark.parametrize("location", ["state", "question"])
def test_reserved_tokens_rejected_before_inference(upstream, location):
    backend = ModelBackend("cpu")
    backend.load()
    calls = upstream.calls
    with pytest.raises(ValueError, match="reserved"):
        backend.evaluate("<reserved>" if location == "state" else "state",
                         {"q": {"instructions": "<reserved>" if location == "question" else "choose"}})
    assert upstream.calls == calls


def test_token_budgets_reject_before_inference(upstream):
    backend = ModelBackend("cpu", max_tokens=10)
    backend.load()
    calls = upstream.calls
    upstream.tokens = 11
    with pytest.raises(ValueError, match="limit is 10"):
        backend.evaluate("state", {"q": {}})
    upstream.tokens = 10
    with pytest.raises(ValueError, match="Combined"):
        backend.evaluate("state", {str(i): {} for i in range(5)})
    assert upstream.calls == calls


def test_generated_tokens_are_not_accepted(upstream):
    backend = ModelBackend("cpu")
    backend.load()
    upstream.output_tokens = 1
    with pytest.raises(RuntimeError, match="generated tokens"):
        backend.evaluate("state", {"q": {}})
