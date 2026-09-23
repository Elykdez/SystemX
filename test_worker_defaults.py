import sys
from types import SimpleNamespace
import pytest
from workers import default_workers

@pytest.mark.parametrize('gib,expected', [(4,1),(8,1),(12,1),(16,1),(24,3)])
def test_worker_defaults_follow_gpu_memory(monkeypatch,gib,expected):
    monkeypatch.delenv('SYSTEMX_WORKERS',raising=False)
    monkeypatch.delenv('SYSTEMX_QUANTIZATION',raising=False)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(total_memory=gib*2**30))))
    assert default_workers('cuda')==expected

def test_cpu_defaults_to_one_without_importing_torch(monkeypatch):
    monkeypatch.delenv('SYSTEMX_WORKERS',raising=False)
    assert default_workers('cpu')==1

def test_worker_override(monkeypatch):
    monkeypatch.setenv('SYSTEMX_WORKERS','2')
    assert default_workers('cuda')==2


def test_bf16_keeps_two_workers_on_4090(monkeypatch):
    monkeypatch.delenv('SYSTEMX_WORKERS', raising=False)
    monkeypatch.setenv('SYSTEMX_QUANTIZATION', 'none')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _: SimpleNamespace(total_memory=24*2**30))))
    assert default_workers('cuda') == 2


@pytest.mark.parametrize('count', ['0', '4', '8'])
def test_worker_override_rejects_more_than_three(monkeypatch, count):
    monkeypatch.setenv('SYSTEMX_WORKERS', count)
    with pytest.raises(ValueError, match='1-3'):
        default_workers('cuda')
