import json

import pytest

import model_download


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VON_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(model_download, "CHECKPOINT_BYTES", 4)
    return model_download.snapshot_directory()


def create_snapshot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in model_download.FILES:
        (directory / name).write_bytes(b"test" if name == "option_marker.pt" else b"{}")


def events(capsys):
    output = capsys.readouterr()
    return [json.loads(line) for line in output.out.splitlines()], output.err


def test_complete_cache_is_ready_without_downloading(cache, monkeypatch, capsys):
    create_snapshot(cache)
    monkeypatch.setattr(model_download, "download_model", lambda: pytest.fail("Existing cache must not use the network"))
    assert model_download.main([]) == 0
    assert events(capsys)[0] == [{"status": "checking"}, {"status": "ready"}]


def test_missing_check_only_does_not_download(cache, monkeypatch, capsys):
    monkeypatch.setattr(model_download, "download_model", lambda: pytest.fail("Check-only must not use the network"))
    assert model_download.main(["--check-only"]) == 2
    assert events(capsys)[0] == [{"status": "checking"}, {"status": "missing"}]


def test_download_then_verify_and_keep_stdout_json_only(cache, monkeypatch, capsys):
    def download():
        print("Library diagnostic")
        create_snapshot(cache)
        return cache

    monkeypatch.setattr(model_download, "download_model", download)
    assert model_download.main([]) == 0
    output, stderr = events(capsys)
    assert output == [{"status": "checking"}, {"status": "downloading"}, {"status": "ready"}]
    assert "Library diagnostic" in stderr


@pytest.mark.parametrize("damage", ["missing_metadata", "bad_json", "short_checkpoint", "partial_checkpoint"])
def test_incomplete_cache_is_not_ready(cache, damage):
    create_snapshot(cache)
    if damage == "missing_metadata":
        (cache / "config.json").unlink()
    elif damage == "bad_json":
        (cache / "tokenizer.json").write_text("{")
    elif damage == "short_checkpoint":
        (cache / "option_marker.pt").write_bytes(b"x")
    else:
        (cache / "option_marker.pt").rename(cache / "option_marker.pt.incomplete")
    assert not model_download.snapshot_ready(cache)


def test_failed_download_emits_friendly_error_and_preserves_partial(cache, monkeypatch, capsys):
    partial = cache.parent / "download.incomplete"

    def download():
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        raise OSError("Simulated network interruption")

    monkeypatch.setattr(model_download, "download_model", download)
    assert model_download.main([]) == 1
    output, stderr = events(capsys)
    assert [event["status"] for event in output] == ["checking", "downloading", "error"]
    assert "retry" in output[-1]["message"]
    assert "Simulated network interruption" in stderr
    assert partial.read_bytes() == b"partial"


def test_download_cannot_report_ready_with_incomplete_result(cache, monkeypatch, capsys):
    monkeypatch.setattr(model_download, "download_model", lambda: cache)
    assert model_download.main([]) == 1
    assert [event["status"] for event in events(capsys)[0]] == ["checking", "downloading", "error"]


def test_check_only_imports_neither_torch_nor_huggingface(cache, monkeypatch, capsys):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
            raise AssertionError("Local readiness must not import model/download libraries")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert model_download.main(["--check-only"]) == 2
    assert events(capsys)[0][-1] == {"status": "missing"}
