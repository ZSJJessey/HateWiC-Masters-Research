import pytest
import torch
from hatewic_formal import training


def test_explicit_cpu_does_not_satisfy_formal_gpu_gate():
    with pytest.raises(RuntimeError, match="GPU"):
        training.resolve_device("cpu", mode="formal")


def test_explicit_cpu_smoke_is_preserved():
    assert training.resolve_device("cpu", mode="smoke").type == "cpu"


def test_mps_random_state_roundtrip(monkeypatch):
    saved=torch.tensor([4,7],dtype=torch.uint8)
    seen=[]
    monkeypatch.setattr(torch.mps,"get_rng_state",lambda:saved.clone())
    monkeypatch.setattr(torch.mps,"set_rng_state",lambda x:seen.append(x.clone()))
    state=training._rng(torch.device("mps"))
    training._restore_rng(state)
    assert torch.equal(state["mps"],saved)
    assert len(seen)==1 and torch.equal(seen[0],saved)


def test_unavailable_explicit_mps_fails_instead_of_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:False)
    with pytest.raises(RuntimeError, match="MPS"):
        training.resolve_device("mps",mode="benchmark")
