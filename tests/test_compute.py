"""Tests for the opt-in compute backend resolver."""

from __future__ import annotations

from simple_ai_bitcoin_trading_binance.compute import (
    BackendInfo,
    describe_backend,
    resolve_backend,
)


def test_resolve_backend_defaults_to_cpu_when_unspecified() -> None:
    info = resolve_backend(None)
    assert isinstance(info, BackendInfo)
    assert info.kind == "cpu"
    assert info.device == "cpu"
    assert info.vendor == "Python stdlib"
    assert info.reason == ""


def test_resolve_backend_cpu_is_always_available() -> None:
    info = resolve_backend("cpu")
    assert info.kind == "cpu"
    assert info.requested == "cpu"


def test_resolve_backend_cuda_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("cuda")
    # In the unit-test environment torch is not present so we expect a graceful
    # fallback rather than an exception.
    if info.kind == "cpu":
        assert "CUDA" in info.reason
        assert info.requested == "cuda"
    else:
        assert info.kind == "cuda"
        assert info.device.startswith("cuda")


def test_resolve_backend_cuda_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("cuda")

    assert info.kind == "cpu"
    assert info.reason == "CUDA unavailable (torch missing or no CUDA device)"


def test_resolve_backend_rocm_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("rocm")
    if info.kind == "cpu":
        assert "ROCm" in info.reason
        assert info.requested == "rocm"
    else:
        assert info.kind == "rocm"


def test_resolve_backend_rocm_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("rocm")

    assert info.kind == "cpu"
    assert info.reason == "ROCm unavailable (torch missing or not a ROCm build)"


def test_resolve_backend_directml_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("directml")
    if info.kind == "cpu":
        assert "DirectML" in info.reason
        assert info.requested == "directml"
    else:
        assert info.kind == "directml"


def test_resolve_backend_auto_falls_back_to_cpu_without_torch() -> None:
    info = resolve_backend("auto")
    if info.kind == "cpu":
        assert "GPU" in info.reason or "CPU" in info.reason
        assert info.requested == "auto"


def test_resolve_backend_auto_returns_cpu_when_every_probe_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (None, "directml missing in test"),
    )

    info = resolve_backend("auto")

    assert info.kind == "cpu"
    assert info.reason == "No GPU backend available; running on CPU"


def test_resolve_backend_unknown_value_is_cpu_with_explanation() -> None:
    info = resolve_backend("ferrari")
    assert info.kind == "cpu"
    assert "ferrari" in info.reason


def test_describe_backend_includes_components() -> None:
    info = resolve_backend("cpu")
    text = describe_backend(info)
    assert "compute=cpu" in text
    assert "device=cpu" in text
    assert "vendor=Python stdlib" in text


def test_describe_backend_includes_reason_when_present() -> None:
    info = resolve_backend("cuda")
    text = describe_backend(info)
    if info.kind == "cpu":
        assert info.reason in text
    else:
        assert info.reason == ""


class _FakeCuda:
    def __init__(self, *, available: bool = True, count: int = 1, name: str = "Test GPU") -> None:
        self._available = available
        self._count = count
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return self._name


class _FakeMpsBackend:
    def __init__(self, *, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps: _FakeMpsBackend | None = None) -> None:
        if mps is not None:
            self.mps = mps


def _make_fake_torch(
    *,
    cuda: _FakeCuda | None = None,
    hip: str | None = None,
    mps: _FakeMpsBackend | None = None,
    include_version: bool = True,
):
    class _FakeTorch:
        pass

    fake = _FakeTorch()
    if cuda is not None:
        fake.cuda = cuda
    if include_version:
        class _Version:
            pass

        version = _Version()
        if hip is not None:
            version.hip = hip
        fake.version = version
    fake.backends = _FakeBackends(mps=mps)
    return fake


def test_resolve_backend_cuda_success_uses_torch(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="NVIDIA Test"))
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cuda"
    assert info.device == "cuda:0"
    assert info.vendor == "NVIDIA Test"
    assert info.reason == ""


def test_resolve_backend_cuda_returns_cpu_when_not_available(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(available=False))
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_returns_cpu_when_zero_devices(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(count=0))
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_returns_cpu_for_rocm_torch_build(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="AMD Via CUDA Namespace"), hip="6.0")
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_swallows_torch_exceptions(monkeypatch) -> None:
    class _ExplodingCuda:
        def is_available(self):
            raise RuntimeError("driver missing")

    fake = _make_fake_torch(cuda=_ExplodingCuda())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_success_via_torch_hip(monkeypatch) -> None:
    fake = _make_fake_torch(
        cuda=_FakeCuda(name="AMD Test"),
        hip="6.0",
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "rocm"
    assert info.device == "cuda:0"


def test_resolve_backend_rocm_returns_cpu_when_no_hip(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_returns_cpu_when_cuda_namespace_missing(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(available=False), hip="6.0")
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_returns_cpu_when_no_devices(monkeypatch) -> None:
    cuda = _FakeCuda()
    cuda._count = 0  # is_available=True, but no device count
    fake = _make_fake_torch(cuda=cuda, hip="6.0")
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


class _FakeDirectML:
    def __init__(self, *, available: bool = True, explode: bool = False) -> None:
        self._available = available
        self._explode = explode

    def is_available(self) -> bool:
        if self._explode:
            raise RuntimeError("driver")
        return self._available

    def device(self) -> str:
        return "privateuseone:0"


def test_resolve_backend_directml_success(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (_FakeDirectML(), ""),
    )
    info = resolve_backend("directml")
    assert info.kind == "directml"
    assert info.device == "privateuseone:0"
    assert info.vendor == "DirectML"


def test_resolve_backend_directml_returns_cpu_when_unavailable(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (_FakeDirectML(available=False), ""),
    )
    assert resolve_backend("directml").kind == "cpu"


def test_resolve_backend_directml_swallows_exceptions(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (_FakeDirectML(explode=True), ""),
    )
    assert resolve_backend("directml").kind == "cpu"


def test_resolve_backend_mps_success(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "mps"
    assert info.device == "mps"


def test_resolve_backend_mps_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("mps")

    assert info.kind == "cpu"
    assert info.reason == "MPS unavailable (torch missing or not Apple Silicon)"


def test_resolve_backend_mps_returns_cpu_when_unavailable(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend(available=False))
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_mps_returns_cpu_when_attribute_missing(monkeypatch) -> None:
    fake = _make_fake_torch()
    fake.backends = _FakeBackends(mps=None)
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_mps_swallows_exceptions(monkeypatch) -> None:
    class _ExplodingMps:
        def is_available(self):
            raise RuntimeError("not on this OS")

    fake = _make_fake_torch(mps=_ExplodingMps())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_auto_picks_first_available(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="NVIDIA Auto"))
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("auto")
    assert info.kind == "cuda"
    assert info.requested == "auto"


def test_resolve_backend_auto_picks_mps_when_cuda_and_rocm_missing(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (None, "torch-directml unavailable in this test"),
    )
    info = resolve_backend("auto")
    assert info.kind == "mps"
    assert info.requested == "auto"


def test_resolve_backend_auto_picks_directml_before_mps(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch_directml",
        lambda: (_FakeDirectML(), ""),
    )
    info = resolve_backend("auto")
    assert info.kind == "directml"
    assert info.requested == "auto"


def test_probe_torch_returns_tuple_when_torch_missing(monkeypatch) -> None:
    """Direct exercise of the _probe_torch fallback path."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Reload the function reference
    from simple_ai_bitcoin_trading_binance import compute as compute_mod

    importlib.reload(compute_mod)
    torch_obj, reason = compute_mod._probe_torch()
    assert torch_obj is None
    assert "torch" in reason.lower()
    importlib.reload(compute_mod)  # restore for other tests


def test_probe_torch_returns_torch_when_importable(monkeypatch) -> None:
    """Cover the success branch of _probe_torch via a fake sys.modules entry."""
    import sys
    import types

    fake_torch = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from simple_ai_bitcoin_trading_binance import compute as compute_mod

    obj, reason = compute_mod._probe_torch()
    assert obj is fake_torch
    assert reason == ""


def test_probe_torch_directml_success_and_missing(monkeypatch) -> None:
    import importlib
    import types

    fake_directml = types.SimpleNamespace()
    original = importlib.import_module

    def fake_import(name: str):
        if name == "torch_directml":
            return fake_directml
        return original(name)

    from simple_ai_bitcoin_trading_binance import compute as compute_mod

    monkeypatch.setattr(compute_mod.importlib, "import_module", fake_import)
    obj, reason = compute_mod._probe_torch_directml()
    assert obj is fake_directml
    assert reason == ""

    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(compute_mod.importlib, "import_module", missing_import)
    obj, reason = compute_mod._probe_torch_directml()
    assert obj is None
    assert "torch-directml" in reason


def test_resolve_backend_rocm_hip_attribute_falsy_returns_cpu(monkeypatch) -> None:
    """torch.version exists but torch.version.hip is missing/None — must fall back without raising."""
    fake = _make_fake_torch(cuda=_FakeCuda(), include_version=True, hip=None)
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"

