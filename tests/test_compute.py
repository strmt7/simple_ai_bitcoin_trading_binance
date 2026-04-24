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
    assert info.vendor == "NumPy"
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


def test_resolve_backend_rocm_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("rocm")
    if info.kind == "cpu":
        assert "ROCm" in info.reason
        assert info.requested == "rocm"
    else:
        assert info.kind == "rocm"


def test_resolve_backend_auto_falls_back_to_cpu_without_torch() -> None:
    info = resolve_backend("auto")
    if info.kind == "cpu":
        assert "GPU" in info.reason or "CPU" in info.reason
        assert info.requested == "auto"


def test_resolve_backend_unknown_value_is_cpu_with_explanation() -> None:
    info = resolve_backend("ferrari")
    assert info.kind == "cpu"
    assert "ferrari" in info.reason


def test_describe_backend_includes_components() -> None:
    info = resolve_backend("cpu")
    text = describe_backend(info)
    assert "compute=cpu" in text
    assert "device=cpu" in text
    assert "vendor=NumPy" in text


def test_describe_backend_includes_reason_when_present() -> None:
    info = resolve_backend("cuda")
    text = describe_backend(info)
    if info.kind == "cpu":
        assert info.reason in text
    else:
        assert info.reason == ""
