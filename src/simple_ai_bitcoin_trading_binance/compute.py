"""Opt-in compute backend selection for training and inference.

The default backend is pure NumPy on CPU so no heavy dependency is required. A
user may opt into GPU acceleration by setting ``RuntimeConfig.compute_backend``
to one of:

    * ``"cpu"``   — NumPy only (default, always available).
    * ``"cuda"``  — NVIDIA GPU via PyTorch (requires ``torch`` with a CUDA build).
    * ``"rocm"``  — AMD GPU via PyTorch (requires ``torch`` with a ROCm build).
    * ``"auto"``  — probe CUDA, then ROCm, then MPS (Apple), else fall back to CPU.

The selection never silently installs anything; if the requested backend is not
usable on the current host, :func:`resolve_backend` returns a ``BackendInfo``
whose ``kind`` is ``"cpu"`` and whose ``reason`` explains why, so the caller can
surface that to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendKind = Literal["cpu", "cuda", "rocm", "mps"]


@dataclass(frozen=True)
class BackendInfo:
    """Resolved backend.

    Attributes:
        requested: The value supplied by the operator.
        kind: What was actually selected and is safe to use.
        device: A device identifier usable with torch (e.g. ``"cuda:0"``).
        vendor: Best-effort vendor label, for display.
        reason: Human-readable explanation of fallbacks, blank on success.
    """

    requested: str
    kind: BackendKind
    device: str
    vendor: str
    reason: str


def _probe_torch() -> tuple[object | None, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environmental
        return None, f"torch not importable ({exc.__class__.__name__})"
    return torch, ""


def _try_cuda() -> BackendInfo | None:
    torch, err = _probe_torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return None
        name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - driver corner case
        return None
    return BackendInfo(
        requested="cuda",
        kind="cuda",
        device="cuda:0",
        vendor=str(name),
        reason="",
    )


def _try_rocm() -> BackendInfo | None:
    torch, err = _probe_torch()
    if torch is None:
        return None
    try:
        # ROCm builds of PyTorch still expose their devices under the "cuda" namespace.
        version = getattr(torch.version, "hip", None)
        if not version:
            return None
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "AMD ROCm"
    except Exception:  # pragma: no cover
        return None
    return BackendInfo(
        requested="rocm",
        kind="rocm",
        device="cuda:0",
        vendor=str(name),
        reason="",
    )


def _try_mps() -> BackendInfo | None:
    torch, err = _probe_torch()
    if torch is None:
        return None
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return None
    try:
        if not mps.is_available():
            return None
    except Exception:  # pragma: no cover
        return None
    return BackendInfo(
        requested="mps",
        kind="mps",
        device="mps",
        vendor="Apple MPS",
        reason="",
    )


def _cpu(requested: str, reason: str = "") -> BackendInfo:
    return BackendInfo(
        requested=requested,
        kind="cpu",
        device="cpu",
        vendor="NumPy",
        reason=reason,
    )


def resolve_backend(requested: str | None) -> BackendInfo:
    """Resolve a requested backend name to a usable ``BackendInfo``.

    The function never raises on unsupported input; it falls back to CPU and
    includes a reason in the return value.
    """

    name = (requested or "cpu").strip().lower()
    if name == "cpu":
        return _cpu("cpu")

    if name == "cuda":
        info = _try_cuda()
        if info is not None:
            return info
        return _cpu("cuda", reason="CUDA unavailable (torch missing or no CUDA device)")

    if name == "rocm":
        info = _try_rocm()
        if info is not None:
            return info
        return _cpu("rocm", reason="ROCm unavailable (torch missing or not a ROCm build)")

    if name == "mps":
        info = _try_mps()
        if info is not None:
            return info
        return _cpu("mps", reason="MPS unavailable (torch missing or not Apple Silicon)")

    if name == "auto":
        for probe in (_try_cuda, _try_rocm, _try_mps):
            info = probe()
            if info is not None:
                return info
        return _cpu("auto", reason="No GPU backend available; running on CPU")

    return _cpu(name, reason=f"Unknown backend {requested!r}; defaulting to CPU")


def describe_backend(info: BackendInfo) -> str:
    """Return a compact one-line description of the resolved backend."""
    suffix = f" — {info.reason}" if info.reason else ""
    return f"compute={info.kind} device={info.device} vendor={info.vendor}{suffix}"
