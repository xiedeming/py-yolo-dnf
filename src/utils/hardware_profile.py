"""Detect the local GPU and select the appropriate runtime profile.

The module deliberately keeps the optional PyTorch import inside the detection
function.  Low-spec machines can therefore use the ONNX/CPU profile without
having PyTorch installed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


GIB = 1024 ** 3
HIGH_SPEC_MEMORY_BYTES = 4 * GIB


@dataclass(frozen=True)
class GPUInfo:
    """The stable GPU properties needed for profile selection."""

    index: int
    name: str
    total_memory_bytes: int

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / GIB


@dataclass(frozen=True)
class HardwareInfo:
    """Result of the best-effort CUDA hardware probe."""

    cuda_available: bool
    gpus: Tuple[GPUInfo, ...] = ()
    reason: str = ""

    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        return max(self.gpus, key=lambda gpu: (gpu.total_memory_bytes, -gpu.index), default=None)

    @property
    def max_memory_bytes(self) -> int:
        gpu = self.best_gpu
        return gpu.total_memory_bytes if gpu else 0

    @property
    def is_high_spec(self) -> bool:
        return self.cuda_available and self.max_memory_bytes >= HIGH_SPEC_MEMORY_BYTES


@dataclass(frozen=True)
class RuntimeProfile:
    """Selected configuration and device for one application start."""

    name: str
    config_path: Path
    device: str
    hardware: HardwareInfo
    reason: str

    @property
    def is_high_spec(self) -> bool:
        return self.name == "high"

    def summary(self) -> str:
        """Return a concise human-readable explanation for startup output."""
        gpu = self.hardware.best_gpu
        if gpu is None:
            gpu_text = "未检测到可用 CUDA 显卡"
        else:
            gpu_text = f"{gpu.name} ({gpu.total_memory_gib:.2f} GiB)"
        return f"{self.name} 配置，设备 {self.device}，显卡 {gpu_text}；{self.reason}"


def _load_torch() -> Any:
    """Import PyTorch only when a hardware probe is actually requested."""
    try:
        import torch
    except Exception:
        return None
    return torch


def detect_hardware(torch_module: Any = None) -> HardwareInfo:
    """Probe CUDA availability and total memory without allocating GPU memory.

    ``torch_module`` is injectable so the boundary and failure paths can be
    tested without requiring a CUDA machine.
    """
    torch_module = _load_torch() if torch_module is None else torch_module
    if torch_module is None:
        return HardwareInfo(False, reason="PyTorch 不可用")

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return HardwareInfo(False, reason="PyTorch 未提供 CUDA 接口")

    try:
        if not bool(cuda.is_available()):
            return HardwareInfo(False, reason="CUDA 不可用")
        device_count = int(cuda.device_count())
    except Exception as error:
        return HardwareInfo(False, reason=f"CUDA 探测失败: {error}")

    gpus = []
    failures = []
    for index in range(max(device_count, 0)):
        try:
            properties = cuda.get_device_properties(index)
            total_memory = int(getattr(properties, "total_memory"))
            name = getattr(properties, "name", None)
            if not name:
                name = cuda.get_device_name(index)
            gpus.append(GPUInfo(index=index, name=str(name or f"CUDA:{index}"), total_memory_bytes=total_memory))
        except Exception as error:
            failures.append(f"GPU {index}: {error}")

    if not gpus:
        suffix = f" ({'; '.join(failures)})" if failures else ""
        return HardwareInfo(True, reason=f"未能读取 CUDA 显卡属性{suffix}")

    hardware = HardwareInfo(True, tuple(gpus))
    best_gpu = hardware.best_gpu
    if hardware.is_high_spec:
        reason = f"最大显存达到 {best_gpu.total_memory_gib:.2f} GiB（阈值 4 GiB）"
    else:
        reason = f"最大显存仅 {best_gpu.total_memory_gib:.2f} GiB（低于 4 GiB 阈值）"
    return HardwareInfo(True, tuple(gpus), reason=reason)


def select_runtime_profile(
    project_root: Optional[str] = None,
    requested_device: Optional[str] = None,
    hardware: Optional[HardwareInfo] = None,
) -> RuntimeProfile:
    """Select high/low configuration, honoring an explicit device override.

    With no override, only a CUDA GPU with at least 4 GiB total memory selects
    the high profile.  ``-d cpu`` and ``-d cuda`` are explicit user choices and
    therefore select their corresponding profile regardless of the probe result.
    """
    if requested_device not in (None, "cpu", "cuda"):
        raise ValueError(f"Unsupported requested device: {requested_device}")

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if hardware is None:
        hardware = (
            detect_hardware()
            if requested_device is None
            else HardwareInfo(False, reason="手动设备覆盖，未进行自动探测")
        )

    if requested_device == "cpu":
        name = "low"
        device = "cpu"
        reason = "使用了手动 CPU 覆盖"
    elif requested_device == "cuda":
        name = "high"
        device = "cuda"
        reason = "使用了手动 CUDA 覆盖"
    elif hardware.is_high_spec:
        name = "high"
        gpu = hardware.best_gpu
        device = f"cuda:{gpu.index}" if gpu and gpu.index else "cuda"
        reason = hardware.reason
    else:
        name = "low"
        device = "cpu"
        reason = hardware.reason or "未满足高配条件"

    config_name = "settings.yaml" if name == "high" else "settings.cpu.yaml"
    return RuntimeProfile(
        name=name,
        config_path=root / "config" / config_name,
        device=device,
        hardware=hardware,
        reason=reason,
    )
