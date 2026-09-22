"""屏幕捕获后端。"""
import logging

from .stats import CaptureStats
from .mss_capture import MSSCapture
from .window_manager import WindowManager
from .bettercam_capture import BetterCamCapture

logger = logging.getLogger(__name__)

__all__ = [
    'MSSCapture',
    'BetterCamCapture',
    'WindowManager',
    'CaptureStats',
    'create_capture',
]

CAPTURE_METHODS = ('mss', 'bettercam')


def create_capture(method: str = 'mss', monitor_index: int = 1, target_fps: int = 60):
    """按配置创建捕获后端。

    bettercam 走 DXGI 桌面复制，本机实测全屏 2560x1440 捕获从 31ms 降到 5ms。
    若初始化失败（显示器不支持 DXGI、独占全屏、已被占用，或未安装 bettercam），
    自动回退到 mss，避免因为捕获后端导致程序无法启动。
    """
    method = (method or 'mss').lower()
    if method not in CAPTURE_METHODS:
        raise ValueError(f"不支持的捕获方式: {method}，可选: {CAPTURE_METHODS}")

    if method == 'bettercam':
        try:
            return BetterCamCapture(monitor_index=monitor_index, target_fps=target_fps)
        except Exception as e:
            logger.warning("BetterCam 初始化失败，回退到 mss: %s", e)

    return MSSCapture(monitor_index=monitor_index, target_fps=target_fps)
