"""
屏幕捕获模块 - 基于 BetterCam (DXGI Desktop Duplication) 的高性能捕获

比 MSS 快得多（本机实测全屏 2560x1440：31ms -> 5ms），适合游戏实时推理。
代价是 DXGI 只在画面内容变化时才交付新帧，`grab()` 可能返回 None；本类会缓存
上一帧并照常返回，因此对调用方而言接口与 MSSCapture 完全一致。
"""
import logging
import time
from typing import Optional, Tuple

import numpy as np

from .stats import CaptureStats

logger = logging.getLogger(__name__)


class BetterCamCapture:
    """基于 BetterCam (DXGI) 的高性能屏幕捕获"""

    def __init__(self, monitor_index: int = 1, target_fps: int = 60,
                 output_color: str = "BGR", first_frame_timeout: float = 1.0):
        """
        初始化屏幕捕获

        Args:
            monitor_index: 显示器索引 (1 = 主显示器，与 MSSCapture 保持一致)
            target_fps: 目标帧率
            output_color: 输出颜色格式，BGR 与 OpenCV/ultralytics 一致
            first_frame_timeout: 首次抓取等待画面变化的超时秒数
        """
        import bettercam

        self.monitor_index = monitor_index
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self._first_frame_timeout = first_frame_timeout

        # bettercam 使用 0 基的 output_idx，MSSCapture 使用 1 基的 monitor_index
        self._output_idx = max(monitor_index - 1, 0)
        self._cam = bettercam.create(output_idx=self._output_idx, output_color=output_color)
        if self._cam is None:
            raise RuntimeError(
                f"BetterCam 初始化失败 (output_idx={self._output_idx})。"
                "常见原因：显示器不支持 DXGI 桌面复制、独占全屏，或已被其他程序占用。"
            )

        self._last_frame: Optional[np.ndarray] = None
        self._reused_frames = 0

        self._stats = CaptureStats()
        self._frame_times: list = []

    def capture(self) -> np.ndarray:
        """
        捕获整个显示器

        Returns:
            BGR 格式的 numpy 数组
        """
        return self._grab(None)

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        捕获指定区域

        Args:
            region: (left, top, width, height) 格式，与 MSSCapture 一致

        Returns:
            BGR 格式的 numpy 数组
        """
        left, top, width, height = region
        # bettercam 的区域是 (left, top, right, bottom) 绝对坐标
        dxgi_region = (int(left), int(top), int(left + width), int(top + height))
        return self._grab(dxgi_region)

    def capture_window(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """捕获指定窗口区域"""
        return self.capture_region((left, top, width, height))

    def _grab(self, dxgi_region) -> np.ndarray:
        start = time.perf_counter()

        frame = self._cam.grab(region=dxgi_region)
        if frame is None:
            if self._last_frame is not None:
                # 画面自上次抓取以来没有变化；DXGI 不重复交付，复用上一帧
                self._reused_frames += 1
            else:
                # 启动时画面可能静止，首次抓取会返回 None。短暂轮询直到有帧，
                # 否则直接抛错会让程序在静止界面上无法启动。
                deadline = time.perf_counter() + self._first_frame_timeout
                while frame is None and time.perf_counter() < deadline:
                    frame = self._cam.grab(region=dxgi_region)
                if frame is None:
                    raise RuntimeError(
                        f"BetterCam 在 {self._first_frame_timeout}s 内未能取得任何帧。"
                        "屏幕长时间完全静止，或捕获不可用。"
                    )

        if frame is not None:
            # bettercam 在部分驱动下返回 RGBA，统一裁成 BGR
            self._last_frame = frame[:, :, :3] if frame.shape[2] == 4 else frame

        self._update_stats(start)
        return self._last_frame

    def get_monitor_info(self) -> dict:
        """获取显示器信息"""
        device = getattr(self._cam, "device", None)
        return {
            "monitor_index": self.monitor_index,
            "output_idx": self._output_idx,
            "resolution": tuple(getattr(device, "resolution", ())) if device else None,
        }

    def get_reused_frames(self) -> int:
        """返回因画面未变化而复用上一帧的次数"""
        return self._reused_frames

    def _update_stats(self, capture_start: float) -> None:
        """更新统计信息"""
        current_time = time.perf_counter()
        self._stats.frame_count += 1
        self._stats.last_capture_time = current_time - capture_start

        self._frame_times.append(current_time)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._stats.fps = (len(self._frame_times) - 1) / elapsed

    def get_fps(self) -> float:
        """获取当前捕获帧率"""
        return self._stats.fps

    def get_stats(self) -> CaptureStats:
        """获取捕获统计信息"""
        return self._stats

    def close(self) -> None:
        """释放资源"""
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
