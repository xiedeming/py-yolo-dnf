"""
屏幕捕获模块 - 基于MSS的高性能屏幕捕获
"""
import mss
import numpy as np
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    """捕获统计信息"""
    fps: float = 0.0
    frame_count: int = 0
    last_capture_time: float = 0.0


class MSSCapture:
    """基于MSS的高性能屏幕捕获"""

    def __init__(self, monitor_index: int = 1, target_fps: int = 60):
        """
        初始化屏幕捕获

        Args:
            monitor_index: 显示器索引 (1 = 主显示器)
            target_fps: 目标帧率
        """
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0

        # 统计信息
        self._stats = CaptureStats()
        self._frame_times: list = []
        self._last_capture_time = 0.0

    def capture(self) -> np.ndarray:
        """
        捕获整个显示器

        Returns:
            BGR格式的numpy数组
        """
        capture_start = time.perf_counter()

        monitor = self.sct.monitors[self.monitor_index]
        screenshot = self.sct.grab(monitor)
        image = np.array(screenshot)[:, :, :3]  # BGRA -> BGR

        self._update_stats(capture_start)
        return image

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        捕获指定区域的屏幕

        Args:
            region: (left, top, width, height) 格式

        Returns:
            BGR格式的numpy数组
        """
        capture_start = time.perf_counter()

        left, top, width, height = region

        monitor = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height)
        }

        logger.debug(f"MSS捕获参数: left={left}, top={top}, width={width}, height={height}")

        screenshot = self.sct.grab(monitor)
        image = np.array(screenshot)[:, :, :3]

        # 验证捕获结果
        h, w = image.shape[:2]
        if w != width or h != height:
            logger.warning(f"捕获尺寸不匹配! 请求: {width}x{height}, 实际: {w}x{h}")

        self._update_stats(capture_start)
        return image

    def capture_window(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """
        捕获指定窗口区域

        Args:
            left: 窗口左边界
            top: 窗口上边界
            width: 窗口宽度
            height: 窗口高度

        Returns:
            BGR格式的numpy数组
        """
        return self.capture_region((left, top, width, height))

    def get_monitor_info(self) -> dict:
        """
        获取显示器信息

        Returns:
            显示器信息字典
        """
        return {
            "monitors": self.sct.monitors,
            "current_monitor": self.sct.monitors[self.monitor_index]
        }

    def _update_stats(self, capture_start: float) -> None:
        """更新统计信息"""
        current_time = time.perf_counter()
        self._stats.frame_count += 1
        self._stats.last_capture_time = current_time - capture_start

        # 计算FPS
        self._frame_times.append(current_time)
        # 保留最近60帧的时间
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
        if self.sct:
            self.sct.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
