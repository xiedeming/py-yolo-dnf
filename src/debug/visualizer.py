"""
调试可视化模块
"""
import zlib

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time

from ..detection.detector import Detection


class DebugVisualizer:
    """调试可视化器"""

    # 当前模型的实际类别（顺序见 config/settings.yaml 的 detection.models）
    CLASS_COLORS = {
        'people': (0, 200, 255),      # 橙黄
        'door': (0, 255, 255),        # 黄
        'monster': (0, 0, 255),       # 红
        'brand': (255, 0, 255),       # 品红
        'menu': (255, 128, 0),        # 蓝
        'article': (0, 255, 0),       # 绿
        'purple_card': (200, 0, 180), # 紫
        'hero': (0, 255, 128),        # 青绿
        'elite': (0, 128, 255),       # 橙
        'boss-n': (128, 0, 255),      # 粉紫
        'boss-m': (0, 0, 200),        # 深红
    }

    # 兼容早期的语义类别名（enemy/item/ui 等）
    DEFAULT_COLORS = {
        'enemy': (0, 0, 255),       # 红色
        'enemies': (0, 0, 255),     # 红色
        'item': (0, 255, 0),        # 绿色
        'items': (0, 255, 0),       # 绿色
        'obstacle': (0, 255, 255),  # 黄色
        'obstacles': (0, 255, 255), # 黄色
        'ui': (255, 0, 0),          # 蓝色
        'ui_elements': (255, 0, 0), # 蓝色
        'button': (255, 0, 255),    # 紫色
        'health_bar': (0, 165, 255) # 橙色
    }

    def __init__(
        self,
        show_fps: bool = True,
        show_confidence: bool = True,
        show_center: bool = True,
        show_state: bool = True
    ):
        """
        初始化可视化器

        Args:
            show_fps: 是否显示FPS
            show_confidence: 是否显示置信度
            show_center: 是否显示中心点
            show_state: 是否显示状态
        """
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.show_center = show_center
        self.show_state = show_state

        # 截图保存
        self.save_dir = Path("data/screenshots")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_count = 0

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        获取类别对应的颜色

        未登记的类别按名字哈希取一个稳定的颜色。早期实现只有 enemy/item/ui 等
        语义名，与模型实际类别（people/door/monster/...）一个都不匹配，于是所有
        检测框都落到兜底的白色。

        Args:
            class_name: 类别名称

        Returns:
            BGR颜色元组
        """
        key = (class_name or '').lower()
        if key in self.CLASS_COLORS:
            return self.CLASS_COLORS[key]
        for name, color in self.DEFAULT_COLORS.items():
            if name in key:
                return color

        # 未知类别：由名字决定色相，保证同名稳定、异名可区分
        hue = zlib.crc32(key.encode('utf-8')) % 180  # OpenCV 色相范围 0-179
        bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    @staticmethod
    def label_text_color(background: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        按背景亮度选择黑或白文字。

        标签背景用的是类别颜色，文字若固定为白色，在浅色（尤其是兜底白色）背景上
        会完全看不见。

        Args:
            background: 标签背景的 BGR 颜色

        Returns:
            BGR文字颜色
        """
        b, g, r = background
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        return (0, 0, 0) if luminance > 140 else (255, 255, 255)

    def draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        绘制单个检测框

        Args:
            image: 输入图像
            detection: 检测结果
            color: BGR颜色，None则自动选择

        Returns:
            绘制后的图像
        """
        result = image.copy()
        x1, y1, x2, y2 = detection.bbox

        if color is None:
            color = self.get_color(detection.class_name)

        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = detection.class_name
        if self.show_confidence:
            label += f" {detection.confidence:.2f}"

        # 标签背景
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result,
            (x1, y1 - label_h - 5),
            (x1 + label_w, y1),
            color, -1
        )

        # 标签文字
        cv2.putText(
            result, label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.label_text_color(color), 1
        )

        # 绘制中心点
        if self.show_center:
            cv2.circle(result, detection.center, 3, color, -1)
            cv2.circle(result, detection.center, 5, (255, 255, 255), 1)

        return result

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        category: Optional[str] = None
    ) -> np.ndarray:
        """
        绘制多个检测框

        Args:
            image: 输入图像
            detections: 检测结果列表
            category: 类别名称（用于选择颜色）

        Returns:
            绘制后的图像
        """
        result = image.copy()

        color = None
        if category:
            color = self.get_color(category)

        for det in detections:
            result = self.draw_detection(result, det, color)

        return result

    def draw_all(
        self,
        image: np.ndarray,
        all_detections: Dict[str, List[Detection]],
        fps: float = 0.0,
        state: str = "",
        extra_info: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        绘制所有检测结果和信息

        Args:
            image: 输入图像
            all_detections: {category: [Detection, ...]}
            fps: 帧率
            state: 游戏状态
            extra_info: 额外信息字典

        Returns:
            绘制后的图像
        """
        result = image.copy()

        # 绘制所有检测框
        for category, detections in all_detections.items():
            result = self.draw_detections(result, detections, category)

        # 绘制信息面板
        info_y = 30
        info_spacing = 25

        # FPS
        if self.show_fps and fps > 0:
            fps_color = (0, 255, 0) if fps >= 25 else (0, 255, 255) if fps >= 15 else (0, 0, 255)
            cv2.putText(
                result, f"FPS: {fps:.1f}",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                fps_color, 2
            )
            info_y += info_spacing

        # 状态
        if self.show_state and state:
            cv2.putText(
                result, f"State: {state}",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
            info_y += info_spacing

        # 额外信息
        if extra_info:
            for key, value in extra_info.items():
                cv2.putText(
                    result, f"{key}: {value}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1
                )
                info_y += info_spacing - 5

        return result

    def draw_crosshair(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        size: int = 20,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        绘制准星

        Args:
            image: 输入图像
            center: 中心位置
            size: 十字大小
            color: BGR颜色

        Returns:
            绘制后的图像
        """
        result = image.copy()
        x, y = center

        # 水平线
        cv2.line(result, (x - size, y), (x + size, y), color, 1)
        # 垂直线
        cv2.line(result, (x, y - size), (x, y + size), color, 1)
        # 中心圆
        cv2.circle(result, center, 3, color, -1)

        return result

    def draw_target_lock(
        self,
        image: np.ndarray,
        target: Detection
    ) -> np.ndarray:
        """
        绘制目标锁定框

        Args:
            image: 输入图像
            target: 目标检测结果

        Returns:
            绘制后的图像
        """
        result = image.copy()
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center

        # 绘制角标
        corner_length = 15
        color = (0, 0, 255)  # 红色

        # 左上角
        cv2.line(result, (x1, y1), (x1 + corner_length, y1), color, 2)
        cv2.line(result, (x1, y1), (x1, y1 + corner_length), color, 2)

        # 右上角
        cv2.line(result, (x2, y1), (x2 - corner_length, y1), color, 2)
        cv2.line(result, (x2, y1), (x2, y1 + corner_length), color, 2)

        # 左下角
        cv2.line(result, (x1, y2), (x1 + corner_length, y2), color, 2)
        cv2.line(result, (x1, y2), (x1, y2 - corner_length), color, 2)

        # 右下角
        cv2.line(result, (x2, y2), (x2 - corner_length, y2), color, 2)
        cv2.line(result, (x2, y2), (x2, y2 - corner_length), color, 2)

        # 中心点
        cv2.circle(result, (cx, cy), 5, color, -1)

        # 标签
        label = f"[TARGET] {target.class_name}"
        cv2.putText(
            result, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 2
        )

        return result

    def save_screenshot(
        self,
        image: np.ndarray,
        prefix: str = "debug"
    ) -> str:
        """
        保存截图

        Args:
            image: 要保存的图像
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        self.screenshot_count += 1
        filename = f"{prefix}_{int(time.time())}_{self.screenshot_count}.png"
        filepath = self.save_dir / filename
        cv2.imwrite(str(filepath), image)
        return str(filepath)

    def create_info_panel(
        self,
        image: np.ndarray,
        context: dict
    ) -> np.ndarray:
        """
        创建信息面板

        Args:
            image: 输入图像
            context: 上下文信息字典

        Returns:
            带信息面板的图像
        """
        h, w = image.shape[:2]
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)

        y = 20
        for key, value in context.items():
            text = f"{key}: {value}"
            cv2.putText(
                panel, text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1
            )
            y += 20

        # 将面板添加到图像底部
        result = np.vstack([image, panel])
        return result
