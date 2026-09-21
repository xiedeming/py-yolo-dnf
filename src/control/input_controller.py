"""
输入控制模块 - 键盘和鼠标控制
"""
import time
import random
import math
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button


logger = logging.getLogger(__name__)


@dataclass
class InputConfig:
    """输入控制配置"""
    humanize: bool = True
    random_delay_range: Tuple[float, float] = (0.02, 0.08)
    mouse_speed: float = 0.2
    smooth_mouse: bool = True


class InputController:
    """统一输入控制器"""

    def __init__(self, config: Optional[InputConfig] = None):
        """
        初始化输入控制器

        Args:
            config: 输入配置，None则使用默认配置
        """
        self.config = config or InputConfig()
        self.keyboard = KeyboardController()
        self.mouse = MouseController()

        # 当前移动方向状态: None, 'left', 'right'
        self._current_moving_direction: Optional[str] = None
        self._pressed_keys = set()
        self._pressed_mouse_buttons = set()

        # 特殊键映射
        self._special_keys = {
            'enter': Key.enter,
            'space': Key.space,
            'tab': Key.tab,
            'escape': Key.esc,
            'esc': Key.esc,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'home': Key.home,
            'end': Key.end,
            'page_up': Key.page_up,
            'page_down': Key.page_down,
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
        }

    def _get_random_delay(self) -> float:
        """生成随机延迟"""
        if self.config.humanize:
            min_delay, max_delay = self.config.random_delay_range
            return random.uniform(min_delay, max_delay)
        return 0.01

    def _delay(self) -> None:
        """执行随机延迟"""
        time.sleep(self._get_random_delay())

    # ========== 键盘控制 ==========

    def _get_key(self, key: str):
        """
        获取键对象

        Args:
            key: 键名称或字符

        Returns:
            pynput键对象
        """
        key_lower = key.lower()
        if key_lower in self._special_keys:
            return self._special_keys[key_lower]
        elif len(key) == 1:
            return key
        else:
            raise ValueError(f"Unknown key: {key}")

    def key_press(self, key: str, duration: float = 0.1) -> None:
        """
        按下并释放按键

        Args:
            key: 键名称
            duration: 按住时间(秒)
        """
        key_obj = self._get_key(key)
        self.keyboard.press(key_obj)
        self._pressed_keys.add(key_obj)
        try:
            time.sleep(duration)
        finally:
            try:
                self.keyboard.release(key_obj)
            except Exception as error:
                logger.warning("Failed to release key %r: %s", key_obj, error)
            else:
                self._pressed_keys.discard(key_obj)
        self._delay()

    def key_down(self, key: str) -> None:
        """
        按下按键（持续）

        Args:
            key: 键名称
        """
        key_obj = self._get_key(key)
        self.keyboard.press(key_obj)
        self._pressed_keys.add(key_obj)

    def key_up(self, key: str) -> None:
        """
        释放按键

        Args:
            key: 键名称
        """
        key_obj = self._get_key(key)
        self.keyboard.release(key_obj)
        self._pressed_keys.discard(key_obj)

    def key_combo(self, *keys: str, duration: float = 0.1) -> None:
        """
        组合键

        Args:
            keys: 键名称序列，如 ('ctrl', 'c')
            duration: 按住时间
        """
        pressed_keys = []
        try:
            # 按下所有键
            for key in keys:
                key_obj = self._get_key(key)
                self.keyboard.press(key_obj)
                self._pressed_keys.add(key_obj)
                pressed_keys.append(key_obj)
                time.sleep(0.02)

            time.sleep(duration)
        finally:
            # 释放所有已按下的键（逆序）
            for key_obj in reversed(pressed_keys):
                try:
                    self.keyboard.release(key_obj)
                except Exception as error:
                    logger.warning("Failed to release key %r: %s", key_obj, error)
                else:
                    self._pressed_keys.discard(key_obj)
                time.sleep(0.02)

        self._delay()

    def key_sequence(self, keys: List[str], interval: float = 0.1) -> None:
        """
        按键序列

        Args:
            keys: 键名称列表
            interval: 按键间隔
        """
        for key in keys:
            self.key_press(key, duration=0.05)
            time.sleep(interval)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """
        输入文本

        Args:
            text: 要输入的文本
            interval: 字符间隔
        """
        for char in text:
            self.keyboard.press(char)
            self._pressed_keys.add(char)
            self.keyboard.release(char)
            self._pressed_keys.discard(char)
            if self.config.humanize:
                time.sleep(interval + random.uniform(0, 0.02))
            else:
                time.sleep(interval)

    # ========== 鼠标控制 ==========

    def get_mouse_position(self) -> Tuple[int, int]:
        """
        获取当前鼠标位置

        Returns:
            (x, y)
        """
        return self.mouse.position

    def mouse_move(self, x: int, y: int, smooth: Optional[bool] = None) -> None:
        """
        移动鼠标到指定位置

        Args:
            x: 目标x坐标
            y: 目标y坐标
            smooth: 是否平滑移动，None则使用配置值
        """
        use_smooth = smooth if smooth is not None else self.config.smooth_mouse

        if use_smooth and self.config.humanize:
            self._smooth_move(x, y)
        else:
            self.mouse.position = (x, y)

    def _smooth_move(self, target_x: int, target_y: int) -> None:
        """
        平滑移动鼠标（贝塞尔曲线）

        Args:
            target_x: 目标x坐标
            target_y: 目标y坐标
        """
        start_x, start_y = self.mouse.position

        # 如果距离很近，直接移动
        distance = math.sqrt((target_x - start_x) ** 2 + (target_y - start_y) ** 2)
        if distance < 10:
            self.mouse.position = (target_x, target_y)
            return

        # 生成随机控制点（贝塞尔曲线）
        ctrl_x = (start_x + target_x) / 2 + random.randint(-50, 50)
        ctrl_y = (start_y + target_y) / 2 + random.randint(-50, 50)

        # 根据距离计算移动时间
        duration = min(self.config.mouse_speed * (distance / 500), 0.3)
        steps = int(duration * 60)  # 60fps

        for i in range(steps + 1):
            t = i / steps
            # 二次贝塞尔曲线
            x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * ctrl_x + t ** 2 * target_x
            y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * ctrl_y + t ** 2 * target_y
            self.mouse.position = (int(x), int(y))
            time.sleep(duration / steps)

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        """
        相对移动鼠标

        Args:
            dx: x偏移量
            dy: y偏移量
        """
        current_x, current_y = self.mouse.position
        self.mouse_move(current_x + dx, current_y + dy)

    def mouse_click(
        self,
        button: str = 'left',
        clicks: int = 1,
        interval: float = 0.1
    ) -> None:
        """
        鼠标点击

        Args:
            button: 按键 ('left', 'right', 'middle')
            clicks: 点击次数
            interval: 点击间隔
        """
        btn_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = btn_map.get(button.lower(), Button.left)

        for _ in range(clicks):
            self.mouse.press(btn)
            self._pressed_mouse_buttons.add(btn)
            try:
                if self.config.humanize:
                    time.sleep(0.05 + random.uniform(0, 0.02))
                else:
                    time.sleep(0.05)
            finally:
                try:
                    self.mouse.release(btn)
                except Exception as error:
                    logger.warning("Failed to release mouse button %r: %s", btn, error)
                else:
                    self._pressed_mouse_buttons.discard(btn)
            if clicks > 1:
                time.sleep(interval)

        self._delay()

    def mouse_double_click(self, button: str = 'left') -> None:
        """
        鼠标双击

        Args:
            button: 按键
        """
        self.mouse_click(button, clicks=2)

    def mouse_down(self, button: str = 'left') -> None:
        """
        按下鼠标按键

        Args:
            button: 按键
        """
        btn_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = btn_map.get(button.lower(), Button.left)
        self.mouse.press(btn)
        self._pressed_mouse_buttons.add(btn)

    def mouse_up(self, button: str = 'left') -> None:
        """
        释放鼠标按键

        Args:
            button: 按键
        """
        btn_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = btn_map.get(button.lower(), Button.left)
        self.mouse.release(btn)
        self._pressed_mouse_buttons.discard(btn)

    def release_all_inputs(self) -> None:
        """Release every key and mouse button held by this controller."""
        for key_obj in tuple(self._pressed_keys):
            try:
                self.keyboard.release(key_obj)
            except Exception as error:
                logger.warning("Failed to release key %r: %s", key_obj, error)
            else:
                self._pressed_keys.discard(key_obj)

        for button in tuple(self._pressed_mouse_buttons):
            try:
                self.mouse.release(button)
            except Exception as error:
                logger.warning("Failed to release mouse button %r: %s", button, error)
            else:
                self._pressed_mouse_buttons.discard(button)

        if not self._pressed_keys:
            self._current_moving_direction = None

    def mouse_scroll(self, direction: str, amount: int = 1) -> None:
        """
        鼠标滚轮

        Args:
            direction: 方向 ('up' 或 'down')
            amount: 滚动量
        """
        delta = amount if direction.lower() == 'up' else -amount
        self.mouse.scroll(0, delta)
        self._delay()

    def mouse_drag(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        button: str = 'left'
    ) -> None:
        """
        鼠标拖拽

        Args:
            start: 起始位置 (x, y)
            end: 结束位置 (x, y)
            button: 按键
        """
        self.mouse_move(start[0], start[1], smooth=False)
        self.mouse_down(button)
        self._smooth_move(end[0], end[1])
        self.mouse_up(button)
        self._delay()

    # ========== 高级操作 ==========

    def click_at(self, x: int, y: int, button: str = 'left') -> None:
        """
        移动到指定位置并点击

        Args:
            x: 目标x坐标
            y: 目标y坐标
            button: 按键
        """
        self.mouse_move(x, y)
        self.mouse_click(button)

    def move_and_press(self, x: int, y: int, key: str) -> None:
        """
        移动到指定位置并按键

        Args:
            x: 目标x坐标
            y: 目标y坐标
            key: 键名称
        """
        self.mouse_move(x, y)
        self.key_press(key)

    def is_mouse_near(self, x: int, y: int, threshold: int = 10) -> bool:
        """
        检查鼠标是否在指定位置附近

        Args:
            x: 目标x坐标
            y: 目标y坐标
            threshold: 距离阈值

        Returns:
            是否在附近
        """
        current_x, current_y = self.mouse.position
        distance = math.sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        return distance <= threshold

    # ========== 横版游戏控制 ==========

    def move_left(self, duration: float = 0.1) -> None:
        """
        向左移动

        Args:
            duration: 按住时间(秒)
        """
        self.key_press('left', duration)

    def move_right(self, duration: float = 0.1) -> None:
        """
        向右移动

        Args:
            duration: 按住时间(秒)
        """
        self.key_press('right', duration)

    def start_moving(self, direction: str, run: bool = True) -> None:
        """
        开始持续移动

        Args:
            direction: 'left' 或 'right'
            run: 是否跑动模式（双击+长按），默认True
        """
        if direction not in ['left', 'right']:
            return

        # 如果已经在往同一方向移动，不需要重复按键
        if self._current_moving_direction == direction:
            return

        # 如果正在往反方向移动，先停止
        if self._current_moving_direction is not None:
            self.key_up(self._current_moving_direction)

        if run:
            # 跑动模式：双击方向键然后长按
            # 第一次点击
            self.key_down(direction)
            time.sleep(0.05)
            self.key_up(direction)
            # 短暂延迟后第二次按下并保持
            time.sleep(0.05)
            self.key_down(direction)
        else:
            # 普通移动：直接按下方向键
            self.key_down(direction)

        self._current_moving_direction = direction

    def stop_moving(self, direction: str = None) -> None:
        """
        停止移动

        Args:
            direction: 'left', 'right' 或 None(停止所有)
        """
        if direction is None:
            # 停止所有方向
            if self._current_moving_direction is not None:
                self.key_up(self._current_moving_direction)
                self._current_moving_direction = None
        elif direction in ['left', 'right']:
            # 只停止指定方向
            if self._current_moving_direction == direction:
                self.key_up(direction)
                self._current_moving_direction = None

    def get_moving_direction(self) -> Optional[str]:
        """
        获取当前移动方向

        Returns:
            'left', 'right' 或 None
        """
        return self._current_moving_direction

    def attack_continuous(self, key: str = 'x', count: int = 3, interval: float = 0.15) -> None:
        """
        连续攻击

        Args:
            key: 攻击按键
            count: 攻击次数
            interval: 攻击间隔
        """
        for _ in range(count):
            self.key_press(key, duration=0.05)
            time.sleep(interval)
