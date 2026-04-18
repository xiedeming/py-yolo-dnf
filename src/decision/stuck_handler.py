"""
卡住检测和恢复模块 - 检测游戏中的卡住情况并执行恢复动作
"""
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING
from collections import deque
from enum import Enum

if TYPE_CHECKING:
    from ..control.input_controller import InputController
    from .game_context import GameContext


class StuckType(Enum):
    """卡住类型"""
    DOOR_STUCK = "door_stuck"       # 卡门
    PLAYER_STUCK = "player_stuck"   # 玩家检测异常
    POSITION_STUCK = "position_stuck"  # 位置卡住（帧相似度）


@dataclass
class StuckConfig:
    """卡住检测配置"""
    door_stuck_threshold: int = 5       # door_count >= 5 判定卡门
    player_stuck_threshold: int = 4     # people_count > 4 判定角色卡住
    frame_similarity_threshold: float = 0.95  # 帧相似度阈值
    frame_history_size: int = 10        # 帧历史记录大小
    recovery_cooldown: float = 5.0      # 恢复动作冷却时间


class StuckHandler:
    """
    卡住检测和恢复处理器

    检测三种卡住情况：
    1. 卡门：在门附近无法通过
    2. 玩家检测异常：检测到多个玩家（判定角色卡住）
    3. 位置卡住：连续多帧画面相似度极高
    """

    def __init__(
        self,
        config: StuckConfig,
        controller: 'InputController'
    ):
        """
        初始化卡住检测处理器

        Args:
            config: 检测配置
            controller: 输入控制器
        """
        self.config = config
        self.controller = controller

        # 帧历史记录（用于位置卡住检测）
        self.frame_history: deque = deque(maxlen=config.frame_history_size)

        # 上次恢复时间
        self.last_recovery_time: float = 0.0

        # 当前卡住状态
        self.current_stuck_type: Optional[StuckType] = None

    def detect_door_stuck(self, door_count: int, context: 'GameContext') -> bool:
        """
        检测是否卡门

        Args:
            door_count: 当前检测到的门数量
            context: 游戏上下文

        Returns:
            是否卡门
        """
        if door_count >= self.config.door_stuck_threshold:
            context.increment_door_stuck()
            return context.is_door_stuck(self.config.door_stuck_threshold)
        else:
            context.reset_door_stuck()
            return False

    def detect_player_stuck(self, player_count: int, context: 'GameContext') -> bool:
        """
        检测玩家是否卡住

        Args:
            player_count: 检测到的玩家数量
            context: 游戏上下文

        Returns:
            是否玩家卡住
        """
        if player_count > self.config.player_stuck_threshold:
            context.increment_player_stuck()
            return context.is_player_stuck(self.config.player_stuck_threshold)
        else:
            context.reset_player_stuck()
            return False

    def detect_position_stuck(
        self,
        current_frame: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        通过帧相似度检测位置卡住

        Args:
            current_frame: 当前帧图像
            threshold: 相似度阈值（可选，默认使用配置值）

        Returns:
            是否位置卡住
        """
        if threshold is None:
            threshold = self.config.frame_similarity_threshold

        # 添加当前帧到历史
        self.frame_history.append(current_frame.copy())

        # 历史帧数不足，无法判断
        if len(self.frame_history) < 3:
            return False

        # 比较最近几帧的相似度
        recent_frames = list(self.frame_history)[-3:]
        similarities = []

        for i in range(len(recent_frames) - 1):
            sim = self._calculate_frame_similarity(
                recent_frames[i],
                recent_frames[i + 1]
            )
            similarities.append(sim)

        # 如果平均相似度超过阈值，判定为位置卡住
        avg_similarity = np.mean(similarities)
        return avg_similarity >= threshold

    def execute_recovery(self, stuck_type: StuckType, context: 'GameContext') -> bool:
        """
        执行恢复动作

        Args:
            stuck_type: 卡住类型
            context: 游戏上下文

        Returns:
            是否执行了恢复动作
        """
        # 检查冷却时间
        if time.time() - self.last_recovery_time < self.config.recovery_cooldown:
            return False

        self.current_stuck_type = stuck_type

        # 根据卡住类型执行不同的恢复动作
        if stuck_type == StuckType.DOOR_STUCK:
            self._recover_door_stuck()
        elif stuck_type == StuckType.PLAYER_STUCK:
            self._recover_player_stuck()
        elif stuck_type == StuckType.POSITION_STUCK:
            self._recover_position_stuck()

        self.last_recovery_time = time.time()

        # 重置卡住计数
        context.reset_door_stuck()
        context.reset_player_stuck()
        context.set_custom_data('recovery_done', True)

        return True

    def _recover_door_stuck(self) -> None:
        """卡门恢复：向左跑脱困"""
        # 双击左方向键冲刺
        self.controller.key_press('left')
        time.sleep(0.05)
        self.controller.key_press('left')

        # 持续向左跑
        self.controller.key_down('left')
        time.sleep(0.5)
        self.controller.key_up('left')

    def _recover_player_stuck(self) -> None:
        """玩家卡住恢复：随机移动调整位置"""
        import random

        # 随机选择方向
        direction = random.choice(['left', 'right'])

        # 冲刺移动
        self.controller.key_press(direction)
        time.sleep(0.05)
        self.controller.key_press(direction)

        # 跳跃
        self.controller.key_press('space')
        time.sleep(0.1)

        # 继续移动
        self.controller.key_down(direction)
        time.sleep(0.3)
        self.controller.key_up(direction)

    def _recover_position_stuck(self) -> None:
        """位置卡住恢复：尝试多个方向移动"""
        # 先向右尝试
        self.controller.key_press('right')
        time.sleep(0.05)
        self.controller.key_press('right')

        self.controller.key_down('right')
        time.sleep(0.3)
        self.controller.key_up('right')

        # 跳跃
        self.controller.key_press('space')
        time.sleep(0.1)

        # 再向左尝试
        self.controller.key_press('left')
        time.sleep(0.05)
        self.controller.key_press('left')

        self.controller.key_down('left')
        time.sleep(0.3)
        self.controller.key_up('left')

    def _calculate_frame_similarity(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        计算两帧之间的相似度

        Args:
            frame1: 第一帧
            frame2: 第二帧

        Returns:
            相似度值 (0.0 - 1.0)
        """
        # 转换为灰度图
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # 确保尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        # 计算直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # 归一化
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # 计算相关性
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return max(0.0, min(1.0, similarity))

    def reset(self) -> None:
        """重置所有状态"""
        self.frame_history.clear()
        self.last_recovery_time = 0.0
        self.current_stuck_type = None

    def get_status(self) -> dict:
        """
        获取当前状态

        Returns:
            状态字典
        """
        return {
            'current_stuck_type': self.current_stuck_type.value if self.current_stuck_type else None,
            'frame_history_size': len(self.frame_history),
            'last_recovery_time': self.last_recovery_time,
            'recovery_cooldown_remaining': max(0.0, self.config.recovery_cooldown - (time.time() - self.last_recovery_time))
        }


def create_stuck_handler_from_config(
    config_dict: dict,
    controller: 'InputController'
) -> StuckHandler:
    """
    从配置字典创建卡住检测处理器

    Args:
        config_dict: 配置字典
        controller: 输入控制器

    Returns:
        StuckHandler实例
    """
    config = StuckConfig(
        door_stuck_threshold=config_dict.get('door_threshold', 5),
        player_stuck_threshold=config_dict.get('player_threshold', 4),
        frame_similarity_threshold=config_dict.get('frame_similarity_threshold', 0.95),
        frame_history_size=config_dict.get('frame_history_size', 10),
        recovery_cooldown=config_dict.get('recovery_cooldown', 5.0)
    )

    return StuckHandler(config=config, controller=controller)
