"""
游戏上下文模块 - 管理游戏状态和历史信息
"""
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

from ..detection.detector import Detection


class GameState(Enum):
    """游戏状态枚举"""
    MENU = "menu"
    LOADING = "loading"
    PLAYING = "playing"
    COMBAT = "combat"
    TRANSITIONING = "transitioning"  # 过渡状态：移动到门/等待地图加载
    PAUSED = "paused"
    DEAD = "dead"
    VICTORY = "victory"
    UNKNOWN = "unknown"
    # DNF 新增状态
    CARD_FLIPPING = "card_flipping"      # 翻牌界面
    CHARACTER_SWITCH = "character_switch" # 角色切换中
    STUCK_RECOVERY = "stuck_recovery"     # 卡住恢复
    BUFFING = "buffing"                   # 释放Buff
    MENU_SELECTING = "menu_selecting"     # 菜单选择


@dataclass
class DetectionResult:
    """单帧检测结果封装"""
    timestamp: float = field(default_factory=time.time)
    enemies: List[Detection] = field(default_factory=list)
    items: List[Detection] = field(default_factory=list)
    ui_elements: List[Detection] = field(default_factory=list)
    obstacles: List[Detection] = field(default_factory=list)
    doors: List[Detection] = field(default_factory=list)  # 门检测
    player_health: Optional[float] = None
    player_position: Optional[Tuple[int, int]] = None

    def get_all(self) -> List[Detection]:
        """获取所有检测对象"""
        return self.enemies + self.items + self.ui_elements + self.obstacles + self.doors

    def get_count(self) -> Dict[str, int]:
        """获取各类对象数量"""
        return {
            'enemies': len(self.enemies),
            'items': len(self.items),
            'ui_elements': len(self.ui_elements),
            'obstacles': len(self.obstacles),
            'doors': len(self.doors)
        }


@dataclass
class GameStats:
    """游戏统计数据"""
    enemies_killed: int = 0
    items_collected: int = 0
    deaths: int = 0
    total_frames: int = 0
    combat_time: float = 0.0
    play_time: float = 0.0


class GameContext:
    """游戏上下文 - 存储当前游戏状态和历史信息"""

    def __init__(self, history_size: int = 30):
        """
        初始化游戏上下文

        Args:
            history_size: 历史记录保留数量
        """
        # 当前状态（默认 PLAYING，直接开始寻找目标）
        self.state = GameState.PLAYING
        self.frame_count = 0
        self.last_update: float = time.time()
        self._last_stats_update: float = time.monotonic()
        self.session_start: float = time.time()

        # 当前帧检测结果
        self.current_detections = DetectionResult()

        # 历史记录
        self.detection_history: deque = deque(maxlen=history_size)
        self.state_history: deque = deque(maxlen=history_size)

        # 统计信息
        self.stats = GameStats()

        # 屏幕中心（用于计算距离）
        self.screen_center: Tuple[int, int] = (960, 540)

        # 自定义数据存储
        self._custom_data: Dict[str, Any] = {}

        # ========== DNF 专用字段 ==========
        # 地图追踪
        self.map_index: int = 0                    # 当前房间索引
        self.map_total: int = 0                    # 总房间数
        self.door_stuck_count: int = 0             # 卡门计数
        self.player_stuck_count: int = 0           # 玩家检测异常计数

        # 模式标志
        self.is_abyss_mode: bool = False           # True=深渊, False=白图
        self.dungeon_run_count: int = 0            # 当前刷图次数
        self.max_dungeon_runs: int = 16            # 最大刷图次数

        # 翻牌
        self.purple_card_priority: bool = True     # 紫卡优先
        self.card_positions: List[Tuple[int, int]] = []  # 检测到的卡片位置

        # 多角色
        self.current_character_index: int = 0      # 当前角色索引
        self.character_list: List[str] = []        # 角色ID列表

        # 玩家检测
        self.player_count: int = 0                 # 检测到的玩家数量
        self.hero_position: Optional[Tuple[int, int]] = None  # 小地图英雄位置

        # 菜单检测计数（用于判断是否需要切换角色）
        self.menu_detect_count: int = 0            # menu 连续检测次数
        self.menu_detect_threshold: int = 180      # 超过此次数触发角色切换

    def update(
        self,
        detections: Dict[str, List[Detection]],
        state: Optional[GameState] = None
    ) -> None:
        """
        更新上下文

        Args:
            detections: 检测结果字典 {'enemies': [...], 'items': [...], ...}
            state: 可选的状态更新
        """
        self.frame_count += 1
        self.last_update = time.time()
        now_monotonic = time.monotonic()
        elapsed = max(0.0, now_monotonic - self._last_stats_update)
        self._last_stats_update = now_monotonic
        self.stats.total_frames += 1

        # 更新状态
        if state:
            old_state = self.state
            self.state = state
            if old_state != state:
                self.state_history.append((self.last_update, state))

        # 更新检测结果
        self.current_detections = DetectionResult(
            timestamp=self.last_update,
            enemies=detections.get('enemies', []),
            items=detections.get('items', []),
            ui_elements=detections.get('ui_elements', []),
            obstacles=detections.get('obstacles', []),
            doors=detections.get('doors', [])
        )

        # 添加到历史
        self.detection_history.append(self.current_detections)

        # 更新游戏时间
        if self.state == GameState.PLAYING:
            self.stats.play_time += elapsed
        elif self.state == GameState.COMBAT:
            self.stats.combat_time += elapsed

    def set_screen_center(self, width: int, height: int) -> None:
        """
        设置屏幕中心

        Args:
            width: 屏幕宽度
            height: 屏幕高度
        """
        self.screen_center = (width // 2, height // 2)

    def get_nearest_enemy(self) -> Optional[Detection]:
        """
        获取最近的敌人

        Returns:
            最近的敌人Detection对象，如果没有则返回None
        """
        if not self.current_detections.enemies:
            return None

        return min(
            self.current_detections.enemies,
            key=lambda e: self._distance_to_center(e.center)
        )

    def get_nearest_item(self) -> Optional[Detection]:
        """
        获取最近的道具

        Returns:
            最近的道具Detection对象
        """
        if not self.current_detections.items:
            return None

        return min(
            self.current_detections.items,
            key=lambda i: self._distance_to_center(i.center)
        )

    def get_enemies_in_range(self, max_distance: int) -> List[Detection]:
        """
        获取指定范围内的敌人

        Args:
            max_distance: 最大距离

        Returns:
            范围内的敌人列表
        """
        return [
            e for e in self.current_detections.enemies
            if self._distance_to_center(e.center) <= max_distance
        ]

    def get_health_status(self) -> str:
        """
        获取血量状态

        Returns:
            'healthy', 'wounded', 'critical', 'unknown'
        """
        health = self.current_detections.player_health
        if health is None:
            return "unknown"
        if health > 0.7:
            return "healthy"
        if health > 0.3:
            return "wounded"
        return "critical"

    def get_enemy_count(self) -> int:
        """获取当前敌人数量"""
        return len(self.current_detections.enemies)

    def get_item_count(self) -> int:
        """获取当前道具数量"""
        return len(self.current_detections.items)

    def has_enemies(self) -> bool:
        """是否有敌人"""
        return len(self.current_detections.enemies) > 0

    def has_items(self) -> bool:
        """是否有道具"""
        return len(self.current_detections.items) > 0

    # ========== 横版游戏专用方法 ==========

    def get_enemy_direction(self, enemy: Detection, threshold: int = 50) -> str:
        """
        获取敌人相对于玩家的方向

        Args:
            enemy: 敌人检测对象
            threshold: 判断阈值

        Returns:
            'left', 'right', 'center'
        """
        if enemy is None:
            return 'center'

        player_x = self.screen_center[0]
        enemy_x = enemy.center[0]

        if enemy_x < player_x - threshold:
            return 'left'
        elif enemy_x > player_x + threshold:
            return 'right'
        return 'center'

    def is_enemy_in_attack_range(self, enemy: Detection, attack_range: int = 100) -> bool:
        """
        判断敌人是否在攻击范围内（基于X轴距离）

        Args:
            enemy: 敌人检测对象
            attack_range: 攻击范围

        Returns:
            是否在攻击范围内
        """
        if enemy is None:
            return False

        player_x = self.screen_center[0]
        enemy_x = enemy.center[0]
        return abs(enemy_x - player_x) <= attack_range

    def get_door(self) -> Optional[Detection]:
        """
        获取门的位置

        Returns:
            门Detection对象，如果没有则返回None
        """
        doors = self.current_detections.doors
        if not doors:
            return None
        return doors[0]  # 返回第一个门

    def has_door(self) -> bool:
        """是否检测到门"""
        return len(self.current_detections.doors) > 0

    def is_player_at_door(self, door: Detection, threshold: int = 50) -> bool:
        """
        判断玩家是否站在门上

        Args:
            door: 门检测对象
            threshold: 判断阈值

        Returns:
            是否站在门上
        """
        if door is None:
            return False

        player_x = self.screen_center[0]
        door_x = door.center[0]
        return abs(door_x - player_x) <= threshold

    def get_move_direction_to_target(self, target: Detection, threshold: int = 50) -> str:
        """
        获取移动到目标需要的方向

        Args:
            target: 目标检测对象
            threshold: 判断阈值

        Returns:
            'left', 'right', 'center'
        """
        return self.get_enemy_direction(target, threshold)

    # ========== 通用方法 ==========

    def is_in_combat(self) -> bool:
        """是否在战斗中"""
        return self.state == GameState.COMBAT

    def is_alive(self) -> bool:
        """是否存活"""
        return self.state not in [GameState.DEAD, GameState.UNKNOWN]

    def get_elapsed_time(self) -> float:
        """获取会话持续时间"""
        return time.time() - self.session_start

    def get_fps(self) -> float:
        """估算当前FPS"""
        if len(self.detection_history) < 2:
            return 0.0

        recent = list(self.detection_history)[-30:]  # 最近30帧
        if len(recent) < 2:
            return 0.0

        time_elapsed = recent[-1].timestamp - recent[0].timestamp
        if time_elapsed <= 0:
            return 0.0

        return (len(recent) - 1) / time_elapsed

    def get_trend(self, object_type: str, window: int = 10) -> str:
        """
        获取对象数量趋势

        Args:
            object_type: 对象类型 ('enemies', 'items')
            window: 窗口大小

        Returns:
            'increasing', 'decreasing', 'stable'
        """
        history = list(self.detection_history)[-window:]
        if len(history) < 2:
            return "stable"

        counts = [
            len(getattr(h, object_type, []))
            for h in history
        ]

        if counts[-1] > counts[0] + 1:
            return "increasing"
        elif counts[-1] < counts[0] - 1:
            return "decreasing"
        return "stable"

    def _distance_to_center(self, point: Tuple[int, int]) -> float:
        """计算点到屏幕中心的距离"""
        return math.sqrt(
            (point[0] - self.screen_center[0]) ** 2 +
            (point[1] - self.screen_center[1]) ** 2
        )

    def set_custom_data(self, key: str, value: Any) -> None:
        """设置自定义数据"""
        self._custom_data[key] = value

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """获取自定义数据"""
        return self._custom_data.get(key, default)

    def reset_stats(self) -> None:
        """重置统计数据"""
        self.stats = GameStats()
        self.frame_count = 0
        self.session_start = time.time()

    # ========== DNF 专用方法 ==========

    def increment_map_index(self) -> None:
        """增加房间索引"""
        self.map_index += 1

    def reset_map_index(self) -> None:
        """重置房间索引"""
        self.map_index = 0

    def increment_dungeon_run(self) -> None:
        """增加刷图次数"""
        self.dungeon_run_count += 1

    def reset_dungeon_run(self) -> None:
        """重置刷图次数"""
        self.dungeon_run_count = 0

    def should_switch_character(self) -> bool:
        """判断是否应该切换角色"""
        return self.dungeon_run_count >= self.max_dungeon_runs

    def increment_door_stuck(self) -> None:
        """增加卡门计数"""
        self.door_stuck_count += 1

    def reset_door_stuck(self) -> None:
        """重置卡门计数"""
        self.door_stuck_count = 0

    def is_door_stuck(self, threshold: int = 5) -> bool:
        """判断是否卡门"""
        return self.door_stuck_count >= threshold

    def increment_player_stuck(self) -> None:
        """增加玩家卡住计数"""
        self.player_stuck_count += 1

    def reset_player_stuck(self) -> None:
        """重置玩家卡住计数"""
        self.player_stuck_count = 0

    def is_player_stuck(self, threshold: int = 4) -> bool:
        """判断玩家是否卡住"""
        return self.player_stuck_count >= threshold

    def increment_menu_detect(self) -> None:
        """增加菜单检测计数"""
        self.menu_detect_count += 1

    def reset_menu_detect(self) -> None:
        """重置菜单检测计数"""
        self.menu_detect_count = 0

    def is_menu_detect_timeout(self) -> bool:
        """判断菜单检测是否超时"""
        return self.menu_detect_count >= self.menu_detect_threshold

    def should_switch_character(self) -> bool:
        """
        判断是否应该切换角色
        条件：刷图次数达到上限
        注意：菜单检测超时单独处理，不触发角色切换
        """
        return self.dungeon_run_count >= self.max_dungeon_runs

    def has_shop(self) -> bool:
        """是否检测到商店"""
        for ui in self.current_detections.ui_elements:
            if ui.class_name == 'shop':
                return True
        return False

    def has_menu(self) -> bool:
        """是否检测到菜单"""
        for ui in self.current_detections.ui_elements:
            if ui.class_name == 'menu':
                return True
        return False

    def get_boss_detection(self) -> Optional[Detection]:
        """
        获取Boss检测结果

        Returns:
            Boss Detection对象，如果没有则返回None
        """
        for enemy in self.current_detections.enemies:
            if enemy.class_name in ['boss', 'boss-m']:
                return enemy
        return None

    def get_elite_detection(self) -> Optional[Detection]:
        """
        获取精英怪检测结果

        Returns:
            精英怪 Detection对象，如果没有则返回None
        """
        for enemy in self.current_detections.enemies:
            if enemy.class_name == 'elite':
                return enemy
        return None

    def has_boss(self) -> bool:
        """是否检测到Boss"""
        return self.get_boss_detection() is not None

    def has_elite(self) -> bool:
        """是否检测到精英怪"""
        return self.get_elite_detection() is not None

    def to_dict(self) -> dict:
        """转换为字典（用于日志/调试）"""
        return {
            'state': self.state.value,
            'frame_count': self.frame_count,
            'elapsed_time': self.get_elapsed_time(),
            'enemy_count': self.get_enemy_count(),
            'item_count': self.get_item_count(),
            'health_status': self.get_health_status(),
            'stats': {
                'enemies_killed': self.stats.enemies_killed,
                'items_collected': self.stats.items_collected,
                'deaths': self.stats.deaths
            }
        }
