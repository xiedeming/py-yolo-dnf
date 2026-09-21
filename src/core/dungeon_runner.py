"""
副本运行器模块 - 管理DNF副本的主循环逻辑
"""
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Callable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .engine import GameEngine
    from ..decision.game_context import GameContext
    from ..decision.buff_manager import BuffManager
    from ..decision.map_navigator import MapNavigator
    from ..decision.card_flipper import CardFlipper
    from ..decision.stuck_handler import StuckHandler
    from ..detection.detector import Detection


class DungeonMode(Enum):
    """副本模式"""
    ABYSS = "abyss"           # 深渊模式 (run_dnf_sy)
    WHITE_MAP = "white_map"   # 白图模式 (run_dnf)
    NEW_ABYSS = "new_abyss"   # 新深渊


@dataclass
class DungeonConfig:
    """副本配置"""
    mode: DungeonMode = DungeonMode.WHITE_MAP
    max_runs: int = 16            # 最大刷图次数
    auto_sell: bool = False       # 自动卖装备
    sell_after_runs: int = 5      # 刷图多少次后卖装备
    collect_items: bool = True    # 自动拾取


class DungeonRunner:
    """
    副本运行器

    管理副本的主循环逻辑，区分深渊模式和白图模式

    深渊模式特点：
    - 无门导航，直线探索
    - boss-m 优先
    - 简化的移动逻辑

    白图模式特点：
    - 门导航，地图路线
    - hero 小地图追踪
    - 复杂的房间切换逻辑
    """

    def __init__(
        self,
        engine: 'GameEngine',
        mode: DungeonMode = DungeonMode.WHITE_MAP,
        config: Optional[DungeonConfig] = None
    ):
        """
        初始化副本运行器

        Args:
            engine: 游戏引擎
            mode: 副本模式
            config: 副本配置
        """
        self.engine = engine
        self.mode = mode
        self.config = config or DungeonConfig(mode=mode)

        # 运行状态
        self.is_running = False
        self.room_count = 0
        self.run_count = 0

        # 回调函数
        self.on_run_complete: Optional[Callable] = None
        self.on_all_complete: Optional[Callable] = None

    def run_abyss_cycle(self, context: 'GameContext') -> bool:
        """
        执行深渊模式副本循环

        Args:
            context: 游戏上下文

        Returns:
            是否继续运行
        """
        from ..decision.game_context import GameState

        # 检查状态
        if context.state == GameState.CARD_FLIPPING:
            # 翻牌状态
            self._handle_card_flipping(context)
            return True

        if context.state == GameState.MENU:
            # 结算菜单
            self._handle_menu_state(context)
            return True

        # 战斗逻辑
        if context.has_enemies():
            # 检查是否有Boss
            boss = context.get_boss_detection()
            if boss:
                self._attack_target(boss, context)
            else:
                # 攻击最近的敌人
                nearest = context.get_nearest_enemy()
                if nearest:
                    self._attack_target(nearest, context)
        else:
            # 无敌人，向右探索
            self._explore_right(context)

        return True

    def run_white_map_cycle(self, context: 'GameContext') -> bool:
        """
        执行白图模式副本循环

        Args:
            context: 游戏上下文

        Returns:
            是否继续运行
        """
        from ..decision.game_context import GameState

        # 检查状态
        if context.state == GameState.CARD_FLIPPING:
            self._handle_card_flipping(context)
            return True

        if context.state == GameState.MENU:
            self._handle_menu_state(context)
            return True

        if context.state == GameState.TRANSITIONING:
            self._handle_door_transition(context)
            return True

        if context.state == GameState.STUCK_RECOVERY:
            self._handle_stuck_recovery(context)
            return True

        # 战斗逻辑
        if context.has_enemies():
            # 检查是否有Boss或精英
            boss = context.get_boss_detection()
            if boss:
                self._attack_target(boss, context)
            else:
                elite = context.get_elite_detection()
                if elite:
                    self._attack_target(elite, context)
                else:
                    nearest = context.get_nearest_enemy()
                    if nearest:
                        self._attack_target(nearest, context)
        else:
            # 检查门
            if context.has_door():
                self._navigate_to_door(context)
            else:
                # 向右探索
                self._explore_right(context)

        return True

    def handle_menu_state(self, context: 'GameContext') -> None:
        """
        处理结算菜单状态

        Args:
            context: 游戏上下文
        """
        self._handle_menu_state(context)

    def handle_card_flipping(self, context: 'GameContext') -> None:
        """
        处理翻牌状态

        Args:
            context: 游戏上下文
        """
        self._handle_card_flipping(context)

    def should_continue(self, context: 'GameContext') -> bool:
        """
        检查是否应该继续刷图

        Args:
            context: 游戏上下文

        Returns:
            是否继续
        """
        return self.run_count < self.config.max_runs

    def _attack_target(self, target: 'Detection', context: 'GameContext') -> None:
        """
        攻击目标

        Args:
            target: 目标检测对象
            context: 游戏上下文
        """
        # 获取移动方向
        direction = context.get_enemy_direction(target)

        # 移动到攻击范围
        if not context.is_enemy_in_attack_range(target):
            if direction == 'left':
                self.engine.controller.key_down('left')
                time.sleep(0.1)
                self.engine.controller.key_up('left')
            elif direction == 'right':
                self.engine.controller.key_down('right')
                time.sleep(0.1)
                self.engine.controller.key_up('right')

        # 使用技能
        if self.engine.skill_manager:
            self.engine.skill_manager.use_next_available_skill()
        else:
            # 普通攻击
            self.engine.controller.key_press('x')

    def _explore_right(self, context: 'GameContext') -> None:
        """
        向右探索

        Args:
            context: 游戏上下文
        """
        # 冲刺移动
        self.engine.controller.key_press('right')
        time.sleep(0.05)
        self.engine.controller.key_press('right')

        # 持续移动
        self.engine.controller.key_down('right')
        time.sleep(0.3)
        self.engine.controller.key_up('right')

    def _navigate_to_door(self, context: 'GameContext') -> None:
        """
        导航到门

        Args:
            context: 游戏上下文
        """
        door = context.get_door()
        if door is None:
            return

        # 获取门的方向
        direction = context.get_move_direction_to_target(door)

        # 移动到门
        if direction == 'left':
            self.engine.controller.key_down('left')
            time.sleep(0.2)
            self.engine.controller.key_up('left')
        elif direction == 'right':
            self.engine.controller.key_down('right')
            time.sleep(0.2)
            self.engine.controller.key_up('right')

        # 进门
        if context.is_player_at_door(door):
            self.engine.controller.key_press('up')
            context.increment_map_index()

    def _handle_door_transition(self, context: 'GameContext') -> None:
        """
        处理门过渡状态

        Args:
            context: 游戏上下文
        """
        door = context.get_door()
        if door and context.is_player_at_door(door):
            self.engine.controller.key_press('up')

    def _handle_menu_state(self, context: 'GameContext') -> None:
        """
        处理结算菜单

        Args:
            context: 游戏上下文
        """
        # 聚集物品
        for _ in range(3):
            self.engine.controller.key_press('tab')
            time.sleep(0.1)
            self.engine.controller.key_press('x')
            time.sleep(0.1)

        # 判断是否继续
        if self.should_continue(context):
            # 再来一次 (F10)
            self.engine.controller.key_press('f10')
            self.run_count += 1
            context.increment_dungeon_run()
        else:
            # 退出副本 (ESC)
            self.engine.controller.key_press('escape')
            context.set_custom_data('dungeon_complete', True)

    def _handle_card_flipping(self, context: 'GameContext') -> None:
        """
        处理翻牌

        Args:
            context: 游戏上下文
        """
        if self.engine.card_flipper:
            # 获取检测结果
            ui_elements = context.current_detections.ui_elements
            items = context.current_detections.items

            # 执行翻牌
            self.engine.card_flipper.execute_flip(ui_elements + items, context)

    def _handle_stuck_recovery(self, context: 'GameContext') -> None:
        """
        处理卡住恢复

        Args:
            context: 游戏上下文
        """
        if self.engine.stuck_handler:
            from ..decision.stuck_handler import StuckType

            # 根据卡住类型恢复
            if context.is_door_stuck():
                self.engine.stuck_handler.execute_recovery(StuckType.DOOR_STUCK, context)
            elif context.is_player_stuck():
                self.engine.stuck_handler.execute_recovery(StuckType.PLAYER_STUCK, context)

    def start(self) -> None:
        """开始副本运行"""
        self.is_running = True
        self.room_count = 0
        self.run_count = 0

    def stop(self) -> None:
        """停止副本运行"""
        self.is_running = False

    def reset(self) -> None:
        """重置状态"""
        self.room_count = 0
        self.run_count = 0
        self.is_running = False

    def get_status(self) -> dict:
        """
        获取运行状态

        Returns:
            状态字典
        """
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'room_count': self.room_count,
            'run_count': self.run_count,
            'max_runs': self.config.max_runs,
            'progress': self.run_count / self.config.max_runs if self.config.max_runs > 0 else 0
        }


def create_dungeon_runner_from_config(
    engine: 'GameEngine',
    config_dict: dict
) -> DungeonRunner:
    """
    从配置字典创建副本运行器

    Args:
        engine: 游戏引擎
        config_dict: 配置字典

    Returns:
        DungeonRunner实例
    """
    mode_str = config_dict.get('mode', 'white_map')
    mode = DungeonMode(mode_str) if mode_str in [m.value for m in DungeonMode] else DungeonMode.WHITE_MAP

    dungeon_config = DungeonConfig(
        mode=mode,
        max_runs=config_dict.get('max_runs', 16),
        auto_sell=config_dict.get('auto_sell', False),
        sell_after_runs=config_dict.get('sell_after_runs', 5),
        collect_items=config_dict.get('collect_items', True)
    )

    return DungeonRunner(engine=engine, mode=mode, config=dungeon_config)
