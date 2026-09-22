"""
多角色管理模块 - 管理多个角色的依次运行
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..core.engine import GameEngine
    from .character_switcher import CharacterSwitcher
    from .game_context import GameContext


@dataclass
class CharacterRunConfig:
    """单个角色的运行配置"""
    character_id: str                     # 角色ID
    name: str = ""                        # 角色名称
    dungeon_runs: int = 16                # 刷图次数
    dungeon_mode: str = "white_map"       # 副本模式
    art: List[List[str]] = field(default_factory=list)     # 技能按键列表
    art_time: Dict[str, float] = field(default_factory=dict)  # 技能冷却时间
    buff: List[List[str]] = field(default_factory=list)    # Buff按键列表
    move_speed: float = 1.0               # 移动速度系数，1.0 基准，>1 更快
    run_sleep: float = 0.075              # 移动延迟：最小点按/步进时长(秒)
    press_sleep: float = 0.55             # 按键延迟：参考距离对应的按住时长(秒)
    buff_sleep: float = 0.3               # Buff延迟


@dataclass
class CharacterRunStats:
    """角色运行统计"""
    character_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    runs_completed: int = 0
    errors: int = 0

    def get_duration(self) -> float:
        """获取运行时长（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class MultiCharacterManager:
    """
    多角色管理器

    管理多个角色的依次切换和运行

    功能：
    1. 管理角色配置列表
    2. 按顺序切换角色
    3. 追踪每个角色的运行状态
    4. 支持指定开始/结束角色
    """

    def __init__(
        self,
        engine: 'GameEngine',
        role_list: List[CharacterRunConfig],
        character_switcher: Optional['CharacterSwitcher'] = None
    ):
        """
        初始化多角色管理器

        Args:
            engine: 游戏引擎
            role_list: 角色配置列表
            character_switcher: 角色切换器（可选）
        """
        self.engine = engine
        self.role_list = role_list
        self.character_switcher = character_switcher

        # 当前状态
        self.current_index = 0
        self.is_running = False

        # 运行统计
        self.run_stats: Dict[str, CharacterRunStats] = {}

        # 初始化统计
        for config in role_list:
            self.run_stats[config.character_id] = CharacterRunStats(
                character_id=config.character_id
            )

        # 回调函数
        self.on_character_start: Optional[Callable[[CharacterRunConfig], None]] = None
        self.on_character_complete: Optional[Callable[[CharacterRunConfig, CharacterRunStats], None]] = None
        self.on_all_complete: Optional[Callable[[], None]] = None

        # 特殊角色范围（可选）
        self.start_character: Optional[str] = None
        self.end_character: Optional[str] = None

    def set_character_range(self, start: Optional[str], end: Optional[str]) -> None:
        """
        设置角色范围（从哪个角色开始，到哪个角色结束）

        Args:
            start: 开始角色ID
            end: 结束角色ID
        """
        self.start_character = start
        self.end_character = end

        # 调整起始索引
        if start:
            for i, config in enumerate(self.role_list):
                if config.character_id == start:
                    self.current_index = i
                    break

    def get_current_character(self) -> Optional[CharacterRunConfig]:
        """
        获取当前角色配置

        Returns:
            当前角色配置，如果已全部完成则返回None
        """
        if self.current_index >= len(self.role_list):
            return None
        return self.role_list[self.current_index]

    def get_current_character_id(self) -> Optional[str]:
        """
        获取当前角色ID

        Returns:
            当前角色ID
        """
        config = self.get_current_character()
        return config.character_id if config else None

    def switch_to_next(self, context: Optional['GameContext'] = None) -> bool:
        """
        切换到下一个角色

        Args:
            context: 游戏上下文

        Returns:
            是否切换成功（False表示所有角色已完成）
        """
        current = self.get_current_character()
        if current:
            # 记录当前角色完成
            stats = self.run_stats[current.character_id]
            stats.end_time = datetime.now()

            if self.on_character_complete:
                self.on_character_complete(current, stats)

        # 检查是否到达结束角色
        if self.end_character and current and current.character_id == self.end_character:
            self.is_running = False
            if self.on_all_complete:
                self.on_all_complete()
            return False

        # 移动到下一个角色
        self.current_index += 1

        # 检查是否还有角色
        next_char = self.get_current_character()
        if next_char is None:
            self.is_running = False
            if self.on_all_complete:
                self.on_all_complete()
            return False

        # 执行角色切换
        if self.character_switcher:
            success = self.character_switcher.execute_full_switch(
                next_char.character_id,
                context
            )
            if not success:
                return False

        # 开始新角色的统计
        stats = self.run_stats[next_char.character_id]
        stats.start_time = datetime.now()
        stats.runs_completed = 0
        stats.errors = 0

        if self.on_character_start:
            self.on_character_start(next_char)

        return True

    def run_character_cycle(
        self,
        run_func: Callable[[CharacterRunConfig], bool]
    ) -> bool:
        """
        运行当前角色的刷图循环

        Args:
            run_func: 运行函数，接收角色配置，返回是否成功

        Returns:
            是否成功
        """
        current = self.get_current_character()
        if current is None:
            return False

        # 确保统计已开始
        stats = self.run_stats[current.character_id]
        if stats.start_time is None:
            stats.start_time = datetime.now()

        if self.on_character_start:
            self.on_character_start(current)

        return run_func(current)

    def run_all_characters(
        self,
        run_func: Callable[[CharacterRunConfig], bool]
    ) -> None:
        """
        依次运行所有角色

        Args:
            run_func: 运行函数
        """
        self.is_running = True

        while self.is_running:
            current = self.get_current_character()
            if current is None:
                break

            # 运行当前角色
            success = self.run_character_cycle(run_func)

            if success:
                # 切换到下一个角色
                if not self.switch_to_next():
                    break
            else:
                # 运行失败，记录错误
                stats = self.run_stats[current.character_id]
                stats.errors += 1
                break

        self.is_running = False

    def get_progress(self) -> tuple:
        """
        获取进度

        Returns:
            (当前索引, 总数, 当前角色ID)
        """
        return (
            self.current_index,
            len(self.role_list),
            self.get_current_character_id()
        )

    def get_all_stats(self) -> Dict[str, CharacterRunStats]:
        """
        获取所有角色的运行统计

        Returns:
            统计字典
        """
        return self.run_stats

    def get_summary(self) -> dict:
        """
        获取运行摘要

        Returns:
            摘要字典
        """
        total_runs = sum(s.runs_completed for s in self.run_stats.values())
        total_errors = sum(s.errors for s in self.run_stats.values())
        total_time = sum(s.get_duration() for s in self.run_stats.values())

        return {
            'total_characters': len(self.role_list),
            'completed_characters': self.current_index,
            'total_runs': total_runs,
            'total_errors': total_errors,
            'total_time_seconds': total_time,
            'current_character': self.get_current_character_id(),
            'is_running': self.is_running
        }

    def reset(self) -> None:
        """重置所有状态"""
        self.current_index = 0
        self.is_running = False

        for stats in self.run_stats.values():
            stats.start_time = None
            stats.end_time = None
            stats.runs_completed = 0
            stats.errors = 0

    def stop(self) -> None:
        """停止运行"""
        self.is_running = False


def create_multi_character_manager_from_config(
    engine: 'GameEngine',
    config_dict: dict,
    character_switcher: Optional['CharacterSwitcher'] = None
) -> Optional[MultiCharacterManager]:
    """
    从配置字典创建多角色管理器

    Args:
        engine: 游戏引擎
        config_dict: 配置字典
        character_switcher: 角色切换器

    Returns:
        MultiCharacterManager实例，如果未启用则返回None
    """
    if not config_dict.get('enabled', False):
        return None

    role_list_data = config_dict.get('role_list', [])
    if not role_list_data:
        return None

    role_list = []
    for role_data in role_list_data:
        config = CharacterRunConfig(
            character_id=role_data.get('id', ''),
            name=role_data.get('name', ''),
            dungeon_runs=role_data.get('dungeon_runs', 16),
            dungeon_mode=role_data.get('dungeon_mode', 'white_map'),
            art=role_data.get('art', []),
            art_time=role_data.get('art_time', {}),
            buff=role_data.get('buff', []),
            move_speed=role_data.get('move_speed', 1.0),
            run_sleep=role_data.get('run_sleep', 0.075),
            press_sleep=role_data.get('press_sleep', 0.55),
            buff_sleep=role_data.get('buff_sleep', 0.3)
        )
        role_list.append(config)

    manager = MultiCharacterManager(
        engine=engine,
        role_list=role_list,
        character_switcher=character_switcher
    )

    # 设置角色范围
    if 'start_name' in config_dict:
        manager.start_character = config_dict['start_name']
    if 'end_name' in config_dict:
        manager.end_character = config_dict['end_name']

    manager.set_character_range(
        manager.start_character,
        manager.end_character
    )

    return manager
