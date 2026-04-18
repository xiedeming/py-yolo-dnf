"""
Buff技能管理模块 - 独立管理Buff技能的释放和冷却
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..control.input_controller import InputController
    from .game_context import GameContext


@dataclass
class BuffConfig:
    """Buff技能配置"""
    keys: List[str]              # 按键序列，如 ["shift", "1"]
    cooldown: float = 30.0       # 重新施放间隔（秒）
    apply_on_start: bool = True  # 进图时是否自动释放
    name: str = ""               # Buff名称（可选）


@dataclass
class BuffInstance:
    """Buff运行时状态"""
    config: BuffConfig
    last_cast_time: float = 0.0  # 上次释放时间
    is_active: bool = False      # 是否已激活

    def is_ready(self) -> bool:
        """判断Buff是否就绪（冷却结束）"""
        return time.time() - self.last_cast_time >= self.config.cooldown

    def time_until_ready(self) -> float:
        """返回剩余冷却时间"""
        remaining = self.config.cooldown - (time.time() - self.last_cast_time)
        return max(0.0, remaining)

    def mark_cast(self) -> None:
        """标记为已释放"""
        self.last_cast_time = time.time()
        self.is_active = True


class BuffManager:
    """
    Buff技能管理器

    独立于攻击技能管理，专门处理Buff类技能的释放时机
    """

    def __init__(
        self,
        buff_configs: List[BuffConfig],
        controller: 'InputController'
    ):
        """
        初始化Buff管理器

        Args:
            buff_configs: Buff配置列表
            controller: 输入控制器
        """
        self.controller = controller
        self.buff_instances: List[BuffInstance] = []

        # 初始化Buff实例
        for config in buff_configs:
            self.buff_instances.append(BuffInstance(config=config))

    def apply_buffs(self, context: 'GameContext') -> bool:
        """
        释放所有就绪的Buff

        Args:
            context: 游戏上下文

        Returns:
            是否释放了任何Buff
        """
        any_cast = False

        for instance in self.buff_instances:
            if instance.is_ready():
                self._cast_buff(instance)
                any_cast = True

        return any_cast

    def apply_start_buffs(self, context: 'GameContext') -> bool:
        """
        释放进图时需要自动释放的Buff

        Args:
            context: 游戏上下文

        Returns:
            是否释放了任何Buff
        """
        any_cast = False

        for instance in self.buff_instances:
            if instance.config.apply_on_start and instance.is_ready():
                self._cast_buff(instance)
                any_cast = True

        return any_cast

    def should_reapply(self) -> bool:
        """
        检查是否有Buff需要重新施放

        Returns:
            是否有Buff需要重新施放
        """
        return any(instance.is_ready() for instance in self.buff_instances)

    def get_ready_buffs(self) -> List[BuffInstance]:
        """
        获取所有就绪的Buff

        Returns:
            就绪的Buff实例列表
        """
        return [inst for inst in self.buff_instances if inst.is_ready()]

    def update(self, context: 'GameContext') -> None:
        """
        更新Buff状态（每帧调用）

        Args:
            context: 游戏上下文
        """
        # 检查是否需要重新施放Buff
        if self.should_reapply():
            context.set_custom_data('need_buff', True)
        else:
            context.set_custom_data('need_buff', False)

    def _cast_buff(self, instance: BuffInstance) -> None:
        """
        执行Buff释放

        Args:
            instance: Buff实例
        """
        keys = instance.config.keys

        if len(keys) == 1:
            # 单键Buff
            self.controller.key_press(keys[0])
        else:
            # 组合键Buff（如 shift+1）
            self.controller.key_combo(keys)

        instance.mark_cast()

    def reset_all(self) -> None:
        """重置所有Buff状态"""
        for instance in self.buff_instances:
            instance.last_cast_time = 0.0
            instance.is_active = False

    def get_status(self) -> dict:
        """
        获取Buff状态信息

        Returns:
            状态字典
        """
        return {
            'total_buffs': len(self.buff_instances),
            'ready_buffs': len(self.get_ready_buffs()),
            'buffs': [
                {
                    'name': inst.config.name or f"Buff_{i}",
                    'keys': inst.config.keys,
                    'ready': inst.is_ready(),
                    'time_until_ready': inst.time_until_ready(),
                    'is_active': inst.is_active
                }
                for i, inst in enumerate(self.buff_instances)
            ]
        }


def create_buff_manager_from_config(
    buff_configs: List[dict],
    controller: 'InputController'
) -> BuffManager:
    """
    从配置字典创建Buff管理器

    Args:
        buff_configs: Buff配置字典列表
        controller: 输入控制器

    Returns:
        BuffManager实例
    """
    configs = []
    for cfg in buff_configs:
        configs.append(BuffConfig(
            keys=cfg.get('keys', []),
            cooldown=cfg.get('cooldown', 30.0),
            apply_on_start=cfg.get('apply_on_start', True),
            name=cfg.get('name', '')
        ))

    return BuffManager(buff_configs=configs, controller=controller)
