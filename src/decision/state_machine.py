"""
状态机模块 - 游戏状态转换逻辑
"""
from enum import Enum
from typing import Dict, Callable, Optional, List, Any
from dataclasses import dataclass


@dataclass
class Transition:
    """状态转换"""
    target_state: Enum
    condition: Callable[[Any], bool]  # 接受GameContext，返回是否满足条件
    action: Optional[Callable[[Any], None]] = None  # 转换时执行的动作
    priority: int = 0  # 优先级，数字越大优先级越高


class StateMachine:
    """有限状态机"""

    def __init__(self, initial_state: Enum):
        """
        初始化状态机

        Args:
            initial_state: 初始状态
        """
        self.current_state = initial_state
        self.previous_state: Optional[Enum] = None

        # 转换规则: {from_state: [Transition, ...]}
        self.transitions: Dict[Enum, List[Transition]] = {}

        # 状态动作: {state: action_function}
        self.state_actions: Dict[Enum, Callable[[Any], None]] = {}

        # 进入/离开状态动作
        self.on_enter: Dict[Enum, Callable[[Any], None]] = {}
        self.on_exit: Dict[Enum, Callable[[Any], None]] = {}

        # 状态变化历史
        self.state_history: List[tuple] = []
        self.max_history = 100

    def add_transition(
        self,
        from_state: Enum,
        to_state: Enum,
        condition: Callable[[Any], bool],
        action: Optional[Callable[[Any], None]] = None,
        priority: int = 0
    ) -> 'StateMachine':
        """
        添加状态转换

        Args:
            from_state: 源状态
            to_state: 目标状态
            condition: 转换条件函数，接受GameContext，返回bool
            action: 转换时执行的动作
            priority: 优先级

        Returns:
            self（支持链式调用）
        """
        if from_state not in self.transitions:
            self.transitions[from_state] = []

        self.transitions[from_state].append(Transition(
            target_state=to_state,
            condition=condition,
            action=action,
            priority=priority
        ))

        # 按优先级排序
        self.transitions[from_state].sort(key=lambda t: -t.priority)

        return self

    def set_state_action(
        self,
        state: Enum,
        action: Callable[[Any], None]
    ) -> 'StateMachine':
        """
        设置状态动作（每帧执行）

        Args:
            state: 状态
            action: 动作函数

        Returns:
            self
        """
        self.state_actions[state] = action
        return self

    def set_enter_action(
        self,
        state: Enum,
        action: Callable[[Any], None]
    ) -> 'StateMachine':
        """
        设置进入状态动作

        Args:
            state: 状态
            action: 动作函数

        Returns:
            self
        """
        self.on_enter[state] = action
        return self

    def set_exit_action(
        self,
        state: Enum,
        action: Callable[[Any], None]
    ) -> 'StateMachine':
        """
        设置离开状态动作

        Args:
            state: 状态
            action: 动作函数

        Returns:
            self
        """
        self.on_exit[state] = action
        return self

    def update(self, context: Any) -> Enum:
        """
        更新状态机

        Args:
            context: 游戏上下文

        Returns:
            当前状态
        """
        # 检查转换条件
        if self.current_state in self.transitions:
            for transition in self.transitions[self.current_state]:
                try:
                    if transition.condition(context):
                        # 执行离开动作
                        if self.current_state in self.on_exit:
                            self.on_exit[self.current_state](context)

                        # 执行转换动作
                        if transition.action:
                            transition.action(context)

                        # 更新状态
                        self.previous_state = self.current_state
                        self.current_state = transition.target_state

                        # 记录历史
                        import time
                        self.state_history.append((
                            time.time(),
                            self.previous_state,
                            self.current_state
                        ))
                        if len(self.state_history) > self.max_history:
                            self.state_history.pop(0)

                        # 执行进入动作
                        if self.current_state in self.on_enter:
                            self.on_enter[self.current_state](context)

                        return self.current_state
                except Exception as e:
                    print(f"Error in transition condition: {e}")
                    continue

        # 执行当前状态动作
        if self.current_state in self.state_actions:
            try:
                self.state_actions[self.current_state](context)
            except Exception as e:
                print(f"Error in state action: {e}")

        return self.current_state

    def force_state(self, state: Enum, context: Any = None) -> None:
        """
        强制切换到指定状态

        Args:
            state: 目标状态
            context: 游戏上下文
        """
        if self.current_state in self.on_exit and context:
            self.on_exit[self.current_state](context)

        self.previous_state = self.current_state
        self.current_state = state

        if state in self.on_enter and context:
            self.on_enter[state](context)

    def get_state(self) -> Enum:
        """获取当前状态"""
        return self.current_state

    def get_previous_state(self) -> Optional[Enum]:
        """获取上一个状态"""
        return self.previous_state

    def is_state(self, state: Enum) -> bool:
        """检查是否是指定状态"""
        return self.current_state == state

    def was_state(self, state: Enum) -> bool:
        """检查上一个状态是否是指定状态"""
        return self.previous_state == state

    def get_state_history(self, count: int = 10) -> List[tuple]:
        """
        获取状态变化历史

        Args:
            count: 返回的记录数量

        Returns:
            [(timestamp, from_state, to_state), ...]
        """
        return self.state_history[-count:]


def create_game_state_machine() -> StateMachine:
    """
    创建游戏状态机（预定义常用状态转换）

    Returns:
        配置好的状态机
    """
    from .game_context import GameState

    # 默认初始状态为 PLAYING，直接开始寻找目标
    sm = StateMachine(GameState.PLAYING)

    # MENU -> PLAYING: 不再检测到菜单，且检测到敌人或门
    sm.add_transition(
        GameState.MENU,
        GameState.PLAYING,
        condition=lambda ctx: not ctx.has_menu() and (ctx.has_enemies() or ctx.has_door()),
        priority=10
    )

    # MENU -> LOADING: 不再检测到菜单（可能是加载中）
    sm.add_transition(
        GameState.MENU,
        GameState.LOADING,
        condition=lambda ctx: not ctx.has_menu(),
        priority=5
    )

    # PLAYING -> COMBAT: 检测到敌人（高优先级）
    sm.add_transition(
        GameState.PLAYING,
        GameState.COMBAT,
        condition=lambda ctx: ctx.has_enemies(),
        priority=10
    )

    # PLAYING -> TRANSITIONING: 无敌人且有门
    sm.add_transition(
        GameState.PLAYING,
        GameState.TRANSITIONING,
        condition=lambda ctx: not ctx.has_enemies() and ctx.has_door()
    )

    # PLAYING -> BUFFING: 需要释放Buff（高优先级）
    sm.add_transition(
        GameState.PLAYING,
        GameState.BUFFING,
        condition=lambda ctx: ctx.get_custom_data('need_buff', False),
        priority=15
    )

    # PLAYING -> CARD_FLIPPING: 检测到翻牌界面（brand）
    sm.add_transition(
        GameState.PLAYING,
        GameState.CARD_FLIPPING,
        condition=lambda ctx: any(
            d.class_name == 'brand' for d in ctx.current_detections.ui_elements
        ),
        priority=12
    )

    # COMBAT -> CARD_FLIPPING: 检测到翻牌界面（brand）
    sm.add_transition(
        GameState.COMBAT,
        GameState.CARD_FLIPPING,
        condition=lambda ctx: any(
            d.class_name == 'brand' for d in ctx.current_detections.ui_elements
        ),
        priority=15
    )

    # PLAYING / COMBAT / TRANSITIONING -> STUCK_RECOVERY: 卡住检测
    # 三个移动相关状态都能进入 —— 实际最常见的卡住就是"走向门走不动"和"战斗中被卡角落"。
    # 优先级 16 高于 BUFFING(15) 和 COMBAT 转移(10)，低于 MENU(20)/DEAD(30)：
    # 出菜单和死亡本来就该优先于解卡。
    for _stuck_source in (GameState.PLAYING, GameState.COMBAT, GameState.TRANSITIONING):
        sm.add_transition(
            _stuck_source,
            GameState.STUCK_RECOVERY,
            condition=lambda ctx: ctx.get_custom_data('stuck_detected', False),
            priority=16,
        )

    # PLAYING -> MENU: 检测到菜单（高优先级）
    sm.add_transition(
        GameState.PLAYING,
        GameState.MENU,
        condition=lambda ctx: ctx.has_menu(),
        priority=20
    )

    # COMBAT -> PLAYING: 没有敌人了
    sm.add_transition(
        GameState.COMBAT,
        GameState.PLAYING,
        condition=lambda ctx: not ctx.has_enemies()
    )

    # COMBAT -> MENU: 检测到菜单（高优先级）
    sm.add_transition(
        GameState.COMBAT,
        GameState.MENU,
        condition=lambda ctx: ctx.has_menu(),
        priority=20
    )

    # COMBAT -> BUFFING: 战斗中需要Buff
    sm.add_transition(
        GameState.COMBAT,
        GameState.BUFFING,
        condition=lambda ctx: ctx.get_custom_data('need_buff', False),
        priority=15
    )

    # TRANSITIONING -> LOADING: 玩家站在门上
    sm.add_transition(
        GameState.TRANSITIONING,
        GameState.LOADING,
        condition=lambda ctx: ctx.is_player_at_door(ctx.get_door())
    )

    # TRANSITIONING -> MENU: 检测到菜单（高优先级）
    sm.add_transition(
        GameState.TRANSITIONING,
        GameState.MENU,
        condition=lambda ctx: ctx.has_menu(),
        priority=20
    )

    # TRANSITIONING -> PLAYING: 门消失
    # 原有的出口只有 ->LOADING（需要门）和 ->MENU，门一旦消失就会永久卡在 TRANSITIONING，
    # 而解卡正好可能返回到这个状态，所以必须补一个出口。优先级 0，让上面两条优先。
    sm.add_transition(
        GameState.TRANSITIONING,
        GameState.PLAYING,
        condition=lambda ctx: not ctx.has_door(),
    )

    # LOADING -> PLAYING: 加载完成，检测到敌人或门
    sm.add_transition(
        GameState.LOADING,
        GameState.PLAYING,
        condition=lambda ctx: ctx.has_enemies() or ctx.has_door()
    )

    # LOADING -> MENU: 检测到菜单（高优先级）
    sm.add_transition(
        GameState.LOADING,
        GameState.MENU,
        condition=lambda ctx: ctx.has_menu(),
        priority=20
    )

    # LOADING -> BUFFING: 进图后释放Buff
    sm.add_transition(
        GameState.LOADING,
        GameState.BUFFING,
        condition=lambda ctx: ctx.get_custom_data('need_buff', False),
        priority=10
    )

    # BUFFING -> PLAYING: Buff释放完成
    sm.add_transition(
        GameState.BUFFING,
        GameState.PLAYING,
        condition=lambda ctx: ctx.get_custom_data('buff_done', False)
    )

    # BUFFING -> COMBAT: Buff中检测到敌人
    sm.add_transition(
        GameState.BUFFING,
        GameState.COMBAT,
        condition=lambda ctx: ctx.has_enemies(),
        priority=10
    )

    # BUFFING -> MENU: 检测到菜单（高优先级）
    sm.add_transition(
        GameState.BUFFING,
        GameState.MENU,
        condition=lambda ctx: ctx.has_menu(),
        priority=20
    )

    # CARD_FLIPPING -> MENU: 翻牌完成
    sm.add_transition(
        GameState.CARD_FLIPPING,
        GameState.MENU,
        condition=lambda ctx: ctx.get_custom_data('card_flip_done', False)
    )

    # STUCK_RECOVERY -> 卡住前的状态: 恢复完成（成功或耗尽后放弃）
    # 按 stuck_return_state 回到原状态。条件只接收 ctx，所以用闭包把 target 捕获进去。
    for _stuck_target in (GameState.PLAYING, GameState.COMBAT, GameState.TRANSITIONING):
        sm.add_transition(
            GameState.STUCK_RECOVERY,
            _stuck_target,
            condition=(lambda target: lambda ctx: (
                ctx.get_custom_data('recovery_done', False)
                and ctx.get_custom_data('stuck_return_state') is target
            ))(_stuck_target)
        )

    # CHARACTER_SWITCH -> MENU: 角色切换完成
    sm.add_transition(
        GameState.CHARACTER_SWITCH,
        GameState.MENU,
        condition=lambda ctx: ctx.get_custom_data('switch_done', False)
    )

    # MENU -> CHARACTER_SWITCH: 需要切换角色
    sm.add_transition(
        GameState.MENU,
        GameState.CHARACTER_SWITCH,
        condition=lambda ctx: ctx.should_switch_character()
    )

    # PLAYING/COMBAT -> DEAD: 血量为0或检测到死亡UI
    sm.add_transition(
        GameState.PLAYING,
        GameState.DEAD,
        condition=lambda ctx: ctx.get_health_status() == "critical",
        priority=30
    )
    sm.add_transition(
        GameState.COMBAT,
        GameState.DEAD,
        condition=lambda ctx: ctx.get_health_status() == "critical",
        priority=30
    )

    # DEAD -> PLAYING: 复活
    sm.add_transition(
        GameState.DEAD,
        GameState.PLAYING,
        condition=lambda ctx: ctx.get_health_status() not in ["critical", "unknown"]
    )

    # PAUSED -> PLAYING: 取消暂停
    sm.add_transition(
        GameState.PAUSED,
        GameState.PLAYING,
        condition=lambda ctx: ctx.has_enemies() or ctx.has_items()
    )

    # PLAYING/COMBAT -> PAUSED: 暂停
    sm.add_transition(
        GameState.PLAYING,
        GameState.PAUSED,
        condition=lambda ctx: ctx.get_custom_data('is_paused', False)
    )
    sm.add_transition(
        GameState.COMBAT,
        GameState.PAUSED,
        condition=lambda ctx: ctx.get_custom_data('is_paused', False)
    )

    return sm
