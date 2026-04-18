"""
技能管理模块 - 管理技能队列、冷却和释放逻辑
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class SkillState(Enum):
    """技能状态"""
    READY = "ready"           # 就绪
    COOLING = "cooling"       # 冷却中


@dataclass
class SkillInfo:
    """技能信息"""
    keys: List[str]  # 按键列表（支持组合键）
    cooldown: float  # 冷却时间
    last_used_time: float = 0.0
    state: SkillState = SkillState.READY

    def get_remaining_cooldown(self) -> float:
        """获取剩余冷却时间"""
        elapsed = time.time() - self.last_used_time
        remaining = self.cooldown - elapsed
        return max(0.0, remaining)

    def is_ready(self) -> bool:
        """技能是否就绪"""
        return self.get_remaining_cooldown() <= 0

    def use(self) -> None:
        """使用技能（记录时间）"""
        self.last_used_time = time.time()
        self.state = SkillState.COOLING


class SkillQueue:
    """技能队列管理器 - 支持后台冷却监控"""

    def __init__(self, controller):
        """
        初始化技能队列

        Args:
            controller: InputController实例
        """
        self.controller = controller
        self.skills: Dict[str, SkillInfo] = {}  # 技能名 -> 技能信息
        self.queue: Queue = Queue()  # 就绪技能队列
        self._running = False
        self._cooldown_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._check_interval = 0.1  # 冷却检查间隔（秒）

    def load_skills(self, art_list: List[List[str]], art_time: Dict[str, float]) -> None:
        """
        加载技能配置

        Args:
            art_list: 技能列表，每个元素是按键列表
            art_time: 技能冷却时间字典
        """
        with self._lock:
            self.skills.clear()
            self.queue = Queue()

            for keys in art_list:
                if not keys:
                    continue

                # 使用第一个按键作为技能标识
                skill_name = keys[0] if len(keys) == 1 else '+'.join(keys)
                cooldown = art_time.get(skill_name, 10.0)

                # 如果是组合键，尝试用组合键名查找冷却时间
                if len(keys) > 1:
                    combo_name = '+'.join(keys)
                    if combo_name in art_time:
                        cooldown = art_time[combo_name]

                self.skills[skill_name] = SkillInfo(
                    keys=keys,
                    cooldown=cooldown
                )
                # 初始时所有技能都加入队列
                self.queue.put(skill_name)

            logger.info(f"已加载 {len(self.skills)} 个技能到队列")

    def start(self) -> None:
        """启动冷却监控线程"""
        if self._running:
            return

        self._running = True
        self._cooldown_thread = threading.Thread(target=self._cooldown_monitor, daemon=True)
        self._cooldown_thread.start()
        logger.debug("技能冷却监控线程已启动")

    def stop(self) -> None:
        """停止冷却监控线程"""
        self._running = False
        if self._cooldown_thread:
            self._cooldown_thread.join(timeout=1.0)
            self._cooldown_thread = None
        logger.debug("技能冷却监控线程已停止")

    def _cooldown_monitor(self) -> None:
        """后台冷却监控线程"""
        while self._running:
            try:
                with self._lock:
                    for skill_name, skill_info in self.skills.items():
                        if skill_info.state == SkillState.COOLING and skill_info.is_ready():
                            skill_info.state = SkillState.READY
                            self.queue.put(skill_name)
                            logger.debug(f"技能 {skill_name} 冷却完成，已加入队列")

                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"冷却监控错误: {e}")

    def use_next_skill(self) -> bool:
        """
        使用队列中的下一个技能

        Returns:
            是否成功使用技能
        """
        with self._lock:
            if self.queue.empty():
                return False

            # 获取下一个技能
            skill_name = self.queue.get()
            skill_info = self.skills.get(skill_name)

            if skill_info is None:
                return False

            if not skill_info.is_ready():
                # 技能还在冷却，放回队列末尾
                self.queue.put(skill_name)
                return False

            # 执行技能按键
            self._execute_skill(skill_info)
            skill_info.use()

            logger.debug(f"使用技能: {skill_name}")
            return True

    def _execute_skill(self, skill_info: SkillInfo) -> None:
        """执行技能按键"""
        keys = skill_info.keys

        if len(keys) == 1:
            # 单键
            self.controller.key_press(keys[0])
        else:
            # 组合键：依次按下，然后依次释放
            for key in keys:
                self.controller.key_down(key)
            time.sleep(0.05)
            for key in reversed(keys):
                self.controller.key_up(key)

    def reset(self) -> None:
        """重置所有技能状态和队列"""
        with self._lock:
            self.queue = Queue()
            for skill_name, skill_info in self.skills.items():
                skill_info.last_used_time = 0.0
                skill_info.state = SkillState.READY
                self.queue.put(skill_name)
            logger.debug("技能队列已重置")

    def clear(self) -> None:
        """清空所有技能"""
        with self._lock:
            self.skills.clear()
            self.queue = Queue()

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()

    def get_skill_status(self) -> Dict[str, Dict]:
        """获取所有技能状态（调试用）"""
        with self._lock:
            return {
                name: {
                    "keys": info.keys,
                    "cooldown": info.cooldown,
                    "remaining": info.get_remaining_cooldown(),
                    "state": info.state.value
                }
                for name, info in self.skills.items()
            }


class BuffManager:
    """Buff管理器"""

    def __init__(self, controller):
        """
        初始化Buff管理器

        Args:
            controller: InputController实例
        """
        self.controller = controller
        self.buffs: List[List[str]] = []  # buff按键列表
        self._last_cast_time: float = 0.0
        self._buff_cooldown: float = 2.0  # buff释放间隔

    def load_buffs(self, buff_list: List[List[str]]) -> None:
        """
        加载Buff配置

        Args:
            buff_list: Buff按键列表
        """
        self.buffs = buff_list
        logger.info(f"已加载 {len(self.buffs)} 个Buff")

    def apply_buffs(self) -> bool:
        """
        释放所有Buff

        Returns:
            是否成功释放
        """
        if not self.buffs:
            return False

        # 检查释放间隔
        if time.time() - self._last_cast_time < self._buff_cooldown:
            return False

        for buff_keys in self.buffs:
            self._execute_buff(buff_keys)
            time.sleep(0.1)  # buff之间的间隔

        self._last_cast_time = time.time()
        logger.info(f"已释放 {len(self.buffs)} 个Buff")
        return True

    def _execute_buff(self, keys: List[str]) -> None:
        """执行Buff按键"""
        if len(keys) == 1:
            self.controller.key_press(keys[0])
        else:
            # 组合键
            for key in keys:
                self.controller.key_down(key)
            time.sleep(0.05)
            for key in reversed(keys):
                self.controller.key_up(key)

    def reset(self) -> None:
        """重置Buff状态"""
        self._last_cast_time = 0.0


# ============ 兼容旧版 SkillManager ============

class SkillConditionChecker:
    """技能条件检查器（保留兼容性）"""

    def __init__(self, context):
        self.context = context

    def check(self, condition_type: str, params: dict) -> bool:
        return True


class SkillManager:
    """技能管理器（兼容旧版接口）"""

    def __init__(
        self,
        character_config,
        context,
        controller
    ):
        self.character = character_config
        self.context = context
        self.controller = controller
        self.skills = {}
        self.condition_checker = SkillConditionChecker(context)
        self.strategy = "priority"

        # 新的技能队列
        self.skill_queue: Optional[SkillQueue] = None
        self.buff_manager: Optional[BuffManager] = None

    def update(self) -> None:
        """更新技能状态"""
        pass

    def use_next_available_skill(self) -> Optional[str]:
        """使用下一个可用技能"""
        if self.skill_queue:
            if self.skill_queue.use_next_skill():
                return "skill"
        return None

    def get_default_attack_key(self) -> str:
        return "x"

    def get_attack_type(self) -> str:
        return "melee"

    def get_attack_range(self) -> int:
        return 100

    def to_dict(self) -> dict:
        return {"character": self.character.name if self.character else "unknown"}
