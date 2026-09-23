"""
地下城（白图）流程 —— 房间推进与 BOSS 房识别

完整流程：

    进图 → 往右走 → 打怪 → 找门 → 循环 → BOSS房 → 打BOSS
                                              ↓
                        翻牌 → tab聚物 → 再来一次 → 到次数/无法再来 → 切角色

本模块只负责**房间这一层**的状态：当前是在清怪、还是该往下一间推进、是不是 BOSS 房，
以及"该往哪走"（交给 `PathPlanner`）。通关提示之后的环节（翻牌/聚物/再来一次/切角色）
由 `engine._menu_action` 与切换流程负责。

与深渊流程的区别：深渊没有房间网格，白图是一间一间推进的 —— 这正是本模块存在的理由。
"""
import logging
from enum import Enum
from typing import Callable, Optional

from .path_planner import PathPlanner, RightwardPlanner

logger = logging.getLogger(__name__)


class RoomState(Enum):
    """当前房间的状态"""
    CLEARING = "clearing"     # 房间里还有怪，继续打
    ADVANCING = "advancing"   # 房间已清空，朝出口方向推进
    BOSS = "boss"             # BOSS 房（检测到 boss-m）


class DungeonFlow:
    """
    地下城房间推进。

    信号来源（当前模型能提供的）：
    - `monster` / `boss-m` 检测 → 是否有怪、是否 BOSS 房
    - 画面大幅变化 → 推进过程中切换到了新房间

    门与小地图暂不可用（原训练数据丢失、模型检不出，现有截图又是深渊场景没有小地图），
    所以"往哪走"由 `PathPlanner` 决定，默认 `RightwardPlanner` 向右。
    """

    def __init__(
        self,
        room_clear_frames: int = 8,
        planner: Optional[PathPlanner] = None,
        room_change_motion: float = 25.0,
        motion_sampler: Optional[Callable] = None,
        motion_diff: Optional[Callable] = None,
    ):
        """
        Args:
            room_clear_frames: 连续多少帧没有怪才算"本间清空"
            planner: 路径规划器，默认向右
            room_change_motion: 推进中帧差超过此值视为换了一间
            motion_sampler / motion_diff: 帧差实现（默认复用 stuck_handler 的通用实现）
        """
        self.room_clear_frames = room_clear_frames
        self.planner = planner or RightwardPlanner()
        self.room_change_motion = room_change_motion

        if motion_sampler is None or motion_diff is None:
            # 复用 stuck_handler 里的通用帧差工具，避免同一套数学写两遍
            from .stuck_handler import frame_motion, to_motion_sample
            motion_sampler = motion_sampler or to_motion_sample
            motion_diff = motion_diff or frame_motion
        self._sample = motion_sampler
        self._diff = motion_diff

        self.state = RoomState.CLEARING
        self.room_index = 0
        self.rooms_cleared = 0
        self.boss_rooms_seen = 0
        self._empty_frames = 0
        self._prev_sample = None

    # ---------- 每帧更新 ----------

    def observe(self, context, image=None) -> RoomState:
        """
        每帧调用一次，更新房间状态。

        Args:
            context: 游戏上下文（用 has_enemies / get_boss_detection 判断怪况）
            image: 当前帧，用于推进时检测房间切换；为 None 则跳过房间切换检测

        Returns:
            更新后的 RoomState
        """
        motion = self._update_motion(image)

        if self._has_boss(context):
            self._empty_frames = 0
            self.state = RoomState.BOSS
            return self.state

        if context.has_enemies():
            self._empty_frames = 0
            self.state = RoomState.CLEARING
            return self.state

        # 没有怪：累计空帧，够了就算本间清空
        self._empty_frames += 1
        if self._empty_frames < self.room_clear_frames:
            self.state = RoomState.CLEARING
            return self.state

        was_clearing = self.state is not RoomState.ADVANCING
        self.state = RoomState.ADVANCING
        if was_clearing:
            self.rooms_cleared += 1
            logger.info(f"第 {self.room_index} 间已清空，准备推进（共清 {self.rooms_cleared} 间）")
            # 刚进入推进态，把当前帧作为基准，避免把"清空瞬间"误判成换房间
            self._prev_sample = self._sample(image) if image is not None else self._prev_sample
            return self.state

        # 推进过程中画面大幅变化 → 进入新房间
        if motion is not None and motion >= self.room_change_motion:
            self._enter_next_room()
        return self.state

    def _update_motion(self, image) -> Optional[float]:
        """更新帧差并返回本次的 motion（image 为 None 时返回 None）。"""
        if image is None:
            return None
        sample = self._sample(image)
        motion = self._diff(self._prev_sample, sample) if self._prev_sample is not None else None
        self._prev_sample = sample
        return motion

    def _has_boss(self, context) -> bool:
        getter = getattr(context, 'get_boss_detection', None)
        if not callable(getter):
            return False
        if getter() is not None:
            self.boss_rooms_seen = max(self.boss_rooms_seen, 1)
            return True
        return False

    def _enter_next_room(self) -> None:
        self.room_index += 1
        self._empty_frames = 0
        self.state = RoomState.CLEARING
        logger.info(f"进入第 {self.room_index} 间")

    # ---------- 决策 ----------

    def advance_direction(self, context, image=None) -> str:
        """清完本间后该往哪走；小地图规划器需要当前帧 image。"""
        return self.planner.next_direction(context, self.room_index, image)

    def should_advance(self) -> bool:
        return self.state is RoomState.ADVANCING

    def is_boss_room(self) -> bool:
        return self.state is RoomState.BOSS

    # ---------- 生命周期 ----------

    def start_new_run(self) -> None:
        """开始新的一次刷图（按 F10 再来一次之后调用）。"""
        self.room_index = 0
        self.rooms_cleared = 0
        self.boss_rooms_seen = 0
        self._empty_frames = 0
        self._prev_sample = None
        self.state = RoomState.CLEARING

    def reset(self) -> None:
        """换角色/重进图时重置全部状态。"""
        self.start_new_run()

    def get_status(self) -> dict:
        return {
            'state': self.state.value,
            'room_index': self.room_index,
            'rooms_cleared': self.rooms_cleared,
            'planner': self.planner.describe(),
        }
