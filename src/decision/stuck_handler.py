"""
卡住检测与分级恢复模块

检测信号：**正在命令移动，画面却几乎不动**。
- 帧差用「灰度 + 下采样 + 裁掉上下 HUD 带」后的逐像素平均绝对差(MAD)，
  下采样让小型 UI 动画被平均掉，而角色/背景真实滚动会产生大块空间一致的差异。
- **只在 movement.is_moving() 为真时累计** —— 这一条排除了过场、翻牌、暂停、
  静态菜单等"画面本来就不该动"的场景，是误判的主要来源。

恢复策略：探测 → 动作 → 复检 → 升级（四级确定性升级，不是随机游走）：

| 阶段 | 动作                | 意图                       |
|-----|---------------------|----------------------------|
| 0   | 反方向定时点按       | 脱离正压着的墙/角           |
| 1   | 换向点按(1.5x 参考)  | 障碍可能在另一侧            |
| 2   | 原方向点按 + 跳跃    | DNF 最常见的卡点是台阶/边缘  |
| 3   | 释放所有按键后静置   | 清掉可能卡住的按键状态       |

全程**非阻塞**：每级只发出命令并设定观察窗口，由主循环逐帧推进 `observe`/`step_recovery`，
没有任何 `time.sleep()`。复检依据就是同一路 `motion` 信号（窗口内画面恢复即成功）。
"""
import logging
import time
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StuckType(Enum):
    """卡住类型（保留原有取值，用于日志与阶段选择）"""
    DOOR_STUCK = "door_stuck"          # 卡门：在门/过渡状态下走不动
    PLAYER_STUCK = "player_stuck"      # 其他：空旷处走不动
    POSITION_STUCK = "position_stuck"  # 战斗中走不动


# 分级恢复的动作名，索引即阶段号
STAGE_NAMES = ("back_off", "alternate", "jump", "release")


def to_motion_sample(image: np.ndarray, sample_width: int = 160,
                     crop_top_ratio: float = 0.0,
                     crop_bottom_ratio: float = 0.0) -> np.ndarray:
    """把一帧压成用于帧差的小灰度图：裁掉上下 HUD 带后下采样。"""
    height = image.shape[0]
    top = int(height * crop_top_ratio)
    bottom = int(height * (1.0 - crop_bottom_ratio))
    if bottom <= top:
        top, bottom = 0, height

    roi = image[top:bottom]
    if roi.shape[2] == 4:
        roi = roi[:, :, :3]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    scale = sample_width / gray.shape[1]
    return cv2.resize(
        gray, (sample_width, max(1, int(gray.shape[0] * scale))),
        interpolation=cv2.INTER_AREA
    )


def frame_motion(prev_gray: Optional[np.ndarray],
                 curr_gray: Optional[np.ndarray]) -> float:
    """两张同尺寸灰度图之间的平均绝对差(0-255)。越大表示画面变化越大。"""
    if prev_gray is None or curr_gray is None:
        return 0.0
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))


def _opposite(direction: Optional[str]) -> str:
    return 'right' if direction == 'left' else 'left'


def _resolve(direction: Optional[str]) -> str:
    return direction if direction in ('left', 'right') else 'right'


class StuckHandler:
    """卡住检测 + 分级恢复处理器"""

    def __init__(self, config, controller, movement=None,
                 now: Callable[[], float] = time.monotonic):
        """
        Args:
            config: StuckRecoveryConfig
            controller: InputController
            movement: MovementController（可选；分级动作用它按角色速度换算出时长）
            now: 时钟，注入以便测试
        """
        self.config = config
        self.controller = controller
        self.movement = movement
        self._now = now

        # 检测状态
        self._prev_sample: Optional[np.ndarray] = None
        self._last_motion: float = 0.0
        self._stall_frames: int = 0
        self.stuck_type: Optional[StuckType] = None
        self._stuck_direction: Optional[str] = None

        # 恢复会话状态（StateMachine 没有时长概念，状态只能放在这里）
        self._session_start: Optional[float] = None
        self._stage: int = 0
        self._stage_deadline: float = 0.0
        self._stage_saw_motion: bool = False
        self._failed_sessions: int = 0
        self._cooldown_until: float = 0.0
        self._good_since: Optional[float] = None

    # ---------- 检测 ----------

    def observe(self, image: np.ndarray, context, state, movement=None) -> None:
        """每帧调用一次：更新帧差、累计卡住帧数、必要时置 stuck_detected。"""
        movement = movement or self.movement
        now = self._now()

        sample = to_motion_sample(
            image, self.config.sample_width,
            self.config.crop_top_ratio, self.config.crop_bottom_ratio
        )
        motion = frame_motion(self._prev_sample, sample)
        self._prev_sample = sample
        self._last_motion = motion
        logger.debug("stuck motion=%.2f stall=%d", motion, self._stall_frames)

        moving = bool(movement is not None and movement.is_moving())

        # 恢复窗口内只记录"画面是否恢复"，不做新的卡住判定
        if self._session_start is not None:
            if motion >= self.config.motion_threshold:
                self._stage_saw_motion = True
            return

        self._update_escalation_decay(moving, motion, now)

        if not moving or now < self._cooldown_until:
            self._stall_frames = 0
            context.set_custom_data('stuck_detected', False)
            return

        self._stall_frames = self._stall_frames + 1 if motion < self.config.motion_threshold else 0

        if self._stall_frames >= self.config.stuck_frames:
            self._stuck_direction = movement.get_direction()
            self.stuck_type = self._classify(context, state)
            # 触发后清零，避免状态机反应之前重复触发
            self._stall_frames = 0
            context.set_custom_data('stuck_return_state', state)
            context.set_custom_data('stuck_detected', True)
            logger.info("检测到卡住: type=%s dir=%s motion=%.2f",
                        self.stuck_type.value, self._stuck_direction, motion)
        else:
            context.set_custom_data('stuck_detected', False)

    def _update_escalation_decay(self, moving: bool, motion: float, now: float) -> None:
        """正常移动满 attempt_reset 秒后把升级级数降回 0，避免一次倒霉污染整轮。"""
        if not (moving and motion >= self.config.motion_threshold):
            self._good_since = None
            return
        if self._good_since is None:
            self._good_since = now
        if self._failed_sessions and now - self._good_since >= self.config.attempt_reset:
            self._failed_sessions = 0
            self._good_since = None

    def _classify(self, context, state) -> StuckType:
        """按当前状态与画面内容给卡住分类（三个枚举成员因此都有了活的产生者）。"""
        state_name = getattr(state, 'name', '')
        has_door = getattr(context, 'has_door', None)
        has_enemies = getattr(context, 'has_enemies', None)

        if state_name == 'TRANSITIONING' or (callable(has_door) and has_door()):
            return StuckType.DOOR_STUCK
        if state_name == 'COMBAT' or (callable(has_enemies) and has_enemies()):
            return StuckType.POSITION_STUCK
        return StuckType.PLAYER_STUCK

    # ---------- 分级恢复 ----------

    def step_recovery(self, context, movement=None) -> bool:
        """
        每帧推进一次分级恢复。

        Returns:
            True 表示本次恢复结束（成功或耗尽），调用方据此设置 recovery_done
        """
        movement = movement or self.movement
        now = self._now()

        if self._session_start is None:
            self._begin_session(movement, now)
            return False

        # 总超时兜底：阶段动作一直"看到一点动"但始终不收敛时强制结束
        if now - self._session_start >= self.config.recovery_timeout:
            return self._finish(context, movement, now, exhausted=True)

        if now < self._stage_deadline:
            return False

        if self._stage_saw_motion:
            return self._finish(context, movement, now, exhausted=False)

        self._stage += 1
        if self._stage >= self.config.max_stages:
            return self._finish(context, movement, now, exhausted=True)

        self._arm_stage(movement, now)
        return False

    def _begin_session(self, movement, now: float) -> None:
        self._session_start = now
        # 跨会话升级：惯犯直接从更高阶段起步
        self._stage = min(self._failed_sessions, self.config.max_stages - 1)
        self._stage_saw_motion = False
        self._failed_sessions += 1
        logger.info("开始卡住恢复: stage=%d dir=%s type=%s",
                    self._stage, self._stuck_direction,
                    self.stuck_type.value if self.stuck_type else None)
        self._arm_stage(movement, now)

    def _arm_stage(self, movement, now: float) -> None:
        """发出本阶段的动作并设定观察窗口。动作立即返回，不阻塞。"""
        stage = STAGE_NAMES[min(self._stage, len(STAGE_NAMES) - 1)]
        reference = self._reference_distance(movement)
        desired = _resolve(self._stuck_direction)

        if movement is not None:
            if stage == "back_off":
                movement.step(_opposite(desired), reference)
            elif stage == "alternate":
                movement.step(desired, reference * 1.5)
            elif stage == "jump":
                movement.step(desired, reference)
            else:  # release
                movement.stop()
                self.controller.release_all_inputs()

        if stage == "jump":
            self.controller.key_press(self.config.jump_key)

        hold = movement.hold_duration(reference) if movement is not None else 0.0
        self._stage_deadline = now + max(self.config.stage_duration, hold)
        self._stage_saw_motion = False

    def _reference_distance(self, movement) -> float:
        model = getattr(movement, 'model', None)
        return float(getattr(model, 'reference_distance', 150.0))

    def _finish(self, context, movement, now: float, exhausted: bool) -> bool:
        if movement is not None:
            movement.stop()
        self._session_start = None
        self._stage_saw_motion = False
        self._cooldown_until = now + self.config.recovery_cooldown

        if exhausted:
            logger.warning("卡住恢复失败(已用尽 %d 级): type=%s dir=%s",
                           self.config.max_stages,
                           self.stuck_type.value if self.stuck_type else None,
                           self._stuck_direction)
        else:
            self._failed_sessions = 0
            logger.info("卡住恢复成功: stage=%d", self._stage)

        context.set_custom_data('recovery_done', True)
        return True

    def end_session(self) -> None:
        """退出 STUCK_RECOVERY 时清理会话状态。"""
        self._session_start = None
        self._stage_saw_motion = False

    # ---------- 状态 ----------

    def reset(self) -> None:
        self._prev_sample = None
        self._last_motion = 0.0
        self._stall_frames = 0
        self.stuck_type = None
        self._stuck_direction = None
        self.end_session()
        self._failed_sessions = 0
        self._cooldown_until = 0.0
        self._good_since = None

    def get_status(self) -> dict:
        return {
            'last_motion': round(self._last_motion, 2),
            'stall_frames': self._stall_frames,
            'stuck_type': self.stuck_type.value if self.stuck_type else None,
            'in_recovery': self._session_start is not None,
            'stage': self._stage if self._session_start is not None else None,
            'failed_sessions': self._failed_sessions,
            'cooldown_remaining': max(0.0, self._cooldown_until - self._now()),
        }


def create_stuck_handler_from_config(config, controller, movement=None,
                                     now: Callable[[], float] = time.monotonic) -> StuckHandler:
    """
    从 StuckRecoveryConfig 直接创建处理器。

    不再像以前那样先用 dict 重新包一层 —— 那正是 StuckConfig 与 StuckRecoveryConfig
    两份重复配置的来源。
    """
    return StuckHandler(config=config, controller=controller, movement=movement, now=now)
