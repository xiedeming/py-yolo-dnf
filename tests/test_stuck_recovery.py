"""卡住检测与分级恢复的回归测试。

用**真实的小尺寸 numpy 帧**驱动检测（而不是注入假的帧差函数），这样裁剪与下采样
这条真实路径也被覆盖；时钟注入，测试里没有任何 time.sleep。
"""
import types
import unittest

import numpy as np

from src.decision.stuck_handler import (
    StuckHandler, StuckType, frame_motion, to_motion_sample,
)
from src.utils.config_loader import StuckRecoveryConfig


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class RecordingController:
    def __init__(self):
        self.pressed = []
        self.released = 0

    def key_press(self, key):
        self.pressed.append(key)

    def release_all_inputs(self):
        self.released += 1


class FakeMovement:
    """记录 step/stop，is_moving 可切换。"""

    def __init__(self, moving: bool = True, direction: str = 'right'):
        self.moving = moving
        self.direction = direction
        self.steps = []
        self.stops = 0
        self.model = types.SimpleNamespace(reference_distance=150.0, move_speed=1.0)

    def is_moving(self):
        return self.moving

    def get_direction(self):
        return self.direction if self.moving else None

    def step(self, direction, distance_px):
        self.steps.append((direction, round(float(distance_px), 1)))
        return True

    def stop(self):
        self.stops += 1
        self.moving = False

    def hold_duration(self, distance_px):
        return 0.2


class FakeContext:
    def __init__(self, has_door: bool = False, has_enemies: bool = False):
        self.data = {}
        self._has_door = has_door
        self._has_enemies = has_enemies

    def set_custom_data(self, key, value):
        self.data[key] = value

    def get_custom_data(self, key, default=None):
        return self.data.get(key, default)

    def has_door(self):
        return self._has_door

    def has_enemies(self):
        return self._has_enemies


def state(name: str = 'PLAYING'):
    return types.SimpleNamespace(name=name)


def static_frame(value: int = 0):
    return np.full((90, 160, 3), value, dtype=np.uint8)


def moving_frame(offset: int = 20):
    """带一块白色区域的帧；交替不同 offset 可保证连续两帧都在"动"。"""
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[40:60, offset:offset+20] = 255
    return frame


def make_handler(**overrides):
    clock = FakeClock()
    controller = RecordingController()
    movement = FakeMovement()
    config = StuckRecoveryConfig(**overrides)
    handler = StuckHandler(config, controller, movement=movement, now=clock)
    return handler, controller, movement, clock, config


def drive_to_stuck(handler, context, movement, frames=3):
    """喂足够的"不动"帧把 handler 推到触发状态。"""
    for _ in range(frames):
        handler.observe(static_frame(), context, state(), movement)


class FrameMotionTests(unittest.TestCase):
    def test_identical_frames_have_zero_motion(self):
        gray = to_motion_sample(static_frame(120))
        self.assertEqual(frame_motion(gray, gray.copy()), 0.0)

    def test_changed_content_has_positive_motion(self):
        a = to_motion_sample(static_frame(0))
        b = to_motion_sample(moving_frame())
        self.assertGreater(frame_motion(a, b), 0.0)

    def test_hud_crop_ignores_top_band(self):
        # 只有顶部 12% 不同：裁掉该带后应当完全无差异
        base = np.zeros((100, 160, 3), dtype=np.uint8)
        top_changed = base.copy()
        top_changed[:12] = 255

        cropped = to_motion_sample(top_changed, crop_top_ratio=0.12)
        uncropped = to_motion_sample(top_changed)
        base_cropped = to_motion_sample(base, crop_top_ratio=0.12)
        base_uncropped = to_motion_sample(base)

        self.assertEqual(frame_motion(base_cropped, cropped), 0.0)
        self.assertGreater(frame_motion(base_uncropped, uncropped), 0.0)


class DetectionTests(unittest.TestCase):
    def test_never_triggers_while_not_commanded_to_move(self):
        # 关键闸门：没在命令移动时，画面静止多久都不算卡住
        handler, _, movement, _, _ = make_handler(stuck_frames=3)
        movement.moving = False
        context = FakeContext()
        for _ in range(30):
            handler.observe(static_frame(), context, state(), movement)
        self.assertFalse(context.get_custom_data('stuck_detected', False))

    def test_triggers_after_configured_frames(self):
        handler, _, movement, _, _ = make_handler(stuck_frames=3)
        context = FakeContext()

        handler.observe(static_frame(), context, state(), movement)
        handler.observe(static_frame(), context, state(), movement)
        self.assertFalse(context.get_custom_data('stuck_detected', False))

        handler.observe(static_frame(), context, state(), movement)
        self.assertTrue(context.get_custom_data('stuck_detected', False))
        self.assertIs(context.get_custom_data('stuck_return_state').name, 'PLAYING')

    def test_motion_resets_the_stall_counter(self):
        handler, _, movement, _, _ = make_handler(stuck_frames=3)
        context = FakeContext()
        handler.observe(static_frame(), context, state(), movement)
        handler.observe(static_frame(), context, state(), movement)
        handler.observe(moving_frame(), context, state(), movement)   # 动了 → 清零
        handler.observe(static_frame(), context, state(), movement)
        self.assertFalse(context.get_custom_data('stuck_detected', False))

    def test_classification_by_state_and_content(self):
        for name, has_door, has_enemies, expected in (
            ('TRANSITIONING', False, False, StuckType.DOOR_STUCK),
            ('PLAYING', True, False, StuckType.DOOR_STUCK),
            ('COMBAT', False, False, StuckType.POSITION_STUCK),
            ('PLAYING', False, True, StuckType.POSITION_STUCK),
            ('PLAYING', False, False, StuckType.PLAYER_STUCK),
        ):
            with self.subTest(state=name, door=has_door, enemies=has_enemies):
                handler, _, movement, _, _ = make_handler(stuck_frames=1)
                context = FakeContext(has_door=has_door, has_enemies=has_enemies)
                handler.observe(static_frame(), context, state(name), movement)
                self.assertIs(handler.stuck_type, expected)


class EscalationTests(unittest.TestCase):
    def test_escalates_through_all_stages_then_gives_up(self):
        handler, controller, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()

        drive_to_stuck(handler, context, movement, frames=1)
        self.assertTrue(context.get_custom_data('stuck_detected', False))

        # 阶段 0：反方向脱离（卡住方向是 right → 反向 left）
        self.assertFalse(handler.step_recovery(context, movement))
        self.assertEqual(movement.steps, [('left', 150.0)])

        # 阶段 1：换向绕行，1.5 倍参考距离
        clock.advance(config.stage_duration + 0.01)
        self.assertFalse(handler.step_recovery(context, movement))
        self.assertEqual(movement.steps[-1], ('right', 225.0))

        # 阶段 2：原方向 + 跳跃
        clock.advance(config.stage_duration + 0.01)
        self.assertFalse(handler.step_recovery(context, movement))
        self.assertEqual(movement.steps[-1], ('right', 150.0))
        self.assertIn(config.jump_key, controller.pressed)

        # 阶段 3：释放按键静置
        clock.advance(config.stage_duration + 0.01)
        self.assertFalse(handler.step_recovery(context, movement))
        self.assertGreaterEqual(controller.released, 1)

        # 用尽 → 结束并置 recovery_done
        clock.advance(config.stage_duration + 0.01)
        with self.assertLogs('src.decision.stuck_handler', level='WARNING'):
            self.assertTrue(handler.step_recovery(context, movement))
        self.assertTrue(context.get_custom_data('recovery_done', False))
        # _failed_sessions 统计的是"连续失败的会话数"，每次会话只 +1
        self.assertEqual(handler._failed_sessions, 1)

    def test_recovers_as_soon_as_motion_returns(self):
        handler, controller, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()
        drive_to_stuck(handler, context, movement, frames=1)

        self.assertFalse(handler.step_recovery(context, movement))
        issued = list(movement.steps)

        # 观察窗口内画面恢复 → 复检通过
        handler.observe(moving_frame(), context, state(), movement)
        clock.advance(config.stage_duration + 0.01)
        self.assertTrue(handler.step_recovery(context, movement))

        self.assertTrue(context.get_custom_data('recovery_done', False))
        # 成功路径不该继续发出后续阶段的动作
        self.assertEqual(movement.steps, issued)
        self.assertEqual(handler._failed_sessions, 0)

    def test_cooldown_suppresses_retrigger(self):
        handler, _, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()
        drive_to_stuck(handler, context, movement, frames=1)
        handler.step_recovery(context, movement)
        clock.advance(config.stage_duration + 0.01)
        handler.observe(moving_frame(), context, state(), movement)
        handler.step_recovery(context, movement)
        self.assertTrue(context.get_custom_data('recovery_done', False))

        # 冷却期内即使一直不动也不该再触发
        clock.advance(config.recovery_cooldown - 0.01)
        for _ in range(5):
            handler.observe(static_frame(), context, state(), movement)
        self.assertFalse(context.get_custom_data('stuck_detected', False))

    def test_repeat_offender_starts_at_a_higher_stage(self):
        handler, _, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()

        # 先把一次会话跑到用尽
        drive_to_stuck(handler, context, movement, frames=1)
        for _ in range(config.max_stages + 1):
            handler.step_recovery(context, movement)
            clock.advance(config.stage_duration + 0.01)
        self.assertEqual(handler._failed_sessions, 1)

        # 冷却结束后再次卡住 → 第一次动作就来自更高阶段（不再是阶段 0 的 150px 反向）
        clock.advance(config.recovery_cooldown + 0.01)
        movement.moving = True
        movement.steps.clear()
        drive_to_stuck(handler, context, movement, frames=1)
        handler.step_recovery(context, movement)
        self.assertNotEqual(movement.steps[0], ('left', 150.0))

    def test_timeout_forces_finish(self):
        handler, _, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()
        drive_to_stuck(handler, context, movement, frames=1)
        handler.step_recovery(context, movement)

        clock.advance(config.recovery_timeout + 0.01)
        self.assertTrue(handler.step_recovery(context, movement))
        self.assertTrue(context.get_custom_data('recovery_done', False))

    def test_escalation_decays_after_sustained_good_movement(self):
        handler, _, movement, clock, config = make_handler(stuck_frames=1)
        context = FakeContext()
        drive_to_stuck(handler, context, movement, frames=1)
        for _ in range(config.max_stages + 1):
            handler.step_recovery(context, movement)
            clock.advance(config.stage_duration + 0.01)
        self.assertGreater(handler._failed_sessions, 0)

        # 持续正常移动达到 attempt_reset 后，升级级数衰减回 0
        # （阶段 3 会 stop()，真实循环里下一帧的移动指令会把 is_moving 重新置真）
        clock.advance(config.recovery_cooldown + 0.01)
        movement.moving = True
        # 交替两张不同的帧，否则连续喂同一帧会让帧差变成 0
        for i in range(3):
            handler.observe(moving_frame(20 if i % 2 == 0 else 60), context, state(), movement)
            clock.advance(config.attempt_reset)
        handler.observe(moving_frame(20), context, state(), movement)
        self.assertEqual(handler._failed_sessions, 0)

    def test_end_session_clears_in_recovery_flag(self):
        handler, _, movement, _, _ = make_handler(stuck_frames=1)
        context = FakeContext()
        drive_to_stuck(handler, context, movement, frames=1)
        handler.step_recovery(context, movement)
        self.assertTrue(handler.get_status()['in_recovery'])

        handler.end_session()
        self.assertFalse(handler.get_status()['in_recovery'])


if __name__ == '__main__':
    unittest.main()
