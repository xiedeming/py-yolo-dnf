"""按角色移动速度的回归测试。

覆盖 `MoveSpeedModel` 的像素→时长换算，以及 `MovementController` 的非阻塞点按
（时钟注入 —— 这些用例里没有任何 time.sleep，`update()` 由假时钟驱动）。
"""
import unittest

from src.control.movement_controller import MovementController, MoveSpeedModel


class FakeClock:
    """可推进的假时钟，替代 time.monotonic。"""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class RecordingController:
    """记录按键调用，不产生真实输入。"""

    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(('down', key))

    def key_up(self, key):
        self.events.append(('up', key))


def make(dash_gap: float = 0.0, **model_kwargs):
    clock = FakeClock()
    controller = RecordingController()
    movement = MovementController(
        controller, MoveSpeedModel(**model_kwargs), now=clock, dash_gap=dash_gap
    )
    return movement, controller, clock


class MoveSpeedModelTests(unittest.TestCase):
    def test_hold_duration_scales_inversely_with_speed(self):
        # 参考距离 150px、press_sleep 0.55s：速度越快，同样的距离按得越短
        for speed, expected in ((1.0, 0.55), (1.5, 0.366667), (2.0, 0.275)):
            with self.subTest(speed=speed):
                model = MoveSpeedModel(move_speed=speed)
                self.assertAlmostEqual(model.hold_duration(150), expected, places=5)

    def test_hold_duration_clamps_to_bounds(self):
        model = MoveSpeedModel()
        # 距离为 0 → 0（不产生空点按）
        self.assertEqual(model.hold_duration(0), 0.0)
        self.assertEqual(model.hold_duration(-5), 0.0)
        # 极小距离 → 钳到 run_sleep
        self.assertAlmostEqual(model.hold_duration(1), model.run_sleep, places=6)
        # 极大距离 → 钳到 max_hold
        self.assertAlmostEqual(model.hold_duration(10 ** 6), model.max_hold, places=6)

    def test_clamps_are_applied_before_speed_division(self):
        # 钳制以"基准角色"为单位，所以慢/快角色拿到的边界值同样按速度缩放
        fast = MoveSpeedModel(move_speed=2.0)
        self.assertAlmostEqual(fast.hold_duration(10 ** 6), fast.max_hold / 2.0, places=6)
        self.assertAlmostEqual(fast.hold_duration(1), fast.run_sleep / 2.0, places=6)

    def test_scale_threshold_grows_with_speed(self):
        self.assertEqual(MoveSpeedModel(move_speed=1.0).scale_threshold(50), 50)
        self.assertEqual(MoveSpeedModel(move_speed=1.5).scale_threshold(50), 75.0)

    def test_needs_hold_only_for_long_distances(self):
        model = MoveSpeedModel()
        self.assertFalse(model.needs_hold(100))
        self.assertTrue(model.needs_hold(400))


class MovementControllerTests(unittest.TestCase):
    def test_custom_direction_keys_are_used(self):
        # 方向键来自配置，不再硬编码 'left'/'right'
        clock = FakeClock()
        controller = RecordingController()
        movement = MovementController(
            controller, MoveSpeedModel(), left_key='a', right_key='d',
            now=clock, dash_gap=0.0,
        )
        movement.step('left', 100)
        self.assertEqual(controller.events, [('down', 'a')])

    def test_step_does_not_dash(self):
        # dash_gap 非 0 时 step 也不该走双击宏（双击只属于跑动），且这里不会真 sleep
        movement, controller, _ = make(dash_gap=0.05)
        self.assertTrue(movement.step('right', 150))
        self.assertEqual(controller.events, [('down', 'right')])

    def test_step_is_released_by_update_not_by_sleeping(self):
        movement, controller, clock = make()
        duration = movement.hold_duration(150)

        self.assertTrue(movement.step('right', 150))
        self.assertTrue(movement.is_stepping())
        self.assertTrue(movement.is_moving())

        # 到期前不释放 —— 说明 step 没有阻塞等待
        clock.advance(duration - 0.001)
        movement.update()
        self.assertEqual(controller.events, [('down', 'right')])
        self.assertTrue(movement.is_moving())

        # 到期后由主循环的 update() 释放
        clock.advance(0.002)
        movement.update()
        self.assertEqual(controller.events, [('down', 'right'), ('up', 'right')])
        self.assertFalse(movement.is_moving())
        self.assertFalse(movement.is_stepping())

    def test_step_rejects_zero_distance(self):
        movement, controller, _ = make()
        self.assertFalse(movement.step('left', 0))
        self.assertEqual(controller.events, [])

    def test_hold_uses_dash_macro(self):
        # 跑动 = 双击后长按（与 InputController.start_moving(run=True) 一致）
        movement, controller, _ = make()
        movement.hold('left')
        self.assertEqual(
            controller.events, [('down', 'left'), ('up', 'left'), ('down', 'left')]
        )
        self.assertTrue(movement.is_moving())
        self.assertFalse(movement.is_stepping())

    def test_hold_same_direction_is_noop(self):
        movement, controller, _ = make()
        movement.hold('left')
        before = list(controller.events)
        movement.hold('left')
        self.assertEqual(controller.events, before)

    def test_direction_change_releases_previous(self):
        movement, controller, _ = make()
        movement.hold('left')
        movement.hold('right')
        # 换向必须先松开旧方向，再对"已按住"的新方向做一次双击
        self.assertIn(('up', 'left'), controller.events)
        self.assertEqual(movement.get_direction(), 'right')

    def test_approach_holds_when_far_and_steps_when_near(self):
        far, far_controller, _ = make()
        far.approach('right', 400)          # raw 0.55*400/150 ≈ 1.47 > max_hold
        self.assertFalse(far.is_stepping())
        self.assertEqual(len(far_controller.events), 3)   # 双击宏

        near, near_controller, clock = make()
        near.approach('right', 100)
        self.assertTrue(near.is_stepping())
        self.assertEqual(near_controller.events, [('down', 'right')])

    def test_stop_releases_and_clears(self):
        movement, controller, _ = make()
        movement.step('left', 150)
        movement.stop()
        self.assertEqual(controller.events, [('down', 'left'), ('up', 'left')])
        self.assertFalse(movement.is_moving())
        self.assertIsNone(movement.get_direction())

    def test_set_speed_ignores_invalid_values(self):
        movement, _, _ = make()
        movement.set_speed(0)
        self.assertEqual(movement.model.move_speed, 1.0)
        movement.set_speed(-2)
        self.assertEqual(movement.model.move_speed, 1.0)
        movement.set_speed(1.5, press_sleep=0.4)
        self.assertEqual(movement.model.move_speed, 1.5)
        self.assertEqual(movement.model.press_sleep, 0.4)
        # 非正值不覆盖已有基准时长
        movement.set_speed(1.5, press_sleep=0)
        self.assertEqual(movement.model.press_sleep, 0.4)

    def test_set_speed_changes_hold_duration(self):
        movement, _, _ = make()
        before = movement.hold_duration(150)
        movement.set_speed(2.0)
        self.assertAlmostEqual(movement.hold_duration(150), before / 2.0, places=6)


if __name__ == '__main__':
    unittest.main()
