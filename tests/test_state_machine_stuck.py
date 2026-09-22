"""卡住恢复相关的状态机转移回归测试。

沿用 tests/test_decision_regressions.py 的写法：真实 GameContext + 真实状态机，
不 mock 模块。
"""
import unittest
from types import SimpleNamespace

from src.decision.game_context import GameContext, GameState
from src.decision.state_machine import create_game_state_machine

MOVEMENT_STATES = (GameState.PLAYING, GameState.COMBAT, GameState.TRANSITIONING)


class StuckEntryTests(unittest.TestCase):
    def test_all_movement_states_can_enter_stuck_recovery(self):
        # 实际最常见的卡住是"走向门走不动"(TRANSITIONING) 和"战斗中被卡角落"(COMBAT)，
        # 原实现只有 PLAYING 能进入
        for source in MOVEMENT_STATES:
            with self.subTest(source=source):
                sm = create_game_state_machine()
                sm.force_state(source)
                context = GameContext()
                context.set_custom_data('stuck_detected', True)
                sm.update(context)
                self.assertEqual(sm.get_state(), GameState.STUCK_RECOVERY)

    def test_menu_outranks_stuck_recovery(self):
        sm = create_game_state_machine()
        sm.force_state(GameState.PLAYING)
        context = GameContext()
        context.set_custom_data('stuck_detected', True)
        context.has_menu = lambda: True
        sm.update(context)
        self.assertEqual(sm.get_state(), GameState.MENU)

    def test_no_stuck_flag_keeps_normal_flow(self):
        sm = create_game_state_machine()
        sm.force_state(GameState.PLAYING)
        context = GameContext()
        sm.update(context)
        self.assertNotEqual(sm.get_state(), GameState.STUCK_RECOVERY)


class StuckExitTests(unittest.TestCase):
    def test_returns_to_the_state_it_came_from(self):
        for target in MOVEMENT_STATES:
            with self.subTest(target=target):
                sm = create_game_state_machine()
                sm.force_state(GameState.STUCK_RECOVERY)
                context = GameContext()
                context.set_custom_data('recovery_done', True)
                context.set_custom_data('stuck_return_state', target)
                sm.update(context)
                self.assertEqual(sm.get_state(), target)

    def test_stays_in_recovery_until_done(self):
        sm = create_game_state_machine()
        sm.force_state(GameState.STUCK_RECOVERY)
        context = GameContext()
        context.set_custom_data('stuck_return_state', GameState.PLAYING)
        sm.update(context)
        self.assertEqual(sm.get_state(), GameState.STUCK_RECOVERY)


class TransitioningExitTests(unittest.TestCase):
    def test_no_door_falls_back_to_playing(self):
        # 补这个出口是为了避免：门消失后永久卡在 TRANSITIONING
        # （解卡恰好可能把状态返回到 TRANSITIONING）
        sm = create_game_state_machine()
        sm.force_state(GameState.TRANSITIONING)
        context = GameContext()
        context.has_door = lambda: False
        sm.update(context)
        self.assertEqual(sm.get_state(), GameState.PLAYING)

    def test_at_door_still_loads(self):
        sm = create_game_state_machine()
        sm.force_state(GameState.TRANSITIONING)
        context = GameContext()
        context.set_screen_center(1920, 1080)
        context.has_door = lambda: True
        context.get_door = lambda: SimpleNamespace(center=(960, 540))
        sm.update(context)
        self.assertEqual(sm.get_state(), GameState.LOADING)

    def test_menu_outranks_transitioning_fallback(self):
        sm = create_game_state_machine()
        sm.force_state(GameState.TRANSITIONING)
        context = GameContext()
        context.has_menu = lambda: True
        context.has_door = lambda: False
        sm.update(context)
        self.assertEqual(sm.get_state(), GameState.MENU)


if __name__ == '__main__':
    unittest.main()
