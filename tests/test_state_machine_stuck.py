"""卡住恢复相关的状态机转移回归测试。

沿用 tests/test_decision_regressions.py 的写法：真实 GameContext + 真实状态机，
不 mock 模块。
"""
import inspect
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


class EngineStuckExitWiringTests(unittest.TestCase):
    """引擎把退出回调注册进状态机这条路径。

    test_engine_control 把状态机整个替换成 None，本文件其余用例只用裸状态机，
    两者都覆盖不到"GameEngine 注册的 on_exit 回调被状态机以 (context) 调用"——
    `_on_exit_stuck_recovery` 曾漏掉 context 参数，退出恢复时抛 TypeError，
    而异常发生在 current_state 赋值之前，状态机于是永久卡在 STUCK_RECOVERY，
    每帧重复失败。
    """

    def _build_engine(self):
        from src.core.engine import GameEngine

        engine = GameEngine.__new__(GameEngine)
        engine.movement = None
        engine.stuck_handler = None
        engine.context = GameContext()
        engine.state_machine = create_game_state_machine()
        # 与 GameEngine.__init__ 中一致的注册方式
        engine.state_machine.set_exit_action(
            GameState.STUCK_RECOVERY, engine._on_exit_stuck_recovery
        )
        return engine

    def test_exit_callback_takes_a_context_argument(self):
        from src.core.engine import GameEngine

        parameters = list(inspect.signature(GameEngine._on_exit_stuck_recovery).parameters)
        self.assertEqual(parameters, ['self', 'context'])

    def test_exiting_recovery_returns_to_previous_state_and_clears_flags(self):
        engine = self._build_engine()
        context = engine.context
        engine.state_machine.force_state(GameState.STUCK_RECOVERY)
        context.set_custom_data('stuck_detected', False)
        context.set_custom_data('recovery_done', True)
        context.set_custom_data('stuck_return_state', GameState.PLAYING)

        engine.state_machine.update(context)

        self.assertEqual(engine.state_machine.get_state(), GameState.PLAYING)
        self.assertFalse(context.get_custom_data('recovery_done', True))
        self.assertFalse(context.get_custom_data('stuck_detected', True))


if __name__ == '__main__':
    unittest.main()
