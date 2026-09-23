import types
import unittest
from unittest.mock import Mock, patch

from src.decision.game_context import GameContext, GameState
from test_engine_control import load_engine


class GameContextTimingTests(unittest.TestCase):
    def test_play_time_uses_elapsed_monotonic_time_not_assumed_fps(self):
        with patch('src.decision.game_context.time.monotonic', side_effect=(10.0, 10.4)):
            context = GameContext()
            context.update({})
        self.assertAlmostEqual(context.stats.play_time, 0.4)


class EngineLowSpecSafetyTests(unittest.TestCase):
    def make_hotkey_engine(self):
        GameEngine, _ = load_engine()
        engine = GameEngine.__new__(GameEngine)
        engine.dungeon_flow = None
        engine._hotkey_last_pressed = {}
        engine._paused = False
        engine._stop_requested = False
        engine.config = types.SimpleNamespace(
            hotkeys=types.SimpleNamespace(start='f1', pause='f2', stop='f3')
        )
        engine.pause = Mock()
        engine.resume = Mock()
        return engine

    def test_pause_hotkey_uses_pause_lifecycle(self):
        engine = self.make_hotkey_engine()
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._on_hotkey_press(types.SimpleNamespace(name='f2'))
        engine.pause.assert_called_once_with()

    def test_stop_hotkey_releases_inputs_through_pause_before_main_loop_stops(self):
        engine = self.make_hotkey_engine()
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._on_hotkey_press(types.SimpleNamespace(name='f3'))
        self.assertTrue(engine._stop_requested)
        engine.pause.assert_called_once_with()

    def test_menu_timeout_no_longer_triggers_character_switch(self):
        """
        回归守卫：通关提示（右上角「是否继续?」）停留超过 menu_timeout_seconds 曾经会触发
        切角色，但该提示每次通关都会出现 —— 只要没被及时关掉就会误切，刷图次数根本没到。
        现在超时不再触发切换，改由 _menu_action 按继续键推进 + ESC 自愈。
        """
        GameEngine, _ = load_engine()
        engine = GameEngine.__new__(GameEngine)
        engine.dungeon_flow = None
        engine.context = types.SimpleNamespace(
            has_menu=lambda: True,
            increment_menu_detect=Mock(),
            reset_menu_detect=Mock(),
            should_switch_character=lambda: False,
            dungeon_run_count=0,
            max_dungeon_runs=16,
        )
        engine.config = types.SimpleNamespace(
            dungeon=types.SimpleNamespace(menu_timeout_seconds=6.0)
        )
        engine._menu_detect_started_at = 4.0
        engine._execute_character_switch_flow = Mock()
        engine._playing_action = Mock()
        engine.controller = types.SimpleNamespace(key_press=Mock())
        engine.logger = types.SimpleNamespace(
            warning=Mock(), success=Mock(), info=Mock(), debug=Mock()
        )
        engine._running = True

        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._execute_action(GameState.PLAYING)

        engine._execute_character_switch_flow.assert_not_called()
        engine._playing_action.assert_called_once()

    def make_prompt_engine(self, attempts=0):
        GameEngine, _ = load_engine()
        engine = GameEngine.__new__(GameEngine)
        engine.dungeon_flow = None
        engine.config = types.SimpleNamespace(dungeon=types.SimpleNamespace(
            continue_key='f10', gather_key='tab', prompt_advance_delay=1.2,
            prompt_retry_interval=1.5, prompt_max_retries=3,
        ))
        engine.controller = types.SimpleNamespace(key_press=Mock())
        engine.context = types.SimpleNamespace(
            increment_dungeon_run=Mock(), dungeon_run_count=1, max_dungeon_runs=16
        )
        engine.logger = types.SimpleNamespace(info=Mock(), debug=Mock(), warning=Mock())
        engine._menu_detect_started_at = 0.0
        engine._prompt_attempts = attempts
        engine._prompt_last_press_at = None
        return engine

    def test_menu_action_continues_and_counts_the_run_once(self):
        engine = self.make_prompt_engine()

        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._menu_action()
        engine.controller.key_press.assert_called_once_with('f10')
        engine.context.increment_dungeon_run.assert_called_once_with()

        # 重试按继续键，但**不重复计数**
        engine.controller.key_press.reset_mock()
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=12.0):
            engine._menu_action()
        engine.controller.key_press.assert_called_once_with('f10')
        self.assertEqual(engine.context.increment_dungeon_run.call_count, 1)

    def test_menu_action_waits_for_gather_animation(self):
        engine = self.make_prompt_engine()
        # 刚进提示、聚集掉落还没走完 → 什么都不按
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=0.5):
            engine._menu_action()
        engine.controller.key_press.assert_not_called()

    def test_menu_action_falls_back_to_escape_after_max_retries(self):
        engine = self.make_prompt_engine(attempts=3)   # == prompt_max_retries
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._menu_action()
        engine.controller.key_press.assert_called_once_with('escape')
        self.assertEqual(engine._prompt_attempts, 0)   # 计数复位，重新计时


def GameEngineTime(engine):
    return engine._on_hotkey_press.__globals__['time']


if __name__ == '__main__':
    unittest.main()
