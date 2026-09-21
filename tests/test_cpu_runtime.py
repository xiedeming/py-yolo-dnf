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

    def test_menu_timeout_uses_seconds_when_low_spec_profile_configures_it(self):
        GameEngine, _ = load_engine()
        engine = GameEngine.__new__(GameEngine)
        engine.context = types.SimpleNamespace(
            has_menu=lambda: True,
            increment_menu_detect=Mock(),
            reset_menu_detect=Mock(),
            is_menu_detect_timeout=Mock(return_value=False),
            should_switch_character=lambda: False,
            menu_detect_count=1,
            menu_detect_threshold=180,
        )
        engine.config = types.SimpleNamespace(dungeon=types.SimpleNamespace(menu_timeout_seconds=6.0))
        engine._menu_detect_started_at = 4.0
        engine.character_switcher = None
        engine.controller = types.SimpleNamespace(key_press=Mock())
        engine.logger = types.SimpleNamespace(warning=Mock(), success=Mock(), info=Mock())
        engine._running = True
        with patch.object(GameEngineTime(engine), 'monotonic', return_value=10.0):
            engine._execute_action(GameState.PLAYING)
        engine.controller.key_press.assert_called_once_with('tab')
        engine.context.reset_menu_detect.assert_called_once_with()


def GameEngineTime(engine):
    return engine._on_hotkey_press.__globals__['time']


if __name__ == '__main__':
    unittest.main()
