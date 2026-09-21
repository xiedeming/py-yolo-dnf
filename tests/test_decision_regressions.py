import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace


class DecisionImportTests(unittest.TestCase):
    def test_game_context_is_importable(self):
        from src.decision.game_context import GameContext

        self.assertIsNotNone(GameContext)


class StateMachineRegressionTests(unittest.TestCase):
    def test_critical_health_takes_priority_over_door_transition(self):
        from src.decision.game_context import GameContext, GameState
        from src.decision.state_machine import create_game_state_machine

        context = GameContext()
        context.current_detections.doors = [SimpleNamespace(center=(1100, 540))]
        context.current_detections.player_health = 0.0

        self.assertEqual(create_game_state_machine().update(context), GameState.DEAD)

    def test_critical_health_overrides_combat_exit_and_boss_combat(self):
        from src.decision.game_context import GameContext, GameState
        from src.decision.state_machine import create_game_state_machine

        for enemies in ([], [SimpleNamespace(center=(1000, 540), class_name='boss')]):
            with self.subTest(enemies=enemies):
                context = GameContext()
                context.current_detections.enemies = enemies
                context.current_detections.player_health = 0.0
                state_machine = create_game_state_machine()
                state_machine.force_state(GameState.COMBAT)

                self.assertEqual(state_machine.update(context), GameState.DEAD)


class DungeonRunnerRegressionTests(unittest.TestCase):
    def test_attack_target_uses_skill_manager_without_context_argument(self):
        from src.decision.game_context import GameState

        module_path = Path(__file__).resolve().parents[1] / 'src/core/dungeon_runner.py'
        spec = importlib.util.spec_from_file_location('dungeon_runner_under_test', module_path)
        dungeon_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dungeon_runner)
        DungeonRunner = dungeon_runner.DungeonRunner

        class SkillManager:
            def __init__(self):
                self.calls = 0

            def use_next_available_skill(self):
                self.calls += 1
                return 'skill'

        skill_manager = SkillManager()
        runner = DungeonRunner(engine=SimpleNamespace(skill_manager=skill_manager))
        context = SimpleNamespace(
            state=GameState.PLAYING,
            has_enemies=lambda: True,
            get_boss_detection=lambda: None,
            get_nearest_enemy=lambda: SimpleNamespace(center=(960, 540)),
            get_enemy_direction=lambda target: 'center',
            is_enemy_in_attack_range=lambda target: True,
        )

        runner._attack_target(context.get_nearest_enemy(), context)
        self.assertEqual(skill_manager.calls, 1)


if __name__ == '__main__':
    unittest.main()
