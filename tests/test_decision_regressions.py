import unittest
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


if __name__ == '__main__':
    unittest.main()
