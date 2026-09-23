"""引擎移动调用点与角色速度接线的回归测试。

用鸭子类型的假 self 直接调用 GameEngine 的方法，避开桌面依赖：
movement / context / controller 全部替换为记录器。
"""
import types
import unittest

from src.core.engine import GameEngine


class FakeContext:
    """只实现被测代码用到的那几个查询。"""

    def __init__(self, enemy=None, in_range=False, direction='right',
                 door=None, at_door=False):
        self.enemy = enemy
        self.in_range = in_range
        self.direction = direction
        self.door = door
        self.at_door = at_door
        self.screen_center = (960, 540)
        self.movement_speed = 1.0

    def get_nearest_enemy(self):
        return self.enemy

    def get_enemy_direction(self, target, threshold=50):
        return self.direction

    def get_move_direction_to_target(self, target, threshold=50):
        return self.direction

    def is_enemy_in_attack_range(self, target, attack_range=100):
        return self.in_range

    def is_player_at_door(self, door, threshold=50):
        return self.at_door

    def get_door(self):
        return self.door

    def has_enemies(self):
        return self.enemy is not None

    def has_door(self):
        return self.door is not None

    def set_movement_speed(self, speed):
        self.movement_speed = speed


def enemy_at(x, class_name='monster'):
    return types.SimpleNamespace(center=(x, 540), class_name=class_name)


class RecordingMovement:
    """记录 approach/hold/step/stop 调用。"""

    def __init__(self):
        self.calls = []
        self.model = types.SimpleNamespace(reference_distance=150.0, move_speed=1.0)

    def approach(self, direction, distance_px):
        self.calls.append(('approach', direction, round(distance_px, 2)))

    def hold(self, direction):
        self.calls.append(('hold', direction))

    def step(self, direction, distance_px):
        self.calls.append(('step', direction, round(distance_px, 2)))
        return True

    def stop(self):
        self.calls.append(('stop',))

    def set_speed(self, **kwargs):
        self.calls.append(('set_speed', kwargs))


def make_engine(context, movement=None):
    """
    构造只带被测方法所需属性的假引擎。

    用 GameEngine.__new__ 而非 SimpleNamespace，这样类上的真实方法（如 _approach_target）
    仍然可解析；只覆盖真正需要替换的 _perform_attack。
    """
    attacks = []
    engine = GameEngine.__new__(GameEngine)
    engine.context = context
    engine.dungeon_flow = None
    engine.movement = movement or RecordingMovement()
    engine.logger = types.SimpleNamespace(
        debug=lambda *_: None, info=lambda *_: None, warning=lambda *_: None
    )
    engine.config = types.SimpleNamespace(side_scroller=types.SimpleNamespace(
        attack_range=150, attack_key='x', approach_threshold=150
    ))
    engine.current_character = None
    engine._last_enemy_direction = None
    engine._combat_attack_count = 0
    engine._max_blind_attacks = 10
    engine._perform_attack = lambda key: attacks.append(key)
    return engine, attacks


class CombatMovementTests(unittest.TestCase):
    def test_in_range_target_stops_and_attacks(self):
        context = FakeContext(enemy=enemy_at(1000), in_range=True)
        engine, attacks = make_engine(context)

        GameEngine._combat_action_side_scroller(engine)

        self.assertEqual(engine.movement.calls, [('stop',)])
        self.assertEqual(attacks, ['x'])

    def test_out_of_range_target_approaches_with_remaining_gap(self):
        context = FakeContext(enemy=enemy_at(1300), direction='right', in_range=False)
        engine, attacks = make_engine(context)

        GameEngine._combat_action_side_scroller(engine)

        self.assertEqual(engine.movement.calls, [('approach', 'right', 340.0)])
        self.assertEqual(attacks, [])

    def test_centered_but_not_in_range_picks_a_side(self):
        # 原实现在 direction == 'center' 时什么都不做，会永久停在原地
        context = FakeContext(enemy=enemy_at(900), direction='center', in_range=False)
        engine, _ = make_engine(context)

        GameEngine._combat_action_side_scroller(engine)

        self.assertEqual(engine.movement.calls, [('approach', 'left', 60.0)])

    def test_lost_target_blind_attacks_then_stops(self):
        context = FakeContext(enemy=None)
        engine, attacks = make_engine(context)
        engine._last_enemy_direction = 'right'
        engine._combat_attack_count = 0

        GameEngine._combat_action_side_scroller(engine)

        self.assertEqual(engine.movement.calls, [('stop',)])
        self.assertEqual(attacks, ['x'])
        self.assertEqual(engine._combat_attack_count, 1)

    def test_lost_target_exhausts_blind_attacks(self):
        context = FakeContext(enemy=None)
        engine, attacks = make_engine(context)
        engine._last_enemy_direction = 'right'
        engine._combat_attack_count = engine._max_blind_attacks

        GameEngine._combat_action_side_scroller(engine)

        self.assertEqual(engine.movement.calls, [('stop',)])
        self.assertEqual(attacks, [])
        self.assertIsNone(engine._last_enemy_direction)
        self.assertEqual(engine._combat_attack_count, 0)


class TransitioningMovementTests(unittest.TestCase):
    def test_not_at_door_approaches(self):
        context = FakeContext(door=enemy_at(1300), direction='right', at_door=False)
        engine, _ = make_engine(context)

        GameEngine._transitioning_action(engine)

        self.assertEqual(engine.movement.calls, [('approach', 'right', 340.0)])

    def test_at_door_stops(self):
        context = FakeContext(door=enemy_at(960), direction='center', at_door=True)
        engine, _ = make_engine(context)

        GameEngine._transitioning_action(engine)

        self.assertEqual(engine.movement.calls, [('stop',)])


class PlayingMovementTests(unittest.TestCase):
    def test_enemies_stop_movement(self):
        context = FakeContext(enemy=enemy_at(1000))
        engine, _ = make_engine(context)

        GameEngine._playing_action(engine)

        self.assertEqual(engine.movement.calls, [('stop',)])

    def test_door_approaches(self):
        context = FakeContext(door=enemy_at(1200), direction='right')
        engine, _ = make_engine(context)

        GameEngine._playing_action(engine)

        self.assertEqual(engine.movement.calls, [('approach', 'right', 240.0)])

    def test_no_enemy_no_door_searches_right(self):
        context = FakeContext()
        engine, _ = make_engine(context)

        GameEngine._playing_action(engine)

        self.assertEqual(engine.movement.calls, [('hold', 'right')])


class RoleMovementConfigTests(unittest.TestCase):
    def test_applies_role_speed_to_context_and_movement(self):
        context = FakeContext()
        movement = RecordingMovement()
        engine, _ = make_engine(context, movement)
        role = types.SimpleNamespace(move_speed=1.5, press_sleep=0.4, run_sleep=0.05)

        GameEngine._apply_role_movement_config(engine, role)

        self.assertEqual(context.movement_speed, 1.5)
        self.assertEqual(movement.calls, [('set_speed', {
            'move_speed': 1.5, 'press_sleep': 0.4, 'run_sleep': 0.05
        })])

    def test_missing_role_is_ignored(self):
        context = FakeContext()
        engine, _ = make_engine(context)

        GameEngine._apply_role_movement_config(engine, None)

        self.assertEqual(engine.movement.calls, [])
        self.assertEqual(context.movement_speed, 1.0)

    def test_role_without_speed_fields_defaults_to_baseline(self):
        # 老配置里没有这些字段：getattr 兜底，速度回落到 1.0
        context = FakeContext()
        engine, _ = make_engine(context)
        role = types.SimpleNamespace(id='old')

        GameEngine._apply_role_movement_config(engine, role)

        self.assertEqual(context.movement_speed, 1.0)
        self.assertEqual(engine.movement.calls, [('set_speed', {
            'move_speed': 1.0, 'press_sleep': None, 'run_sleep': None
        })])


if __name__ == '__main__':
    unittest.main()
