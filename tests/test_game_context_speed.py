"""GameContext 按角色速度缩放像素阈值的回归测试。

关键区分：**移动相关的判定要缩放**（方向死区、是否到达门），
**攻击范围不能缩放**（它是攻击本身的属性）。
"""
import unittest
from types import SimpleNamespace

from src.decision.game_context import GameContext


def target_at(x: int):
    return SimpleNamespace(center=(x, 540))


class GameContextSpeedTests(unittest.TestCase):
    def setUp(self):
        self.context = GameContext()
        self.context.set_screen_center(1920, 1080)   # screen_center = (960, 540)

    def test_direction_dead_zone_scales_with_speed(self):
        # 偏移 60px：基准速度下判定为 right；1.5 倍速度下死区变成 75px，落在死区内
        for speed, expected in ((1.0, 'right'), (1.5, 'center')):
            with self.subTest(speed=speed):
                self.context.set_movement_speed(speed)
                self.assertEqual(
                    self.context.get_enemy_direction(target_at(1020), 50), expected
                )

    def test_move_direction_to_target_inherits_scaling(self):
        self.context.set_movement_speed(1.5)
        self.assertEqual(
            self.context.get_move_direction_to_target(target_at(1020), 50), 'center'
        )
        self.context.set_movement_speed(1.0)
        self.assertEqual(
            self.context.get_move_direction_to_target(target_at(1020), 50), 'right'
        )

    def test_door_threshold_scales_with_speed(self):
        door = target_at(1020)   # 距中心 60px
        self.context.set_movement_speed(1.0)
        self.assertFalse(self.context.is_player_at_door(door, 50))
        self.context.set_movement_speed(1.5)
        self.assertTrue(self.context.is_player_at_door(door, 50))

    def test_attack_range_is_not_scaled(self):
        # 回归守卫：攻击范围是攻击的属性，缩放它会让快角色停在自己够不到的位置
        enemy = target_at(1140)   # 距中心 180px
        for speed in (1.0, 2.0):
            with self.subTest(speed=speed):
                self.context.set_movement_speed(speed)
                self.assertFalse(self.context.is_enemy_in_attack_range(enemy, 100))
                self.assertTrue(self.context.is_enemy_in_attack_range(enemy, 200))

    def test_set_movement_speed_rejects_invalid(self):
        self.context.set_movement_speed(0)
        self.assertEqual(self.context.movement_speed, 1.0)
        self.context.set_movement_speed(None)
        self.assertEqual(self.context.movement_speed, 1.0)
        self.context.set_movement_speed(1.7)
        self.assertEqual(self.context.movement_speed, 1.7)

    def test_default_speed_preserves_previous_behaviour(self):
        # move_speed 默认 1.0 → 与加缩放之前逐像素一致
        self.assertEqual(self.context.movement_speed, 1.0)
        self.assertEqual(self.context.get_enemy_direction(target_at(1000), 50), 'center')
        self.assertEqual(self.context.get_enemy_direction(target_at(1020), 50), 'right')
        self.assertEqual(self.context.get_enemy_direction(target_at(900), 50), 'left')


if __name__ == '__main__':
    unittest.main()
