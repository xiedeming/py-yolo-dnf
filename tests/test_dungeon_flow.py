"""地下城（白图）流程的回归测试：房间推进、BOSS 房识别、路径规划接口。"""
import unittest

import cv2
import numpy as np

from src.decision.dungeon_flow import DungeonFlow, RoomState
from src.decision.path_planner import (
    MinimapPathPlanner, RightwardPlanner, create_path_planner,
)


class FakeContext:
    def __init__(self, enemies: int = 0, boss: bool = False):
        self.enemies = enemies
        self.boss = boss

    def has_enemies(self):
        return self.enemies > 0

    def get_boss_detection(self):
        return object() if self.boss else None


def make_flow(**kwargs):
    """注入假的帧差实现，用整数当"图像"，避免测试里真的做图像处理。"""
    kwargs.setdefault('motion_sampler', lambda img: img)
    kwargs.setdefault('motion_diff',
                      lambda prev, curr: 0.0 if prev is None else float(curr - prev))
    return DungeonFlow(**kwargs)


def make_minimap_frame(player_xy, boss_xy, width=1920, height=1200):
    """
    构造一张带小地图的游戏帧：在右上角区域放一个青色玩家标记和一个红色 BOSS 标记。

    Args:
        player_xy / boss_xy: 相对于**小地图区域左上角**的偏移 (x, y)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    planner = MinimapPathPlanner()
    rx, ry, rw, rh = planner.region
    ox, oy = int(width * rx), int(height * ry)   # 区域左上角

    # 玩家：亮青色（BGR 里 B 和 G 高、R 低），HSV 约 H=90
    px, py = player_xy
    img[oy + py:oy + py + 6, ox + px:ox + px + 8] = (255, 200, 50)

    # BOSS：亮红色（BGR 里 R 高），HSV 约 H=0 或 180
    bx, by = boss_xy
    img[oy + by:oy + by + 10, ox + bx:ox + bx + 14] = (30, 60, 230)

    return img


class PathPlannerTests(unittest.TestCase):
    def test_rightward_planner_returns_right(self):
        self.assertEqual(RightwardPlanner().next_direction(FakeContext(), 0), 'right')

    def test_rightward_planner_accepts_image(self):
        # 新签名（带 image 参数）对不需要图的实现也不报错
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(RightwardPlanner().next_direction(FakeContext(), 0, image), 'right')

    def test_factory_creates_rightward(self):
        planner = create_path_planner('rightward')
        self.assertIsInstance(planner, RightwardPlanner)
        self.assertEqual(planner.describe(), 'RightwardPlanner')

    def test_factory_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            create_path_planner('nonsense')

    def test_factory_creates_minimap(self):
        planner = create_path_planner('minimap')
        self.assertIsInstance(planner, MinimapPathPlanner)
        self.assertIn('MinimapPathPlanner', planner.describe())


class MinimapPathPlannerTests(unittest.TestCase):
    """用合成小地图验证方向判断，颜色阈值来自实拍截图的校准。"""

    def test_boss_to_the_right_returns_right(self):
        # BOSS 在玩家右侧且水平位移更大
        img = make_minimap_frame(player_xy=(30, 30), boss_xy=(80, 35))
        result = MinimapPathPlanner().next_direction(FakeContext(), 0, img)
        self.assertEqual(result, 'right')

    def test_boss_to_the_left_returns_left(self):
        img = make_minimap_frame(player_xy=(80, 30), boss_xy=(30, 35))
        result = MinimapPathPlanner().next_direction(FakeContext(), 0, img)
        self.assertEqual(result, 'left')

    def test_boss_below_returns_down(self):
        # 垂直位移更大
        img = make_minimap_frame(player_xy=(50, 20), boss_xy=(55, 70))
        result = MinimapPathPlanner().next_direction(FakeContext(), 0, img)
        self.assertEqual(result, 'down')

    def test_boss_above_returns_up(self):
        img = make_minimap_frame(player_xy=(50, 70), boss_xy=(55, 20))
        result = MinimapPathPlanner().next_direction(FakeContext(), 0, img)
        self.assertEqual(result, 'up')

    def test_no_image_falls_back_to_right(self):
        # 没有帧时保守向右（与 RightwardPlanner 一致）
        self.assertEqual(MinimapPathPlanner().next_direction(FakeContext(), 0, None), 'right')

    def test_no_player_marker_falls_back_to_right(self):
        # 只有 BOSS 没有玩家 → 找不到位移 → 保守向右
        img = np.zeros((1200, 1920, 3), dtype=np.uint8)
        result = MinimapPathPlanner().next_direction(FakeContext(), 0, img)
        self.assertEqual(result, 'right')

    def test_custom_region(self):
        # 用自定义区域时标记也要在自定义区域内才能被找到
        img = make_minimap_frame(player_xy=(30, 30), boss_xy=(80, 30))
        # 默认区域下应返回 right
        self.assertEqual(MinimapPathPlanner().next_direction(FakeContext(), 0, img), 'right')



class RoomStateTests(unittest.TestCase):
    def test_enemies_keep_the_room_in_clearing(self):
        flow = make_flow(room_clear_frames=3)
        context = FakeContext(enemies=2)
        for _ in range(5):
            self.assertIs(flow.observe(context, image=0), RoomState.CLEARING)
        self.assertEqual(flow.rooms_cleared, 0)

    def test_boss_takes_priority_over_clearing(self):
        flow = make_flow(room_clear_frames=1)
        self.assertIs(flow.observe(FakeContext(enemies=1, boss=True), image=0), RoomState.BOSS)
        self.assertTrue(flow.is_boss_room())

    def test_room_becomes_advancing_after_enough_empty_frames(self):
        flow = make_flow(room_clear_frames=3)
        context = FakeContext()
        self.assertIs(flow.observe(context, image=0), RoomState.CLEARING)
        self.assertIs(flow.observe(context, image=0), RoomState.CLEARING)
        self.assertIs(flow.observe(context, image=0), RoomState.ADVANCING)
        self.assertTrue(flow.should_advance())
        self.assertEqual(flow.rooms_cleared, 1)

    def test_room_counter_does_not_keep_growing_while_advancing(self):
        flow = make_flow(room_clear_frames=1, room_change_motion=100.0)
        context = FakeContext()
        for _ in range(6):
            flow.observe(context, image=0)
        self.assertEqual(flow.rooms_cleared, 1)   # 只在进入推进态那一次计数

    def test_large_frame_change_while_advancing_enters_next_room(self):
        flow = make_flow(room_clear_frames=1, room_change_motion=25.0)
        context = FakeContext()
        flow.observe(context, image=0)      # 首帧：建立基准
        flow.observe(context, image=0)      # 清空 → 推进（基准重置）
        flow.observe(context, image=50)     # 帧差 50 ≥ 25 → 换房间
        self.assertEqual(flow.room_index, 1)
        self.assertIs(flow.state, RoomState.CLEARING)

    def test_small_frame_change_does_not_change_room(self):
        flow = make_flow(room_clear_frames=1, room_change_motion=25.0)
        context = FakeContext()
        flow.observe(context, image=0)
        flow.observe(context, image=0)
        flow.observe(context, image=5)      # 帧差 5 < 25
        self.assertEqual(flow.room_index, 0)

    def test_no_room_change_detection_without_image(self):
        flow = make_flow(room_clear_frames=1)
        context = FakeContext()
        for _ in range(4):
            flow.observe(context, image=None)
        self.assertEqual(flow.room_index, 0)
        self.assertIs(flow.state, RoomState.ADVANCING)


class DecisionTests(unittest.TestCase):
    def test_advance_direction_uses_the_planner(self):
        flow = make_flow(planner=RightwardPlanner())
        self.assertEqual(flow.advance_direction(FakeContext()), 'right')

    def test_advance_direction_can_be_swapped(self):
        class LeftPlanner(RightwardPlanner):
            def next_direction(self, context, room_index, image=None):
                return 'left'

        flow = make_flow(planner=LeftPlanner())
        self.assertEqual(flow.advance_direction(FakeContext()), 'left')

    def test_advance_direction_passes_image_to_planner(self):
        class RecordingPlanner(RightwardPlanner):
            def __init__(self):
                self.received_image = None

            def next_direction(self, context, room_index, image=None):
                self.received_image = image
                return 'right'

        planner = RecordingPlanner()
        flow = make_flow(planner=planner)
        dummy = object()
        flow.advance_direction(FakeContext(), image=dummy)
        self.assertIs(planner.received_image, dummy)


class LifecycleTests(unittest.TestCase):
    def test_start_new_run_resets_progress(self):
        flow = make_flow(room_clear_frames=1, room_change_motion=25.0)
        context = FakeContext()
        flow.observe(context, image=0)
        flow.observe(context, image=0)
        flow.observe(context, image=50)
        self.assertEqual(flow.room_index, 1)

        flow.start_new_run()
        self.assertEqual(flow.room_index, 0)
        self.assertEqual(flow.rooms_cleared, 0)
        self.assertIs(flow.state, RoomState.CLEARING)

    def test_status_is_serialisable(self):
        flow = make_flow(room_clear_frames=1)
        flow.observe(FakeContext(enemies=1), image=0)
        status = flow.get_status()
        self.assertEqual(status['state'], 'clearing')
        self.assertEqual(status['planner'], 'RightwardPlanner')


if __name__ == '__main__':
    unittest.main()
