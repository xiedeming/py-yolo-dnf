"""遇到 monster / boss-m 时按目标类型释放技能的回归测试。

直接调用 GameEngine._cast_skills_for_target，用鸭子类型的假 self 避开
桌面和 ML 依赖，只有技能队列和输入控制器是真的（控制器被替换为记录器）。
"""
import logging
import types
import unittest

from src.core.engine import GameEngine
from src.decision.skill_manager import SkillQueue


class RecordingController:
    """记录按键调用，不产生真实输入。"""

    def __init__(self):
        self.pressed = []

    def key_press(self, key):
        self.pressed.append(key)

    def key_down(self, key):
        self.pressed.append(("down", key))

    def key_up(self, key):
        self.pressed.append(("up", key))


class FakeTarget:
    def __init__(self, class_name):
        self.class_name = class_name


def make_engine(skills=(("q", 10.0), ("w", 10.0), ("e", 10.0)),
                count_by_class=None, interval=1.0, use_skills=True,
                with_queue=True):
    """构造只带 _cast_skills_for_target 所需属性的假引擎。"""
    controller = RecordingController()
    queue = None
    if with_queue:
        queue = SkillQueue(controller)
        queue.load_skills([[key] for key, _ in skills],
                          {key: cd for key, cd in skills})

    combat = types.SimpleNamespace(
        use_skills=use_skills,
        skill_count_by_class=({"monster": 1, "boss-m": 2}
                              if count_by_class is None else count_by_class),
        skill_trigger_interval=interval,
    )
    engine = types.SimpleNamespace(
        skill_queue=queue,
        config=types.SimpleNamespace(decision=types.SimpleNamespace(combat=combat)),
        logger=logging.getLogger("test_skill_trigger"),
        _last_skill_cast_time=0.0,
    )
    return engine, controller, queue


def cast(engine, class_name):
    return GameEngine._cast_skills_for_target(engine, FakeTarget(class_name))


class SkillTriggerTests(unittest.TestCase):
    def test_monster_casts_one_skill(self):
        engine, controller, _ = make_engine()
        self.assertTrue(cast(engine, "monster"))
        self.assertEqual(controller.pressed, ["q"])

    def test_boss_m_casts_two_skills(self):
        engine, controller, _ = make_engine()
        self.assertTrue(cast(engine, "boss-m"))
        self.assertEqual(controller.pressed, ["q", "w"])

    def test_class_not_in_mapping_casts_nothing(self):
        engine, controller, _ = make_engine()
        for class_name in ("hero", "elite", "boss", "people", "menu"):
            engine._last_skill_cast_time = 0.0
            self.assertFalse(cast(engine, class_name), class_name)
        self.assertEqual(controller.pressed, [])

    def test_custom_mapping_is_respected(self):
        engine, controller, _ = make_engine(
            count_by_class={"hero": 3}, skills=(("q", 10.0), ("w", 10.0), ("e", 10.0))
        )
        self.assertTrue(cast(engine, "hero"))
        self.assertEqual(controller.pressed, ["q", "w", "e"])
        # 未列出的类别不受自定义映射影响
        engine._last_skill_cast_time = 0.0
        self.assertFalse(cast(engine, "monster"))

    def test_throttle_blocks_retrigger_within_interval(self):
        engine, controller, _ = make_engine()
        self.assertTrue(cast(engine, "monster"))
        self.assertFalse(cast(engine, "monster"))  # 同一节流窗口内
        self.assertEqual(controller.pressed, ["q"])

    def test_zero_interval_retriggers_immediately(self):
        engine, controller, _ = make_engine(interval=0.0)
        self.assertTrue(cast(engine, "monster"))
        self.assertTrue(cast(engine, "monster"))
        self.assertEqual(controller.pressed, ["q", "w"])

    def test_cooling_skill_is_skipped(self):
        # 只有一个技能：放出去后立刻进入冷却，再次触发不应放出任何技能
        engine, controller, _ = make_engine(skills=(("q", 60.0),))
        self.assertTrue(cast(engine, "monster"))
        engine._last_skill_cast_time = 0.0  # 排除节流影响，单独验证冷却
        self.assertFalse(cast(engine, "monster"))
        self.assertEqual(controller.pressed, ["q"])

    def test_cooling_skill_is_skipped_but_others_fill_the_count(self):
        # q 在冷却，boss-m 想要 2 个，应由 w/e 补上而不是卡在 q 上
        engine, controller, queue = make_engine()
        queue.skills["q"].use()
        self.assertTrue(cast(engine, "boss-m"))
        self.assertNotIn("q", controller.pressed)
        self.assertEqual(controller.pressed, ["w", "e"])

    def test_use_skills_false_casts_nothing(self):
        engine, controller, _ = make_engine(use_skills=False)
        self.assertFalse(cast(engine, "boss-m"))
        self.assertEqual(controller.pressed, [])

    def test_without_skill_queue_casts_nothing(self):
        engine, controller, _ = make_engine(with_queue=False)
        self.assertFalse(cast(engine, "boss-m"))
        self.assertEqual(controller.pressed, [])


if __name__ == "__main__":
    unittest.main()
