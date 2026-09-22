"""调试窗口配色回归测试。

曾出现的问题：颜色表只登记了 enemy/item/ui 等语义名，与模型实际类别
（people/door/monster/...）一个都不匹配，于是所有检测框都落到兜底的白色；标签
文字又固定用白色画在同一背景上，最终表现为"满屏白框、看不到类别"。
"""
import unittest

from src.debug.visualizer import DebugVisualizer

# config/settings.yaml 里 detection.models 的类别顺序
MODEL_CLASSES = [
    'people', 'door', 'monster', 'brand', 'menu', 'article',
    'purple_card', 'hero', 'elite', 'boss-n', 'boss-m',
]


class ClassColorTests(unittest.TestCase):
    def test_every_model_class_gets_its_own_non_white_color(self):
        viz = DebugVisualizer()
        colors = [viz.get_color(name) for name in MODEL_CLASSES]
        self.assertEqual(len(set(colors)), len(MODEL_CLASSES))
        self.assertNotIn((255, 255, 255), colors)

    def test_unknown_class_color_is_stable(self):
        viz = DebugVisualizer()
        self.assertEqual(
            viz.get_color('some_new_class'), viz.get_color('some_new_class')
        )

    def test_legacy_semantic_names_still_resolve(self):
        viz = DebugVisualizer()
        self.assertNotEqual(viz.get_color('enemy'), viz.get_color('item'))


class LabelTextColorTests(unittest.TestCase):
    def test_label_text_never_matches_its_background(self):
        # 白底白字正是标签看不见的原因
        viz = DebugVisualizer()
        for name in MODEL_CLASSES:
            background = viz.get_color(name)
            with self.subTest(class_name=name):
                self.assertNotEqual(viz.label_text_color(background), background)

    def test_light_background_uses_black_text(self):
        self.assertEqual(DebugVisualizer.label_text_color((255, 255, 255)), (0, 0, 0))
        self.assertEqual(DebugVisualizer.label_text_color((0, 255, 255)), (0, 0, 0))

    def test_dark_background_uses_white_text(self):
        self.assertEqual(DebugVisualizer.label_text_color((0, 0, 0)), (255, 255, 255))
        self.assertEqual(DebugVisualizer.label_text_color((0, 0, 255)), (255, 255, 255))


if __name__ == '__main__':
    unittest.main()
