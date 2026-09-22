import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.config_loader import (
    CharacterRunConfigData, Config, ConfigLoader, SideScrollerConfig
)


class ConfigLoaderTests(unittest.TestCase):
    def test_empty_yaml_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.yaml'
            for content in ('', '# comment only\n', 'null\n'):
                with self.subTest(content=content):
                    path.write_text(content, encoding='utf-8')
                    self.assertEqual(ConfigLoader.load(str(path)), Config())

    def test_invalid_root_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.yaml'
            for content in ('[]', 'false', '42', 'hello'):
                with self.subTest(content=content):
                    path.write_text(content, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'mapping'):
                        ConfigLoader.load(str(path))

    def test_all_config_sections_survive_save_load(self):
        config = ConfigLoader.load(str(Path(__file__).resolve().parents[1] / 'config/settings.yaml'))
        # Exercise the YAML alias for the dataclass's condition_type field.
        config.characters = ConfigLoader._parse_characters_config({
            'current': 'tester',
            'presets': {'tester': {'skills': [{
                'id': 'heal', 'condition': 'low_hp', 'condition_params': {'threshold': 0.2}
            }]}}
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/settings.yaml'
            ConfigLoader.save(config, str(path))
            saved = yaml.safe_load(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['characters']['presets']['tester']['skills'][0]['condition'], 'low_hp')
            self.assertEqual(ConfigLoader.load(str(path)), config)

    def test_extends_merges_nested_cpu_profile_without_resetting_base_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base.yaml').write_text(
                'game:\n  target_fps: 30\ndetection:\n  models:\n    main:\n      path: base.pt\n      conf_threshold: 0.5\n',
                encoding='utf-8'
            )
            (root / 'cpu.yaml').write_text(
                'extends: base.yaml\ngame:\n  target_fps: 10\ndetection:\n  backend: onnxruntime\n  models:\n    main:\n      path: cpu.onnx\n',
                encoding='utf-8'
            )
            config = ConfigLoader.load(str(root / 'cpu.yaml'))
        self.assertEqual(config.game.target_fps, 10)
        self.assertEqual(config.detection.backend, 'onnxruntime')
        self.assertEqual(config.detection.models['main'], {
            'path': 'cpu.onnx', 'conf_threshold': 0.5
        })


    def test_movement_speed_fields_parse_from_settings(self):
        config = ConfigLoader.load(str(Path(__file__).resolve().parents[1] / 'config/settings.yaml'))
        self.assertEqual(config.side_scroller.reference_distance, 150)
        self.assertEqual(config.side_scroller.max_hold, 1.2)
        role = config.multi_character.role_list[0]
        self.assertEqual(role.move_speed, 1.0)
        self.assertEqual(role.press_sleep, 0.55)
        self.assertEqual(role.run_sleep, 0.075)

    def test_movement_speed_defaults_when_absent(self):
        # 旧配置没有这些字段时必须仍能加载，且回落到基准值
        default_role = CharacterRunConfigData()
        self.assertEqual(default_role.move_speed, 1.0)
        self.assertEqual(default_role.press_sleep, 0.55)
        self.assertEqual(default_role.run_sleep, 0.075)
        self.assertEqual(SideScrollerConfig().reference_distance, 150)
        self.assertEqual(SideScrollerConfig().max_hold, 1.2)


if __name__ == '__main__':
    unittest.main()
