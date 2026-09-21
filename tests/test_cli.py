import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.config_loader import Config


class CliTests(unittest.TestCase):
    def run_main(self, arguments):
        config = Config()
        config.detection.device = 'cpu'
        config.game.target_fps = 12
        config.detection.models = {'main': {
            'path': 'old.pt', 'conf_threshold': 0.15, 'iou_threshold': 0.25,
            'classes': [1]
        }}
        engine_module = types.ModuleType('src.core.engine')
        engine_module.GameEngine = Mock()
        engine_module.EngineConfig = Mock()
        engine_module.GameEngine.return_value.is_running.return_value = False
        with patch.dict(sys.modules, {'src.core.engine': engine_module}):
            spec = importlib.util.spec_from_file_location(
                'cli_under_test', Path(__file__).resolve().parents[1] / 'main.py'
            )
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            with patch.object(sys, 'argv', ['main.py'] + arguments), \
                    patch.object(cli.ConfigLoader, 'load', return_value=config), \
                    patch.object(cli, 'init_logger'), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = cli.main()
        return config, result

    def test_omitted_flags_preserve_yaml_device_and_fps(self):
        config, result = self.run_main([])
        self.assertEqual(result, 0)
        self.assertEqual(config.detection.device, 'cpu')
        self.assertEqual(config.game.target_fps, 12)

    def test_explicit_flags_override_yaml(self):
        config, result = self.run_main(['-d', 'cuda', '-f', '60'])
        self.assertEqual(result, 0)
        self.assertEqual(config.detection.device, 'cuda')
        self.assertEqual(config.game.target_fps, 60)

    def test_model_override_preserves_thresholds_and_class_filter(self):
        config, _ = self.run_main(['-m', 'new.pt'])
        self.assertEqual(config.detection.models['main'], {
            'path': 'new.pt', 'conf_threshold': 0.15,
            'iou_threshold': 0.25, 'classes': [1]
        })

    def test_non_positive_fps_is_rejected(self):
        for fps in ('0', '-1'):
            with self.subTest(fps=fps), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_main(['-f', fps])
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
