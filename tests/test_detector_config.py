import unittest
from unittest.mock import patch

from src.detection.detector import MultiModelDetector


class DetectorConfigTests(unittest.TestCase):
    def test_long_threshold_keys_are_applied(self):
        with patch('src.detection.detector.YOLODetector') as detector:
            MultiModelDetector({'main': {
                'path': 'fake.pt', 'conf_threshold': 0.1,
                'iou_threshold': 0.2, 'classes': [1, 2]
            }}, device='cpu')
            detector.assert_called_once_with(
                model_path='fake.pt', device='cpu', conf_threshold=0.1,
                iou_threshold=0.2, classes=[1, 2]
            )

    def test_legacy_keys_remain_supported(self):
        with patch('src.detection.detector.YOLODetector') as detector:
            MultiModelDetector({'main': {'path': 'fake.pt', 'conf': 0.3, 'iou': 0.4}})
            detector.assert_called_once_with(
                model_path='fake.pt', device='cuda', conf_threshold=0.3,
                iou_threshold=0.4, classes=None
            )

    def test_long_keys_take_precedence_even_when_zero(self):
        with patch('src.detection.detector.YOLODetector') as detector:
            MultiModelDetector({'main': {
                'path': 'fake.pt', 'conf_threshold': 0.0, 'conf': 0.8,
                'iou_threshold': 0.0, 'iou': 0.9
            }})
            self.assertEqual(detector.call_args.kwargs['conf_threshold'], 0.0)
            self.assertEqual(detector.call_args.kwargs['iou_threshold'], 0.0)


if __name__ == '__main__':
    unittest.main()
