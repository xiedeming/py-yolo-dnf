import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.detection.detector import create_detector
from src.detection.onnx_detector import ONNXDetector
from src.utils.config_loader import ConfigLoader


class ONNXDetectorUnitTests(unittest.TestCase):
    def make_detector(self):
        detector = ONNXDetector.__new__(ONNXDetector)
        detector.input_size = 416
        detector.conf_threshold = 0.5
        detector.iou_threshold = 0.45
        detector.classes = None
        detector.max_det = 300
        detector.class_names = {0: 'people', 1: 'door'}
        return detector

    def test_letterbox_keeps_aspect_ratio_and_produces_nchw_tensor(self):
        detector = self.make_detector()
        tensor, scale, pad_x, pad_y = detector._preprocess(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertAlmostEqual(scale, 416 / 1280)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 91)

    def test_decode_transposes_raw_yolo_output_and_restores_client_coordinates(self):
        detector = self.make_detector()
        output = np.zeros((1, 6, 20), dtype=np.float32)
        # First candidate is xywh at the letterboxed input scale and class 1.
        output[0, :4, 0] = (208, 208, 65, 65)
        output[0, 5, 0] = 0.9
        detections = detector._decode(output, (720, 1280), 416 / 1280, 0, 91)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.class_name, 'door')
        self.assertEqual(detection.bbox, (540, 260, 740, 460))

    def test_nms_does_not_suppress_overlapping_different_classes(self):
        detector = self.make_detector()
        output = np.zeros((1, 6, 20), dtype=np.float32)
        for index, class_id in enumerate((0, 1)):
            output[0, :4, index] = (208, 208, 65, 65)
            output[0, 4 + class_id, index] = 0.9
        detections = detector._decode(output, (416, 416), 1.0, 0, 0)
        self.assertEqual({item.class_name for item in detections}, {'people', 'door'})


class DetectionFactoryTests(unittest.TestCase):
    def test_onnx_factory_passes_cpu_specific_options(self):
        config = {
            'main': {
                'path': 'model.onnx', 'imgsz': 416, 'class_names': ['people'],
                'conf_threshold': 0.1, 'iou_threshold': 0.2, 'max_det': 25,
            }
        }
        with patch('src.detection.onnx_detector.ONNXDetector') as detector:
            create_detector(config, device='cpu', backend='onnxruntime', cpu_threads=2)
        detector.assert_called_once_with(
            model_path='model.onnx', device='cpu', input_size=416, class_names=['people'],
            conf_threshold=0.1, iou_threshold=0.2, classes=None, max_det=25, cpu_threads=2,
        )

    def test_cpu_profile_is_parseable(self):
        config = ConfigLoader.load(str(Path('config/settings.cpu.yaml')))
        self.assertEqual(config.detection.backend, 'onnxruntime')
        self.assertEqual(config.detection.device, 'cpu')
        self.assertEqual(config.detection.cpu_threads, 2)
        self.assertEqual(config.detection.models['main']['imgsz'], 416)
        self.assertIn('dungeon_1', config.maps.presets)


if __name__ == '__main__':
    unittest.main()
