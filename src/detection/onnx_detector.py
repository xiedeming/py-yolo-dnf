"""CPU-oriented YOLO ONNX Runtime detector.

The module intentionally imports onnxruntime only when a detector is created so
the existing PyTorch/GPU workflow does not acquire a CPU-runtime dependency.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection, DetectionResult


logger = logging.getLogger(__name__)


class ONNXDetector:
    """Run a static-shape, raw-output YOLO detection model with ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        input_size: Optional[int] = None,
        max_det: int = 300,
        class_names: Optional[Sequence[str]] = None,
        cpu_threads: int = 0,
    ):
        if device != "cpu":
            raise ValueError("ONNXDetector currently supports only device='cpu'")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError("ONNX model file not found: {}".format(model_path))
        if input_size is not None and input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if max_det <= 0:
            raise ValueError("max_det must be a positive integer")
        if cpu_threads < 0:
            raise ValueError("cpu_threads cannot be negative")

        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is required for backend='onnxruntime'. "
                "Install requirements-cpu.txt."
            ) from error

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cpu_threads:
            options.intra_op_num_threads = cpu_threads
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.input_size = self._resolve_input_size(self.input.shape, input_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = set(classes) if classes is not None else None
        self.max_det = max_det
        self.class_names = self._resolve_class_names(class_names)
        logger.info(
            "Loaded ONNX CPU model %s (input=%dx%d, threads=%s)",
            self.model_path, self.input_size, self.input_size, cpu_threads or "runtime default"
        )

    def _resolve_input_size(self, shape: Sequence, configured_size: Optional[int]) -> int:
        height, width = shape[2], shape[3]
        if isinstance(height, str) or isinstance(width, str) or height is None or width is None:
            if configured_size is None:
                raise ValueError("Dynamic ONNX input requires detection.models.<name>.imgsz")
            return configured_size
        if height != width:
            raise ValueError("Only square YOLO ONNX inputs are supported, got {}x{}".format(width, height))
        if configured_size is not None and configured_size != height:
            raise ValueError(
                "Configured imgsz {} does not match ONNX input {}".format(configured_size, height)
            )
        return int(height)

    def _resolve_class_names(self, configured_names: Optional[Sequence[str]]) -> Dict[int, str]:
        names = configured_names
        if names is None:
            metadata_path = self.model_path.with_suffix(".json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                names = metadata.get("class_names")
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(names)}
        raise ValueError(
            "ONNX model requires class_names in its JSON sidecar or model configuration"
        )

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR HxWx3 array")
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        letterboxed = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        tensor = np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        return tensor[None] / 255.0, scale, pad_x, pad_y

    @staticmethod
    def _nms(boxes: List[List[float]], scores: List[float], iou_threshold: float, max_det: int) -> List[int]:
        order = np.argsort(np.asarray(scores))[::-1]
        keep: List[int] = []
        boxes_array = np.asarray(boxes, dtype=np.float32)
        while len(order) and len(keep) < max_det:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            x1 = np.maximum(boxes_array[current, 0], boxes_array[remaining, 0])
            y1 = np.maximum(boxes_array[current, 1], boxes_array[remaining, 1])
            x2 = np.minimum(boxes_array[current, 2], boxes_array[remaining, 2])
            y2 = np.minimum(boxes_array[current, 3], boxes_array[remaining, 3])
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            current_area = (boxes_array[current, 2] - boxes_array[current, 0]) * (boxes_array[current, 3] - boxes_array[current, 1])
            remaining_area = (boxes_array[remaining, 2] - boxes_array[remaining, 0]) * (boxes_array[remaining, 3] - boxes_array[remaining, 1])
            union = current_area + remaining_area - intersection
            ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[ious <= iou_threshold]
        return keep

    def _decode(
        self, output: np.ndarray, original_shape: Tuple[int, int], scale: float, pad_x: int, pad_y: int
    ) -> List[Detection]:
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError("Only batch=1 ONNX outputs are supported")
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError("Unsupported ONNX output shape {}".format(predictions.shape))
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise ValueError("Raw YOLO output must contain xywh and class scores")

        boxes_by_class: Dict[int, List[List[float]]] = {}
        scores_by_class: Dict[int, List[float]] = {}
        image_height, image_width = original_shape
        for row in predictions:
            if not np.all(np.isfinite(row)):
                continue
            class_id = int(np.argmax(row[4:]))
            confidence = float(row[4 + class_id])
            if confidence < self.conf_threshold or (self.classes is not None and class_id not in self.classes):
                continue
            center_x, center_y, width, height = row[:4]
            x1 = (center_x - width / 2 - pad_x) / scale
            y1 = (center_y - height / 2 - pad_y) / scale
            x2 = (center_x + width / 2 - pad_x) / scale
            y2 = (center_y + height / 2 - pad_y) / scale
            x1, x2 = np.clip((x1, x2), 0, image_width)
            y1, y2 = np.clip((y1, y2), 0, image_height)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_by_class.setdefault(class_id, []).append([float(x1), float(y1), float(x2), float(y2)])
            scores_by_class.setdefault(class_id, []).append(confidence)

        detections: List[Detection] = []
        for class_id, boxes in boxes_by_class.items():
            for index in self._nms(boxes, scores_by_class[class_id], self.iou_threshold, self.max_det):
                x1, y1, x2, y2 = boxes[index]
                detections.append(Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, "class_{}".format(class_id)),
                    confidence=scores_by_class[class_id][index],
                    bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                ))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)[:self.max_det]

    def predict(self, image: np.ndarray, conf: Optional[float] = None, iou: Optional[float] = None,
                classes: Optional[List[int]] = None) -> DetectionResult:
        previous_confidence, previous_iou, previous_classes = self.conf_threshold, self.iou_threshold, self.classes
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou
        if classes is not None:
            self.classes = set(classes)
        try:
            start = time.perf_counter()
            tensor, scale, pad_x, pad_y = self._preprocess(image)
            outputs = self.session.run(None, {self.input_name: tensor})
            if len(outputs) != 1:
                raise ValueError("Expected one raw YOLO output, got {}".format(len(outputs)))
            detections = self._decode(outputs[0], image.shape[:2], scale, pad_x, pad_y)
            return DetectionResult(detections=detections, inference_time=time.perf_counter() - start)
        finally:
            self.conf_threshold, self.iou_threshold, self.classes = previous_confidence, previous_iou, previous_classes

    def get_class_names(self) -> Dict[int, str]:
        return self.class_names

    def get_class_id(self, class_name: str) -> Optional[int]:
        for class_id, name in self.class_names.items():
            if name == class_name:
                return class_id
        return None

    def warmup(self, image_size: Tuple[int, int] = (640, 480)) -> float:
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.predict(image)
        return time.perf_counter() - start


class ONNXMultiModelDetector:
    """Multi-model ONNX adapter with the same public behaviour as the legacy adapter."""

    def __init__(self, model_configs: Dict[str, dict], cpu_threads: int = 0):
        self.detectors = {
            name: ONNXDetector(cpu_threads=cpu_threads, **onnx_detector_options(config))
            for name, config in model_configs.items()
        }

    def detect_all(self, image: np.ndarray) -> Dict[str, DetectionResult]:
        return {name: detector.predict(image) for name, detector in self.detectors.items()}

    def warmup_all(self) -> None:
        for detector in self.detectors.values():
            detector.warmup()


def onnx_detector_options(config: dict) -> dict:
    """Extract ONNX-specific model settings without leaking them to PyTorch."""
    if "path" not in config:
        raise ValueError("Each model configuration requires a path")
    return {
        "model_path": config["path"],
        "device": "cpu",
        "conf_threshold": config.get("conf_threshold", config.get("conf", 0.5)),
        "iou_threshold": config.get("iou_threshold", config.get("iou", 0.45)),
        "classes": config.get("classes"),
        "input_size": config.get("imgsz"),
        "max_det": config.get("max_det", 300),
        "class_names": config.get("class_names"),
    }
