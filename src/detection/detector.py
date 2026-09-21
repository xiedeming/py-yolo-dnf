"""
YOLOv8检测模块 - 目标检测核心
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging

# 模块日志
logger = logging.getLogger(__name__)


def detector_options(config: dict) -> dict:
    """统一单模型和多模型配置，同时兼容旧版 conf/iou 字段。"""
    return {
        'model_path': config['path'],
        'conf_threshold': config.get('conf_threshold', config.get('conf', 0.5)),
        'iou_threshold': config.get('iou_threshold', config.get('iou', 0.45)),
        'classes': config.get('classes'),
    }


def create_detector(model_configs: Dict[str, dict], device: str = 'cuda', backend: str = 'ultralytics',
                    cpu_threads: int = 0):
    """Create a detector adapter while keeping game logic backend-agnostic."""
    if not model_configs:
        return None
    backend = backend.lower()
    if backend == 'ultralytics':
        if len(model_configs) == 1:
            _, config = next(iter(model_configs.items()))
            return YOLODetector(device=device, **detector_options(config))
        return MultiModelDetector(model_configs=model_configs, device=device)
    if backend == 'onnxruntime':
        from .onnx_detector import ONNXDetector, ONNXMultiModelDetector, onnx_detector_options

        if device != 'cpu':
            raise ValueError("backend='onnxruntime' requires detection.device='cpu'")
        if len(model_configs) == 1:
            _, config = next(iter(model_configs.items()))
            return ONNXDetector(cpu_threads=cpu_threads, **onnx_detector_options(config))
        return ONNXMultiModelDetector(model_configs=model_configs, cpu_threads=cpu_threads)
    raise ValueError("Unsupported detection backend: {}".format(backend))


@dataclass
class Detection:
    """单个检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int] = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    area: int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height

    @classmethod
    def from_yolo_result(cls, box, names: Dict[int, str]) -> 'Detection':
        """
        从YOLOv8结果创建Detection对象

        Args:
            box: YOLOv8检测框
            names: 类别名称映射

        Returns:
            Detection对象
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())

        return cls(
            class_id=cls_id,
            class_name=names.get(cls_id, f"class_{cls_id}"),
            confidence=float(conf),
            bbox=(int(x1), int(y1), int(x2), int(y2))
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


@dataclass
class DetectionResult:
    """单帧检测结果"""
    timestamp: float = field(default_factory=time.time)
    detections: List[Detection] = field(default_factory=list)
    inference_time: float = 0.0

    def get_by_class(self, class_name: str) -> List[Detection]:
        """获取指定类别的检测结果"""
        return [d for d in self.detections if d.class_name == class_name]

    def get_by_class_id(self, class_id: int) -> List[Detection]:
        """获取指定类别ID的检测结果"""
        return [d for d in self.detections if d.class_id == class_id]

    def filter_by_confidence(self, min_conf: float) -> List[Detection]:
        """按置信度过滤"""
        return [d for d in self.detections if d.confidence >= min_conf]

    def sort_by_confidence(self, descending: bool = True) -> List[Detection]:
        """按置信度排序"""
        return sorted(self.detections, key=lambda d: d.confidence, reverse=descending)

    def sort_by_area(self, descending: bool = True) -> List[Detection]:
        """按面积排序"""
        return sorted(self.detections, key=lambda d: d.area, reverse=descending)


class YOLODetector:
    """YOLOv8检测器"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None
    ):
        """
        初始化YOLOv8检测器

        Args:
            model_path: 模型文件路径
            device: 推理设备 ('cuda[:index]' 或 'cpu')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            classes: 要检测的类别ID列表，None表示检测所有类别
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"加载模型: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

        # 获取类别名称
        self.class_names = self.model.names
        logger.info(f"模型类别: {list(self.class_names.values())}")

        # 验证设备。自动分档可能选择 cuda:1 等非默认 CUDA 设备。
        if str(device).lower().startswith('cuda'):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA 不可用，切换到 CPU")
                    self.device = 'cpu'
                else:
                    logger.info(f"使用设备: {device}")
            except ImportError:
                logger.warning("torch 未安装，使用 CPU")
                self.device = 'cpu'
        else:
            logger.info(f"使用设备: CPU")

    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> DetectionResult:
        """
        执行推理

        Args:
            image: BGR格式的图像
            conf: 置信度阈值，None则使用默认值
            iou: IOU阈值，None则使用默认值
            classes: 要检测的类别ID列表，None则使用默认值

        Returns:
            DetectionResult对象
        """
        start_time = time.perf_counter()

        # 使用传入参数或默认值
        conf_threshold = conf if conf is not None else self.conf_threshold
        iou_threshold = iou if iou is not None else self.iou_threshold
        detect_classes = classes if classes is not None else self.classes

        # 执行推理
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=detect_classes,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        # 解析结果
        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                detection = Detection.from_yolo_result(box, self.class_names)
                detections.append(detection)

        # 输出检测日志
        if detections:
            class_counts = {}
            for det in detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

            summary = ', '.join(f"{name}({count})" for name, count in class_counts.items())
            logger.debug(f"检测到 {len(detections)} 个目标: {summary} | 推理耗时: {inference_time*1000:.1f}ms")
        else:
            logger.debug(f"未检测到目标 | 推理耗时: {inference_time*1000:.1f}ms")

        return DetectionResult(
            timestamp=time.time(),
            detections=detections,
            inference_time=inference_time
        )

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        批量推理

        Args:
            images: BGR格式的图像列表
            conf: 置信度阈值
            iou: IOU阈值

        Returns:
            DetectionResult列表
        """
        start_time = time.perf_counter()

        conf_threshold = conf if conf is not None else self.conf_threshold
        iou_threshold = iou if iou is not None else self.iou_threshold

        results = self.model.predict(
            source=images,
            device=self.device,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        all_results = []
        for result in results:
            detections = []
            for box in result.boxes:
                detection = Detection.from_yolo_result(box, self.class_names)
                detections.append(detection)

            all_results.append(DetectionResult(
                timestamp=time.time(),
                detections=detections,
                inference_time=inference_time / len(images) if images else 0
            ))

        return all_results

    def get_class_names(self) -> Dict[int, str]:
        """获取所有类别名称"""
        return self.class_names

    def get_class_id(self, class_name: str) -> Optional[int]:
        """根据类别名称获取类别ID"""
        for id, name in self.class_names.items():
            if name == class_name:
                return id
        return None

    def warmup(self, image_size: Tuple[int, int] = (640, 480)) -> float:
        """
        预热模型（用于首次推理加速）

        Args:
            image_size: 预热图像大小

        Returns:
            预热推理时间
        """
        logger.debug(f"预热模型，图像尺寸: {image_size}")
        dummy_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        start = time.perf_counter()
        _ = self.predict(dummy_image)
        warmup_time = time.perf_counter() - start
        logger.debug(f"预热完成，耗时: {warmup_time*1000:.1f}ms")
        return warmup_time


class MultiModelDetector:
    """多模型检测器 - 支持同时使用多个模型"""

    def __init__(self, model_configs: Dict[str, dict], device: str = 'cuda'):
        """
        初始化多模型检测器

        Args:
            model_configs: 模型配置字典
                {
                    'enemies': {'path': 'models/enemies.pt', 'conf': 0.5},
                    'items': {'path': 'models/items.pt', 'conf': 0.6},
                }
            device: 推理设备
        """
        logger.info(f"初始化多模型检测器，共 {len(model_configs)} 个模型")
        self.detectors: Dict[str, YOLODetector] = {}

        for name, config in model_configs.items():
            logger.info(f"加载模型 [{name}]: {config['path']}")
            self.detectors[name] = YOLODetector(
                device=device,
                **detector_options(config)
            )

    def detect(
        self,
        image: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, DetectionResult]:
        """
        使用指定模型进行检测

        Args:
            image: BGR格式图像
            model_names: 要使用的模型名称列表，None表示使用所有模型

        Returns:
            {model_name: DetectionResult}
        """
        results = {}
        names = model_names if model_names else list(self.detectors.keys())

        for name in names:
            if name in self.detectors:
                results[name] = self.detectors[name].predict(image)

        return results

    def detect_all(self, image: np.ndarray) -> Dict[str, DetectionResult]:
        """
        使用所有模型进行检测

        Args:
            image: BGR格式图像

        Returns:
            {model_name: DetectionResult}
        """
        return self.detect(image)

    def get_all_detections(self, image: np.ndarray) -> List[Detection]:
        """
        获取所有模型的检测结果合并列表

        Args:
            image: BGR格式图像

        Returns:
            所有检测结果的合并列表
        """
        all_detections = []
        results = self.detect_all(image)

        for model_name, result in results.items():
            for det in result.detections:
                # 添加模型来源标记
                det.model_source = model_name
                all_detections.append(det)

        return all_detections

    def get_detector(self, name: str) -> Optional[YOLODetector]:
        """获取指定名称的检测器"""
        return self.detectors.get(name)

    def get_class_names(self, model_name: str) -> Optional[Dict[int, str]]:
        """获取指定模型的类别名称"""
        detector = self.detectors.get(model_name)
        return detector.get_class_names() if detector else None

    def warmup_all(self, image_size: Tuple[int, int] = (640, 480)) -> None:
        """预热所有模型"""
        for name, detector in self.detectors.items():
            logger.debug(f"预热模型: {name}")
            detector.warmup(image_size)
