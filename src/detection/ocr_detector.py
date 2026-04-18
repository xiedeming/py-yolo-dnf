"""
OCR检测模块 - 使用RapidOCR (基于ONNX Runtime) 检测游戏界面文字
"""
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    logger.warning("RapidOCR未安装，OCR功能将禁用。安装命令: pip install rapidocr-onnxruntime")


@dataclass
class TextDetection:
    """文字检测结果"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


class ShopDetector:
    """
    商店检测器

    使用RapidOCR检测游戏界面中是否包含商店相关的文字
    """

    # 商店相关关键词（支持中英文）
    SHOP_KEYWORDS = [
        '商店', '商店界面', '商店购买', '购买', '出售', '修理',
        'shop', 'store', 'buy', 'sell', 'repair',
        '金币', '金币商店', '点券', '兑换'
    ]

    def __init__(
        self,
        det_model_dir: str = '',
        rec_model_dir: str = '',
        cls_model_dir: str = '',
        **kwargs  # 忽略use_gpu等不再需要的参数
    ):
        """
        初始化商店检测器

        Args:
            det_model_dir: 文本检测模型路径
            rec_model_dir: 文本识别模型路径
            cls_model_dir: 文本方向分类模型路径
            **kwargs: 其他参数（忽略，保持向后兼容）
        """
        self.ocr = None
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir

        if RAPIDOCR_AVAILABLE:
            try:
                # 构建RapidOCR参数
                ocr_params = {}

                # 添加自定义模型路径
                if det_model_dir and Path(det_model_dir).exists():
                    ocr_params['det_model_path'] = det_model_dir
                    logger.info(f"使用自定义检测模型: {det_model_dir}")
                if rec_model_dir and Path(rec_model_dir).exists():
                    ocr_params['rec_model_path'] = rec_model_dir
                    logger.info(f"使用自定义识别模型: {rec_model_dir}")
                if cls_model_dir and Path(cls_model_dir).exists():
                    ocr_params['cls_model_path'] = cls_model_dir
                    logger.info(f"使用自定义分类模型: {cls_model_dir}")

                self.ocr = RapidOCR(**ocr_params) if ocr_params else RapidOCR()
                logger.info("RapidOCR初始化成功")

            except Exception as e:
                logger.error(f"RapidOCR初始化失败: {e}")
                self.ocr = None

    def detect_text(self, image) -> List[TextDetection]:
        """
        检测图像中的所有文字

        Args:
            image: BGR格式的图像

        Returns:
            文字检测结果列表
        """
        if self.ocr is None:
            return []

        try:
            start_time = time.time()
            # RapidOCR返回: (result, elapse)
            # result: 每行为 [bbox, text, confidence]
            # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 四个角点
            result, elapse = self.ocr(image)
            elapsed = time.time() - start_time

            detections = []

            if result:
                for line in result:
                    # line格式: [bbox, text, confidence]
                    points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1]
                    confidence = line[2]

                    # 计算边界框
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    detections.append(TextDetection(
                        text=text,
                        confidence=float(confidence),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=center
                    ))

            logger.debug(f"OCR检测到 {len(detections)} 个文字区域，耗时: {elapsed*1000:.1f}ms")

            return detections

        except Exception as e:
            logger.error(f"OCR检测失败: {e}")
            return []

    def detect_shop(self, image) -> Tuple[bool, Optional[str]]:
        """
        检测图像中是否有商店界面

        Args:
            image: BGR格式的图像

        Returns:
            (是否检测到商店, 匹配的文字)
        """
        if self.ocr is None:
            logger.warning("OCR未初始化，无法检测商店")
            return False, None

        detections = self.detect_text(image)

        for det in detections:
            text_lower = det.text.lower()
            for keyword in self.SHOP_KEYWORDS:
                if keyword.lower() in text_lower:
                    logger.info(f"检测到商店关键词: '{det.text}' (匹配: '{keyword}')")
                    return True, det.text

        return False, None

    def is_available(self) -> bool:
        """检查OCR是否可用"""
        return self.ocr is not None

    def detect_text_position(
        self,
        image,
        keywords: List[str]
    ) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """
        检测图像中指定关键词的位置

        Args:
            image: BGR格式的图像
            keywords: 关键词列表

        Returns:
            (是否检测到, 中心坐标(x, y), 匹配的文字)
        """
        if self.ocr is None:
            logger.warning("OCR未初始化，无法检测文字位置")
            return False, None, None

        detections = self.detect_text(image)

        for det in detections:
            for keyword in keywords:
                if keyword.lower() in det.text.lower():
                    logger.info(f"检测到关键词: '{det.text}' (匹配: '{keyword}'), 位置: {det.center}")
                    return True, det.center, det.text

        return False, None, None

    def get_all_text(self, image) -> str:
        """
        获取图像中所有文字（用于调试）

        Args:
            image: BGR格式的图像

        Returns:
            所有文字拼接成的字符串
        """
        detections = self.detect_text(image)
        return ' | '.join([d.text for d in detections])


class OCRManager:
    """OCR管理器 - 单例模式"""

    _instance: Optional['OCRManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        det_model_dir: str = '',
        rec_model_dir: str = '',
        cls_model_dir: str = '',
        **kwargs  # 忽略use_gpu等不再需要的参数
    ):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.shop_detector = ShopDetector(
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir
        )

    def detect_shop(self, image) -> Tuple[bool, Optional[str]]:
        """检测商店"""
        return self.shop_detector.detect_shop(image)

    def is_available(self) -> bool:
        """检查OCR是否可用"""
        return self.shop_detector.is_available()

    def detect_text_position(
        self,
        image,
        keywords: List[str]
    ) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """
        检测图像中指定关键词的位置

        Args:
            image: BGR格式的图像
            keywords: 关键词列表

        Returns:
            (是否检测到, 中心坐标(x, y), 匹配的文字)
        """
        return self.shop_detector.detect_text_position(image, keywords)


def create_ocr_manager(
    use_gpu: bool = False,  # 保留参数以保持向后兼容，但忽略
    lang: str = 'ch',       # 保留参数以保持向后兼容，但忽略
    det_model_dir: str = '',
    rec_model_dir: str = '',
    cls_model_dir: str = ''
) -> OCRManager:
    """
    创建或获取OCR管理器实例

    Args:
        use_gpu: 已弃用，ONNX Runtime自动处理设备选择
        lang: 已弃用，RapidOCR使用多语言模型
        det_model_dir: 文本检测模型路径
        rec_model_dir: 文本识别模型路径
        cls_model_dir: 文本方向分类模型路径

    Returns:
        OCRManager实例
    """
    return OCRManager(
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=cls_model_dir
    )
