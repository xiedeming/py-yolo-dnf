"""
翻牌处理模块 - 自动处理副本结算后的翻牌选择
"""
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..control.input_controller import InputController
    from ..detection.detector import Detection
    from .game_context import GameContext


@dataclass
class CardInfo:
    """卡片信息"""
    position: Tuple[int, int]  # 卡片中心位置
    is_purple: bool = False    # 是否为紫卡
    confidence: float = 0.0    # 检测置信度


class CardFlipper:
    """
    自动翻牌处理器

    功能：
    1. 检测翻牌界面中的卡片位置
    2. 识别紫卡（高价值卡片）
    3. 按优先级选择卡片（紫卡优先）
    4. 执行翻牌动作
    """

    def __init__(
        self,
        controller: 'InputController',
        purple_priority: bool = True,
        random_selection: bool = True
    ):
        """
        初始化翻牌处理器

        Args:
            controller: 输入控制器
            purple_priority: 是否优先选择紫卡
            random_selection: 无紫卡时是否随机选择
        """
        self.controller = controller
        self.purple_priority = purple_priority
        self.random_selection = random_selection

        # 翻牌状态
        self.flip_complete: bool = False
        self.last_flip_time: float = 0.0

    def detect_cards(self, detections: List['Detection']) -> List[CardInfo]:
        """
        从检测结果中提取卡片信息

        Args:
            detections: 检测结果列表

        Returns:
            卡片信息列表
        """
        cards = []

        for det in detections:
            if det.class_name in ['purple_card', 'card', 'brand']:
                card = CardInfo(
                    position=det.center,
                    is_purple=(det.class_name == 'purple_card'),
                    confidence=det.confidence
                )
                cards.append(card)

        return cards

    def select_card(self, cards: List[CardInfo]) -> Optional[Tuple[int, int]]:
        """
        选择一张卡片

        Args:
            cards: 卡片信息列表

        Returns:
            选中的卡片位置，如果没有卡片则返回None
        """
        if not cards:
            return None

        # 紫卡优先
        if self.purple_priority:
            purple_cards = [c for c in cards if c.is_purple]
            if purple_cards:
                # 选择置信度最高的紫卡
                best = max(purple_cards, key=lambda c: c.confidence)
                return best.position

        # 无紫卡或不需要紫卡优先
        if self.random_selection and len(cards) > 1:
            # 随机选择（更像人类行为）
            selected = random.choice(cards)
            return selected.position
        else:
            # 选择第一张
            return cards[0].position

    def execute_flip(
        self,
        detections: List['Detection'],
        context: Optional['GameContext'] = None
    ) -> bool:
        """
        执行翻牌动作

        Args:
            detections: 检测结果列表
            context: 游戏上下文（可选）

        Returns:
            是否执行了翻牌
        """
        # 检测卡片
        cards = self.detect_cards(detections)

        if not cards:
            return False

        # 选择卡片
        target = self.select_card(cards)

        if target is None:
            return False

        # 移动鼠标到卡片位置
        self.controller.mouse_move(target[0], target[1])

        # 短暂延迟（模拟人类反应）
        time.sleep(random.uniform(0.1, 0.3))

        # 点击翻牌
        self.controller.mouse_click()

        self.last_flip_time = time.time()
        self.flip_complete = True

        # 更新上下文
        if context:
            context.set_custom_data('card_flip_done', True)

        return True

    def handle_brand_detection(
        self,
        brand_detection: 'Detection',
        purple_cards: List['Detection'] = None,
        context: Optional['GameContext'] = None
    ) -> bool:
        """
        处理brand检测结果（翻牌界面）

        Args:
            brand_detection: brand检测结果
            purple_cards: 紫卡检测结果列表
            context: 游戏上下文

        Returns:
            是否执行了翻牌
        """
        # 如果检测到紫卡，优先点击
        if purple_cards and self.purple_priority:
            # 选择最近的紫卡
            nearest = min(purple_cards, key=lambda c: c.center[0])
            target = nearest.center
        else:
            # 根据brand位置推断卡片区域
            # DNF中brand通常在屏幕中央区域，卡片分布在两侧
            brand_x = brand_detection.center[0]
            brand_y = brand_detection.center[1]

            # 假设卡片在brand上方一定距离
            # 实际位置需要根据游戏UI调整
            target = (brand_x, brand_y - 100)

        # 移动并点击
        self.controller.mouse_move(target[0], target[1])
        time.sleep(random.uniform(0.1, 0.2))
        self.controller.mouse_click()

        self.flip_complete = True
        if context:
            context.set_custom_data('card_flip_done', True)

        return True

    def reset(self) -> None:
        """重置翻牌状态"""
        self.flip_complete = False
        self.last_flip_time = 0.0

    def is_flip_complete(self) -> bool:
        """
        检查翻牌是否完成

        Returns:
            是否完成翻牌
        """
        return self.flip_complete

    def get_status(self) -> dict:
        """
        获取当前状态

        Returns:
            状态字典
        """
        return {
            'flip_complete': self.flip_complete,
            'purple_priority': self.purple_priority,
            'last_flip_time': self.last_flip_time
        }


class MultiCardFlipper(CardFlipper):
    """
    多卡翻牌处理器

    处理需要选择多张卡片的情况
    """

    def __init__(
        self,
        controller: 'InputController',
        purple_priority: bool = True,
        max_flips: int = 4
    ):
        """
        初始化多卡翻牌处理器

        Args:
            controller: 输入控制器
            purple_priority: 是否优先选择紫卡
            max_flips: 最大翻牌数量
        """
        super().__init__(controller, purple_priority)
        self.max_flips = max_flips
        self.flips_done: int = 0

    def execute_multi_flip(
        self,
        detections: List['Detection'],
        num_flips: int = None,
        context: Optional['GameContext'] = None
    ) -> bool:
        """
        执行多次翻牌

        Args:
            detections: 检测结果列表
            num_flips: 翻牌次数（None则使用max_flips）
            context: 游戏上下文

        Returns:
            是否执行了翻牌
        """
        if num_flips is None:
            num_flips = self.max_flips

        cards = self.detect_cards(detections)
        if not cards:
            return False

        # 优先选择紫卡
        selected_cards = []
        if self.purple_priority:
            purple_cards = [c for c in cards if c.is_purple]
            selected_cards.extend(purple_cards[:num_flips])

        # 剩余选择普通卡
        remaining = num_flips - len(selected_cards)
        if remaining > 0:
            other_cards = [c for c in cards if not c.is_purple]
            selected_cards.extend(other_cards[:remaining])

        # 依次翻牌
        for card in selected_cards[:num_flips]:
            self.controller.mouse_move(card.position[0], card.position[1])
            time.sleep(random.uniform(0.05, 0.15))
            self.controller.mouse_click()
            time.sleep(random.uniform(0.1, 0.2))
            self.flips_done += 1

        self.flip_complete = True
        if context:
            context.set_custom_data('card_flip_done', True)

        return True

    def reset(self) -> None:
        """重置状态"""
        super().reset()
        self.flips_done = 0


def create_card_flipper_from_config(
    config_dict: dict,
    controller: 'InputController'
) -> CardFlipper:
    """
    从配置字典创建翻牌处理器

    Args:
        config_dict: 配置字典
        controller: 输入控制器

    Returns:
        CardFlipper实例
    """
    return CardFlipper(
        controller=controller,
        purple_priority=config_dict.get('purple_priority', True),
        random_selection=config_dict.get('random_selection', True)
    )
