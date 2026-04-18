"""
角色切换模块 - 处理DNF多角色切换流程
"""
import time
import random
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..control.input_controller import InputController
    from .game_context import GameContext

logger = logging.getLogger(__name__)


class SwitchState(Enum):
    """切换状态"""
    IDLE = "idle"
    STARTING = "starting"       # 按F12
    CONFIRMING = "confirming"   # 按ESC
    NAVIGATING = "navigating"   # 导航到角色
    SELECTING = "selecting"     # 选择角色
    COMPLETE = "complete"


@dataclass
class CharacterPosition:
    """角色在换人界面的位置"""
    character_id: str
    name: str
    position: Tuple[int, int]   # 角色图标位置
    row: int = 0                # 行索引
    col: int = 0                # 列索引


class CharacterSwitcher:
    """
    角色切换处理器

    处理DNF换人流程: F12 -> ESC -> 鼠标点击选择角色

    流程：
    1. 按F12打开换人界面
    2. 按ESC确认
    3. 鼠标移动到目标角色位置
    4. 点击选择角色
    """

    def __init__(
        self,
        controller: 'InputController',
        character_positions: Dict[str, CharacterPosition] = None,
        ocr_config: dict = None,
        capture = None,
        window_manager = None
    ):
        """
        初始化角色切换器

        Args:
            controller: 输入控制器
            character_positions: 角色位置配置字典
            ocr_config: OCR配置字典 (det_model_dir, rec_model_dir, cls_model_dir)
            capture: 屏幕捕获对象
            window_manager: 窗口管理器
        """
        self.controller = controller
        self.character_positions = character_positions or {}
        self.ocr_config = ocr_config or {}
        self.capture = capture
        self.window_manager = window_manager

        # 切换状态
        self.state = SwitchState.IDLE

        # 切换序列延迟配置
        self.delay_f12 = 0.3      # F12后等待时间
        self.delay_esc = 0.5      # ESC后等待时间
        self.delay_click = 0.3    # 点击后等待时间

        # 当前目标角色
        self.target_character: Optional[str] = None

    def start_switch(self) -> None:
        """
        开始角色切换流程

        执行第一步：按F12打开换人界面
        """
        self.state = SwitchState.STARTING
        self.controller.key_press('f12')
        time.sleep(self.delay_f12)

    def confirm_interface(self) -> None:
        """
        确认换人界面

        执行第二步：按ESC确认
        """
        self.state = SwitchState.CONFIRMING
        self.controller.key_press('escape')
        time.sleep(self.delay_esc)

    def navigate_to_character(self, character_id: str) -> bool:
        """
        导航到指定角色

        Args:
            character_id: 角色ID

        Returns:
            是否导航成功
        """
        if character_id not in self.character_positions:
            return False

        self.state = SwitchState.NAVIGATING
        self.target_character = character_id

        char_pos = self.character_positions[character_id]

        # 使用方向键导航（如果配置了行列）
        self.controller.key_press('right')
        return True

    def select_character(self, character_id: str) -> bool:
        """
        选择指定角色

        Args:
            character_id: 角色ID

        Returns:
            是否选择成功
        """
        if character_id not in self.character_positions:
            return False

        self.state = SwitchState.SELECTING

        char_pos = self.character_positions[character_id]

        # 移动鼠标到角色位置
        self.controller.mouse_move(char_pos.position[0], char_pos.position[1])

        # 短暂延迟（模拟人类）
        time.sleep(random.uniform(0.1, 0.2))

        # 点击选择
        self.controller.mouse_click()

        time.sleep(self.delay_click)

        return True

    def execute_full_switch(
        self,
        next_character: str,
        context: Optional['GameContext'] = None
    ) -> bool:
        """
        执行完整的角色切换流程

        Args:
            next_character: 下一个角色ID
            context: 游戏上下文（可选）

        Returns:
            是否切换成功
        """
        if next_character not in self.character_positions:
            return False

        try:
            # Step 1: F12
            self.start_switch()

            # Step 2: ESC
            self.confirm_interface()

            # Step 3: 选择角色
            if not self.select_character(next_character):
                return False

            # Step 4: 确认选择（再次点击或按Enter）
            time.sleep(0.2)
            self.controller.key_press('enter')

            self.state = SwitchState.COMPLETE

            # 更新上下文
            if context:
                context.set_custom_data('switch_done', True)
                context.dungeon_run_count = 0  # 重置刷图次数

            return True

        except Exception as e:
            self.state = SwitchState.IDLE
            return False

    def execute_return_to_character_selection(
        self,
        character_button_position: Tuple[int, int],
        current_image: Optional[np.ndarray] = None,
        context: Optional['GameContext'] = None
    ) -> bool:
        """
        执行返回角色选择菜单流程

        流程：
        1. 使用OCR检测是否有商店，如果有则按 ESC
        2. 按 F12，等待 0.25s
        3. 按 ESC，等待 0.25s
        4. 获取截图，使用OCR检测"选择角色"按钮位置
        5. 移动鼠标到选择角色按钮位置
        6. 鼠标左键点击
        7. 等待 0.25s
        8. 按 right（选择角色）

        Args:
            character_button_position: 选择角色按钮的默认位置 (x, y)，OCR检测失败时使用
            current_image: 当前屏幕图像（用于OCR检测商店）
            context: 游戏上下文（可选）

        Returns:
            是否切换成功
        """
        try:
            logger.info("开始执行返回角色选择菜单流程...")

            # Step 1: 使用OCR检测是否有商店
            has_shop = False
            if current_image is not None:
                has_shop = self._detect_shop_with_ocr(current_image)

            if has_shop:
                logger.info("OCR检测到商店界面，按下 ESC")
                self.controller.key_press('escape')
                time.sleep(0.25)
            else:
                logger.debug("未检测到商店界面")

            # Step 2: 按 F12，等待 0.25s
            logger.debug("按下 F12")
            self.controller.key_press('f12')
            time.sleep(0.25)

            # Step 3: 按 ESC，等待 0.25s
            logger.debug("按下 ESC")
            self.controller.key_press('escape')
            time.sleep(0.3)  # 等待界面加载

            # Step 4: 获取截图，使用OCR检测"选择角色"按钮位置
            button_pos = character_button_position  # 默认使用配置的位置

            if self.capture and self.window_manager:
                # 获取新截图
                new_image = self._capture_screen()
                if new_image is not None:
                    # 使用OCR检测"选择角色"按钮
                    detected, pos, text = self._detect_button_position(new_image)
                    if detected and pos:
                        button_pos = pos
                        logger.info(f"OCR检测到'{text}'按钮位置: {button_pos}")
                    else:
                        logger.warning(f"OCR未检测到'选择角色'按钮，使用默认位置: {button_pos}")
                else:
                    logger.warning("获取截图失败，使用默认按钮位置")
            else:
                logger.debug("capture或window_manager未设置，使用默认按钮位置")

            # Step 5: 移动鼠标到选择角色按钮位置
            logger.debug(f"移动鼠标到选择角色按钮: {button_pos}")
            self.controller.mouse_move(button_pos[0], button_pos[1])
            time.sleep(0.1)

            # Step 6: 鼠标左键点击
            logger.debug("鼠标左键点击")
            self.controller.mouse_click()

            # Step 7: 等待 0.25s
            time.sleep(0.25)

            # Step 8: 按 right（选择角色）
            logger.debug("按下 right")
            self.controller.key_press('right')

            # 更新上下文
            if context:
                context.set_custom_data('switch_done', True)
                context.dungeon_run_count = 0  # 重置刷图次数
                context.reset_menu_detect()    # 重置菜单检测计数

            self.state = SwitchState.COMPLETE
            logger.info("返回角色选择菜单流程完成")

            return True

        except Exception as e:
            logger.error(f"返回角色选择菜单流程失败: {e}")
            self.state = SwitchState.IDLE
            return False

    def _detect_shop_with_ocr(self, image: np.ndarray) -> bool:
        """
        使用OCR检测图像中是否有商店界面

        Args:
            image: BGR格式的图像

        Returns:
            是否检测到商店
        """
        try:
            from ..detection.ocr_detector import create_ocr_manager

            # 使用配置参数创建OCR管理器
            ocr_manager = create_ocr_manager(
                det_model_dir=self.ocr_config.get('det_model_dir', ''),
                rec_model_dir=self.ocr_config.get('rec_model_dir', ''),
                cls_model_dir=self.ocr_config.get('cls_model_dir', '')
            )
            if not ocr_manager.is_available():
                logger.debug("OCR不可用，跳过商店检测")
                return False

            has_shop, matched_text = ocr_manager.detect_shop(image)
            if has_shop:
                logger.info(f"OCR检测到商店文字: {matched_text}")

            return has_shop

        except ImportError:
            logger.debug("OCR模块未安装，跳过商店检测")
            return False
        except Exception as e:
            logger.error(f"OCR商店检测失败: {e}")
            return False

    def _capture_screen(self) -> Optional[np.ndarray]:
        """
        获取当前屏幕截图

        Returns:
            BGR格式的图像，失败返回None
        """
        try:
            if not self.capture or not self.window_manager:
                return None

            # 获取窗口客户区屏幕坐标
            rect = self.window_manager.get_client_screen_rect()
            if rect is None:
                logger.warning("无法获取窗口区域")
                return None

            left, top, right, bottom = rect
            width = right - left
            height = bottom - top

            # 捕获窗口区域
            image = self.capture.capture_region((left, top, width, height))
            return image

        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def _detect_button_position(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """
        使用OCR检测"选择角色"按钮的位置

        Args:
            image: BGR格式的图像

        Returns:
            (是否检测到, 中心坐标(x, y), 匹配的文字)
        """
        try:
            from ..detection.ocr_detector import create_ocr_manager

            # 使用配置参数创建OCR管理器
            ocr_manager = create_ocr_manager(
                det_model_dir=self.ocr_config.get('det_model_dir', ''),
                rec_model_dir=self.ocr_config.get('rec_model_dir', ''),
                cls_model_dir=self.ocr_config.get('cls_model_dir', '')
            )

            if not ocr_manager.is_available():
                logger.debug("OCR不可用，跳过按钮位置检测")
                return False, None, None

            # 检测"选择角色"相关关键词
            keywords = ['选择角色', '角色选择', '选择']
            detected, pos, text = ocr_manager.detect_text_position(image, keywords)

            return detected, pos, text

        except ImportError:
            logger.debug("OCR模块未安装，跳过按钮位置检测")
            return False, None, None
        except Exception as e:
            logger.error(f"OCR按钮位置检测失败: {e}")
            return False, None, None

    def add_character_position(
        self,
        character_id: str,
        position: Tuple[int, int],
        name: str = "",
        row: int = 0,
        col: int = 0
    ) -> None:
        """
        添加角色位置配置

        Args:
            character_id: 角色ID
            position: 角色图标位置
            name: 角色名称
            row: 行索引
            col: 列索引
        """
        self.character_positions[character_id] = CharacterPosition(
            character_id=character_id,
            name=name,
            position=position,
            row=row,
            col=col
        )

    def get_next_character(self, current_index: int, character_list: List[str]) -> Optional[str]:
        """
        获取下一个角色ID

        Args:
            current_index: 当前角色索引
            character_list: 角色ID列表

        Returns:
            下一个角色ID，如果已到最后则返回None
        """
        next_index = current_index + 1
        if next_index >= len(character_list):
            return None  # 所有角色已完成
        return character_list[next_index]

    def reset(self) -> None:
        """重置切换状态"""
        self.state = SwitchState.IDLE
        self.target_character = None

    def get_state(self) -> SwitchState:
        """获取当前状态"""
        return self.state

    def is_complete(self) -> bool:
        """是否切换完成"""
        return self.state == SwitchState.COMPLETE

    def get_status(self) -> dict:
        """
        获取当前状态

        Returns:
            状态字典
        """
        return {
            'state': self.state.value,
            'target_character': self.target_character,
            'available_characters': list(self.character_positions.keys())
        }


class CharacterSequenceManager:
    """
    角色序列管理器

    管理多个角色的依次切换和运行
    """

    def __init__(
        self,
        switcher: CharacterSwitcher,
        character_list: List[str]
    ):
        """
        初始化角色序列管理器

        Args:
            switcher: 角色切换器
            character_list: 角色ID列表
        """
        self.switcher = switcher
        self.character_list = character_list
        self.current_index = 0

        # 每个角色的运行配置
        self.character_configs: Dict[str, dict] = {}

    def set_character_config(self, character_id: str, config: dict) -> None:
        """
        设置角色配置

        Args:
            character_id: 角色ID
            config: 配置字典（包含dungeon_runs等）
        """
        self.character_configs[character_id] = config

    def get_current_character(self) -> Optional[str]:
        """
        获取当前角色ID

        Returns:
            当前角色ID
        """
        if self.current_index < len(self.character_list):
            return self.character_list[self.current_index]
        return None

    def switch_to_next(self, context: Optional['GameContext'] = None) -> bool:
        """
        切换到下一个角色

        Args:
            context: 游戏上下文

        Returns:
            是否切换成功（False表示所有角色已完成）
        """
        next_character = self.switcher.get_next_character(
            self.current_index,
            self.character_list
        )

        if next_character is None:
            return False  # 所有角色已完成

        if self.switcher.execute_full_switch(next_character, context):
            self.current_index += 1
            return True

        return False

    def has_more_characters(self) -> bool:
        """
        是否还有未处理的角色

        Returns:
            是否还有角色
        """
        return self.current_index < len(self.character_list) - 1

    def reset(self) -> None:
        """重置到第一个角色"""
        self.current_index = 0

    def get_progress(self) -> Tuple[int, int]:
        """
        获取进度

        Returns:
            (当前索引, 总数)
        """
        return (self.current_index, len(self.character_list))


def create_character_switcher_from_config(
    controller: 'InputController',
    characters_config: List[dict],
    ocr_config: dict = None,
    capture = None,
    window_manager = None
) -> CharacterSwitcher:
    """
    从配置字典创建角色切换器

    Args:
        controller: 输入控制器
        characters_config: 角色配置列表
        ocr_config: OCR配置字典
        capture: 屏幕捕获对象
        window_manager: 窗口管理器

    Returns:
        CharacterSwitcher实例
    """
    positions = {}

    for i, char_config in enumerate(characters_config):
        char_id = char_config.get('id', f'char_{i}')
        position = char_config.get('position', (100 + i * 100, 200))
        name = char_config.get('name', '')
        row = char_config.get('row', 0)
        col = char_config.get('col', i)

        positions[char_id] = CharacterPosition(
            character_id=char_id,
            name=name,
            position=tuple(position),
            row=row,
            col=col
        )

    return CharacterSwitcher(
        controller=controller,
        character_positions=positions,
        ocr_config=ocr_config,
        capture=capture,
        window_manager=window_manager
    )
