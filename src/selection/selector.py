"""
选择器模块 - 角色和地图选择功能
"""
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from ..control.input_controller import InputController
from ..utils.config_loader import CharacterConfig, MapConfig, Config
from ..utils.logger import GameLogger, init_logger


@dataclass
class SelectionResult:
    """选择结果"""
    success: bool
    message: str
    elapsed_time: float = 0.0


class CharacterSelector:
    """角色选择器"""

    def __init__(
        self,
        controller: InputController,
        config: Config,
        logger: Optional[GameLogger] = None
    ):
        self.controller = controller
        self.config = config
        self.logger = logger or init_logger()

    def select(self, character_id: str) -> SelectionResult:
        """
        选择角色

        Args:
            character_id: 角色ID

        Returns:
            SelectionResult
        """
        start_time = time.time()

        # 获取角色配置
        char_config = self.config.characters.presets.get(character_id)
        if not char_config:
            return SelectionResult(
                success=False,
                message=f"Character not found: {character_id}"
            )

        try:
            selection = char_config.selection

            # 移动到角色图标位置
            self.controller.mouse_move(
                selection.icon_position[0],
                selection.icon_position[1]
            )
            time.sleep(0.2)

            # 点击选择
            self.controller.mouse_click('left')
            time.sleep(0.3)

            # 确认选择
            self.controller.key_press(selection.confirm_key)
            time.sleep(0.5)

            elapsed = time.time() - start_time

            return SelectionResult(
                success=True,
                message=f"Selected character: {char_config.name}",
                elapsed_time=elapsed
            )

        except Exception as e:
            return SelectionResult(
                success=False,
                message=f"Error selecting character: {e}"
            )

    def enter_dungeon(self) -> SelectionResult:
        """
        进入副本流程（DNF专用）

        流程：
        1. 按下空格，等待0.15秒
        2. 按下right键3秒
        3. 按下空格，等待0.15秒
        4. 按下right键0.25秒
        5. 按下left键2秒
        6. 按下space键0.08秒
        Returns:
            SelectionResult
        """
        start_time = time.time()

        try:
            self.logger.info("开始进入副本流程...")

            # Step 1: 按下空格，等待0.15秒
            self.logger.debug("按下空格")
            self.controller.key_press('space')
            time.sleep(0.15)

            # Step 2: 按下right键5秒
            self.logger.debug("按下right键3秒")
            self.controller.key_down('right')
            time.sleep(3)
            self.controller.key_up('right')

            # Step 3: 按下空格，等待0.15秒
            self.logger.debug("按下空格")
            self.controller.key_press('space')
            time.sleep(0.25)

            # Step 4: 按下right键0.25秒
            self.logger.debug("按下right键0.25秒")
            self.controller.key_down('right')
            time.sleep(0.25)
            self.controller.key_up('right')
            # Step 5: 按下left键1.5秒
            self.logger.debug("按下left键2秒")
            self.controller.key_down('left')
            time.sleep(1.5)
            self.controller.key_up('left')

            # Step 6: 按下space键0.08秒
            self.logger.debug("按下space键0.08秒")
            self.controller.key_down('space')
            time.sleep(0.08)
            self.controller.key_up('space')

            elapsed = time.time() - start_time

            self.logger.info(f"进入副本完成，耗时: {elapsed:.1f}秒")

            return SelectionResult(
                success=True,
                message="进入副本完成",
                elapsed_time=elapsed
            )

        except Exception as e:
            self.logger.error(f"进入副本失败: {e}")
            return SelectionResult(
                success=False,
                message=f"Error entering dungeon: {e}"
            )

    def list_available(self) -> list:
        """列出可用角色"""
        if not self.config.characters:
            return []
        return [
            {"id": char_id, "name": char.name, "description": char.description}
            for char_id, char in self.config.characters.presets.items()
        ]


class MapSelector:
    """地图选择器"""

    def __init__(
        self,
        controller: InputController,
        config: Config,
        logger: Optional[GameLogger] = None
    ):
        self.controller = controller
        self.config = config
        self.logger = logger or init_logger()

    def select(self, map_id: str) -> SelectionResult:
        """
        选择地图

        Args:
            map_id: 地图ID

        Returns:
            SelectionResult
        """
        start_time = time.time()

        # 获取地图配置
        map_config = self.config.maps.presets.get(map_id)
        if not map_config:
            return SelectionResult(
                success=False,
                message=f"Map not found: {map_id}"
            )

        try:
            selection = map_config.selection

            # 打开地图菜单
            self.controller.key_press(selection.menu_key)
            time.sleep(0.5)

            # 如果需要滚动
            for _ in range(selection.scroll_count):
                self.controller.mouse_scroll('down')
                time.sleep(0.1)

            # 移动到地图图标位置
            self.controller.mouse_move(
                selection.icon_position[0],
                selection.icon_position[1]
            )
            time.sleep(0.2)

            # 点击选择
            self.controller.mouse_click('left')
            time.sleep(0.3)

            # 确认选择
            self.controller.key_press(selection.confirm_key)
            time.sleep(0.5)

            elapsed = time.time() - start_time

            return SelectionResult(
                success=True,
                message=f"Selected map: {map_config.name}",
                elapsed_time=elapsed
            )

        except Exception as e:
            return SelectionResult(
                success=False,
                message=f"Error selecting map: {e}"
            )

    def list_available(self) -> list:
        """列出可用地图"""
        if not self.config.maps:
            return []
        return [
            {"id": map_id, "name": map.name, "description": map.description}
            for map_id, map in self.config.maps.presets.items()
        ]


class SelectionManager:
    """选择管理器 - 统一管理角色和地图选择"""

    def __init__(
        self,
        controller: InputController,
        config: Config,
        logger: Optional[GameLogger] = None
    ):
        self.controller = controller
        self.config = config
        self.logger = logger or init_logger()

        self.character_selector = CharacterSelector(controller, config, logger)
        self.map_selector = MapSelector(controller, config, logger)

    def auto_select(self) -> Tuple[SelectionResult, SelectionResult]:
        """
        自动选择配置中的角色和地图

        Returns:
            (character_result, map_result)
        """
        char_result = SelectionResult(success=True, message="No character configured")
        map_result = SelectionResult(success=True, message="No map configured")

        # 选择角色
        if self.config.characters and self.config.characters.current:
            char_result = self.character_selector.select(self.config.characters.current)
            if not char_result.success:
                self.logger.error(f"Character selection failed: {char_result.message}")

        # 等待一下
        time.sleep(1.0)

        # 选择地图
        if self.config.maps and self.config.maps.current:
            map_result = self.map_selector.select(self.config.maps.current)
            if not map_result.success:
                self.logger.error(f"Map selection failed: {map_result.message}")

        return char_result, map_result

    def quick_start(self, character_id: str, map_id: str) -> bool:
        """
        快速开始 - 选择角色和地图后开始游戏

        Args:
            character_id: 角色ID
            map_id: 地图ID

        Returns:
            是否成功
        """
        # 选择角色
        char_result = self.character_selector.select(character_id)
        if not char_result.success:
            return False

        time.sleep(1.0)

        # 选择地图
        map_result = self.map_selector.select(map_id)
        if not map_result.success:
            return False

        # 更新配置
        self.config.characters.current = character_id
        self.config.maps.current = map_id

        return True

    def list_characters(self) -> list:
        """列出可用角色"""
        return self.character_selector.list_available()

    def list_maps(self) -> list:
        """列出可用地图"""
        return self.map_selector.list_available()

    def enter_dungeon(self) -> SelectionResult:
        """
        进入副本流程（DNF专用）

        Returns:
            SelectionResult
        """
        return self.character_selector.enter_dungeon()
