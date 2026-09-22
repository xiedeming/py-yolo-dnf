"""
主引擎模块 - 协调所有模块运行
"""
import time
import cv2
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..capture import create_capture
from ..capture.window_manager import WindowManager
from ..detection.detector import Detection, create_detector
from ..control.input_controller import InputController, InputConfig
from ..control.movement_controller import MovementController, MoveSpeedModel
from ..decision.game_context import GameContext, GameState
from ..decision.state_machine import StateMachine, create_game_state_machine
from ..decision.skill_manager import SkillManager, SkillQueue, BuffManager
from ..decision.map_navigator import MapNavigator, create_map_navigator_from_config
from ..decision.card_flipper import CardFlipper, create_card_flipper_from_config
from ..decision.stuck_handler import StuckHandler, create_stuck_handler_from_config
from ..decision.character_switcher import CharacterSwitcher, create_character_switcher_from_config
from ..decision.multi_character_manager import MultiCharacterManager, create_multi_character_manager_from_config
from ..selection.selector import SelectionManager
from ..utils.logger import GameLogger, init_logger
from ..utils.config_loader import Config, ConfigLoader, CharacterConfig
from ..debug.visualizer import DebugVisualizer


@dataclass
class EngineStats:
    """引擎统计信息"""
    fps: float = 0.0
    frame_count: int = 0
    capture_time: float = 0.0
    detection_time: float = 0.0
    decision_time: float = 0.0
    total_time: float = 0.0


class GameEngine:
    """主游戏引擎"""

    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        """
        初始化游戏引擎

        Args:
            config: 配置对象
            config_path: 配置文件路径
        """
        # 加载配置
        if config:
            self.config = config
        elif config_path:
            self.config = ConfigLoader.load(config_path)
        else:
            # Keep direct GameEngine users on the same automatic profile as the
            # CLI.  The import is lazy so CPU-only test doubles do not need the
            # optional hardware probing dependencies.
            from ..utils.hardware_profile import select_runtime_profile

            profile = select_runtime_profile()
            self.config = ConfigLoader.load(str(profile.config_path))
            self.config.detection.device = profile.device

        # 初始化日志
        self.logger = init_logger()
        self.logger.info("Initializing Game Engine...")

        # 组件初始化前必须先声明这些占位属性：
        # _init_components() -> _init_dnf_modules() 会读取 skill_manager，
        # 而它只在"配置了角色"时才会被 _init_character_system() 赋值。
        # 声明放在 _init_components() 之后会导致 multi_character 模式下直接
        # AttributeError（之前的顺序就是这么错的）。
        # 移动控制：把像素距离换算成按键时长，并按当前角色速度系数缩放
        self.movement: Optional[MovementController] = None

        # 技能和Buff管理
        self.skill_manager: Optional[SkillManager] = None
        self.skill_queue: Optional[SkillQueue] = None  # 新的技能队列
        self.buff_manager: Optional[BuffManager] = None  # 新的Buff管理器
        self.current_character: Optional[CharacterConfig] = None
        self._current_role_config: Optional[Any] = None  # 当前角色配置

        # DNF 专用模块
        self.map_navigator: Optional[MapNavigator] = None
        self.card_flipper: Optional[CardFlipper] = None
        self.stuck_handler: Optional[StuckHandler] = None
        self.character_switcher: Optional[CharacterSwitcher] = None
        self.multi_char_manager: Optional[MultiCharacterManager] = None
        self.dungeon_runner = None  # 将在所有模块初始化后创建
        self.scheduler = None

        # 初始化组件
        self._init_components()

        # 控制变量
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._hotkey_listener = None
        self._hotkey_last_pressed: Dict[str, float] = {}
        self._stats = EngineStats()
        self._frame_times: list = []
        self._debug_window_size: tuple = None  # 调试窗口尺寸缓存

        # 检查 OpenCV 是否支持 GUI（headless 环境不支持）
        self._opencv_gui_available = self._check_opencv_gui()

        # 战斗状态追踪（用于处理角色遮挡怪物的情况）
        self._last_enemy_direction: Optional[str] = None  # 上一次敌人方向
        self._combat_attack_count: int = 0  # 连续攻击计数
        self._max_blind_attacks: int = 10   # 丢失目标后最多盲攻次数
        self._last_skill_cast_time: float = 0.0  # 上次按目标类型释放技能的时间（节流用）

        # 截图保存相关
        self._save_screenshots = self.config.debug.save_screenshots
        self._screenshot_dir = Path(self.config.debug.screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_screenshot_time: float = 0.0
        self._menu_detect_started_at: Optional[float] = None
        # 配置中是帧数，转换为秒数
        target_fps = self.config.game.target_fps if self.config.game.target_fps > 0 else 30
        self._screenshot_interval: float = self.config.debug.screenshot_interval / target_fps
        if self._save_screenshots:
            self.logger.debug(f"截图功能已启用，保存目录: {self._screenshot_dir}，间隔: {self._screenshot_interval:.1f}秒")

        # 自定义动作回调
        self._state_callbacks: Dict[GameState, Callable] = {}


    def _init_components(self) -> None:
        """初始化所有组件"""
        # 窗口管理器
        self.window_manager = WindowManager(self.config.game.window_title)

        # 屏幕捕获（后端由 capture.method 决定，bettercam 失败会自动回退到 mss）
        self.capture = create_capture(
            method=getattr(self.config.capture, 'method', 'mss'),
            monitor_index=self.config.capture.monitor_index,
            target_fps=self.config.game.target_fps
        )

        # YOLO检测器
        self._init_detector()

        # 输入控制器
        input_config = InputConfig(
            humanize=self.config.control.humanize,
            random_delay_range=self.config.control.random_delay_range
        )
        self.controller = InputController(input_config)

        # 移动控制器：像素距离 → 按键时长；方向键取 side_scroller 配置
        # （config 用 getattr 兜底，便于用 SimpleNamespace 鸭子类型构造引擎的测试）
        ss_config = getattr(self.config, 'side_scroller', None)
        self.movement = MovementController(
            self.controller,
            MoveSpeedModel(
                reference_distance=getattr(ss_config, 'reference_distance', 150),
                max_hold=getattr(ss_config, 'max_hold', 1.2),
            ),
            left_key=getattr(ss_config, 'move_left_key', 'left'),
            right_key=getattr(ss_config, 'move_right_key', 'right'),
        )

        # 游戏上下文
        self.context = GameContext()

        # 设置菜单检测阈值（从配置读取）
        if self.config.dungeon:
            self.context.menu_detect_threshold = self.config.dungeon.menu_detect_threshold

        # 状态机
        self.state_machine = create_game_state_machine()

        # 设置状态进入回调
        self.state_machine.set_enter_action(GameState.MENU, self._on_enter_menu)
        self.state_machine.set_enter_action(GameState.PLAYING, self._on_enter_playing)
        self.state_machine.set_enter_action(GameState.COMBAT, self._on_enter_combat)
        # 退出卡住恢复时清标记，避免 recovery_done / stuck_detected 泄漏到下一帧
        self.state_machine.set_exit_action(
            GameState.STUCK_RECOVERY, self._on_exit_stuck_recovery
        )

        # 调试可视化
        if self.config.debug.enabled:
            self.visualizer = DebugVisualizer(
                show_fps=self.config.debug.show_fps,
                show_state=self.config.debug.show_state
            )
        else:
            self.visualizer = None

        # 初始化角色和技能系统
        self._init_character_system()

        # 初始化DNF专用模块
        self._init_dnf_modules()

        # 初始化选择管理器
        self.selection_manager = SelectionManager(
            controller=self.controller,
            config=self.config,
            logger=self.logger
        )

        self.logger.success("All components initialized")

    def _init_detector(self) -> None:
        """初始化检测器"""
        models = self.config.detection.models

        if not models:
            self.logger.warning("No models configured, detection will be disabled")
            self.detector = None
            return

        backend = getattr(self.config.detection, 'backend', 'ultralytics')
        cpu_threads = getattr(self.config.detection, 'cpu_threads', 0)
        self.detector = create_detector(
            model_configs=models,
            device=self.config.detection.device,
            backend=backend,
            cpu_threads=cpu_threads,
        )
        self.logger.info(
            f"Loaded {len(models)} model(s) with {backend} backend"
        )

    def _init_character_system(self) -> None:
        """初始化角色系统"""
        if self.config.characters and self.config.characters.current:
            char_id = self.config.characters.current
            char_config = self.config.characters.presets.get(char_id)

            if char_config:
                self.current_character = char_config

                # 创建技能管理器
                self.skill_manager = SkillManager(
                    character_config=self.current_character,
                    context=self.context,
                    controller=self.controller
                )
                self.logger.info(f"Loaded character: {self.current_character.name} with {len(char_config.skills)} skills")
            else:
                self.logger.warning(f"Character not found: {char_id}")
        else:
            self.logger.info("No character configured, using default combat")

    def _apply_role_movement_config(self, role_config) -> None:
        """
        把当前角色的移动参数应用到速度模型与上下文阈值。

        用 getattr 兜底，这样用 SimpleNamespace 鸭子类型构造的测试角色也能工作。
        """
        if role_config is None:
            return

        move_speed = getattr(role_config, 'move_speed', 1.0) or 1.0
        if self.context is not None:
            self.context.set_movement_speed(move_speed)
        if self.movement is not None:
            self.movement.set_speed(
                move_speed=move_speed,
                press_sleep=getattr(role_config, 'press_sleep', None),
                run_sleep=getattr(role_config, 'run_sleep', None),
            )

    def _init_dnf_modules(self) -> None:
        """初始化DNF专用模块"""
        # 初始化技能队列和Buff管理器
        if self.config.multi_character and self.config.multi_character.enabled:
            # 获取第一个角色的配置
            if self.config.multi_character.role_list:
                first_role = self.config.multi_character.role_list[0]
                self._current_role_config = first_role
                self._apply_role_movement_config(first_role)

                # 初始化技能队列
                self.skill_queue = SkillQueue(self.controller)
                self.skill_queue.load_skills(first_role.art, first_role.art_time)
                if self.skill_manager:
                    self.skill_manager.skill_queue = self.skill_queue
                self.logger.info(f"技能队列已初始化: {len(first_role.art)} 个技能")

                # 初始化Buff管理器
                self.buff_manager = BuffManager(self.controller)
                self.buff_manager.load_buffs(first_role.buff)
                self.logger.info(f"Buff管理器已初始化: {len(first_role.buff)} 个Buff")

        # 初始化地图导航器
        if self.config.map_routes:
            self.map_navigator = create_map_navigator_from_config(
                self.config.map_routes,
                self.controller
            )
            self.logger.info(f"Map navigator initialized with {len(self.config.map_routes)} routes")

        # 初始化翻牌处理器
        if self.config.card_flip and self.config.card_flip.enabled:
            self.card_flipper = create_card_flipper_from_config(
                {
                    'purple_priority': self.config.card_flip.purple_priority,
                    'random_selection': self.config.card_flip.random_selection
                },
                self.controller
            )
            self.logger.info("Card flipper initialized")

        # 初始化卡住检测与分级恢复处理器
        # 直接把 StuckRecoveryConfig 交给它，不再用 dict 重新包一层
        # （那正是过去 StuckConfig / StuckRecoveryConfig 两份重复配置的来源）
        if self.config.stuck_recovery:
            self.stuck_handler = create_stuck_handler_from_config(
                self.config.stuck_recovery,
                self.controller,
                movement=self.movement,
            )
            self.logger.info("Stuck handler initialized")

        # 初始化角色切换器
        if self.config.multi_character and self.config.multi_character.enabled:
            role_list_config = [
                {
                    'id': role.id,
                    'name': role.name,
                    'position': list(role.position)
                }
                for role in self.config.multi_character.role_list
            ]
            # 构建OCR配置
            ocr_config = {}
            if self.config.ocr:
                ocr_config = {
                    'det_model_dir': self.config.ocr.det_model_dir,
                    'rec_model_dir': self.config.ocr.rec_model_dir,
                    'cls_model_dir': self.config.ocr.cls_model_dir
                }
            self.character_switcher = create_character_switcher_from_config(
                self.controller,
                role_list_config,
                ocr_config=ocr_config,
                capture=self.capture,
                window_manager=self.window_manager
            )
            self.logger.info(f"Character switcher initialized with {len(role_list_config)} characters")

            # 初始化多角色管理器
            multi_char_config = {
                'enabled': self.config.multi_character.enabled,
                'start_name': self.config.multi_character.start_name,
                'end_name': self.config.multi_character.end_name,
                # 完整转发角色字段。此前只转发 4 个字段，导致 manager 造出的
                # CharacterRunConfig 是空壳（art/art_time 缺失 → 技能队列为空）
                'role_list': [
                    {
                        'id': role.id,
                        'name': role.name,
                        'dungeon_runs': role.dungeon_runs,
                        'buff': role.buff,  # 新格式：直接传递buff按键列表
                        'art': getattr(role, 'art', []),
                        'art_time': getattr(role, 'art_time', {}),
                        'move_speed': getattr(role, 'move_speed', 1.0),
                        'run_sleep': getattr(role, 'run_sleep', 0.075),
                        'press_sleep': getattr(role, 'press_sleep', 0.55),
                        'buff_sleep': getattr(role, 'buff_sleep', 0.3),
                    }
                    for role in self.config.multi_character.role_list
                ]
            }
            self.multi_char_manager = create_multi_character_manager_from_config(
                self,
                multi_char_config,
                self.character_switcher
            )
            self.logger.info("Multi-character manager initialized")

        # 初始化副本运行器
        from .dungeon_runner import create_dungeon_runner_from_config
        if self.config.dungeon:
            dungeon_config = {
                'mode': self.config.dungeon.mode,
                'max_runs': self.config.dungeon.max_runs,
                'auto_sell': self.config.dungeon.auto_sell,
                'sell_after_runs': self.config.dungeon.sell_after_runs,
                'collect_items': self.config.dungeon.collect_items
            }
            self.dungeon_runner = create_dungeon_runner_from_config(self, dungeon_config)
            self.logger.info(f"Dungeon runner initialized (mode: {self.config.dungeon.mode})")

        # 初始化定时调度器
        if self.config.schedule and self.config.schedule.enabled:
            try:
                from .scheduler import create_scheduler_from_config, ScheduleConfig
                schedule_config = ScheduleConfig(
                    enabled=self.config.schedule.enabled,
                    hour=self.config.schedule.hour,
                    minute=self.config.schedule.minute,
                    recurring=self.config.schedule.recurring
                )
                self.scheduler = create_scheduler_from_config(
                    schedule_config,
                    self.start  # 定时启动引擎
                )
                self.logger.info(f"Scheduler initialized ({self.config.schedule.hour}:{self.config.schedule.minute})")
            except ImportError:
                self.logger.warning("APScheduler not installed, scheduling disabled")

    def find_game_window(self) -> bool:
        """
        查找游戏窗口

        Returns:
            是否找到窗口
        """
        self.logger.info(f"Looking for window: {self.config.game.window_title}")

        if self.window_manager.find_window():
            self.logger.success(f"Found window: {self.window_manager._window_info.title}")
            return True
        else:
            self.logger.error("Game window not found!")
            # 列出所有窗口帮助调试
            self.logger.info("Available windows:")
            for hwnd, title in WindowManager.list_all_windows()[:10]:
                self.logger.info(f"  - {title}")
            return False

    def _run_init_flow(self) -> bool:
        """
        执行初始化流程

        流程:
        1. 按下ESC键
        2. 等待0.2秒
        3. 使用OCR识别"选择角色"按钮
        4. 移动鼠标到按钮上并左键点击
        5. 等待0.2秒
        6. 按下空格键

        Returns:
            是否成功
        """
        try:
            # Step 1: 按下ESC键
            self.logger.info("初始化流程: 按下ESC键")
            time.sleep(6.5)
            self.controller.key_press('esc')
            time.sleep(0.2)

            # Step 2: 获取截图用于OCR检测
            client_rect = self.window_manager.get_client_screen_rect()
            left, top, right, bottom = client_rect
            width = right - left
            height = bottom - top
            image = self.capture.capture_region((left, top, width, height))

            # Step 3: 使用OCR识别"选择角色"按钮
            button_pos = None
            try:
                from ..detection.ocr_detector import create_ocr_manager

                # 获取OCR配置
                det_model_dir = ''
                rec_model_dir = ''
                cls_model_dir = ''
                if self.config.ocr:
                    det_model_dir = self.config.ocr.det_model_dir
                    rec_model_dir = self.config.ocr.rec_model_dir
                    cls_model_dir = self.config.ocr.cls_model_dir

                ocr_manager = create_ocr_manager(
                    det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
                    cls_model_dir=cls_model_dir
                )

                if ocr_manager.is_available():
                    keywords = ['选择角色', '角色选择']
                    detected, pos, text = ocr_manager.detect_text_position(image, keywords)

                    if detected and pos:
                        button_pos = pos
                        self.logger.info(f"OCR检测到'{text}'按钮位置: {button_pos}")
                    else:
                        self.logger.warning("OCR未检测到'选择角色'按钮")
                else:
                    self.logger.warning("OCR不可用")

            except ImportError:
                self.logger.warning("OCR模块未安装，跳过按钮检测")
            except Exception as e:
                self.logger.error(f"OCR检测失败: {e}")

            # 如果OCR检测失败，使用配置中的默认位置
            if button_pos is None:
                if self.config.dungeon and self.config.dungeon.character_button_pos:
                    button_pos = self.config.dungeon.character_button_pos
                    self.logger.info(f"使用默认按钮位置: {button_pos}")
                else:
                    self.logger.error("无法确定'选择角色'按钮位置")
                    return False

            # Step 4: 移动鼠标到按钮上并左键点击
            # 将窗口客户区坐标转换为屏幕坐标
            screen_x = left + button_pos[0]
            screen_y = top + button_pos[1]

            self.logger.debug(f"移动鼠标到按钮位置: ({screen_x}, {screen_y})")
            self.controller.mouse_move(screen_x, screen_y)
            time.sleep(0.1)
            self.controller.mouse_click('left')
            time.sleep(0.2)

            # Step 5: 按下空格键
            self.logger.debug("初始化流程: 按下空格键")
            self.controller.key_press('space')

            self.logger.success("初始化流程执行完成")
            return True

        except Exception as e:
            self.logger.error(f"初始化流程失败: {e}")
            return False

    def set_state_callback(self, state: GameState, callback: Callable) -> None:
        """
        设置状态回调函数

        Args:
            state: 游戏状态
            callback: 回调函数，接受GameContext参数
        """
        self._state_callbacks[state] = callback

    def start(self, auto_enter_dungeon: bool = True, run_init_flow: bool = True) -> None:
        """
        启动引擎

        Args:
            auto_enter_dungeon: 是否自动执行进入副本流程 (默认True)
            run_init_flow: 是否执行初始化流程 (默认True)
        """
        if self._running:
            self.logger.warning("Engine is already running")
            return

        # 查找游戏窗口
        if not self.find_game_window():
            return

        # 将窗口置顶
        self.window_manager.bring_to_front()

        # 执行初始化流程
        if run_init_flow:
            self.logger.info("执行初始化流程...")
            if not self._run_init_flow():
                self.logger.error("初始化流程失败")
                return
            self.logger.success("初始化流程完成!")

        # 预热检测器
        if self.detector:
            self.logger.info("Warming up detector...")
            if hasattr(self.detector, 'warmup_all'):
                self.detector.warmup_all()
            else:
                self.detector.warmup()

        # 执行进入副本流程
        if auto_enter_dungeon:
            self.logger.info("执行进入副本流程...")
            enter_result = self.selection_manager.enter_dungeon()
            if not enter_result.success:
                self.logger.error(f"进入副本失败: {enter_result.message}")
                return
            self.logger.success("进入副本成功!")

        # 启动技能队列和释放初始Buff
        if self.skill_queue:
            self.skill_queue.start()
            self.logger.info("技能队列已启动")

        if self.buff_manager:
            self.buff_manager.apply_buffs()
            self.logger.info("初始Buff已释放")

        self._running = True
        self._paused = False
        self._stop_requested = False
        self._start_hotkey_listener()
        self.logger.success("Engine started!")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()

    def _main_loop(self) -> None:
        """主循环"""
        frame_interval = 1.0 / self.config.game.target_fps if self.config.game.target_fps > 0 else 0

        self.logger.info("Entering main loop...")

        while self._running:
            loop_start = time.perf_counter()

            if getattr(self, '_stop_requested', False):
                self._running = False
                break

            if not self._paused:
                try:
                    self._process_frame()
                except Exception as e:
                    self.logger.error(f"Error processing frame: {e}")

            # 帧率控制
            loop_time = time.perf_counter() - loop_start
            if frame_interval > 0 and loop_time < frame_interval:
                time.sleep(frame_interval - loop_time)

            # 更新FPS
            self._update_fps(loop_start)

            # 处理OpenCV窗口事件（仅在支持GUI的环境）
            if self.visualizer and self._opencv_gui_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    self.logger.info("Exit requested via debug window")
                    break
                elif key == ord('p'):  # P
                    if self._paused:
                        self.resume()
                    else:
                        self.pause()

    def _process_frame(self) -> None:
        """处理单帧"""
        frame_start = time.perf_counter()

        # 1. 屏幕捕获 - 使用客户区坐标
        capture_start = time.perf_counter()
        try:
            # 获取客户区屏幕坐标（不包含窗口边框和标题栏）
            client_rect = self.window_manager.get_client_screen_rect()
            left, top, right, bottom = client_rect
            width = right - left
            height = bottom - top

            self.logger.debug(f"捕获区域: ({left}, {top}) 大小: {width}x{height}")

            # 传入 (left, top, width, height) 格式
            image = self.capture.capture_region((left, top, width, height))

            # 验证捕获结果
            img_h, img_w = image.shape[:2]
            if img_w != width or img_h != height:
                self.logger.warning(f"捕获尺寸异常! 期望: {width}x{height}, 实际: {img_w}x{img_h}")

        except Exception as e:
            self.logger.error(f"Capture error: {e}")
            return
        self._stats.capture_time = time.perf_counter() - capture_start

        # 更新屏幕中心
        h, w = image.shape[:2]
        self.context.set_screen_center(w, h)

        # 2. 目标检测
        detection_start = time.perf_counter()
        detections = {}
        if self.detector:
            detections = self._run_detection(image)
        self._stats.detection_time = time.perf_counter() - detection_start

        # 3. 更新上下文
        self.context.update(detections)

        # 3.5 卡住检测：必须早于状态机，转移条件才能当帧生效
        if self.stuck_handler:
            self.stuck_handler.observe(
                image, self.context, self.state_machine.get_state(), self.movement
            )

        # 3.6 更新技能状态
        if self.skill_manager:
            self.skill_manager.update()

        # 4. 状态机决策
        decision_start = time.perf_counter()
        self.state_machine.update(self.context)
        current_state = self.state_machine.get_state()
        self.context.state = current_state
        self._stats.decision_time = time.perf_counter() - decision_start

        # 5. 执行状态回调
        if current_state in self._state_callbacks:
            try:
                self._state_callbacks[current_state](self.context)
            except Exception as e:
                self.logger.error(f"Error in state callback: {e}")

        # 5.5 推进定时移动：到期释放方向键（替代阻塞 sleep）
        if self.movement:
            self.movement.update()

        # 6. 执行默认动作
        self._execute_action(current_state, image)

        # 7. 调试输出
        if self.visualizer:
            self._show_debug(image, detections)

        # 8. 定时保存截图
        self._save_screenshot_periodically(image, detections)

        self._stats.total_time = time.perf_counter() - frame_start
        self._stats.frame_count += 1

    def _save_screenshot_periodically(self, image, detections: Dict[str, list]) -> None:
        """
        定时保存截图

        Args:
            image: 当前帧图像
            detections: 检测结果
        """
        # 检查截图功能是否启用
        if not self._save_screenshots:
            return

        current_time = time.time()

        # 检查是否到达保存时间
        if current_time - self._last_screenshot_time >= self._screenshot_interval:
            self._last_screenshot_time = current_time

            # 生成文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')

            try:
                # 1. 首先保存原始图片（无标注）
                raw_filename = f"screenshot_{timestamp}_raw.jpg"
                raw_filepath = self._screenshot_dir / raw_filename
                cv2.imwrite(str(raw_filepath), image)
                self.logger.debug(f"原始截图已保存: {raw_filepath}")

                # 2. 保存带标注的图片
                annotated_filename = f"screenshot_{timestamp}_annotated.jpg"
                annotated_filepath = self._screenshot_dir / annotated_filename
                save_image = image.copy()

                if detections:
                    # 定义颜色
                    colors = {
                        'enemies': (0, 0, 255),      # 红色
                        'doors': (255, 0, 0),        # 蓝色
                        'items': (0, 255, 0),        # 绿色
                        'ui_elements': (255, 255, 0), # 青色
                        'players': (255, 0, 255)      # 紫色
                    }

                    # 绘制所有检测框
                    for category, det_list in detections.items():
                        color = colors.get(category, (255, 255, 255))
                        for det in det_list:
                            x1, y1, x2, y2 = det.bbox
                            # 绘制矩形
                            cv2.rectangle(save_image, (x1, y1), (x2, y2), color, 2)
                            # 绘制标签
                            label = f"{det.class_name} {det.confidence:.2f}"
                            cv2.putText(save_image, label, (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 保存带标注的图像
                cv2.imwrite(str(annotated_filepath), save_image)
                self.logger.debug(f"标注截图已保存: {annotated_filepath}")

            except Exception as e:
                self.logger.error(f"保存截图失败: {e}")

    def _run_detection(self, image) -> Dict[str, list]:
        """
        执行检测并将模型分类映射到游戏逻辑分类

        模型分类 -> 游戏逻辑分类映射:
        - monster, hero, elite, boss -> enemies (敌人)
        - door -> doors (门)
        - article, purple_card -> items (道具)
        - brand, menu -> ui_elements (UI元素)
        - people -> players (玩家/NPC，可选处理)
        """
        raw_detections = {}

        # 获取原始检测结果
        if hasattr(self.detector, 'detect_all'):
            results = self.detector.detect_all(image)
            for name, result in results.items():
                raw_detections[name] = result.detections
        else:
            result = self.detector.predict(image)
            raw_detections['main'] = result.detections

        # 分类映射
        CLASS_MAPPING = {
            'enemies': ['monster', 'hero', 'elite', 'boss', 'boss-m'],
            'doors': ['door'],
            'items': ['article', 'purple_card'],
            'ui_elements': ['brand', 'menu', 'shop'],  # shop: 商店
            'players': ['people'],
            'minimap_hero': ['hero'],      # 小地图英雄图标
            'cards': ['purple_card']       # 翻牌卡片
        }

        # 转换检测结果
        detections = {
            'enemies': [],
            'doors': [],
            'items': [],
            'ui_elements': [],
            'players': [],
            'minimap_hero': [],
            'cards': []
        }

        # 遍历所有检测结果并分类
        for model_name, det_list in raw_detections.items():
            for det in det_list:
                class_name = det.class_name.lower()
                matched = False

                # 查找匹配的分类
                for category, class_list in CLASS_MAPPING.items():
                    if class_name in class_list:
                        detections[category].append(det)
                        matched = True
                        break

                # 未匹配的分类放入others（可选）
                if not matched:
                    # 可以选择忽略或放入其他分类
                    pass

        return detections

    def _execute_action(self, state: GameState, image=None) -> None:
        """执行状态对应的动作"""
        # 检测菜单并更新计数。低配配置可用真实秒数，避免检测频率变化改变超时语义。
        has_menu = self.context.has_menu()
        if has_menu:
            self.context.increment_menu_detect()
            if self._menu_detect_started_at is None:
                self._menu_detect_started_at = time.monotonic()
        else:
            self.context.reset_menu_detect()
            self._menu_detect_started_at = None

        timeout_seconds = getattr(self.config.dungeon, 'menu_timeout_seconds', None)
        menu_timeout = (
            has_menu and self._menu_detect_started_at is not None and
            timeout_seconds is not None and
            time.monotonic() - self._menu_detect_started_at >= timeout_seconds
        )
        if timeout_seconds is None:
            menu_timeout = has_menu and self.context.is_menu_detect_timeout()

        # 检查菜单检测是否超时（单独处理）
        if menu_timeout:
            if self.character_switcher:
                # 启用了多角色模式，执行角色切换
                self.logger.info(f"菜单检测超时，触发角色切换...")
                self._execute_character_switch_flow(image)
            else:
                # 未启用多角色模式，尝试按Tab键恢复
                self.logger.warning("菜单检测超时，尝试按Tab键恢复...")
                self.controller.key_press('tab')
                self.context.reset_menu_detect()
                self._menu_detect_started_at = None
            return

        # 检查是否需要切换角色（刷图次数达到上限）
        if self.context.should_switch_character() and has_menu:
            self.logger.info(f"触发角色切换条件: 刷图次数={self.context.dungeon_run_count}/{self.context.max_dungeon_runs}")

            if self.character_switcher:
                # 启用了多角色模式，执行角色切换
                self._execute_character_switch_flow(image)
            else:
                # 未启用多角色模式，停止程序
                self.logger.success(f"已完成 {self.context.dungeon_run_count} 次刷图，程序结束")
                self._running = False
            return

        if state == GameState.COMBAT:
            self._combat_action_side_scroller()
        elif state == GameState.PLAYING:
            self._playing_action()
        elif state == GameState.TRANSITIONING:
            self._transitioning_action()
        elif state == GameState.DEAD:
            self._dead_action()
        elif state == GameState.CARD_FLIPPING:
            self._card_flipping_action()
        elif state == GameState.CHARACTER_SWITCH:
            self._character_switch_action()
        elif state == GameState.STUCK_RECOVERY:
            self._stuck_recovery_action()
        elif state == GameState.BUFFING:
            self._buffing_action()

    def _combat_action(self) -> None:
        """战斗动作 - 原始鼠标版本（保留兼容）"""
        target = self.context.get_nearest_enemy()
        if not target:
            return

        # 瞄准敌人中心
        self.controller.mouse_move(target.center[0], target.center[1])

        # 尝试使用技能
        skill_used = False
        if self.skill_manager:
            # 检查是否启用技能
            use_skills = True
            if self.config.decision and self.config.decision.combat:
                use_skills = self.config.decision.combat.use_skills

            if use_skills:
                used_skill_id = self.skill_manager.use_next_available_skill()
                if used_skill_id:
                    self.logger.debug(f"Used skill: {used_skill_id}")
                    skill_used = True

        # 如果没有使用技能，使用普通攻击
        if not skill_used:
            attack_key = self._get_attack_key()
            self.controller.key_press(attack_key)

    def _combat_action_side_scroller(self) -> None:
        """横版游戏战斗动作 - 键盘控制版本"""
        target = self.context.get_nearest_enemy()

        # 获取配置
        ss_config = self.config.side_scroller
        attack_range = ss_config.attack_range
        attack_key = ss_config.attack_key

        # 如果角色配置有攻击范围和按键，使用角色配置
        if self.current_character and self.current_character.attack:
            attack_range = self.current_character.attack.range

        if target:
            # 检测到敌人，重置盲攻计数
            self._combat_attack_count = 0

            # 记录敌人方向
            direction = self.context.get_enemy_direction(target, ss_config.approach_threshold)
            self._last_enemy_direction = direction

            # 判断距离。注意攻击范围**不按速度缩放** —— 它是攻击本身的属性，
            # 缩了会让快角色停在自己够不到的位置，在"停/走"之间永久震荡
            in_range = self.context.is_enemy_in_attack_range(target, attack_range)

            if in_range:
                # 在攻击范围内，停止移动并攻击
                self.movement.stop()
                self._perform_attack(attack_key)
            else:
                # 不在范围内：按剩余距离决定持续按住还是定时点按（近距离点按不会冲过头）
                self._approach_target(target, ss_config.approach_threshold)
        else:
            # 没有检测到敌人
            self.movement.stop()
            if self._last_enemy_direction and self._combat_attack_count < self._max_blind_attacks:
                # 之前有敌人且正在攻击，继续盲攻（角色可能遮挡了怪物）
                self._combat_attack_count += 1
                self.logger.debug(f"丢失目标，继续盲攻 ({self._combat_attack_count}/{self._max_blind_attacks})")
                self._perform_attack(attack_key)
            else:
                # 超过盲攻次数或之前没有敌人，停止攻击
                self._last_enemy_direction = None
                self._combat_attack_count = 0

    def _perform_attack(self, attack_key: str) -> None:
        """执行攻击动作"""
        target = self.context.get_nearest_enemy()

        # 按目标类型释放技能（monster 1 个 / boss-m 2 个），带节流
        skill_used = self._cast_skills_for_target(target) if target else False

        # 没有技能可用时使用普通攻击
        if not skill_used:
            self.controller.key_press(attack_key)

    def _cast_skills_for_target(self, target: Detection) -> bool:
        """按目标类型释放技能。

        遇到 monster 放 1 个、boss-m 放 2 个，由
        ``config.decision.combat.skill_count_by_class`` 配置；未列出的类别不放技能。
        实际能否放出由 SkillQueue 的冷却决定（冷却中的会被轮转跳过），
        ``skill_trigger_interval`` 则限制触发频率，避免同一波敌人之间反复触发。
        """
        if not self.skill_queue:
            return False

        combat_cfg = getattr(self.config.decision, 'combat', None)
        if combat_cfg is not None and not getattr(combat_cfg, 'use_skills', True):
            return False

        counts = getattr(combat_cfg, 'skill_count_by_class', None) or {}
        count = counts.get(target.class_name, 0)
        if count <= 0:
            return False

        interval = getattr(combat_cfg, 'skill_trigger_interval', 1.0)
        now = time.time()
        if now - self._last_skill_cast_time < interval:
            return False
        # 无论这次是否真的放出技能都推进节流时间，否则技能全在冷却时
        # 会每帧轮询一遍队列
        self._last_skill_cast_time = now

        # 队列里冷却中的技能会被轮到队尾并返回 False，因此多试几轮直到凑够
        # 数量或用完所有技能
        cast = 0
        for _ in range(len(self.skill_queue.skills)):
            if cast >= count:
                break
            if self.skill_queue.use_next_skill():
                cast += 1

        if cast:
            self.logger.debug(f"遇到 {target.class_name}，释放 {cast} 个技能")
        return cast > 0

    def _transitioning_action(self) -> None:
        """过渡状态动作 - 移动到门并等待地图加载"""
        door = self.context.get_door()
        if not door:
            self.logger.warning("No door detected in TRANSITIONING state")
            return

        # 判断是否已站在门上
        if self.context.is_player_at_door(door):
            # 站在门上，停止移动，等待加载
            self.movement.stop()
            self.logger.debug("Standing at door, waiting for map transition...")
            return

        # 移动到门
        self._approach_target(door, self.config.side_scroller.approach_threshold)

    def _approach_target(self, target, threshold: int) -> None:
        """
        朝目标靠近：按角色速度缩放的死区判断方向，按剩余距离决定持续按住还是定时点按。

        `direction == 'center'` 表示已在死区内、但可能还没到"到达"判定（到达阈值更小）。
        原实现在这种情况下什么都不做，会永久停在原地；这里按 X 坐标补一个方向继续靠近，
        因为死区（approach_threshold）比到达阈值大，继续靠近会自然收敛而不会来回震荡。
        """
        direction = self.context.get_move_direction_to_target(target, threshold)
        if direction == 'center':
            direction = 'left' if target.center[0] < self.context.screen_center[0] else 'right'
        gap = abs(target.center[0] - self.context.screen_center[0])
        self.movement.approach(direction, gap)

    def _get_attack_key(self) -> str:
        """获取攻击按键"""
        if self.skill_manager:
            return self.skill_manager.get_default_attack_key()
        elif self.config.decision and self.config.decision.combat:
            return self.config.decision.combat.attack_key
        return "space"  # 默认

    def _playing_action(self) -> None:
        """游玩动作 - 横版游戏版本"""
        # 如果有敌人，停止移动（等待下一帧进入COMBAT状态）
        if self.context.has_enemies():
            self.movement.stop()
            return

        # 如果有门，移动到门那边
        if self.context.has_door():
            self._approach_target(
                self.context.get_door(),
                self.config.side_scroller.approach_threshold
            )
            return

        # 如果没有敌人和门，向右持续移动搜索
        self.movement.hold('right')
        self.logger.debug("No enemies or door detected, moving right to search...")

    def _dead_action(self) -> None:
        """死亡动作"""
        # 等待复活或按下复活键
        self.logger.info("Player dead, waiting for respawn...")
        time.sleep(1)

    def _card_flipping_action(self) -> None:
        """翻牌动作 - 按下3键，等待0.08s，按下ESC键"""
        self.controller.key_press('3')
        time.sleep(0.08)
        self.controller.key_press('esc')
        self.logger.info("翻牌完成：按下3键和ESC键")
        time.sleep(2.25)
        # 标记翻牌完成，触发状态转换
        self.context.set_custom_data('card_flip_done', True)

    def _execute_character_switch_flow(self, image=None) -> None:
        """
        执行角色切换流程

        触发条件：
        1. 刷图次数达到 max_runs 且检测到 menu
        2. menu 连续检测超过阈值次数

        流程：
        1. 使用OCR检测是否有商店，如果有则按 ESC
        2. 按 F12，等待 0.25s
        3. 按 ESC，等待 0.25s
        4. 移动鼠标到选择角色按钮
        5. 点击
        6. 等待 0.25s
        7. 按 right
        8. 运行自动进图逻辑

        Args:
            image: 当前屏幕图像（用于OCR检测商店）
        """
        if not self.character_switcher:
            self.logger.warning("CharacterSwitcher 未初始化")
            return

        # 获取选择角色按钮位置
        char_button_pos = (960, 540)  # 默认位置
        if self.config.dungeon and self.config.dungeon.character_button_pos:
            char_button_pos = self.config.dungeon.character_button_pos

        self.logger.info(f"执行角色切换流程，按钮位置: {char_button_pos}")

        # 执行返回角色选择菜单流程（使用OCR检测商店）
        success = self.character_switcher.execute_return_to_character_selection(
            character_button_position=char_button_pos,
            current_image=image,
            context=self.context
        )

        if success:
            self.logger.info("角色切换流程完成，准备进入副本")

            # 运行自动进图逻辑
            time.sleep(0.5)
            enter_result = self.selection_manager.enter_dungeon()
            if enter_result.success:
                self.logger.success("自动进图完成")
            else:
                self.logger.error(f"自动进图失败: {enter_result.message}")
        else:
            self.logger.error("角色切换流程失败")

    def _character_switch_action(self) -> None:
        """角色切换动作"""
        if self.multi_char_manager and self.character_switcher:
            success = self.multi_char_manager.switch_to_next(self.context)
            if success:
                self.logger.info("Character switch completed")
                # 加载新角色的技能和Buff
                self._load_current_character_skills()
            else:
                self.logger.info("All characters completed")

    def _load_current_character_skills(self) -> None:
        """加载当前角色的技能和Buff配置"""
        if not self.config.multi_character or not self.config.multi_character.enabled:
            return

        # 获取当前角色索引
        current_index = 0
        if self.multi_char_manager:
            current_index = self.multi_char_manager.current_index

        role_list = self.config.multi_character.role_list
        if current_index < len(role_list):
            role_config = role_list[current_index]
            self._current_role_config = role_config
            self._apply_role_movement_config(role_config)

            # 重置并加载技能队列
            if self.skill_queue:
                self.skill_queue.stop()
                self.skill_queue.load_skills(role_config.art, role_config.art_time)
                self.skill_queue.start()
                self.logger.info(f"已加载角色 {role_config.name} 的技能: {len(role_config.art)} 个")

            # 加载Buff
            if self.buff_manager:
                self.buff_manager.load_buffs(role_config.buff)
                self.logger.info(f"已加载角色 {role_config.name} 的Buff: {len(role_config.buff)} 个")

    def _stuck_recovery_action(self) -> None:
        """
        卡住恢复动作 —— 每帧推进一级分级恢复（探测 → 动作 → 复检 → 升级）。

        注意 `recovery_done` 现在只由 handler 在恢复真正结束时设置。原实现无条件置位，
        即使因冷却没执行任何动作也会立刻弹回原状态。
        """
        if self.stuck_handler:
            self.stuck_handler.step_recovery(self.context, self.movement)

    def _on_exit_stuck_recovery(self) -> None:
        """退出卡住恢复状态：清掉标记，避免泄漏到下一帧重复触发。"""
        if self.movement:
            self.movement.stop()
        if self.stuck_handler:
            self.stuck_handler.end_session()
        self.context.set_custom_data('recovery_done', False)
        self.context.set_custom_data('stuck_detected', False)

    def _buffing_action(self) -> None:
        """释放Buff动作"""
        if self.buff_manager:
            if self.buff_manager.apply_buffs():
                self.logger.debug("Buffs applied")

        self.context.set_custom_data('buff_done', True)
        self.context.set_custom_data('need_buff', False)

    def _on_enter_menu(self, context) -> None:
        """进入MENU状态时的回调 - 按Tab键"""
        self.controller.key_press('tab')
        self.logger.debug("进入菜单状态：已按Tab键")

    def _on_enter_playing(self, context) -> None:
        """进入PLAYING状态时的回调 - 释放Buff"""
        if self.buff_manager:
            self.logger.info("进入PLAYING状态，释放Buff...")
            self.buff_manager.apply_buffs()

        # 启动技能队列
        if self.skill_queue and not self.skill_queue._running:
            self.skill_queue.start()

    def _on_enter_combat(self, context) -> None:
        """进入COMBAT状态时的回调 - 释放Buff"""
        if self.buff_manager:
            self.logger.info("进入COMBAT状态，释放Buff...")
            self.buff_manager.apply_buffs()

        # 启动技能队列
        if self.skill_queue and not self.skill_queue._running:
            self.skill_queue.start()

    def _show_debug(self, image, detections) -> None:
        """显示调试信息"""
        debug_image = self.visualizer.draw_all(
            image,
            detections,
            fps=self._stats.fps,
            state=self.state_machine.get_state().value
        )

        # 绘制准星
        # debug_image = self.visualizer.draw_crosshair(
        #     debug_image,
        #     self.context.screen_center
        # )

        # 仅在支持GUI的环境中显示调试窗口
        if not self._opencv_gui_available:
            return

        # 显示调试窗口
        window_name = 'Game Autopilot - Debug'

        # 获取图像尺寸
        h, w = debug_image.shape[:2]

        # 如果窗口尺寸变化，更新窗口大小
        if self._debug_window_size != (w, h):
            self._debug_window_size = (w, h)

            # 创建或更新窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, w, h)
            cv2.moveWindow(window_name, 0, 0)

        cv2.imshow(window_name, debug_image)

    def _update_fps(self, frame_time: float) -> None:
        """更新FPS计算"""
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._stats.fps = (len(self._frame_times) - 1) / elapsed

    def _check_opencv_gui(self) -> bool:
        """检查 OpenCV 是否支持 GUI 功能"""
        try:
            # 尝试创建一个窗口来检查 GUI 支持
            test_window = '__opencv_gui_test__'
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window)
            return True
        except cv2.error:
            self.logger.warning("OpenCV GUI 不可用（headless 模式），调试窗口已禁用")
            return False
        except Exception:
            return False

    def _start_hotkey_listener(self) -> None:
        """Start F1/F2/F3 controls even when the OpenCV debug window is disabled."""
        if self._hotkey_listener is not None:
            return
        try:
            from pynput import keyboard
            self._hotkey_listener = keyboard.Listener(on_press=self._on_hotkey_press)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception as error:
            self.logger.warning(f"Global hotkeys unavailable: {error}")

    def _stop_hotkey_listener(self) -> None:
        listener = getattr(self, '_hotkey_listener', None)
        if listener is not None:
            listener.stop()
            self._hotkey_listener = None

    def _on_hotkey_press(self, key) -> None:
        """Handle configured global hotkeys without doing expensive work in the listener thread."""
        key_name = getattr(key, 'name', None) or str(key).split('.')[-1].lower()
        now = time.monotonic()
        if now - self._hotkey_last_pressed.get(key_name, 0.0) < 0.35:
            return
        self._hotkey_last_pressed[key_name] = now
        hotkeys = self.config.hotkeys
        if key_name == hotkeys.stop:
            self._stop_requested = True
            self.pause()
        elif key_name == hotkeys.pause:
            if self._paused:
                self.resume()
            else:
                self.pause()
        elif key_name == hotkeys.start:
            self.resume()

    def pause(self) -> None:
        """暂停引擎"""
        self._paused = True
        self.controller.release_all_inputs()
        # 同步清空移动状态，否则 is_moving() 会继续报告"正在移动"
        if self.movement:
            self.movement.stop()
        self.logger.info("Engine paused")

    def resume(self) -> None:
        """恢复引擎"""
        self._paused = False
        self.logger.info("Engine resumed")

    def stop(self) -> None:
        """停止引擎"""
        self.logger.info("Stopping engine...")
        self._running = False
        self._stop_requested = True
        self._stop_hotkey_listener()
        self.controller.release_all_inputs()
        if self.movement:
            self.movement.stop()

        # 停止技能队列线程
        if self.skill_queue:
            self.skill_queue.stop()

        # 释放资源
        self.capture.close()

        if self.visualizer and self._opencv_gui_available:
            cv2.destroyAllWindows()

        self.logger.success("Engine stopped")

    def get_stats(self) -> EngineStats:
        """获取引擎统计信息"""
        return self._stats

    def is_running(self) -> bool:
        """引擎是否在运行"""
        return self._running

    def is_paused(self) -> bool:
        """引擎是否暂停"""
        return self._paused

    def set_character(self, character_id: str) -> bool:
        """
        切换角色

        Args:
            character_id: 角色ID

        Returns:
            是否成功切换
        """
        if not self.config.characters:
            self.logger.error("No characters configured")
            return False

        char_config = self.config.characters.presets.get(character_id)
        if not char_config:
            self.logger.error(f"Character not found: {character_id}")
            return False

        self.current_character = char_config
        self.config.characters.current = character_id

        # 重新创建技能管理器
        self.skill_manager = SkillManager(
            character_config=self.current_character,
            context=self.context,
            controller=self.controller
        )

        self.logger.info(f"Switched to character: {char_config.name}")
        return True

    def get_combat_info(self) -> dict:
        """获取战斗信息（调试用）"""
        info = {
            "character": self.current_character.name if self.current_character else None,
            "attack_key": self._get_attack_key(),
        }

        if self.skill_manager:
            info["skills"] = self.skill_manager.to_dict()

        return info


@dataclass
class EngineConfig:
    """引擎配置（简化版，用于快速创建）"""
    window_title: str = "Game Window"
    model_path: str = "models/best.pt"
    target_fps: int = 30
    device: str = "cuda"
    debug: bool = True

    def to_config(self) -> Config:
        """转换为完整配置"""
        config = Config()
        config.game.window_title = self.window_title
        config.game.target_fps = self.target_fps
        config.detection.device = self.device
        config.detection.models = {
            'main': {'path': self.model_path, 'conf': 0.5}
        }
        config.debug.enabled = self.debug
        return config
