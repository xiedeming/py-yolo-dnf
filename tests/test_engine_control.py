"""Regression tests for engine shutdown and controller input release paths."""
import importlib.util
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

from src.detection.detector import detector_options as real_detector_options


ROOT = Path(__file__).resolve().parents[1]


class FakeKeyboard:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))


class FakeMouse(FakeKeyboard):
    pass


def load_input_controller():
    """Load the controller with fake pynput objects so no desktop input is sent."""
    pynput = types.ModuleType("pynput")
    keyboard = types.ModuleType("pynput.keyboard")
    mouse = types.ModuleType("pynput.mouse")
    key_names = (
        "enter space tab esc shift ctrl alt backspace delete up down left right "
        "home end page_up page_down f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12"
    ).split()
    keyboard.Controller = object
    keyboard.Key = types.SimpleNamespace(**{name: name for name in key_names})
    mouse.Controller = object
    mouse.Button = types.SimpleNamespace(left="left", right="right", middle="middle")

    with mock.patch.dict(
        sys.modules,
        {"pynput": pynput, "pynput.keyboard": keyboard, "pynput.mouse": mouse},
    ):
        spec = importlib.util.spec_from_file_location(
            "input_controller_under_test", ROOT / "src" / "control" / "input_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.InputController


def load_engine():
    """Load GameEngine against lightweight fakes for hardware and ML adapters."""
    modules = {}

    def add_module(name, **attributes):
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        modules[name] = module

    add_module("src", __path__=[])
    add_module("src.core", __path__=[])
    add_module("src.core.dungeon_runner", create_dungeon_runner_from_config=lambda *args, **kwargs: None)
    add_module("src.capture", __path__=[])
    add_module("src.detection", __path__=[])
    add_module("src.control", __path__=[])
    add_module("src.decision", __path__=[])
    add_module("src.selection", __path__=[])
    add_module("src.utils", __path__=[])
    add_module("src.debug", __path__=[])
    add_module("cv2", destroyAllWindows=lambda: None, waitKey=lambda _: -1)

    class FakeDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def create_fake_detector(model_configs, device, backend, cpu_threads=0):
        _, config = next(iter(model_configs.items()))
        options = real_detector_options(config)
        options['device'] = device
        return FakeDetector(**options)

    class FakeQueue:
        def __init__(self, controller):
            self.controller = controller
            self.loaded = None
            self.stopped = False

        def load_skills(self, art, art_time):
            self.loaded = (art, art_time)

        def stop(self):
            self.stopped = True

    class FakeBuffManager:
        def __init__(self, controller):
            self.controller = controller

        def load_buffs(self, buffs):
            self.buffs = buffs

    add_module("src.capture.mss_capture", MSSCapture=object)
    add_module("src.capture.window_manager", WindowManager=object)
    add_module(
        "src.detection.detector",
        YOLODetector=FakeDetector,
        MultiModelDetector=FakeDetector,
        Detection=object,
        detector_options=real_detector_options,
        create_detector=create_fake_detector,
    )
    add_module("src.control.input_controller", InputController=object, InputConfig=object)
    add_module("src.decision.game_context", GameContext=object, GameState=object)
    add_module("src.decision.state_machine", StateMachine=object, create_game_state_machine=lambda: None)
    add_module("src.decision.skill_manager", SkillManager=object, SkillQueue=FakeQueue, BuffManager=FakeBuffManager)
    add_module("src.decision.map_navigator", MapNavigator=object, create_map_navigator_from_config=lambda *args: None)
    add_module("src.decision.card_flipper", CardFlipper=object, create_card_flipper_from_config=lambda *args: None)
    add_module("src.decision.stuck_handler", StuckHandler=object, StuckConfig=object, create_stuck_handler_from_config=lambda *args: None)
    add_module("src.decision.character_switcher", CharacterSwitcher=object, create_character_switcher_from_config=lambda *args, **kwargs: None)
    add_module("src.decision.multi_character_manager", MultiCharacterManager=object, create_multi_character_manager_from_config=lambda *args, **kwargs: None)
    add_module("src.selection.selector", SelectionManager=object)
    add_module("src.utils.logger", GameLogger=object, init_logger=lambda *args: None)
    add_module("src.utils.config_loader", Config=object, ConfigLoader=object, CharacterConfig=object)
    add_module("src.debug.visualizer", DebugVisualizer=object)

    with mock.patch.dict(sys.modules, modules, clear=False):
        spec = importlib.util.spec_from_file_location(
            "src.core.engine_under_test", ROOT / "src" / "core" / "engine.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            return module.GameEngine, FakeDetector
        finally:
            sys.modules.pop(spec.name, None)


class InputControllerTests(unittest.TestCase):
    def test_release_all_inputs_releases_tracked_keyboard_and_mouse_inputs(self):
        InputController = load_input_controller()
        controller = InputController.__new__(InputController)
        controller.keyboard = FakeKeyboard()
        controller.mouse = FakeMouse()
        controller._current_moving_direction = None
        controller._pressed_keys = set()
        controller._pressed_mouse_buttons = set()
        controller._special_keys = {"left": "left"}

        controller.key_down("left")
        controller.mouse_down("right")
        controller.release_all_inputs()

        self.assertIn(("release", "left"), controller.keyboard.events)
        self.assertIn(("release", "right"), controller.mouse.events)
        self.assertIsNone(controller.get_moving_direction())

    def test_release_all_inputs_continues_after_a_release_failure(self):
        InputController = load_input_controller()

        class Keyboard(FakeKeyboard):
            def release(self, key):
                self.events.append(("release", key))
                if key == "left":
                    raise RuntimeError("simulated input backend failure")

        controller = InputController.__new__(InputController)
        controller.keyboard = Keyboard()
        controller.mouse = FakeMouse()
        controller._current_moving_direction = "left"
        controller._pressed_keys = {"left", "right"}
        controller._pressed_mouse_buttons = {"left"}

        controller.release_all_inputs()

        self.assertIn(("release", "right"), controller.keyboard.events)
        self.assertIn(("release", "left"), controller.mouse.events)
        self.assertEqual(controller._pressed_keys, {"left"})
        self.assertFalse(controller._pressed_mouse_buttons)
        self.assertEqual(controller.get_moving_direction(), "left")

        controller.keyboard.release = FakeKeyboard.release.__get__(controller.keyboard, FakeKeyboard)
        controller.release_all_inputs()

        self.assertFalse(controller._pressed_keys)
        self.assertIsNone(controller.get_moving_direction())

    def test_key_combo_releases_remaining_keys_after_a_release_failure(self):
        InputController = load_input_controller()

        class Keyboard(FakeKeyboard):
            def release(self, key):
                self.events.append(("release", key))
                if key == "c":
                    raise RuntimeError("simulated input backend failure")

        controller = InputController.__new__(InputController)
        controller.keyboard = Keyboard()
        controller._pressed_keys = set()
        controller._special_keys = {"ctrl": "ctrl"}
        controller.config = types.SimpleNamespace(humanize=False)

        controller.key_combo("ctrl", "c", duration=0)

        self.assertIn(("release", "ctrl"), controller.keyboard.events)
        self.assertEqual(controller._pressed_keys, {"c"})

    def test_failed_key_and_mouse_up_remain_tracked_for_retry(self):
        InputController = load_input_controller()

        class Keyboard(FakeKeyboard):
            def release(self, key):
                raise RuntimeError("simulated keyboard release failure")

        class Mouse(FakeMouse):
            def release(self, button):
                raise RuntimeError("simulated mouse release failure")

        controller = InputController.__new__(InputController)
        controller.keyboard = Keyboard()
        controller.mouse = Mouse()
        controller._pressed_keys = {"left"}
        controller._pressed_mouse_buttons = {"left"}
        controller._special_keys = {"left": "left"}

        with self.assertRaises(RuntimeError):
            controller.key_up("left")
        with self.assertRaises(RuntimeError):
            controller.mouse_up("left")

        self.assertEqual(controller._pressed_keys, {"left"})
        self.assertEqual(controller._pressed_mouse_buttons, {"left"})


class GameEngineTests(unittest.TestCase):
    def test_pause_and_stop_release_held_inputs(self):
        GameEngine, _ = load_engine()

        class Controller:
            def __init__(self):
                self.release_calls = 0

            def release_all_inputs(self):
                self.release_calls += 1

        class Logger:
            def info(self, *_):
                pass

            def success(self, *_):
                pass

        controller = Controller()
        engine = GameEngine.__new__(GameEngine)
        engine.controller = controller
        engine.logger = Logger()
        engine._paused = False
        engine._running = True
        engine.skill_queue = None
        engine.capture = types.SimpleNamespace(close=lambda: None)
        engine.visualizer = None
        engine._opencv_gui_available = False

        engine.pause()
        engine.stop()

        self.assertTrue(engine._paused)
        self.assertFalse(engine._running)
        self.assertEqual(controller.release_calls, 2)

    def test_single_model_passes_all_detector_options(self):
        GameEngine, FakeDetector = load_engine()
        engine = GameEngine.__new__(GameEngine)
        engine.config = types.SimpleNamespace(
            detection=types.SimpleNamespace(
                device="cpu",
                backend="ultralytics",
                cpu_threads=0,
                models={
                    "main": {
                        "path": "models/main.pt",
                        "conf_threshold": 0.2,
                        "iou_threshold": 0.3,
                        "classes": [1, 2],
                    }
                },
            )
        )
        engine.logger = types.SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)

        engine._init_detector()

        self.assertIsInstance(engine.detector, FakeDetector)
        self.assertEqual(
            engine.detector.kwargs,
            {
                "model_path": "models/main.pt",
                "device": "cpu",
                "conf_threshold": 0.2,
                "iou_threshold": 0.3,
                "classes": [1, 2],
            },
        )

    def test_multi_character_queue_is_available_to_skill_manager(self):
        GameEngine, _ = load_engine()
        role = types.SimpleNamespace(
            id="role-1",
            name="Role 1",
            position=(0, 0),
            dungeon_runs=1,
            art=[["x"]],
            art_time={"x": 1},
            buff=[],
        )
        engine = GameEngine.__new__(GameEngine)
        engine.controller = object()
        engine.capture = object()
        engine.window_manager = object()
        engine.logger = types.SimpleNamespace(info=lambda *_: None)
        engine.config = types.SimpleNamespace(
            multi_character=types.SimpleNamespace(
                enabled=True, role_list=[role], start_name="", end_name=""
            ),
            map_routes=None,
            card_flip=None,
            stuck_recovery=None,
            ocr=None,
            dungeon=None,
            schedule=None,
        )
        engine.skill_manager = types.SimpleNamespace(skill_queue=None)
        engine.skill_queue = None
        engine.buff_manager = None
        engine.map_navigator = None
        engine.card_flipper = None
        engine.stuck_handler = None
        engine.character_switcher = None
        engine.multi_char_manager = None

        dungeon_runner = types.ModuleType("src.core.dungeon_runner")
        dungeon_runner.create_dungeon_runner_from_config = lambda *args, **kwargs: None
        with mock.patch.dict(sys.modules, {"src.core.dungeon_runner": dungeon_runner}):
            engine._init_dnf_modules()

        self.assertIs(engine.skill_manager.skill_queue, engine.skill_queue)

    def test_process_frame_syncs_state_before_callbacks_and_actions(self):
        GameEngine, _ = load_engine()
        next_state = object()
        engine = GameEngine.__new__(GameEngine)
        engine.window_manager = types.SimpleNamespace(get_client_screen_rect=lambda: (0, 0, 1, 1))
        engine.capture = types.SimpleNamespace(capture_region=lambda _: types.SimpleNamespace(shape=(1, 1)))
        engine.detector = None
        engine.context = types.SimpleNamespace(
            update=lambda _: None,
            set_screen_center=lambda *_: None,
            state=None,
        )
        engine.skill_manager = None
        engine.state_machine = types.SimpleNamespace(update=lambda _: None, get_state=lambda: next_state)
        engine._stats = types.SimpleNamespace(capture_time=0, detection_time=0, decision_time=0, total_time=0, frame_count=0)
        engine._state_callbacks = {next_state: lambda context: self.assertIs(context.state, next_state)}
        engine._execute_action = lambda state, image: self.assertIs(state, next_state)
        engine.visualizer = None
        engine._save_screenshots = False
        engine.logger = types.SimpleNamespace(debug=lambda *_: None, error=lambda *_: None)

        engine._process_frame()

        self.assertIs(engine.context.state, next_state)

    def test_debug_window_pause_key_uses_pause_lifecycle_method(self):
        GameEngine, _ = load_engine()
        keys = iter((ord("p"), ord("q")))
        GameEngine._main_loop.__globals__["cv2"].waitKey = lambda _: next(keys)
        engine = GameEngine.__new__(GameEngine)
        engine.config = types.SimpleNamespace(game=types.SimpleNamespace(target_fps=0))
        engine.logger = types.SimpleNamespace(info=lambda *_: None)
        engine._running = True
        engine._paused = False
        engine.visualizer = object()
        engine._opencv_gui_available = True
        engine._process_frame = lambda: None
        engine._update_fps = lambda _: None
        pause_calls = []
        engine.pause = lambda: (pause_calls.append(True), setattr(engine, "_paused", True))
        engine.resume = lambda: self.fail("P must pause an unpaused engine")

        engine._main_loop()

        self.assertEqual(pause_calls, [True])


if __name__ == "__main__":
    unittest.main()
