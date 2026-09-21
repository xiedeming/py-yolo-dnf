#!/usr/bin/env python
"""
Game Autopilot - Python + YOLOv8 PC游戏自动通关程序

主入口文件
"""
import sys
import argparse
import ctypes
from pathlib import Path

# 启用 DPI 感知（必须在程序启动时设置，解决 Windows 显示缩放问题）
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.utils.hardware_profile import select_runtime_profile
from src.utils.logger import init_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Game Autopilot - PC游戏自动通关程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           # 按显卡显存自动选择高低配
  python main.py -c config/settings.yaml   # 指定配置文件
  python main.py -w "Game Title"           # 指定窗口标题
  python main.py -m models/best.pt         # 指定模型文件
  python main.py --list-windows            # 列出所有窗口
  python main.py --character warrior       # 指定角色
  python main.py --list-characters         # 列出可用角色
  python main.py --no-enter                # 跳过自动进入副本

热键:
  P - 暂停/恢复
  Q / ESC - 退出
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='配置文件路径（默认根据显卡显存自动选择高低配）'
    )

    parser.add_argument(
        '-w', '--window',
        type=str,
        help='游戏窗口标题'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        help='YOLOv8模型文件路径'
    )

    parser.add_argument(
        '-d', '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='推理设备（默认使用配置文件）'
    )

    parser.add_argument(
        '-f', '--fps',
        type=int,
        default=None,
        help='目标帧率，必须为正整数（默认使用配置文件）'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='禁用调试窗口'
    )

    parser.add_argument(
        '--list-windows',
        action='store_true',
        help='列出所有可见窗口'
    )

    # 角色和地图选择
    parser.add_argument(
        '-C', '--character',
        type=str,
        help='指定角色ID'
    )

    parser.add_argument(
        '-s', '--stage',
        type=str,
        help='指定地图ID'
    )

    parser.add_argument(
        '--list-characters',
        action='store_true',
        help='列出可用角色'
    )

    parser.add_argument(
        '--list-maps',
        action='store_true',
        help='列出可用地图'
    )

    parser.add_argument(
        '--no-enter',
        action='store_true',
        help='跳过自动进入副本流程'
    )

    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error('--fps must be a positive integer')
    return args


def list_windows():
    """列出所有可见窗口"""
    from src.capture.window_manager import WindowManager

    print("\n可见窗口列表:")
    print("-" * 60)

    windows = WindowManager.list_all_windows()
    for i, (hwnd, title) in enumerate(windows, 1):
        print(f"  {i:3d}. [{hwnd}] {title}")

    print("-" * 60)
    print(f"共 {len(windows)} 个可见窗口\n")


def list_characters(config):
    """列出可用角色"""
    print("\n可用角色列表:")
    print("-" * 60)

    if not config.characters or not config.characters.presets:
        print("  没有配置角色")
        return

    current = config.characters.current
    for char_id, char in config.characters.presets.items():
        marker = " *" if char_id == current else ""
        print(f"  {char_id:15s} - {char.name}{marker}")
        if char.description:
            print(f"                   {char.description}")
        if char.skills:
            print(f"                   技能: {len(char.skills)}个")

    print("-" * 60)
    print(f"共 {len(config.characters.presets)} 个角色")
    if current:
        print(f"当前选择: {current}")
    print()


def list_maps(config):
    """列出可用地图"""
    print("\n可用地图列表:")
    print("-" * 60)

    if not config.maps or not config.maps.presets:
        print("  没有配置地图")
        return

    current = config.maps.current
    for map_id, map_config in config.maps.presets.items():
        marker = " *" if map_id == current else ""
        print(f"  {map_id:15s} - {map_config.name}{marker}")
        if map_config.description:
            print(f"                   {map_config.description}")

    print("-" * 60)
    print(f"共 {len(config.maps.presets)} 个地图")
    if current:
        print(f"当前选择: {current}")
    print()


def _load_config_for_startup(args):
    """Load an explicit config or select the local hardware profile."""
    # Listing commands should remain usable on machines without PyTorch/CUDA.
    listing_mode = args.list_windows or args.list_characters or args.list_maps
    if args.config is not None:
        config_path = Path(args.config)
        profile = None
    elif listing_mode:
        config_path = project_root / 'config' / 'settings.yaml'
        profile = None
    else:
        profile = select_runtime_profile(
            project_root=str(project_root),
            requested_device=args.device,
        )
        config_path = profile.config_path

    if config_path.exists():
        config = ConfigLoader.load(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = ConfigLoader.create_default()

    if profile is not None:
        # The high profile may target the largest GPU (cuda:1, etc.).
        config.detection.device = profile.device
        print(f"自动选择运行配置: {profile.summary()}")
    return config


def _validate_detector_selection(config):
    """Reject combinations that the selected detector backend cannot load."""
    backend = str(getattr(config.detection, 'backend', 'ultralytics')).lower()
    device = str(getattr(config.detection, 'device', 'cpu')).lower()
    if backend == 'onnxruntime' and device != 'cpu':
        raise ValueError("ONNX Runtime 配置必须使用 CPU；请移除 -d cuda 或选择高配配置")

    if backend != 'onnxruntime':
        return
    for name, model in (config.detection.models or {}).items():
        model_path = str(model.get('path', ''))
        suffix = Path(model_path).suffix.lower()
        if suffix and suffix != '.onnx':
            raise ValueError(
                f"模型 {name} 使用 ONNX Runtime，但文件不是 .onnx: {model_path}"
            )


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = _load_config_for_startup(args)

    # 列出窗口模式
    if args.list_windows:
        list_windows()
        return 0

    # 列出角色模式
    if args.list_characters:
        list_characters(config)
        return 0

    # 列出地图模式
    if args.list_maps:
        list_maps(config)
        return 0

    # 命令行参数覆盖配置
    if args.window:
        config.game.window_title = args.window
    if args.model:
        if not config.detection.models:
            config.detection.models = {}
        model_config = dict(config.detection.models.get('main', {}))
        model_config['path'] = args.model
        config.detection.models['main'] = model_config
    if args.device:
        config.detection.device = args.device
    if args.fps is not None:
        config.game.target_fps = args.fps
    if args.no_debug:
        config.debug.enabled = False

    try:
        _validate_detector_selection(config)
    except ValueError as error:
        print(f"配置错误: {error}")
        return 2

    # 角色和地图选择
    if args.character:
        if config.characters and args.character in config.characters.presets:
            config.characters.current = args.character
            print(f"Selected character: {args.character}")
        else:
            print(f"Warning: Character '{args.character}' not found")

    if args.stage:
        if config.maps and args.stage in config.maps.presets:
            config.maps.current = args.stage
            print(f"Selected map: {args.stage}")
        else:
            print(f"Warning: Map '{args.stage}' not found")

    # 初始化日志
    init_logger()

    # 打印配置信息
    print("\n" + "=" * 60)
    print("Game Autopilot - PC游戏自动通关程序")
    print("=" * 60)
    print(f"窗口标题: {config.game.window_title}")
    print(f"目标帧率: {config.game.target_fps} FPS")
    print(f"推理设备: {config.detection.device}")
    print(f"调试模式: {'启用' if config.debug.enabled else '禁用'}")
    if config.detection.models:
        print(f"模型数量: {len(config.detection.models)}")
        for name, model in config.detection.models.items():
            print(f"  - {name}: {model.get('path', 'N/A')}")
    # 显示角色信息
    if config.characters and config.characters.current:
        char = config.characters.presets.get(config.characters.current)
        if char:
            print(f"当前角色: {char.name} ({config.characters.current})")
            print(f"  技能数量: {len(char.skills)}")
    # 显示地图信息
    if config.maps and config.maps.current:
        map_config = config.maps.presets.get(config.maps.current)
        if map_config:
            print(f"当前地图: {map_config.name} ({config.maps.current})")
    print("=" * 60 + "\n")

    engine = None
    try:
        # 创建引擎。将导入和初始化放入异常处理，模型/后端缺失时给出可读错误。
        from src.core.engine import GameEngine

        engine = GameEngine(config=config)
        print("按 P 暂停/恢复, Q 或 ESC 退出\n")
        # 是否自动进入副本
        auto_enter = not args.no_enter
        engine.start(auto_enter_dungeon=auto_enter)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if engine is not None and engine.is_running():
            engine.stop()

    print("\n程序已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
