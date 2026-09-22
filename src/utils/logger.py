"""
日志工具模块
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from colorama import init, Fore, Style

# 初始化colorama
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """彩色日志格式化器"""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # 添加颜色
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class GameLogger:
    """游戏日志器"""

    _instance: Optional['GameLogger'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        初始化日志器

        Args:
            log_dir: 日志文件目录
            log_level: 日志级别
            console_output: 是否输出到控制台
        """
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 创建logger
        self.logger = logging.getLogger("GameAutopilot")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 清除已存在的处理器

        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)  # 控制台也输出DEBUG级别日志
            console_format = ColorFormatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # 配置根日志器，让所有模块的日志都输出到控制台
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # bettercam 依赖 comtypes，后者把每次 COM 指针释放都记成 DEBUG。
        # 根日志器是 DEBUG，这些行会按帧刷屏（实测占日志一半以上），故单独压制。
        logging.getLogger("comtypes").setLevel(logging.WARNING)

        # 为根日志器添加控制台处理器（如果还没有）
        has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                         for h in root_logger.handlers)
        if not has_console:
            root_console = logging.StreamHandler(sys.stdout)
            root_console.setLevel(logging.DEBUG)
            root_console.setFormatter(ColorFormatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%H:%M:%S'
            ))
            root_logger.addHandler(root_console)

        # 文件处理器
        log_file = self.log_dir / f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def debug(self, msg: str) -> None:
        """调试日志"""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """信息日志"""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """警告日志"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """错误日志"""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """严重错误日志"""
        self.logger.critical(msg)

    def success(self, msg: str) -> None:
        """成功日志（绿色）"""
        self.logger.info(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")

    def game_event(self, event: str) -> None:
        """游戏事件日志（蓝色）"""
        self.logger.info(f"{Fore.BLUE}[EVENT] {event}{Style.RESET_ALL}")

    def combat(self, msg: str) -> None:
        """战斗日志（红色）"""
        self.logger.info(f"{Fore.RED}[COMBAT] {msg}{Style.RESET_ALL}")

    def detection(self, msg: str) -> None:
        """检测日志（青色）"""
        self.logger.debug(f"{Fore.CYAN}[DETECTION] {msg}{Style.RESET_ALL}")


# 全局日志实例
_logger: Optional[GameLogger] = None


def get_logger() -> GameLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = GameLogger()
    return _logger


def init_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> GameLogger:
    """
    初始化全局日志器

    Args:
        log_dir: 日志目录
        log_level: 日志级别

    Returns:
        GameLogger实例
    """
    global _logger
    _logger = GameLogger(log_dir=log_dir, log_level=log_level)
    return _logger
