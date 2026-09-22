"""
窗口管理模块 - Windows窗口定位和管理
"""
import ctypes
import win32api
import win32gui
import win32con
import win32process
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """窗口信息"""
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    client_rect: Tuple[int, int, int, int]
    is_visible: bool
    is_minimized: bool


class WindowManager:
    """Windows窗口管理器"""

    def __init__(self, window_title: Optional[str] = None):
        """
        初始化窗口管理器

        Args:
            window_title: 窗口标题（支持部分匹配）
        """
        self.window_title = window_title
        self.hwnd: Optional[int] = None
        self._window_info: Optional[WindowInfo] = None

    def find_window(self, title: Optional[str] = None) -> bool:
        """
        查找游戏窗口

        Args:
            title: 窗口标题，如果为None则使用初始化时设置的标题

        Returns:
            是否找到窗口
        """
        if title:
            self.window_title = title

        if not self.window_title:
            return False

        # 尝试精确匹配
        self.hwnd = win32gui.FindWindow(None, self.window_title)

        if not self.hwnd:
            # 尝试部分匹配
            self.hwnd = self._find_window_by_partial_title(self.window_title)

        if self.hwnd:
            self._update_window_info()
            return True

        return False

    def _find_window_by_partial_title(self, partial_title: str) -> Optional[int]:
        """
        通过部分标题查找窗口

        Args:
            partial_title: 部分窗口标题

        Returns:
            窗口句柄或None
        """
        found_hwnd = None

        def enum_callback(hwnd, _):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if partial_title.lower() in title.lower():
                    found_hwnd = hwnd
                    return False  # 停止枚举
            return True

        win32gui.EnumWindows(enum_callback, None)
        return found_hwnd

    def _update_window_info(self) -> None:
        """更新窗口信息"""
        if not self.hwnd:
            return

        rect = win32gui.GetWindowRect(self.hwnd)
        client_rect = win32gui.GetClientRect(self.hwnd)
        title = win32gui.GetWindowText(self.hwnd)

        # 获取窗口状态
        style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_STYLE)
        is_visible = bool(style & win32con.WS_VISIBLE)
        is_minimized = bool(win32gui.IsIconic(self.hwnd))

        self._window_info = WindowInfo(
            hwnd=self.hwnd,
            title=title,
            rect=rect,
            client_rect=client_rect,
            is_visible=is_visible,
            is_minimized=is_minimized
        )

        # 打印窗口信息
        left, top, right, bottom = rect
        cleft, ctop, cright, cbottom = client_rect
        win_width, win_height = right - left, bottom - top
        client_width, client_height = cright - cleft, cbottom - ctop

        # 获取客户区屏幕坐标
        client_screen_left, client_screen_top = win32gui.ClientToScreen(self.hwnd, (0, 0))

        logger.info(f"找到窗口: '{title}'")
        logger.info(f"  句柄: {self.hwnd}")
        logger.info(f"  窗口矩形: ({left}, {top}) - ({right}, {bottom}), 大小: {win_width}x{win_height}")
        logger.info(f"  客户区矩形: ({cleft}, {ctop}) - ({cright}, {cbottom}), 大小: {client_width}x{client_height}")
        logger.info(f"  客户区屏幕坐标: ({client_screen_left}, {client_screen_top}), 大小: {client_width}x{client_height}")
        logger.info(f"  可见: {is_visible}, 最小化: {is_minimized}")

    def get_window_rect(self) -> Tuple[int, int, int, int]:
        """
        获取窗口位置和大小

        Returns:
            (left, top, right, bottom)
        """
        if not self.hwnd:
            raise RuntimeError("窗口未找到")

        rect = win32gui.GetWindowRect(self.hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        logger.debug(f"窗口矩形: left={left}, top={top}, right={right}, bottom={bottom}, size={width}x{height}")

        return rect

    def get_window_size(self) -> Tuple[int, int]:
        """
        获取窗口大小

        Returns:
            (width, height)
        """
        left, top, right, bottom = self.get_window_rect()
        return (right - left, bottom - top)

    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """
        获取客户区位置和大小

        Returns:
            (left, top, right, bottom) - 客户区坐标
        """
        if not self.hwnd:
            raise RuntimeError("窗口未找到")
        return win32gui.GetClientRect(self.hwnd)

    def get_client_screen_rect(self) -> Tuple[int, int, int, int]:
        """
        获取客户区在屏幕上的实际坐标

        Returns:
            (left, top, right, bottom) - 屏幕坐标
        """
        if not self.hwnd:
            raise RuntimeError("窗口未找到")

        # 获取窗口左上角的屏幕坐标
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))

        # 获取客户区大小
        client_rect = win32gui.GetClientRect(self.hwnd)
        width = client_rect[2]
        height = client_rect[3]

        rect = (left, top, left + width, top + height)
        logger.debug(f"客户区屏幕坐标: ({left}, {top}) - ({left + width}, {top + height}), 大小: {width}x{height}")

        return rect

    def is_foreground(self) -> bool:
        """当前窗口是否为前台窗口"""
        if not self.hwnd:
            return False
        try:
            return win32gui.GetForegroundWindow() == self.hwnd
        except Exception:
            return False

    def bring_to_front(self) -> None:
        """将窗口置顶并激活为前台窗口"""
        if not self.hwnd:
            raise RuntimeError("窗口未找到")

        # 如果窗口最小化，先恢复
        if win32gui.IsIconic(self.hwnd):
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

        if win32gui.GetForegroundWindow() == self.hwnd:
            return

        if not self._set_foreground(self.hwnd):
            logger.warning("无法将游戏窗口切到前台，模拟按键可能不会进游戏")

    @staticmethod
    def _set_foreground(hwnd: int) -> bool:
        """
        尽力把 hwnd 变成前台窗口，返回是否成功。

        调用进程不是前台进程时，SetForegroundWindow 会被系统静默拒绝（前台
        锁定）。这里叠加两种业界常用解法：把**当前线程** attach 到前台线程
        （attach 两个外部线程会报"拒绝访问"），以及先发一次合成的 ALT 击键让
        系统认为用户刚有输入。
        """
        user32 = ctypes.windll.user32
        vk_menu = 0x12
        keyeventf_keyup = 0x0002

        current_thread = win32api.GetCurrentThreadId()
        foreground = win32gui.GetForegroundWindow()
        foreground_thread = (
            win32process.GetWindowThreadProcessId(foreground)[0] if foreground else 0
        )

        attached = False
        if foreground_thread and foreground_thread != current_thread:
            try:
                attached = bool(
                    win32process.AttachThreadInput(current_thread, foreground_thread, True)
                )
            except Exception as error:
                logger.debug(f"AttachThreadInput 失败: {error}")

        try:
            user32.keybd_event(vk_menu, 0, 0, 0)
            user32.keybd_event(vk_menu, 0, keyeventf_keyup, 0)
            user32.SetForegroundWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)
        except Exception as error:
            logger.debug(f"SetForegroundWindow 失败: {error}")
        finally:
            if attached:
                try:
                    win32process.AttachThreadInput(current_thread, foreground_thread, False)
                except Exception:
                    pass

        return win32gui.GetForegroundWindow() == hwnd

    @staticmethod
    def make_non_activating(window_title: str) -> None:
        """
        给指定标题的窗口加上 WS_EX_NOACTIVATE 扩展样式。

        调试窗口由 OpenCV 创建，出现时会抢走前台焦点，导致之后模拟的按键全部
        打到调试窗口而不是游戏（表现为"键盘不操作游戏"）。加上该样式后窗口
        显示/更新都不再激活自己。
        """
        try:
            hwnd = win32gui.FindWindow(None, window_title)
            if not hwnd:
                return
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if not ex_style & win32con.WS_EX_NOACTIVATE:
                win32gui.SetWindowLong(
                    hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_NOACTIVATE
                )
        except Exception as error:
            logger.debug(f"设置窗口不抢焦点失败: {error}")

    def maximize(self) -> None:
        """最大化窗口"""
        if not self.hwnd:
            raise RuntimeError("窗口未找到")
        win32gui.ShowWindow(self.hwnd, win32con.SW_MAXIMIZE)

    def minimize(self) -> None:
        """最小化窗口"""
        if not self.hwnd:
            raise RuntimeError("窗口未找到")
        win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)

    def restore(self) -> None:
        """恢复窗口"""
        if not self.hwnd:
            raise RuntimeError("窗口未找到")
        win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

    def get_window_info(self) -> Optional[WindowInfo]:
        """获取窗口信息"""
        return self._window_info

    def is_window_valid(self) -> bool:
        """检查窗口是否仍然有效"""
        if not self.hwnd:
            return False
        try:
            return bool(win32gui.IsWindow(self.hwnd))
        except Exception:
            return False

    @staticmethod
    def list_all_windows() -> List[Tuple[int, str]]:
        """
        列出所有可见窗口

        Returns:
            [(hwnd, title), ...]
        """
        windows = []

        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))
            return True

        win32gui.EnumWindows(enum_callback, None)
        return windows

    @staticmethod
    def print_all_windows() -> None:
        """打印所有可见窗口（用于调试）"""
        windows = WindowManager.list_all_windows()
        print("Visible Windows:")
        print("-" * 60)
        for hwnd, title in windows:
            print(f"  [{hwnd}] {title}")
        print("-" * 60)
