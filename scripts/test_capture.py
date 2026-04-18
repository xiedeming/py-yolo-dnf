#!/usr/bin/env python
"""
屏幕捕获测试脚本
测试MSS屏幕捕获功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time

from src.capture.mss_capture import MSSCapture
from src.capture.window_manager import WindowManager


def test_capture():
    """测试屏幕捕获"""
    print("=" * 60)
    print("屏幕捕获测试")
    print("=" * 60)
    print("\n按 'q' 退出\n")

    capture = MSSCapture(monitor_index=1, target_fps=60)

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            # 捕获屏幕
            image = capture.capture()

            # 计算FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # 显示FPS
            cv2.putText(
                image, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            # 显示图像
            cv2.imshow('Screen Capture Test', image)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        capture.close()
        cv2.destroyAllWindows()

    print(f"\n捕获结束，平均FPS: {fps:.1f}")


def test_window_capture():
    """测试窗口捕获"""
    print("=" * 60)
    print("窗口捕获测试")
    print("=" * 60)

    # 列出所有窗口
    windows = WindowManager.list_all_windows()
    print("\n可见窗口列表:")
    for i, (hwnd, title) in enumerate(windows[:20], 1):
        print(f"  {i:2d}. {title}")

    # 选择窗口
    print("\n输入要捕获的窗口序号（或按Enter跳过）:")
    try:
        choice = input("> ")
        if not choice:
            print("跳过窗口捕获测试")
            return

        idx = int(choice) - 1
        if 0 <= idx < len(windows):
            hwnd, title = windows[idx]
            print(f"\n选择窗口: {title}")

            # 创建窗口管理器
            wm = WindowManager(title)
            if wm.find_window():
                capture = MSSCapture()

                print("按 'q' 退出\n")
                while True:
                    rect = wm.get_window_rect()
                    left, top, right, bottom = rect
                    image = capture.capture_region((left, top, right - left, bottom - top))

                    cv2.imshow('Window Capture', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                capture.close()
                cv2.destroyAllWindows()
    except ValueError:
        print("无效输入")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='屏幕捕获测试')
    parser.add_argument('--window', '-w', action='store_true', help='测试窗口捕获')
    args = parser.parse_args()

    if args.window:
        test_window_capture()
    else:
        test_capture()
