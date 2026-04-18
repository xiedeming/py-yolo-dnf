#!/usr/bin/env python
"""
输入控制测试脚本
测试键盘和鼠标模拟功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import argparse

from src.control.input_controller import InputController, InputConfig


def test_mouse_control():
    """测试鼠标控制"""
    print("=" * 60)
    print("鼠标控制测试")
    print("=" * 60)
    print("\n警告: 此测试会移动您的鼠标!")
    print("请将鼠标移动到左上角以中止测试\n")

    controller = InputController()

    print("3秒后开始测试...")
    time.sleep(3)

    # 获取当前位置
    x, y = controller.get_mouse_position()
    print(f"当前鼠标位置: ({x}, {y})")

    # 测试平滑移动
    print("\n测试平滑移动...")
    screen_width = 1920  # 假设屏幕宽度
    screen_height = 1080  # 假设屏幕高度

    # 移动到四个角
    corners = [
        (100, 100),
        (screen_width - 100, 100),
        (screen_width - 100, screen_height - 100),
        (100, screen_height - 100),
        (screen_width // 2, screen_height // 2)
    ]

    for i, (tx, ty) in enumerate(corners):
        # 检查是否被用户中止
        current_x, current_y = controller.get_mouse_position()
        if current_x < 10 and current_y < 10:
            print("\n用户中止测试")
            return

        print(f"  移动到 ({tx}, {ty})")
        controller.mouse_move(tx, ty)
        time.sleep(0.5)

    print("\n鼠标控制测试完成!")


def test_keyboard_control():
    """测试键盘控制"""
    print("=" * 60)
    print("键盘控制测试")
    print("=" * 60)
    print("\n警告: 此测试会模拟键盘输入!")
    print("请在5秒内切换到一个文本编辑器\n")

    controller = InputController()

    print("5秒后开始测试...")
    time.sleep(5)

    # 测试输入文本
    print("输入测试文本...")
    controller.type_text("Hello, Game Autopilot!", interval=0.05)

    time.sleep(0.5)

    # 测试按键
    print("测试按键...")
    controller.key_press('enter')

    time.sleep(0.2)

    # 测试组合键
    print("测试组合键 (Ctrl+A)...")
    controller.key_combo('ctrl', 'a')

    print("\n键盘控制测试完成!")


def test_click():
    """测试点击"""
    print("=" * 60)
    print("点击测试")
    print("=" * 60)
    print("\n警告: 此测试会模拟鼠标点击!")
    print("请将鼠标移动到左上角以中止测试\n")

    controller = InputController()

    print("3秒后开始测试...")
    time.sleep(3)

    # 获取当前位置并点击
    x, y = controller.get_mouse_position()
    print(f"当前位置: ({x}, {y})")
    print("将在当前位置进行点击测试...")

    for i in range(3):
        # 检查中止
        current_x, current_y = controller.get_mouse_position()
        if current_x < 10 and current_y < 10:
            print("\n用户中止测试")
            return

        print(f"  点击 {i + 1}")
        controller.mouse_click('left')
        time.sleep(0.5)

    print("\n点击测试完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='输入控制测试')
    parser.add_argument('--mouse', '-m', action='store_true', help='测试鼠标控制')
    parser.add_argument('--keyboard', '-k', action='store_true', help='测试键盘控制')
    parser.add_argument('--click', '-c', action='store_true', help='测试点击')
    args = parser.parse_args()

    if not any([args.mouse, args.keyboard, args.click]):
        # 运行所有测试
        print("运行所有测试...\n")
        test_mouse_control()
        print("\n" + "=" * 60 + "\n")
        test_keyboard_control()
        print("\n" + "=" * 60 + "\n")
        test_click()
    else:
        if args.mouse:
            test_mouse_control()
        if args.keyboard:
            test_keyboard_control()
        if args.click:
            test_click()
