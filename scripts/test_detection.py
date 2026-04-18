#!/usr/bin/env python
"""
YOLOv8检测测试脚本
测试模型加载和推理功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import argparse

from src.detection.detector import YOLODetector
from src.capture.mss_capture import MSSCapture


def test_camera_detection(model_path: str, device: str = 'cuda'):
    """测试摄像头检测"""
    print("=" * 60)
    print("YOLOv8摄像头检测测试")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}")

    try:
        detector = YOLODetector(
            model_path=model_path,
            device=device,
            conf_threshold=0.5
        )
        print("模型加载成功!")

        # 打印类别信息
        class_names = detector.get_class_names()
        print(f"\n检测类别 ({len(class_names)}):")
        for id, name in class_names.items():
            print(f"  {id}: {name}")

    except FileNotFoundError:
        print(f"错误: 模型文件不存在: {model_path}")
        return
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 打开摄像头
    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    print("摄像头已打开，按 'q' 退出\n")

    # 预热
    print("预热模型...")
    warmup_time = detector.warmup()
    print(f"预热完成，耗时: {warmup_time:.3f}s\n")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 执行检测
        result = detector.predict(frame)

        # 绘制结果
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{det.class_name} {det.confidence:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        # 计算FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow('YOLOv8 Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n检测结束，平均FPS: {fps:.1f}")


def test_screen_detection(model_path: str, device: str = 'cuda'):
    """测试屏幕检测"""
    print("=" * 60)
    print("YOLOv8屏幕检测测试")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}")

    try:
        detector = YOLODetector(
            model_path=model_path,
            device=device,
            conf_threshold=0.5
        )
        print("模型加载成功!")

        class_names = detector.get_class_names()
        print(f"\n检测类别 ({len(class_names)}):")
        for id, name in class_names.items():
            print(f"  {id}: {name}")

    except FileNotFoundError:
        print(f"错误: 模型文件不存在: {model_path}")
        return
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 初始化屏幕捕获
    capture = MSSCapture(monitor_index=1)

    print("\n预热模型...")
    warmup_time = detector.warmup()
    print(f"预热完成，耗时: {warmup_time:.3f}s\n")

    print("开始屏幕检测，按 'q' 退出\n")

    frame_count = 0
    start_time = time.time()

    while True:
        # 捕获屏幕
        image = capture.capture()

        # 执行检测
        result = detector.predict(image)

        # 绘制结果
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, f"{det.class_name} {det.confidence:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        # 计算FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            image, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow('Screen Detection Test', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.close()
    cv2.destroyAllWindows()
    print(f"\n检测结束，平均FPS: {fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8检测测试')
    parser.add_argument('model', nargs='?', default='models/best_sy.pt', help='模型文件路径')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu'], help='推理设备')
    parser.add_argument('--screen', '-s', action='store_true', help='使用屏幕捕获而非摄像头')
    args = parser.parse_args()

    if args.screen:
        test_screen_detection(args.model, args.device)
    else:
        test_camera_detection(args.model, args.device)
