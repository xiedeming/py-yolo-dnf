#!/usr/bin/env python
"""截屏实时推理 - 直接把屏幕画面喂给模型

捕获当前屏幕（或指定区域），逐帧送入检测模型，输出检测结果与实时帧率。
复用项目的 MSSCapture 和 YOLODetector，因此 .pt / .onnx / .engine 都能直接加载。

用法:
    # 默认使用训练好的 yolo26m
    python scripts/live_infer.py

    # 使用 TensorRT 引擎（推理快约 2.5 倍）
    python scripts/live_infer.py -m models/best_yolo26m_fp16.engine

    # 使用 DXGI 捕获（比 mss 快约 6 倍）
    python scripts/live_infer.py --capture bettercam

    # 显示检测窗口（按 q 退出）
    python scripts/live_infer.py --show

    # 只跑 200 帧后输出统计
    python scripts/live_infer.py -n 200

    # 只截取屏幕区域 left top width height
    python scripts/live_infer.py --region 0 0 1280 720

    # 保存一张标注后的截图
    python scripts/live_infer.py -n 1 --save logs/snap.jpg
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.capture import CAPTURE_METHODS, create_capture
from src.detection.detector import YOLODetector

DEFAULT_MODEL = "models/best_yolo26m.pt"


def resolve_path(path: str) -> Path:
    """相对路径按项目根目录解析，这样在任何工作目录下都能运行。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def configured_capture() -> str:
    """默认捕获后端取自 config/settings.yaml，保证与主程序行为一致。"""
    try:
        from src.utils.config_loader import ConfigLoader

        cfg = ConfigLoader.load(str(ROOT / "config" / "settings.yaml"))
        return getattr(cfg.capture, "method", "bettercam") or "bettercam"
    except Exception:
        return "bettercam"


def draw(image, result):
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det.class_name} {det.confidence:.2f}",
                    (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description="截屏实时推理")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="模型路径 (.pt/.onnx/.engine)")
    parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("-c", "--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("-i", "--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--monitor", type=int, default=1, help="显示器索引 (1 = 主显示器)")
    parser.add_argument("--capture", default=configured_capture(), choices=CAPTURE_METHODS,
                        help="捕获后端: bettercam(DXGI，实测快约 6 倍) 或 mss；默认取 config/settings.yaml")
    parser.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "W", "H"),
                        help="只截取该区域，默认整个显示器")
    parser.add_argument("-n", "--frames", type=int, default=0, help="运行帧数，0 表示一直运行")
    parser.add_argument("--show", action="store_true", help="显示检测窗口（按 q 退出）")
    parser.add_argument("--save", help="保存一张标注后的截图到该路径")
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    print("=" * 64)
    print("截屏实时推理")
    print("=" * 64)
    print(f"模型    : {model_path}")
    print(f"设备    : {args.device}")
    print(f"阈值    : conf={args.conf}  iou={args.iou}")
    print(f"捕获    : {'区域 ' + str(args.region) if args.region else '显示器 ' + str(args.monitor)}")
    print(f"帧数    : {args.frames if args.frames else '无限'}")

    if not model_path.exists():
        print(f"\n错误: 模型文件不存在: {model_path}")
        return 1

    detector = YOLODetector(
        model_path=str(model_path),
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print(f"\n类别 ({len(detector.get_class_names())}): "
          f"{', '.join(detector.get_class_names().values())}")

    capture = create_capture(method=args.capture, monitor_index=args.monitor)
    print(f"捕获后端: {type(capture).__name__}")
    print("\n预热模型...")
    print(f"预热完成: {detector.warmup():.3f}s")
    print("\n开始检测" + ("，按 q 退出" if args.show else "") + "\n")

    capture_times, infer_times, class_totals = [], [], Counter()
    frame_count, start = 0, time.perf_counter()
    snapshot_saved = False

    try:
        while True:
            t0 = time.perf_counter()
            image = (capture.capture_region(tuple(args.region)) if args.region
                     else capture.capture())
            t1 = time.perf_counter()

            result = detector.predict(image)
            t2 = time.perf_counter()

            capture_times.append((t1 - t0) * 1000)
            infer_times.append((t2 - t1) * 1000)
            class_totals.update(d.class_name for d in result.detections)

            frame_count += 1
            elapsed = time.perf_counter() - start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            if result.detections:
                summary = ", ".join(
                    f"{n}({c})" for n, c in
                    Counter(d.class_name for d in result.detections).most_common()
                )
            else:
                summary = "-"

            print(f"[{frame_count:5d}] 捕获 {capture_times[-1]:5.1f}ms  "
                  f"推理 {infer_times[-1]:6.1f}ms  总 {fps:5.1f} FPS  |  {summary}")

            if args.save and not snapshot_saved:
                out = resolve_path(args.save)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), draw(image.copy(), result))
                print(f"       已保存标注截图: {out}")
                snapshot_saved = True

            if args.show:
                cv2.putText(image, f"{fps:.1f} FPS", (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("live_infer", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.frames and frame_count >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        capture.close()
        if args.show:
            cv2.destroyAllWindows()

    if frame_count:
        total = time.perf_counter() - start
        print("\n" + "=" * 64)
        print(f"共 {frame_count} 帧, {total:.1f}s, 平均 {frame_count / total:.1f} FPS")
        print(f"捕获平均 {sum(capture_times) / len(capture_times):.1f}ms  "
              f"推理平均 {sum(infer_times) / len(infer_times):.1f}ms")
        reused = getattr(capture, "get_reused_frames", None)
        if callable(reused) and reused():
            print(f"注意: {reused()} 帧画面未变化，DXGI 未交付新帧，复用了上一帧")
        if class_totals:
            print("累计检测: " + ", ".join(f"{n}={c}" for n, c in class_totals.most_common()))
        else:
            print("累计检测: 无")
    return 0


if __name__ == "__main__":
    sys.exit(main())
