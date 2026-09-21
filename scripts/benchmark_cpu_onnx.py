"""Benchmark a CPU ONNX model against a folder of captured game images."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.onnx_detector import ONNXDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark CPU ONNX detection on image fixtures')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--imgsz', required=True, type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('logs/cpu_benchmark.json'))
    return parser.parse_args()


def percentile(values, percentage):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentage))
    return ordered[index]


def main() -> int:
    args = parse_args()
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [path for path in args.images.rglob('*') if path.suffix.lower() in suffixes]
    if not images:
        raise ValueError('No image fixtures found under {}'.format(args.images))
    detector = ONNXDetector(
        model_path=str(args.model), input_size=args.imgsz, cpu_threads=args.threads,
    )
    loaded = []
    for path in images:
        image = cv2.imread(str(path))
        if image is not None:
            loaded.append(image)
    if not loaded:
        raise ValueError('No readable images found under {}'.format(args.images))
    for index in range(args.warmup):
        detector.predict(loaded[index % len(loaded)])
    elapsed_ms = []
    for image in loaded:
        start = time.perf_counter()
        detector.predict(image)
        elapsed_ms.append((time.perf_counter() - start) * 1000)
    report = {
        'model': str(args.model), 'imgsz': args.imgsz, 'threads': args.threads,
        'images': len(loaded), 'mean_ms': statistics.mean(elapsed_ms),
        'p50_ms': percentile(elapsed_ms, 0.50), 'p95_ms': percentile(elapsed_ms, 0.95),
        'detections_per_second': 1000 / statistics.mean(elapsed_ms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
