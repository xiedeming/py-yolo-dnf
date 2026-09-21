"""Export a trained Ultralytics YOLO model to a static CPU ONNX artifact.

Run this on the GPU training computer, not on the low-spec runtime computer.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description='Export a static raw-output YOLO ONNX model for CPU runtime')
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--imgsz', required=True, type=int, choices=(320, 416, 512, 640))
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError('Weights not found: {}'.format(args.weights))
    if args.output.exists() and not args.overwrite:
        raise FileExistsError('Output exists; pass --overwrite to replace it: {}'.format(args.output))

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    exported = Path(model.export(
        format='onnx', imgsz=args.imgsz, opset=args.opset, dynamic=False,
        simplify=True, nms=False, batch=1,
    ))
    if not exported.is_file():
        raise RuntimeError('Ultralytics export did not produce {}'.format(exported))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)
    names = model.names
    metadata = {
        'source_weights': str(args.weights),
        'source_weights_sha256': sha256(args.weights),
        'onnx_sha256': sha256(args.output),
        'class_names': [names[index] for index in sorted(names)],
        'imgsz': args.imgsz,
        'opset': args.opset,
        'dynamic': False,
        'embedded_nms': False,
        'batch': 1,
    }
    metadata_path = args.output.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ONNX: {}'.format(args.output))
    print('Metadata: {}'.format(metadata_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
