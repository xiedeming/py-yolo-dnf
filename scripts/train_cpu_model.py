"""Train a YOLO11n candidate on the GPU computer using external dataset paths."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train the CPU-oriented YOLO11n candidate')
    parser.add_argument('--data', required=True, type=Path, help='External dataset YAML path')
    parser.add_argument('--project', type=Path, default=Path('runs/cpu_training'))
    parser.add_argument('--name', default='yolo11n_640')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError('Dataset YAML not found: {}'.format(args.data))
    from ultralytics import YOLO

    model = YOLO('yolo11n.pt')
    model.train(
        data=str(args.data), imgsz=640, epochs=args.epochs, patience=args.patience,
        batch=args.batch, device=args.device, seed=args.seed, project=str(args.project),
        name=args.name, cache=False,
    )
    print('Best weights: {}'.format(args.project / args.name / 'weights' / 'best.pt'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
