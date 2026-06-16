import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a soccer YOLO detector.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO weights.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/soccer")
    parser.add_argument("--name", default="detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    run_dir = Path(results.save_dir)
    print(f"Training complete: {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
