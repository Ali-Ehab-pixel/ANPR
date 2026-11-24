"""
Step 2 of the EALPR plan: train the plate detector (vehicle photo -> plate box).

Fine-tunes a small pretrained YOLO model (Ultralytics) on prepared/plates_det,
then tests it ONCE on the held-out test split.

Usage (from the "Claude attempt" folder):
    python train_plate_detector.py                 # full training
    python train_plate_detector.py --epochs 1 --name smoke_test   # quick check
    python train_plate_detector.py --test-only runs/plates_det/yolo11n/weights/best.pt

Results go to runs/plates_det/<name>/ :
    weights/best.pt   the best model (by val mAP) - this is the one to use
    results.csv/png   loss and mAP per epoch
    test/             metrics and plots on the test split
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "prepared" / "plates_det" / "data.yaml"
PROJECT = HERE / "runs" / "plates_det"


def evaluate(weights, name, workers):
    """Evaluate a trained model on the TEST split (never used in training)."""
    model = YOLO(weights)
    m = model.val(data=str(DATA), split="test", imgsz=640, batch=8,
                  workers=workers, project=str(PROJECT), name=f"{name}_test",
                  exist_ok=True, plots=True)
    print("\n===== TEST SPLIT RESULTS =====")
    print(f"precision : {m.box.mp:.4f}")
    print(f"recall    : {m.box.mr:.4f}")
    print(f"mAP50     : {m.box.map50:.4f}   (target > 0.95)")
    print(f"mAP50-95  : {m.box.map:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights to start from (downloaded "
                         "automatically the first time)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8,
                    help="8 fits in 4 GB of GPU memory at imgsz 640")
    ap.add_argument("--workers", type=int, default=1,
                    help="data-loader processes. On Windows each one loads "
                         "PyTorch again (~1-1.5 GB RAM) and validation uses "
                         "twice this many, so keep it low on a 16 GB laptop. "
                         "Use 0 if you still get 'Insufficient memory'.")
    ap.add_argument("--name", default="yolo11n")
    ap.add_argument("--test-only", metavar="WEIGHTS",
                    help="skip training, just evaluate these weights")
    args = ap.parse_args()

    if args.test_only:
        evaluate(args.test_only, args.name, args.workers)
        return

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        patience=20,        # stop early if val mAP hasn't improved in 20 epochs
        imgsz=640,          # photos are ~640-720 px, so no detail is lost
        batch=args.batch,
        workers=args.workers,  # see --workers help: RAM, not GPU, is the limit
        device=0,           # the RTX 3050
        seed=42,
        project=str(PROJECT),
        name=args.name,
        exist_ok=True,
        # --- augmentation -------------------------------------------------
        fliplr=0.0,         # NEVER mirror: flipped Arabic text doesn't exist
        degrees=5.0,        # small rotations only (+-5 degrees)
        # everything else (mosaic, colour jitter, scale, translate) keeps
        # Ultralytics' defaults, which suit this task well.
    )

    best = PROJECT / args.name / "weights" / "best.pt"
    evaluate(best, args.name, args.workers)


# The guard is required on Windows: the data loader starts worker processes
# that re-import this file, and without it they would start training again.
if __name__ == "__main__":
    main()
