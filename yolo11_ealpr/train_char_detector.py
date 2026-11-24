"""
Step 2 of the EALPR plan: train the character recognizer
(plate crop -> one box + class per character, 27 classes).

This is a second YOLO model, trained on prepared/chars_det. The plate crops
are tiny (~125x65 px); imgsz=320 makes Ultralytics enlarge each one so its
long side is 320 px, so no upscaled copy of the dataset is needed on disk.

Usage (from the "Claude attempt" folder):
    python train_char_detector.py                       # full training
    python train_char_detector.py --epochs 1 --name smoke_test
Then measure character / whole-plate accuracy with eval_char_recognizer.py.

Results go to runs/chars_det/<name>/ (weights/best.pt is the model to use).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "prepared" / "chars_det" / "data.yaml"
PROJECT = HERE / "runs" / "chars_det"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32,
                    help="images are small (320 px), so 32 fits in 4 GB")
    ap.add_argument("--workers", type=int, default=1,
                    help="keep low: each worker costs ~1-1.5 GB of RAM on "
                         "Windows (see train_plate_detector.py)")
    ap.add_argument("--name", default="yolo11n")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        patience=30,        # more classes -> slower to settle than plates
        imgsz=320,          # upscales the ~125 px crops to 320 px
        batch=args.batch,
        workers=args.workers,
        device=0,
        seed=42,
        project=str(PROJECT),
        name=args.name,
        exist_ok=True,
        # --- augmentation -------------------------------------------------
        fliplr=0.0,         # NEVER mirror: a mirrored character is wrong
        degrees=5.0,        # plates are almost always level
        # Defaults kept: colour jitter (lighting), scale/translate (crop
        # tightness varies) and mosaic (4 plates per image, more variety).
    )

    best = PROJECT / args.name / "weights" / "best.pt"
    print("\nStandard YOLO metrics on the TEST split:")
    YOLO(best).val(data=str(DATA), split="test", imgsz=320, batch=args.batch,
                   workers=args.workers, project=str(PROJECT),
                   name=f"{args.name}_test", exist_ok=True)
    print("\nNext: python eval_char_recognizer.py --weights", best)


if __name__ == "__main__":     # required on Windows (worker processes)
    main()
