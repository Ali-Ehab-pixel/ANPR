"""Prepare a YOLO license-plate dataset with train/val/test splits.

Default behavior is a dry run: validate the source data and print the split
plan without creating or modifying any files. Add --execute to create the
output dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SEED = 42


@dataclass(frozen=True)  
class Sample:
    image: Path
    label: Path
# This class represents a single sample consisting of an image and its corresponding label, and is immutable after creation.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create the output dataset. Without this flag, only print a plan.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()

#this function sets up command-line argument parsing for the script. It defines several arguments that can be passed when running the script, including:
#--execute: A flag that, when provided, indicates that the script should actually create the output dataset. If this flag is not provided,
# the script will only print a plan without making any changes.
#--seed: An integer argument that allows the user to specify a seed for random number generation, which can be useful for reproducibility.
#--train-ratio: A float argument that specifies the proportion of the dataset to be used for training. The default value is set to 0.70 (70%).
#--val-ratio: A float argument that specifies the proportion of the dataset to be used for validation. The default value is set to 0.15 (15%).
#--test-ratio: A float argument that specifies the proportion of the dataset to be used for testing. The default value is set to 0.15 (15%).
#The function returns the parsed arguments as a Namespace object, which can be accessed later in the script to determine how the dataset
#should be split and whether to execute the creation of the dataset or just print a plan.


def validate_ratios(args: argparse.Namespace) -> None:
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("The train, validation, and test ratios must be positive and sum to 1.0.")
#this function checks that the provided train, validation, and test ratios are valid.


def read_yolo_label(label_path: Path) -> None:
    """Validate a label file without changing it."""

    lines = [line.strip()
             for line in label_path.read_text(encoding="utf-8").splitlines()
               if line.strip()
            ]
    if not lines:
        raise ValueError(f"Empty label file: {label_path}")

    # Validate each line in the label file

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 YOLO fields in {label_path}:{line_number}, found {len(fields)}")

        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric YOLO label in {label_path}:{line_number}") from exc
        
        if class_id != 0:
            raise ValueError(f"Unexpected class ID {class_id} in {label_path}:{line_number}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"Coordinate outside [0, 1] in {label_path}:{line_number}")
        # Validate class ID and coordinates



def collect_samples(project_root: Path) -> list[Sample]:
    image_dir = project_root / "EALPR Vechicles dataset" / "Vehicles"
    label_dir = project_root / "EALPR Vechicles dataset" / "Vehicles Labeling"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    #checks if the specified image and label directories exist.

    samples: list[Sample] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Missing label for image: {image_path}")

        read_yolo_label(label_path)
        samples.append(Sample(image=image_path, label=label_path))

    if not samples:
        raise ValueError("No image/label pairs were found.")

    return samples


def make_split(samples: list[Sample], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Sample]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    train_end = round(len(shuffled) * train_ratio)
    val_end = train_end + round(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_dataset(splits: dict[str, list[Sample]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            shutil.copy2(sample.image, image_dir / sample.image.name)
            shutil.copy2(sample.label, label_dir / sample.label.name)


def main() -> None:
    args = parse_args()
    validate_ratios(args)

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "yolo_plate_dataset"
    samples = collect_samples(project_root)
    splits = make_split(samples, args.seed, args.train_ratio, args.val_ratio)

    print(f"Validated image/label pairs: {len(samples)}")
    print(f"Seed: {args.seed}")
    print("Planned split:")
    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)} samples")
    print(f"Output directory: {output_dir}")

    if args.execute:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Remove it only after reviewing its contents."
            )
        write_dataset(splits, output_dir)
        print("Dataset created successfully.")
    else:
        print("Dry run only. Add --execute to create the split dataset.")


if __name__ == "__main__":
    main()
