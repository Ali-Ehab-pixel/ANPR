from pathlib import Path
import csv
from typing import Optional, Union, Sequence, Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from preprocessing import (
        load_rgb,
        to_hsv,
        to_grayscale,
        split_plate_regions,
        get_hsv_transform,
        get_gray_transform,
        get_rgb_transform,
        PLATE_COLOR_CLASSES,
        PLATE_COLOR_NAMES,
    )
except ImportError:
    from .preprocessing import (
        load_rgb,
        to_hsv,
        to_grayscale,
        split_plate_regions,
        get_hsv_transform,
        get_gray_transform,
        get_rgb_transform,
        PLATE_COLOR_CLASSES,
        PLATE_COLOR_NAMES,
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_path(folder: Union[str, Path]) -> Path:
    target = Path(folder)
    if target.exists():
        return target
    # Fallback to repository root
    root = Path(__file__).resolve().parents[2]
    candidate = root / folder
    if candidate.exists():
        return candidate
    return target


def list_images(folder: Union[str, Path]) -> List[Path]:
    folder_path = resolve_path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    return sorted(
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_class_names(class_file: Union[str, Path, Sequence[str]]) -> List[str]:
    if isinstance(class_file, (list, tuple)):
        return list(class_file)
    path = resolve_path(class_file)
    if not path.exists():
        raise FileNotFoundError(f"Class file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def read_single_class_label(label_path: Path, class_to_idx: Dict[str, int]) -> int:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing character label: {label_path}")

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"{label_path} must contain one class label")

    parts = lines[0].split()
    if len(parts) not in (1, 5):
        raise ValueError(f"{label_path} must contain a class ID or one YOLO row")

    raw_label = parts[0]
    if raw_label in class_to_idx:
        return class_to_idx[raw_label]

    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(f"Unknown class {raw_label!r} in {label_path}") from exc

    if class_to_idx and not 0 <= label < len(class_to_idx):
        raise ValueError(f"Class ID {label} is outside valid range")
    return label


class PlateRegionDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        labels_csv: Optional[Union[str, Path]] = None,
        color_ratio: float = 0.35,
        color_size: Tuple[int, int] = (64, 224),
        character_size: Tuple[int, int] = (160, 224),
        color_transform=None,
        character_transform=None,
        training: bool = False,
    ):
        self.image_dir = resolve_path(image_dir)
        self.images = list_images(self.image_dir)
        self.color_ratio = color_ratio
        self.color_size = color_size
        self.character_size = character_size

        self.color_transform = color_transform or get_hsv_transform(
            color_size, training=False
        )
        self.character_transform = character_transform or get_gray_transform(
            character_size, training=training
        )

        self.labels: Dict[str, int] = {}
        if labels_csv:
            csv_path = resolve_path(labels_csv)
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                if not reader.fieldnames or not {"filename", "label"}.issubset(reader.fieldnames):
                    raise ValueError("CSV must contain filename,label")

                for row in reader:
                    name = row["filename"].strip()
                    raw = row["label"].strip()
                    if raw in PLATE_COLOR_CLASSES:
                        label = PLATE_COLOR_CLASSES[raw]
                    else:
                        label = int(raw)
                    self.labels[name] = label

            self.images = [p for p in self.images if p.name in self.labels]
            if not self.images:
                raise ValueError("No plate images matched the color CSV")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.images[index]

        # Extract HSV plate color and Grayscale characters via preprocessing
        color_hsv, character_gray = split_plate_regions(
            path,
            color_ratio=self.color_ratio,
            as_hsv_and_gray=True,
        )

        color_tensor = self.color_transform(color_hsv)
        character_tensor = self.character_transform(character_gray)

        sample = {
            "color_image": color_tensor,
            "character_image": character_tensor,
            "path": str(path),
        }

        if self.labels:
            sample["color_label"] = torch.tensor(
                self.labels[path.name], dtype=torch.long
            )

        return sample


# Alias for plate dataset
PlateDataset = PlateRegionDataset


class PlateImageDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        labels_csv: Optional[Union[str, Path]] = None,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        extract_regions: bool = False,
        color_ratio: float = 0.35,
        color_size: Tuple[int, int] = (64, 224),
        character_size: Tuple[int, int] = (160, 224),
        training: bool = False,
    ):
        self.image_dir = resolve_path(image_dir)
        self.images = list_images(self.image_dir)
        self.extract_regions = extract_regions

        if extract_regions:
            self.region_ds = PlateRegionDataset(
                image_dir=self.image_dir,
                labels_csv=labels_csv,
                color_ratio=color_ratio,
                color_size=color_size,
                character_size=character_size,
                training=training,
            )
            self.images = self.region_ds.images
        else:
            self.region_ds = None
            self.transform = transform or get_rgb_transform()
            self.class_to_idx = class_to_idx or {}
            self.labels: Dict[str, int] = {}

            if labels_csv:
                csv_path = resolve_path(labels_csv)
                with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
                    reader = csv.DictReader(file)
                    if not reader.fieldnames or not {"filename", "label"}.issubset(reader.fieldnames):
                        raise ValueError("CSV must contain filename,label")

                    for row in reader:
                        raw = row["label"].strip()
                        if raw in self.class_to_idx:
                            label = self.class_to_idx[raw]
                        else:
                            label = int(raw)
                        self.labels[row["filename"].strip()] = label

                self.images = [p for p in self.images if p.name in self.labels]
                if not self.images:
                    raise ValueError("No image filenames matched the CSV")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        if self.extract_regions and self.region_ds is not None:
            return self.region_ds[index]

        path = self.images[index]
        image = load_rgb(path)
        if self.transform:
            image = self.transform(image)

        if self.labels:
            return image, torch.tensor(self.labels[path.name], dtype=torch.long)
        return image, str(path)


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        image_size: Tuple[int, int] = (640, 640),
        transform=None,
        require_labels: bool = False,
    ):
        self.image_dir = resolve_path(image_dir)
        self.label_dir = resolve_path(label_dir)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.transform = transform or get_rgb_transform(self.image_size, training=False)

        raw_images = list_images(self.image_dir)
        if require_labels:
            self.images = [
                p for p in raw_images
                if (self.label_dir / f"{p.stem}.txt").exists()
            ]
        else:
            self.images = raw_images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        path = self.images[index]
        original = load_rgb(path)
        orig_w, orig_h = original.size

        label_path = self.label_dir / f"{path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8-sig") as file:
                for line_no, line in enumerate(file, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        raise ValueError(f"{label_path}:{line_no}: expected class_id xc yc bw bh")

                    cls, xc, yc, bw, bh = map(float, parts)
                    x1 = max(0.0, min(float(orig_w), (xc - bw / 2.0) * orig_w))
                    y1 = max(0.0, min(float(orig_h), (yc - bh / 2.0) * orig_h))
                    x2 = max(0.0, min(float(orig_w), (xc + bw / 2.0) * orig_w))
                    y2 = max(0.0, min(float(orig_h), (yc + bh / 2.0) * orig_h))

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))

        target_h, target_w = self.image_size
        sx = target_w / orig_w
        sy = target_h / orig_h

        scaled_boxes = [
            [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            for x1, y1, x2, y2 in boxes
        ]

        image = self.transform(original)
        boxes_tensor = (
            torch.tensor(scaled_boxes, dtype=torch.float32)
            if scaled_boxes else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels else torch.zeros((0,), dtype=torch.int64)
        )

        target = {
            "boxes": boxes_tensor.reshape(-1, 4),
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([target_h, target_w], dtype=torch.int64),
        }
        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class CharacterClassificationDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Optional[Union[str, Path]] = None,
        classes_file: Optional[Union[str, Path, Sequence[str]]] = None,
        transform=None,
        training: bool = False,
    ):
        self.image_dir = resolve_path(image_dir)
        self.label_dir = resolve_path(label_dir) if label_dir else None

        # Auto-detect classes_file if not provided
        if classes_file is None:
            if self.label_dir and (self.label_dir / "classes.txt").exists():
                classes_file = self.label_dir / "classes.txt"
            elif (self.image_dir.parent / "Characters Labeling" / "classes.txt").exists():
                classes_file = self.image_dir.parent / "Characters Labeling" / "classes.txt"
            elif (self.image_dir / "classes.txt").exists():
                classes_file = self.image_dir / "classes.txt"

        if classes_file is not None:
            self.class_names = read_class_names(classes_file)
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        else:
            self.class_names = []
            self.class_to_idx = {}

        self.images = list_images(self.image_dir)
        self.transform = transform or get_gray_transform((64, 64), training=training)

        self.samples: List[Tuple[Path, int]] = []
        for path in self.images:
            label = None
            # Check txt label file
            if self.label_dir:
                lp = self.label_dir / f"{path.stem}.txt"
                if lp.exists():
                    label = read_single_class_label(lp, self.class_to_idx)

            # Fallback to character symbol encoded in filename
            if label is None:
                parts = path.stem.split("-")
                char_symbol = parts[-2] if len(parts) >= 3 else parts[-1]
                if self.class_to_idx and char_symbol in self.class_to_idx:
                    label = self.class_to_idx[char_symbol]
                else:
                    try:
                        label = int(char_symbol)
                    except ValueError:
                        pass

            if label is not None:
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No character samples could be labeled from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        # Character converted to grayscale via preprocessing
        gray_image = to_grayscale(path)

        if self.transform:
            tensor_image = self.transform(gray_image)
        else:
            tensor_image = get_gray_transform((64, 64))(gray_image)

        return tensor_image, torch.tensor(label, dtype=torch.long)


def get_plate_dataloader(
    image_dir: Union[str, Path] = "EALPR- Plates dataset",
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    ds = PlateRegionDataset(image_dir=image_dir, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_vehicle_dataloader(
    image_dir: Union[str, Path] = "EALPR Vechicles dataset/Vehicles",
    label_dir: Union[str, Path] = "EALPR Vechicles dataset/Vehicles Labeling",
    batch_size: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    ds = YoloDetectionDataset(image_dir=image_dir, label_dir=label_dir, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=detection_collate_fn)


def get_character_dataloader(
    image_dir: Union[str, Path] = "EALPR- LP characters dataset/Characters",
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    ds = CharacterClassificationDataset(image_dir=image_dir, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print("Testing data loading pipelines...")

    plate_loader = get_plate_dataloader(batch_size=4)
    plate_batch = next(iter(plate_loader))
    print(f"Plate batch color (HSV) shape: {plate_batch['color_image'].shape}")
    print(f"Plate batch character (Gray) shape: {plate_batch['character_image'].shape}")

    veh_loader = get_vehicle_dataloader(batch_size=2)
    veh_images, veh_targets = next(iter(veh_loader))
    print(f"Vehicle batch size: {len(veh_images)}, Image 0 shape: {veh_images[0].shape}")
    print(f"Vehicle target 0 boxes: {veh_targets[0]['boxes'].shape}")

    char_loader = get_character_dataloader(batch_size=8)
    char_images, char_labels = next(iter(char_loader))
    print(f"Character batch (Gray) shape: {char_images.shape}, labels: {char_labels.shape}")

    print("All pipelines loaded successfully.")
