
from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset
from preprocessing import (
    load_rgb,
    get_torch_transform,
    split_plate_regions,
    PLATE_COLOR_CLASSES,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def list_images(folder):
    folder = Path('EALPR Vechicles dataset\Vehicles')

    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_class_names(class_file):
    with open(class_file, "r", encoding="utf-8-sig") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def read_single_class_label(label_path, class_to_idx):
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing character label: {label_path}"
        )

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise ValueError(
            f"{label_path} must contain one class label"
        )

    parts = lines[0].split()

    if len(parts) not in (1, 5):
        raise ValueError(
            f"{label_path} must contain a class ID/name "
            "or one YOLO annotation"
        )

    raw_label = parts[0]

    if raw_label in class_to_idx:
        return class_to_idx[raw_label]

    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(
            f"Unknown class {raw_label!r} in {label_path}"
        ) from exc

    if class_to_idx and not 0 <= label < len(class_to_idx):
        raise ValueError(
            f"Class ID {label} is outside the class range"
        )

    return label


class PlateImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        labels_csv=None,
        transform=None,
        class_to_idx=None,
    ):
        self.image_dir = Path(image_dir)
        self.images = list_images(self.image_dir)
        self.transform = transform or get_torch_transform()
        self.class_to_idx = class_to_idx or {}
        self.labels = {}

        if labels_csv:
            with open(
                labels_csv,
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                reader = csv.DictReader(file)

                if not reader.fieldnames or not {
                    "filename", "label"
                }.issubset(reader.fieldnames):
                    raise ValueError(
                        "CSV must contain filename,label"
                    )

                for row in reader:
                    raw_label = row["label"].strip()

                    if raw_label in self.class_to_idx:
                        label = self.class_to_idx[raw_label]
                    else:
                        try:
                            label = int(raw_label)
                        except ValueError as exc:
                            raise ValueError(
                                f"Unknown label: {raw_label}"
                            ) from exc

                    self.labels[row["filename"].strip()] = label

            self.images = [
                path for path in self.images
                if path.name in self.labels
            ]

            if not self.images:
                raise ValueError(
                    "No image filenames matched the CSV"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = load_rgb(path)

        if self.transform:
            image = self.transform(image)

        if self.labels:
            return image, torch.tensor(
                self.labels[path.name],
                dtype=torch.long,
            )

        return image, str(path)


class PlateRegionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        labels_csv=None,
        color_ratio=0.6,
        color_size=(64, 224),
        character_size=(160, 224),
        training=False,
    ):
        self.images = list_images(image_dir)
        self.color_ratio = color_ratio

        self.color_transform = get_torch_transform(
            color_size, training=False
        )
        self.character_transform = get_torch_transform(
            character_size, training=training
        )

        self.labels = {}

        if labels_csv:
            with open(
                labels_csv,
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                reader = csv.DictReader(file)

                if not reader.fieldnames or not {
                    "filename", "label"
                }.issubset(reader.fieldnames):
                    raise ValueError(
                        "CSV must contain filename,label"
                    )

                for row in reader:
                    name = row["filename"].strip()
                    raw_label = row["label"].strip()

                    if raw_label in PLATE_COLOR_CLASSES:
                        label = PLATE_COLOR_CLASSES[raw_label]
                    else:
                        try:
                            label = int(raw_label)
                        except ValueError as exc:
                            raise ValueError(
                                f"Unknown plate color: {raw_label}"
                            ) from exc

                    if not 0 <= label < len(PLATE_COLOR_CLASSES):
                        raise ValueError(
                            f"Invalid plate color ID: {label}"
                        )

                    self.labels[name] = label

            self.images = [
                path for path in self.images
                if path.name in self.labels
            ]

            if not self.images:
                raise ValueError(
                    "No plate images matched the color CSV"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        color_region, character_region = split_plate_regions(
            path,
            color_ratio=self.color_ratio,
        )

        color_tensor = self.color_transform(color_region)
        character_tensor = self.character_transform(
            character_region
        )

        sample = {
            "color_image": color_tensor,
            "character_image": character_tensor,
            "path": str(path),
        }

        if self.labels:
            sample["color_label"] = torch.tensor(
                self.labels[path.name],
                dtype=torch.long,
            )

        return sample


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=(640, 640),
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.images = list_images(self.image_dir)
        self.image_size = (
            int(image_size[0]),
            int(image_size[1]),
        )

        self.transform = transform or get_torch_transform(
            self.image_size,
            training=False,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        original = load_rgb(path)
        orig_w, orig_h = original.size

        label_path = self.label_dir / f"{path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(
                label_path,
                "r",
                encoding="utf-8-sig",
            ) as file:
                for line_no, line in enumerate(file, start=1):
                    parts = line.strip().split()

                    if not parts:
                        continue

                    if len(parts) != 5:
                        raise ValueError(
                            f"{label_path}:{line_no}: "
                            "expected class_id x_center "
                            "y_center width height"
                        )

                    cls, xc, yc, bw, bh = map(float, parts)

                    if not all(
                        0 <= value <= 1
                        for value in (xc, yc, bw, bh)
                    ):
                        raise ValueError(
                            f"{label_path}:{line_no}: "
                            "YOLO coordinates must be normalized"
                        )

                    x1 = (xc - bw / 2) * orig_w
                    y1 = (yc - bh / 2) * orig_h
                    x2 = (xc + bw / 2) * orig_w
                    y2 = (yc + bh / 2) * orig_h

                    x1 = max(0, min(orig_w, x1))
                    x2 = max(0, min(orig_w, x2))
                    y1 = max(0, min(orig_h, y1))
                    y2 = max(0, min(orig_h, y2))

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))

        target_h, target_w = self.image_size
        sx = target_w / orig_w
        sy = target_h / orig_h

        boxes = [
            [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            for x1, y1, x2, y2 in boxes
        ]

        image = self.transform(original)

        target = {
            "boxes": torch.tensor(
                boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.tensor(
                labels, dtype=torch.int64
            ),
            "image_id": torch.tensor(
                [index], dtype=torch.int64
            ),
            "orig_size": torch.tensor(
                [orig_h, orig_w], dtype=torch.int64
            ),
            "size": torch.tensor(
                [target_h, target_w], dtype=torch.int64
            ),
        }

        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class CharacterClassificationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        classes_file,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.class_names = read_class_names(classes_file)
        self.class_to_idx = {
            name: index
            for index, name in enumerate(self.class_names)
        }

        self.images = list_images(self.image_dir)
        self.transform = transform or get_torch_transform(
            (64, 64),
            training=False,
        )

        self.samples = []

        for path in self.images:
            label_path = self.label_dir / f"{path.stem}.txt"

            if not label_path.exists():
                continue

            label = read_single_class_label(
                label_path,
                self.class_to_idx,
            )

            if not 0 <= label < len(self.class_names):
                raise ValueError(
                    f"Invalid class ID {label} in {label_path}"
                )

            self.samples.append((path, label))

        if not self.samples:
            raise ValueError(
                "No character images matched label files"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = load_rgb(path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
