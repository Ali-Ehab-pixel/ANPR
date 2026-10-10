# Egyptian Automatic License Plate Recognition (EALPR) Pipeline

This repository provides a PyTorch dataset and preprocessing pipeline for the Egyptian Automated License Plate Recognition (EALPR) dataset.

---

## Directory Structure

```text
EALPR-master/
├── EALPR Vechicles dataset/
│   ├── Vehicles/                 # Vehicle images (0001.jpg, ...)
│   └── Vehicles Labeling/        # YOLO labels (0001.txt, ...) & classes.txt
├── EALPR- Plates dataset/         # Cropped license plate images (0001_license_plate_1.png, ...)
├── EALPR- LP characters dataset/
│   ├── Characters/               # Cropped characters (0001_license_plate_1-س-0.png, ...)
│   └── Characters Labeling/      # YOLO character bounding boxes & classes.txt
├── src/
│   └── data/
│       ├── preprocessing.py      # Image transforms, color space conversions, plate region splitting
│       └── dataset.py            # PyTorch Datasets and DataLoaders
└── README.md
```

---

## Overview of Modules

### 1. `src/data/preprocessing.py`

Handles image-level operations, transformations, and color-space conversions:

- **`split_plate_regions(image, color_ratio=0.35, as_hsv_and_gray=True)`**:
  - Splits an Egyptian license plate into two regions:
    - **Top region (~35%)**: The colored category band ("EGYPT" / "مصر"). Converted to **HSV** mode.
    - **Bottom region (~65%)**: The character area (letters and digits). Converted to **Grayscale** (`'L'` mode).
- **`to_hsv(image)`**: Converts an image to PIL HSV mode.
- **`to_grayscale(image)`**: Converts an image to single-channel PIL Grayscale (`'L'`) mode.
- **`get_hsv_transform(size=(64, 224), training=False)`**:
  - Resizes to `(64, 224)` and converts HSV image to a float tensor of shape `(3, 64, 224)` with values in `[0.0, 1.0]`.
- **`get_gray_transform(size=(160, 224), training=False)`**:
  - Resizes to `(160, 224)` and converts Grayscale image to a single-channel float tensor of shape `(1, 160, 224)`.
- **`get_torch_transform(size=(224, 224), training=False)` / `get_rgb_transform`**:
  - Standard RGB transform with ImageNet normalization for full vehicle or plate images.
- **`estimate_plate_color(image)`**:
  - Rule-based diagnostic using OpenCV HSV color ranges to classify the plate category.

---

### 2. `src/data/dataset.py`

Provides PyTorch `Dataset` implementations and batch loaders:

- **`PlateRegionDataset` (alias `PlateDataset`)**:
  - Loads cropped license plates from `EALPR- Plates dataset`.
  - Automatically splits each plate into:
    - `"color_image"`: HSV tensor `(3, 64, 224)`.
    - `"character_image"`: Grayscale tensor `(1, 160, 224)`.
    - `"path"`: Path to the original plate image.
- **`PlateImageDataset`**:
  - Loads plate images as full RGB images, or optionally extracts HSV color and Grayscale characters when `extract_regions=True`.
- **`YoloDetectionDataset`**:
  - Loads images and YOLO-formatted bounding boxes (`class_id x_center y_center width height`).
  - Supports vehicle plate detection (`EALPR Vechicles dataset`) or plate character bounding box detection (`Characters Labeling`).
- **`CharacterClassificationDataset`**:
  - Loads individual character crops from `EALPR- LP characters dataset/Characters`.
  - Automatically parses the Arabic character symbol from the filename (e.g., `0001_license_plate_1-س-0.png`) and maps it to the 27 Arabic character classes.
  - Converts characters to **Grayscale** tensors of shape `(1, 64, 64)`.
- **Convenience Loader Functions**:
  - `get_plate_dataloader(image_dir=..., batch_size=32)`
  - `get_vehicle_dataloader(image_dir=..., label_dir=..., batch_size=4)`
  - `get_character_dataloader(image_dir=..., batch_size=32)`

---

## Usage Examples

### 1. Load License Plates (HSV Color & Grayscale Characters)

```python
from src.data.dataset import PlateRegionDataset, get_plate_dataloader
from torch.utils.data import DataLoader

# Using the preconfigured helper
plate_loader = get_plate_dataloader(
    image_dir="EALPR- Plates dataset",
    batch_size=16,
    shuffle=True,
)

batch = next(iter(plate_loader))
color_tensors = batch["color_image"]          # Shape: [16, 3, 64, 224] (HSV)
character_tensors = batch["character_image"]  # Shape: [16, 1, 160, 224] (Grayscale)
image_paths = batch["path"]

print(f"Color (HSV) batch shape: {color_tensors.shape}")
print(f"Character (Gray) batch shape: {character_tensors.shape}")
```

### 2. Load Vehicle Detection (YOLO Format)

```python
from src.data.dataset import YoloDetectionDataset, detection_collate_fn
from torch.utils.data import DataLoader

dataset = YoloDetectionDataset(
    image_dir="EALPR Vechicles dataset/Vehicles",
    label_dir="EALPR Vechicles dataset/Vehicles Labeling",
    image_size=(640, 640),
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=detection_collate_fn,
)

images, targets = next(iter(loader))
print(f"Batch image 0 shape: {images[0].shape}")
print(f"Boxes for image 0: {targets[0]['boxes']}")
print(f"Labels for image 0: {targets[0]['labels']}")
```

### 3. Load Character Classification (Grayscale Characters)

```python
from src.data.dataset import CharacterClassificationDataset, get_character_dataloader

char_loader = get_character_dataloader(
    image_dir="EALPR- LP characters dataset/Characters",
    batch_size=32,
    shuffle=True,
)

char_images, char_labels = next(iter(char_loader))
print(f"Character images shape: {char_images.shape}")  # [32, 1, 64, 64]
print(f"Character labels shape: {char_labels.shape}")  # [32]
```

---

## Plate Color Categories

Egyptian license plates use the top colored band to designate the vehicle type:

| ID | Class Name | Description |
|:---|:---|:---|
| 0 | `private_car_light_blue` | Private cars |
| 1 | `police_car_dark_blue` | Police vehicles |
| 2 | `truck_red` | Trucks |
| 3 | `taxi_or_cab_orange` | Cabs / Taxis |
| 4 | `customs_yellow` | Customs |
| 5 | `government_public_transport_grey` | Government facilities / Public transit |
| 6 | `tourist_bus_or_car_offwhite` | Tourist buses and cars |
| 7 | `diplomatic_green` | Diplomatic vehicles |
