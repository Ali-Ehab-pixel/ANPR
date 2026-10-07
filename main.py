import cv2
import numpy as np
from pathlib import Path


dataset_root = Path(__file__).resolve().parent / "EALPR-master"
characters_dataset = dataset_root / "EALPR- LP characters dataset"
character_images_dir = characters_dataset / "Characters"
character_labels_dir = characters_dataset / "Characters Labeling"
plates_dir = dataset_root / "EALPR- Plates dataset"

image_index = 0
character_size = 32


def prepare_character(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    foreground = cv2.findNonZero(binary)
    if foreground is None:
        raise ValueError("Character image contains no visible character.")

    x, y, width, height = cv2.boundingRect(foreground)
    character = binary[y:y + height, x:x + width]

    scale = (character_size - 8) / max(width, height)
    resized = cv2.resize(
        character,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((character_size, character_size), dtype=np.uint8)
    x_offset = (character_size - resized.shape[1]) // 2
    y_offset = (character_size - resized.shape[0]) // 2
    canvas[
        y_offset:y_offset + resized.shape[0],
        x_offset:x_offset + resized.shape[1],
    ] = resized

    return canvas.reshape(1, -1).astype(np.float32)


def character_from_filename(image_path):
    parts = image_path.stem.rsplit("-", 2)
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"Unexpected character image name: {image_path.name}")
    return parts[1]


def train_character_classifier():
    crop_paths = sorted(character_images_dir.glob("*.png"))
    if not crop_paths:
        raise FileNotFoundError(
            f"No character crop images found in:\n{character_images_dir}"
        )

    class_names = sorted({character_from_filename(path) for path in crop_paths})
    class_ids = {name: index for index, name in enumerate(class_names)}
    samples = []
    responses = []

    for crop_path in crop_paths:
        crop = cv2.imread(str(crop_path))
        if crop is None:
            raise ValueError(f"Could not read character image:\n{crop_path}")

        samples.append(prepare_character(crop))
        responses.append(class_ids[character_from_filename(crop_path)])

    training_data = np.vstack(samples)
    training_labels = np.asarray(responses, dtype=np.float32).reshape(-1, 1)
    classifier = cv2.ml.KNearest_create()
    classifier.setDefaultK(3)
    if not classifier.train(training_data, cv2.ml.ROW_SAMPLE, training_labels):
        raise RuntimeError("Could not train the character classifier.")

    return classifier, class_names


def read_character_boxes(label_path):
    boxes = []
    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            if not line.strip():
                continue

            values = line.split()
            if len(values) != 5:
                raise ValueError(
                    f"Expected 5 values in {label_path.name}, line {line_number}."
                )

            class_id, x_center, y_center, box_width, box_height = map(
                float, values
            )
            boxes.append(
                (int(class_id), x_center, y_center, box_width, box_height)
            )
    return boxes


if not character_labels_dir.is_dir():
    raise FileNotFoundError(f"Character labels directory not found:\n{character_labels_dir}")
if not plates_dir.is_dir():
    raise FileNotFoundError(f"License plate images directory not found:\n{plates_dir}")

labeled_plates = [
    (image_path, character_labels_dir / f"{image_path.stem}.txt")
    for image_path in sorted(plates_dir.glob("*.png"))
    if (character_labels_dir / f"{image_path.stem}.txt").is_file()
]
if not labeled_plates:
    raise FileNotFoundError(
        f"No license plate images with matching character labels found in:\n{plates_dir}"
    )
if not 0 <= image_index < len(labeled_plates):
    raise IndexError(
        f"image_index {image_index} is out of range; "
        f"there are {len(labeled_plates)} labeled plates."
    )

classifier, class_names = train_character_classifier()
image_path, label_path = labeled_plates[image_index]
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not read license plate image:\n{image_path}")

height, width = image.shape[:2]
predictions = []
for class_id, x_center, y_center, box_width, box_height in read_character_boxes(
    label_path
):
    x1 = max(0, int((x_center - box_width / 2) * width))
    y1 = max(0, int((y_center - box_height / 2) * height))
    x2 = min(width, int((x_center + box_width / 2) * width))
    y2 = min(height, int((y_center + box_height / 2) * height))
    character_crop = image[y1:y2, x1:x2]
    if character_crop.size == 0:
        raise ValueError(
            f"Invalid character box in {label_path.name}: {(x1, y1, x2, y2)}"
        )

    sample = prepare_character(character_crop)
    _, result, _, _ = classifier.findNearest(sample, 3)
    predicted_character = class_names[int(result[0, 0])]
    predictions.append((class_id, predicted_character))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"ID {class_id}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

print("Plate:", image_path)
print("Annotations:", label_path)
print(f"Trained character classes: {len(class_names)}")
for index, (class_id, character) in enumerate(predictions, start=1):
    print(f"Character {index}: {character} (annotation class ID {class_id})")
print("Predicted characters in annotation order:", "".join(
    character for _, character in predictions
))
cv2.imshow("Classified License Plate Characters", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
