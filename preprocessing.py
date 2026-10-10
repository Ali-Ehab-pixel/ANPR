
from pathlib import Path
from typing import Optional, Union

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter


PLATE_COLOR_CLASSES = {
    "private_car_light_blue": 0,
    "police_car_dark_blue": 1,
    "truck_red": 2,
    "taxi_or_cab_orange": 3,
    "customs_yellow": 4,
    "government_public_transport_grey": 5,
    "tourist_bus_or_car_offwhite": 6,
    "diplomatic_green": 7,
}

PLATE_COLOR_NAMES = {v: k for k, v in PLATE_COLOR_CLASSES.items()}


def load_rgb(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def preprocess_pil(
    image: Union[str, Path, Image.Image],
    size=(224, 224),
    denoise=False,
    sharpen=False,
    contrast=1.0,
) -> Image.Image:
    im = load_rgb(image)

    if size:
        im = im.resize(
            (int(size[0]), int(size[1])),
            Image.Resampling.BILINEAR,
        )

    if denoise:
        im = im.filter(ImageFilter.MedianFilter(size=3))

    if sharpen:
        im = im.filter(
            ImageFilter.UnsharpMask(
                radius=1,
                percent=120,
                threshold=3,
            )
        )

    if contrast and contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(float(contrast))

    return im


def split_plate_regions(
    image: Union[str, Path, Image.Image],
    color_ratio=0.6,
):
    if not 0.0 < color_ratio < 1.0:
        raise ValueError("color_ratio must be between 0 and 1")

    im = load_rgb(image)
    width, height = im.size
    split_y = max(1, min(height - 1, round(height * color_ratio)))

    color_region = im.crop((0, 0, width, split_y))
    character_region = im.crop((0, split_y, width, height))

    return color_region, character_region


def get_torch_transform(size=(224, 224), training=False):
    from torchvision import transforms

    operations = [
        transforms.Resize((int(size[0]), int(size[1]))),
    ]

    if training:
        operations += [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.12,
                        contrast=0.12,
                        saturation=0.08,
                        hue=0.015,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=3),
        ]

    operations += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]

    return transforms.Compose(operations)


def estimate_plate_color(
    image: Union[str, Path, Image.Image],
) -> Optional[str]:
    rgb = np.asarray(load_rgb(image))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, s, v = [
        channel.reshape(-1)
        for channel in cv2.split(hsv)
    ]

    valid = v > 35

    if valid.sum() == 0:
        return None

    hh, ss, vv = h[valid], s[valid], v[valid]
    chromatic = ss > 45

    if chromatic.mean() < 0.08:
        med_v = float(np.median(vv))

        if med_v >= 175:
            return "tourist_bus_or_car_offwhite"

        if med_v >= 85:
            return "government_public_transport_grey"

        return None

    red = ((hh <= 10) | (hh >= 170)) & (ss > 65)
    orange = (hh >= 11) & (hh <= 23) & (ss > 65)
    yellow = (hh >= 24) & (hh <= 36) & (ss > 65)
    green = (hh >= 37) & (hh <= 90) & (ss > 50)
    blue = (hh >= 91) & (hh <= 135) & (ss > 45)

    scores = {
        "truck_red": float(red.mean()),
        "taxi_or_cab_orange": float(orange.mean()),
        "customs_yellow": float(yellow.mean()),
        "diplomatic_green": float(green.mean()),
        "private_car_light_blue": float(blue.mean()),
    }

    best = max(scores, key=scores.get)

    if scores[best] < 0.12:
        return None

    if best == "private_car_light_blue":
        if float(np.median(vv[blue])) < 125:
            return "police_car_dark_blue"

    return best
