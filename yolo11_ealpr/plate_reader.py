"""
Steps 3 + 4 of the EALPR plan: post-processing and the end-to-end pipeline.

    vehicle photo --(plate detector)--> plate box --crop--> plate image
                  --(character detector)--> character boxes
                  --(post-processing)--> plate text

Post-processing (step 3), in `boxes_to_text`:
  1. sort the character boxes left -> right
  2. find where digits end and letters begin, and fix the one classic
     confusion that breaks this order: alef (أ) and the digit one (١) look
     almost the same, so a "١" among the letters becomes "أ" and vice versa
  3. if there are too many characters (more than 4 digits / 3 letters), drop
     the least confident extra boxes
  4. digits are read left -> right; letters are reversed (read right -> left)
  5. check the format: 1-4 digits and 1-3 letters  -> `valid`
Character normalisation (Urdu heh -> ه, alef maksura ى) is already built into
CLASS_NAMES in prepare_dataset.py.

Use it from Python:
    from plate_reader import PlateReader
    reader = PlateReader()
    for plate in reader.read("car.jpg"):
        print(plate["text"], plate["valid"], plate["box"])

Or from the terminal (prints the text, saves annotated copies):
    python plate_reader.py some_car.jpg another.jpg some_folder/
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from prepare_dataset import CLASS_NAMES, FIRST_DIGIT_CLASS

HERE = Path(__file__).resolve().parent
PLATE_WEIGHTS = HERE / "runs" / "plates_det" / "yolo11n" / "weights" / "best.pt"
CHAR_WEIGHTS = HERE / "runs" / "chars_det" / "yolo11n" / "weights" / "best.pt"

ALEF = CLASS_NAMES.index("أ")
ONE = CLASS_NAMES.index("١")
MAX_DIGITS, MAX_LETTERS = 4, 3

# The EALPR plate crops are ~6% wider and ~11% taller than the labelled plate
# box (measured on 1,979 cars), so the same margin is added around detected
# plates: the character model then sees crops like the ones it learned from.
PAD_X, PAD_Y = 0.03, 0.055


def is_digit(cls):
    return cls >= FIRST_DIGIT_CLASS


# --------------------------------------------------------------------------
# step 3: post-processing
# --------------------------------------------------------------------------
def boxes_to_text(boxes, repair=True):
    """Turn the character boxes of ONE plate into its text.

    boxes: list of (cls, x_center, conf) - any extra fields are ignored.
    Returns a dict:
        digits   "٧٩٨٤"            read left -> right
        letters  ["ع", "و"]        in reading order (right -> left)
        text     "٧٩٨٤ ع و"        same layout as plates_text.csv
        display  "٧٩٨٤ و ع"        as printed on the plate, for drawing
        valid    True/False        1-4 digits and 1-3 letters
        conf     lowest character confidence (a weak link breaks the plate)
        fixes    list of repairs that were applied (empty if none)
    """
    chars = sorted(([int(b[0]), float(b[1]), float(b[2])] for b in boxes),
                    key=lambda b: b[1])
    fixes = []

    if repair and chars:
        # Where do digits end? Try every split point and keep the one where
        # the most boxes are on the expected side (digits left, letters right).
        n = len(chars)
        best_k = max(range(n + 1), key=lambda k:
                     sum(is_digit(c[0]) for c in chars[:k])
                     + sum(not is_digit(c[0]) for c in chars[k:]))
        for i, c in enumerate(chars):
            if i < best_k and c[0] == ALEF:           # alef among digits
                c[0] = ONE
                fixes.append("أ->١")
            elif i >= best_k and c[0] == ONE:          # one among letters
                c[0] = ALEF
                fixes.append("١->أ")

        # Too many of a kind: drop the least confident extras.
        for want_digit, limit in ((True, MAX_DIGITS), (False, MAX_LETTERS)):
            group = [c for c in chars if is_digit(c[0]) == want_digit]
            for c in sorted(group, key=lambda c: c[2])[:max(0, len(group) - limit)]:
                chars.remove(c)
                fixes.append(f"dropped extra {CLASS_NAMES[c[0]]} "
                             f"(conf {c[2]:.2f})")

    digits = "".join(CLASS_NAMES[c[0]] for c in chars if is_digit(c[0]))
    letters_lr = [CLASS_NAMES[c[0]] for c in chars if not is_digit(c[0])]
    letters = letters_lr[::-1]                          # read right -> left
    kinds = "".join("d" if is_digit(c[0]) else "l" for c in chars)
    valid = (1 <= len(digits) <= MAX_DIGITS
             and 1 <= len(letters) <= MAX_LETTERS and "ld" not in kinds)
    return {
        "digits": digits,
        "letters": letters,
        "text": f"{digits} {' '.join(letters)}".strip(),
        "display": f"{digits} {' '.join(letters_lr)}".strip(),
        "valid": valid,
        "conf": min((c[2] for c in chars), default=0.0),
        "fixes": fixes,
    }


# --------------------------------------------------------------------------
# step 4: the end-to-end pipeline
# --------------------------------------------------------------------------
class PlateReader:
    def __init__(self, plate_weights=PLATE_WEIGHTS, char_weights=CHAR_WEIGHTS,
                 plate_conf=0.25, char_conf=0.25, repair=True, device=None):
        from ultralytics import YOLO      # imported here: slow to load
        self.plate_model = YOLO(str(plate_weights))
        self.char_model = YOLO(str(char_weights))
        self.plate_conf, self.char_conf = plate_conf, char_conf
        self.repair, self.device = repair, device

    def read_crop(self, crop):
        """Read the text of one plate image (BGR numpy array)."""
        res = self.char_model.predict(crop, imgsz=320, conf=self.char_conf,
                                      agnostic_nms=True, verbose=False,
                                      device=self.device)[0]
        b = res.boxes
        boxes = list(zip(b.cls.tolist(), b.xywhn[:, 0].tolist(),
                         b.conf.tolist()))
        return boxes_to_text(boxes, repair=self.repair)

    def read(self, image):
        """Find and read every plate in a vehicle photo.

        image: a file path or a BGR numpy array (as from cv2.imread).
        Returns a list of dicts (best plate first): box (x1, y1, x2, y2 in
        pixels), plate_conf, crop, plus everything boxes_to_text returns."""
        if not isinstance(image, np.ndarray):
            # np.fromfile + imdecode also works with non-English paths
            image = cv2.imdecode(np.fromfile(str(image), np.uint8),
                                 cv2.IMREAD_COLOR)
        H, W = image.shape[:2]
        det = self.plate_model.predict(image, imgsz=640, conf=self.plate_conf,
                                       verbose=False, device=self.device)[0]
        plates = []
        for (x1, y1, x2, y2), conf in zip(det.boxes.xyxy.tolist(),
                                          det.boxes.conf.tolist()):
            px, py = (x2 - x1) * PAD_X, (y2 - y1) * PAD_Y
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(W, int(x2 + px + 1)), min(H, int(y2 + py + 1))
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            info = self.read_crop(crop)
            info.update(box=(int(x1), int(y1), int(x2), int(y2)),
                        plate_conf=conf, crop=crop)
            plates.append(info)
        plates.sort(key=lambda p: p["plate_conf"], reverse=True)
        return plates


# --------------------------------------------------------------------------
# drawing (OpenCV can't draw Arabic, so the text is drawn with Pillow)
# --------------------------------------------------------------------------
def plate_pieces(p):
    """The plate text as on the plate, left to right: the digit group, then
    each letter. Drawing these one by one (instead of one Arabic string)
    stops Pillow's/Tk's right-to-left handling from reordering them, which
    proved inconsistent between plates."""
    pieces = ([p["digits"]] if p["digits"] else []) + \
             p["display"].split()[1 if p["digits"] else 0:]
    return pieces or ["?"]


def draw(image, plates):
    from PIL import Image, ImageDraw, ImageFont
    out = image.copy()
    for p in plates:
        x1, y1, x2, y2 = p["box"]
        colour = (0, 200, 0) if p["valid"] else (0, 165, 255)  # green / orange
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    size = max(18, image.shape[1] // 25)
    try:
        font = ImageFont.truetype("arial.ttf", size)   # has Arabic glyphs
    except OSError:
        font = ImageFont.load_default()
    for p in plates:
        x1, y1 = p["box"][:2]
        pieces = plate_pieces(p)
        gap = size // 3
        widths = [d.textlength(s, font=font) for s in pieces]
        tx, ty = x1, max(0, y1 - size - 8)
        total = sum(widths) + gap * (len(pieces) - 1)
        d.rectangle((tx - 4, ty - 4, tx + total + 4, ty + size + 8),
                    fill=(0, 0, 0))
        for s, w in zip(pieces, widths):
            d.text((tx, ty), s, font=font, fill=(255, 255, 255))
            tx += w + gap
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Read Egyptian licence plates.")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "pipeline" / "annotated")
    ap.add_argument("--no-repair", action="store_true",
                    help="turn off the format-based fixes (for comparison)")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files += sorted(q for q in p.iterdir()
                            if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
        else:
            files.append(p)

    reader = PlateReader(repair=not args.no_repair)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in files:
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        t = time.perf_counter()
        plates = reader.read(img)
        ms = (time.perf_counter() - t) * 1000
        texts = [f"{p['text']}{'' if p['valid'] else ' (?)'}" for p in plates]
        lines.append(f"{f.name}: {' | '.join(texts) or 'no plate found'}"
                     f"   [{ms:.0f} ms]")
        cv2.imencode(".jpg", draw(img, plates))[1].tofile(
            str(args.out / (f.stem + ".jpg")))
    report = "\n".join(lines)
    (args.out / "results.txt").write_text(report + "\n", encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(f"Done - see {args.out / 'results.txt'}")
    print(f"Annotated images: {args.out}")


if __name__ == "__main__":
    main()
