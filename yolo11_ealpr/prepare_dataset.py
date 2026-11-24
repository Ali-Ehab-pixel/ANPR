"""
Step 1 of the EALPR plan: clean, pair, split and reorganise the dataset.

What it does
------------
1. Pairs every vehicle image with its plate label, and every plate crop with
   its character label. Unpaired, empty or malformed files are skipped and
   listed in the report.
2. Splits by VEHICLE ID (the "0001" prefix) into train / val / test, so the
   same car never appears in two splits. The same split is used for all
   three outputs below.
3. Writes three ready-to-train datasets under ./prepared/ :
     plates_det/  vehicle photo -> plate box        (YOLO, 1 class)
     chars_det/   plate crop    -> character boxes  (YOLO, 27 classes)
     chars_cls/   single-character crops in class folders (classification)
   plates_det and chars_det each get a data.yaml for Ultralytics.
4. Rebuilds the plate text from the character labels (digits read left to
   right, letters read right to left) and writes plates_text.csv.
5. Writes report.txt: counts, skipped files, class balance per split.

Uses only the Python standard library. Your original files are only read,
never changed: everything is copied into ./prepared/.

Usage (from the "Claude attempt" folder, in VS Code's terminal):
    python prepare_dataset.py
    python prepare_dataset.py --overwrite      # rebuild ./prepared
    python prepare_dataset.py --seed 7 --val 0.15 --test 0.15
"""

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "EALPR-master"
DEFAULT_OUT = HERE / "prepared"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

# Class order exactly as in "Characters Labeling/classes.txt".
# Index 14 in the original file is U+06BE (Urdu heh) and is written here as the
# standard Arabic heh U+0647 so the text output matches normal Arabic typing.
# Index 16 (U+0649, alef maksura) is kept: it is the form used on the plates.
CLASS_NAMES = [
    "أ", "ب", "ج", "د", "ر", "س", "ص",
    "ط", "ع", "ف", "ق", "ل", "م", "ن",
    "ه", "و", "ى",
    "٠", "١", "٢", "٣", "٤",
    "٥", "٦", "٧", "٨", "٩",
]
NUM_CLASSES = len(CLASS_NAMES)          # 27
FIRST_DIGIT_CLASS = 17                   # 0..16 letters, 17..26 digits
# Character spellings that may appear in crop file names -> class index.
CHAR_TO_CLASS = {c: i for i, c in enumerate(CLASS_NAMES)}
CHAR_TO_CLASS["ھ"] = 14             # Urdu heh, as in classes.txt
CHAR_TO_CLASS["ي"] = 16             # yeh -> alef maksura
CHAR_TO_CLASS["ا"] = 0              # bare alef -> alef with hamza
for d in range(10):                      # western digits, just in case
    CHAR_TO_CLASS[str(d)] = FIRST_DIGIT_CLASS + d

VEHICLE_ID = re.compile(r"^(\d+)")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def vehicle_id(stem):
    m = VEHICLE_ID.match(stem)
    return m.group(1) if m else None


def images_in(folder):
    """Image files directly inside folder (NOT subfolders, so augmented
    copies saved in e.g. Vehicles/output/ are ignored)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_yolo(label_path, num_classes, dropped=None):
    """Return (rows, problem). rows = [(cls, xc, yc, w, h), ...].

    A box with zero width or height is an annotation slip (a click instead of
    a drag), not a real object. It is dropped on its own instead of throwing
    away the whole file; if `dropped` is a list, the file name is added to it.
    """
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"unreadable ({exc.__class__.__name__})"
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return None, f"line {n}: expected 5 values, got {len(parts)}"
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            return None, f"line {n}: not numbers"
        if not 0 <= cls < num_classes:
            return None, f"line {n}: class {cls} out of range"
        if 0 <= xc <= 1 and 0 <= yc <= 1 and (w <= 0.001 or h <= 0.001):
            if dropped is not None:
                dropped.append(f"{label_path.name} (line {n})")
            continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None, f"line {n}: box outside 0..1"
        rows.append((cls, xc, yc, w, h))
    if not rows:
        return None, "empty label file"
    return rows, None


def plate_text(rows):
    """Rebuild the plate reading from character boxes.

    Boxes are sorted left to right. Egyptian plates have digits on the left
    (read left to right) and letters on the right (read right to left).
    Returns digits, letters (in reading order), and a list of warnings."""
    ordered = sorted(rows, key=lambda r: r[1])
    digits = [CLASS_NAMES[r[0]] for r in ordered if r[0] >= FIRST_DIGIT_CLASS]
    letters = [CLASS_NAMES[r[0]] for r in ordered if r[0] < FIRST_DIGIT_CLASS]
    letters.reverse()

    warnings = []
    kinds = ["d" if r[0] >= FIRST_DIGIT_CLASS else "l" for r in ordered]
    if "ld" in "".join(kinds):
        warnings.append("a letter sits left of a digit")
    if not 1 <= len(digits) <= 4:
        warnings.append(f"{len(digits)} digits")
    if not 1 <= len(letters) <= 3:
        warnings.append(f"{len(letters)} letters")
    return "".join(digits), " ".join(letters), warnings


def copy_pair(img, rows, out_root, split, label_name=None):
    (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out_root / "images" / split / img.name)
    label_name = label_name or (img.stem + ".txt")
    with open(out_root / "labels" / split / label_name, "w",
              encoding="utf-8", newline="\n") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_yaml(out_root, names):
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: \"{n}\"" for i, n in enumerate(names)]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="folder holding the three EALPR sub-datasets")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="delete an existing output folder first")
    ap.add_argument("--no-crops", action="store_true",
                    help="skip the single-character crops (chars_cls)")
    ap.add_argument("--keep-odd", action="store_true",
                    help="keep plates whose labels have no digits or no "
                         "letters (normally left out, see report)")
    args = ap.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    veh_img_dir = src / "EALPR Vechicles dataset" / "Vehicles"
    veh_lbl_dir = src / "EALPR Vechicles dataset" / "Vehicles Labeling"
    plate_dir = src / "EALPR- Plates dataset"
    chr_dir = src / "EALPR- LP characters dataset"
    chr_lbl_dir = chr_dir / "Characters Labeling"
    crop_dir = chr_dir / "Characters"

    for d in (veh_img_dir, veh_lbl_dir, plate_dir, chr_lbl_dir):
        if not d.is_dir():
            sys.exit(f"Folder not found: {d}")

    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} already exists. Re-run with --overwrite "
                     "to rebuild it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    skipped = defaultdict(list)          # reason -> [file names]
    dropped_boxes = []                   # zero-size boxes removed from labels

    # ---- 1. pair vehicles with plate boxes --------------------------------
    vehicles = []                        # (vid, img, rows)
    for img in images_in(veh_img_dir):
        lbl = veh_lbl_dir / (img.stem + ".txt")
        vid = vehicle_id(img.stem)
        if vid is None:
            skipped["vehicle: name has no numeric id"].append(img.name)
            continue
        if not lbl.is_file():
            skipped["vehicle: no label file"].append(img.name)
            continue
        rows, problem = read_yolo(lbl, 1, dropped_boxes)
        if problem:
            skipped[f"vehicle label: {problem.split(':')[-1].strip()}"].append(
                lbl.name)
            continue
        vehicles.append((vid, img, rows))

    # ---- 2. pair plate crops with character boxes -------------------------
    plates = []                          # (vid, img, rows)
    for img in images_in(plate_dir):
        lbl = chr_lbl_dir / (img.stem + ".txt")
        vid = vehicle_id(img.stem)
        if vid is None:
            skipped["plate: name has no numeric id"].append(img.name)
            continue
        if not lbl.is_file():
            skipped["plate: no character label file"].append(img.name)
            continue
        rows, problem = read_yolo(lbl, NUM_CLASSES, dropped_boxes)
        if problem:
            skipped[f"plate label: {problem.split(':')[-1].strip()}"].append(
                lbl.name)
            continue
        # A real Egyptian plate always has digits AND letters. Checking the
        # images showed that plates labelled with only one group are either
        # half-labelled (e.g. the letters boxed, the digits not) or have the
        # letter alef labelled as the digit 1. Training on them would teach
        # the detector that real characters are background, so they are left
        # out (and listed in the report) unless --keep-odd is given.
        _, _, warns = plate_text(rows)
        odd = [w for w in warns if w in ("0 digits", "0 letters")]
        if odd and not args.keep_odd:
            skipped["plate: incomplete labels (no digits or no letters) - "
                    "check by eye"].append(f"{img.name} [{odd[0]}]")
            continue
        plates.append((vid, img, rows))
    # class ids labelled on each kept plate, used to sanity-check the crops
    plate_classes = {img.stem: {r[0] for r in rows} for _, img, rows in plates}

    plate_imgs = {p.stem for p in images_in(plate_dir)}
    for lbl in sorted(chr_lbl_dir.glob("*.txt")):
        if lbl.stem != "classes" and lbl.stem not in plate_imgs:
            skipped["character label with no plate image"].append(lbl.name)

    # ---- 3. single-character crops (optional) -----------------------------
    crops = []                           # (vid, img, cls)
    if not args.no_crops and crop_dir.is_dir():
        for img in images_in(crop_dir):
            vid = vehicle_id(img.stem)
            cls = None
            # Names are "<plate>-<char>-<n>", e.g. 0001_license_plate_1-س-0
            # (confirmed on the real data). Fall back to the first Arabic
            # character found, just in case.
            parts = img.stem.rsplit("-", 2)
            if len(parts) == 3 and parts[1] in CHAR_TO_CLASS:
                cls = CHAR_TO_CLASS[parts[1]]
            else:
                for part in img.stem.replace("_", "-").split("-"):
                    if part in CHAR_TO_CLASS and not part.isascii():
                        cls = CHAR_TO_CLASS[part]
                        break
            if vid is None or cls is None:
                skipped["crop: could not read id or character from name"
                        ].append(img.name)
                continue
            # The crops were cut from the character labels, so a crop must
            # come from a kept plate and its character must be one of that
            # plate's labelled classes. This drops crops of left-out plates
            # and crops cut from bad boxes (e.g. the zero-size one on 1981).
            plate = parts[0] if len(parts) == 3 else None
            if plate not in plate_classes:
                skipped["crop: its plate was left out or has no label"
                        ].append(img.name)
                continue
            if cls not in plate_classes[plate]:
                skipped["crop: character not in its plate's labels"
                        ].append(img.name)
                continue
            crops.append((vid, img, cls))

    # ---- 4. split by vehicle id -------------------------------------------
    all_ids = sorted({v for v, _, _ in vehicles} | {v for v, _, _ in plates}
                     | {v for v, _, _ in crops})
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_test, n_val = round(n * args.test), round(n * args.val)
    split_of = {}
    for i, vid in enumerate(all_ids):
        split_of[vid] = ("test" if i < n_test else
                         "val" if i < n_test + n_val else "train")
    with open(out / "split_by_vehicle_id.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["vehicle_id", "split"])
        for vid in sorted(split_of):
            w.writerow([vid, split_of[vid]])

    # ---- 5. write the datasets --------------------------------------------
    print("Copying vehicles ...")
    veh_out = out / "plates_det"
    veh_count = Counter()
    for vid, img, rows in vehicles:
        s = split_of[vid]
        copy_pair(img, rows, veh_out, s)
        veh_count[s] += 1
    write_yaml(veh_out, ["plate"])

    print("Copying plates ...")
    chr_out = out / "chars_det"
    plate_count, class_count = Counter(), defaultdict(Counter)
    text_rows = []
    for vid, img, rows in plates:
        s = split_of[vid]
        copy_pair(img, rows, chr_out, s)
        plate_count[s] += 1
        for r in rows:
            class_count[s][r[0]] += 1
        digits, letters, warns = plate_text(rows)
        text_rows.append([img.name, vid, s, digits, letters,
                          f"{digits} {letters}".strip(), len(rows),
                          "; ".join(warns)])
    write_yaml(chr_out, CLASS_NAMES)

    with open(out / "plates_text.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate_image", "vehicle_id", "split", "digits",
                    "letters", "plate_text", "num_chars", "warnings"])
        w.writerows(text_rows)

    crop_count, crop_class = Counter(), defaultdict(Counter)
    if crops:
        print("Copying character crops ...")
        cls_out = out / "chars_cls"
        for vid, img, cls in crops:
            s = split_of[vid]
            dest = cls_out / s / f"{cls:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest / img.name)
            crop_count[s] += 1
            crop_class[s][cls] += 1
        (cls_out / "classes.txt").write_text(
            "\n".join(f"{i:02d} {c}" for i, c in enumerate(CLASS_NAMES))
            + "\n", encoding="utf-8")

    # ---- 6. report ----------------------------------------------------------
    L = []
    L.append("EALPR preparation report")
    L.append(f"source: {src}")
    L.append(f"seed: {args.seed}   split (train/val/test): "
             f"{1 - args.val - args.test:.2f}/{args.val:.2f}/{args.test:.2f}")
    L.append(f"vehicle ids: {n}  ->  " + ", ".join(
        f"{s} {sum(1 for v in split_of.values() if v == s)}" for s in SPLITS))
    L.append("")
    L.append("Kept")
    L.append("  vehicles (plates_det): " + ", ".join(
        f"{s} {veh_count[s]}" for s in SPLITS) + f"  = {len(vehicles)}")
    L.append("  plates   (chars_det):  " + ", ".join(
        f"{s} {plate_count[s]}" for s in SPLITS) + f"  = {len(plates)}")
    if crops:
        L.append("  char crops (chars_cls): " + ", ".join(
            f"{s} {crop_count[s]}" for s in SPLITS) + f"  = {len(crops)}")
    multi = sum(1 for _, _, rows in vehicles if len(rows) > 1)
    L.append(f"  vehicle images with more than one plate box: {multi}")
    warned = sum(1 for r in text_rows if r[-1])
    L.append(f"  plates whose rebuilt text looks unusual: {warned} "
             "(see 'warnings' column in plates_text.csv)")
    L.append("")
    if dropped_boxes:
        L.append(f"  zero-size boxes removed (rest of the file kept): "
                 f"{len(dropped_boxes)}  -> " + ", ".join(dropped_boxes))
    L.append("")
    L.append("Skipped")
    if not skipped:
        L.append("  nothing")
    for reason, files in sorted(skipped.items()):
        shown = ", ".join(files[:30]) + (" ..." if len(files) > 30 else "")
        L.append(f"  {reason}: {len(files)}")
        L.append(f"      {shown}")
    L.append("")
    L.append("Character class balance in chars_det (boxes per class)")
    L.append(f"  {'id':>3} {'char':>4} " + " ".join(f"{s:>6}" for s in SPLITS)
             + f" {'total':>6}")
    for c in range(NUM_CLASSES):
        counts = [class_count[s][c] for s in SPLITS]
        L.append(f"  {c:>3} {CLASS_NAMES[c]:>4} "
                 + " ".join(f"{x:>6}" for x in counts)
                 + f" {sum(counts):>6}")
    totals = {c: sum(class_count[s][c] for s in SPLITS)
              for c in range(NUM_CLASSES)}
    if any(totals.values()):
        rare = [f"{CLASS_NAMES[c]} ({totals[c]})" for c in
                sorted(totals, key=totals.get)[:5]]
        L.append("  rarest: " + ", ".join(rare))
        missing = [f"{CLASS_NAMES[c]} in {s}" for c in range(NUM_CLASSES)
                   for s in ("val", "test") if class_count[s][c] == 0]
        if missing:
            L.append("  classes with NO examples in val/test: "
                     + ", ".join(missing))
        never = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if not totals[c]]
        if never:
            L.append("  classes with NO examples at all: " + ", ".join(never)
                     + "  (index kept so ids match classes.txt; the models "
                     "cannot learn it)")

    report = "\n".join(L) + "\n"
    (out / "report.txt").write_text(report, encoding="utf-8")
    try:
        print("\n" + report)
    except UnicodeEncodeError:          # old Windows consoles
        print("\nDone. Arabic text could not be shown in this console; "
              f"open {out / 'report.txt'} to read the report.")
    print(f"Output written to: {out}")


if __name__ == "__main__":
    main()
