"""
Step 5 (end-to-end part): run the FULL pipeline on whole vehicle photos and
check the text it reads against the labelled plate text.

Unlike eval_char_recognizer.py (which reads the hand-cut plate crops), here
the plate is found by the detector, so detector misses and loose/tight boxes
count against the result - this is the number a real user would see.

A labelled plate counts as read correctly when one of the plates the
pipeline reads in that photo has exactly the same digits and letters.
Photos whose plates were left out of chars_det (no or incomplete character
labels) have no reference text and are skipped.

Usage:
    python eval_pipeline.py                # test split
    python eval_pipeline.py --split val
Outputs: runs/pipeline/eval_<split>/summary.txt, per_image.csv, and
annotated images of every failure in failures/.
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from plate_reader import HERE, PlateReader, draw

PREP = HERE / "prepared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args()

    out = HERE / "runs" / "pipeline" / f"eval_{args.split}"
    (out / "failures").mkdir(parents=True, exist_ok=True)

    # reference texts per vehicle id, from prepare_dataset.py
    truth = defaultdict(list)
    with open(PREP / "plates_text.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] == args.split:
                truth[r["vehicle_id"]].append(r["plate_text"])

    images = sorted((PREP / "plates_det" / "images" / args.split).iterdir())
    reader = PlateReader(repair=not args.no_repair)
    reader.read(cv2.imdecode(np.fromfile(str(images[0]), np.uint8),
                             cv2.IMREAD_COLOR))          # warm-up (GPU init)

    rows, times = [], []
    n_plates = n_correct = n_no_plate = n_skipped = n_extra = 0
    for img_path in images:
        vid = img_path.stem.split("_")[0]
        if vid not in truth:
            n_skipped += 1
            continue
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8),
                           cv2.IMREAD_COLOR)
        t = time.perf_counter()
        plates = reader.read(img)
        times.append(time.perf_counter() - t)

        read = [p["text"] for p in plates]
        hits = [t_ for t_ in truth[vid] if t_ in read]
        n_plates += len(truth[vid])
        n_correct += len(hits)
        n_no_plate += not plates
        n_extra += max(0, len(read) - len(truth[vid]))
        ok = len(hits) == len(truth[vid])
        rows.append([img_path.name, " | ".join(truth[vid]),
                     " | ".join(read) or "(no plate found)",
                     "yes" if ok else "no",
                     " | ".join(f"{p['plate_conf']:.2f}" for p in plates)])
        if not ok:
            cv2.imencode(".jpg", draw(img, plates))[1].tofile(
                str(out / "failures" / (img_path.stem + ".jpg")))

    ms = 1000 * np.array(times)
    L = [f"End-to-end pipeline evaluation - split: {args.split}",
         f"post-processing: {'off' if args.no_repair else 'on'}", "",
         f"vehicle photos evaluated: {len(rows)}  "
         f"(skipped {n_skipped} with no reference plate text)",
         f"labelled plates: {n_plates}",
         f"END-TO-END PLATE ACCURACY: {n_correct}/{n_plates} = "
         f"{n_correct / max(n_plates, 1):.2%}",
         f"photos where no plate was found: {n_no_plate}",
         f"plates read beyond the labelled ones (often unlabelled background "
         f"plates): {n_extra}", "",
         f"speed per photo (detect + read, GPU): mean {ms.mean():.1f} ms, "
         f"median {np.median(ms):.1f} ms  (~{1000 / ms.mean():.0f} photos/s)"]
    report = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(report, encoding="utf-8")
    with open(out / "per_image.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_text", "read_text", "all_correct",
                    "plate_detection_conf"])
        w.writerows(rows)
    try:
        print(report)
    except UnicodeEncodeError:
        print(f"Done - see {out / 'summary.txt'}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
