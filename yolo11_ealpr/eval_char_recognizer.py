"""
Measure how well the character model READS plates (not just YOLO's mAP).

For every plate crop in a split (default: test) it compares the model's
characters with the hand labels and reports:
  * character accuracy - a labelled character counts as correct when a
    predicted box overlaps it (IoU >= 0.5) and has the right class
  * missed characters (no box found) and extra boxes (nothing there)
  * whole-plate exact match - digits AND letters all correct, in order
  * a confusion matrix, to see which characters get mixed up (e.g. ب/ن)

The plate text is rebuilt with the same rule as prepare_dataset.py: sort by
x, digits read left->right, letters reversed (right->left).

Usage:
    python eval_char_recognizer.py --weights runs/chars_det/yolo11n/weights/best.pt
Outputs go to runs/chars_det/<name>_eval/.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from plate_reader import boxes_to_text
from prepare_dataset import CLASS_NAMES, NUM_CLASSES, plate_text

HERE = Path(__file__).resolve().parent
DATASET = HERE / "prepared" / "chars_det"
NONE = NUM_CLASSES          # extra row/column in the matrix: "no character"


def read_labels(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        c, x, y, w, h = line.split()
        rows.append((int(c), float(x), float(y), float(w), float(h)))
    return rows


def iou(a, b):
    """IoU of two (xc, yc, w, h) boxes in the same normalised coordinates."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gt, pred, thr=0.5):
    """Greedily pair labelled and predicted boxes by highest IoU (class is
    ignored here, so a wrong class shows up as a confusion, not a miss).
    Returns [(gt_cls or NONE, pred_cls or NONE), ...]."""
    pairs = sorted(((iou(g[1:], p[1:]), i, j) for i, g in enumerate(gt)
                    for j, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((gt[i][0], pred[j][0]))
    out += [(gt[i][0], NONE) for i in range(len(gt)) if i not in used_g]
    out += [(NONE, pred[j][0]) for j in range(len(pred)) if j not in used_p]
    return out


def save_matrix_png(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]  # Arial has Arabic
    names = CLASS_NAMES + ["∅"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row = 100%
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, fontsize=12)
    ax.set_yticks(range(len(names)), names, fontsize=12)
    ax.set_xlabel("predicted  (∅ = missed: no box found)")
    ax.set_ylabel("true  (∅ = extra box where there is no character)")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Character confusion matrix (counts; colour = share of row)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25,
                    help="minimum confidence for a predicted box")
    ap.add_argument("--name", default=None,
                    help="output folder name (default: from the weights path)")
    args = ap.parse_args()

    weights = Path(args.weights)
    name = args.name or (weights.parent.parent.name + "_eval")
    out = HERE / "runs" / "chars_det" / name
    out.mkdir(parents=True, exist_ok=True)

    images = sorted((DATASET / "images" / args.split).glob("*.png"))
    model = YOLO(str(weights))

    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    plate_rows, exact, exact_fixed = [], 0, 0
    # stream=True handles one image at a time, so RAM use stays small.
    # agnostic_nms: one box per character, even if two classes both fire.
    for img, res in zip(images, model.predict(
            [str(p) for p in images], imgsz=320, conf=args.conf,
            agnostic_nms=True, stream=True, verbose=False)):
        gt = read_labels(DATASET / "labels" / args.split / (img.stem + ".txt"))
        b = res.boxes
        pred = [(int(c), *map(float, xywh)) for c, xywh in
                zip(b.cls.tolist(), b.xywhn.tolist())]
        for g, p in match(gt, pred):
            cm[g, p] += 1

        gd, gl, _ = plate_text(gt)
        if pred:
            pd_, pl, warns = plate_text(pred)
        else:
            pd_, pl, warns = "", "", ["no characters found"]
        ok = (gd, gl) == (pd_, pl)
        exact += ok
        # the same boxes after plate_reader's post-processing (step 3)
        fixed = boxes_to_text(list(zip(b.cls.tolist(), b.xywhn[:, 0].tolist(),
                                       b.conf.tolist())))
        ok_fixed = (gd, gl) == (fixed["digits"], " ".join(fixed["letters"]))
        exact_fixed += ok_fixed
        plate_rows.append([img.name, f"{gd} {gl}".strip(),
                           f"{pd_} {pl}".strip(), "yes" if ok else "no",
                           "; ".join(warns), fixed["text"],
                           "yes" if ok_fixed else "no",
                           "; ".join(fixed["fixes"])])

    # ---- numbers -----------------------------------------------------------
    n_gt = cm[:NUM_CLASSES, :].sum()
    correct = np.trace(cm[:NUM_CLASSES, :NUM_CLASSES])
    missed = cm[:NUM_CLASSES, NONE].sum()
    wrong = n_gt - correct - missed
    extra = cm[NONE, :NUM_CLASSES].sum()
    n_plates = len(plate_rows)

    L = [f"Character recognizer evaluation - split: {args.split}",
         f"weights: {weights}", f"confidence threshold: {args.conf}", "",
         f"plates: {n_plates}",
         f"WHOLE-PLATE EXACT MATCH: {exact}/{n_plates} = "
         f"{exact / max(n_plates, 1):.2%}   (raw: sort + reverse letters only)",
         f"  after post-processing:  {exact_fixed}/{n_plates} = "
         f"{exact_fixed / max(n_plates, 1):.2%}   (alef/one fix, drop extras)",
         "",
         f"labelled characters: {n_gt}",
         f"  CHARACTER ACCURACY: {correct}/{n_gt} = {correct / max(n_gt, 1):.2%}",
         f"  wrong class: {wrong}   missed (no box): {missed}",
         f"  extra boxes (no character there): {extra}", "",
         "Per character (accuracy = share of that character read correctly)",
         f"  {'id':>3} {'char':>4} {'count':>6} {'acc':>7}  most often read as"]
    for c in range(NUM_CLASSES):
        row = cm[c]
        tot = row.sum()
        if not tot:
            L.append(f"  {c:>3} {CLASS_NAMES[c]:>4} {0:>6}       -")
            continue
        errs = [(row[j], j) for j in range(NUM_CLASSES + 1) if j != c and row[j]]
        errs.sort(reverse=True)
        top = ", ".join(f"{'missed' if j == NONE else CLASS_NAMES[j]} x{v}"
                        for v, j in errs[:3])
        L.append(f"  {c:>3} {CLASS_NAMES[c]:>4} {tot:>6} {row[c] / tot:>7.2%}  {top}")

    conf_pairs = Counter()
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j]:
                conf_pairs[(CLASS_NAMES[i], CLASS_NAMES[j])] = int(cm[i, j])
    L += ["", "Most common mix-ups (true -> predicted)"]
    L += [f"  {a} -> {b}: {v}" for (a, b), v in conf_pairs.most_common(10)] or ["  none"]
    report = "\n".join(L) + "\n"

    (out / "summary.txt").write_text(report, encoding="utf-8")
    with open(out / "plate_predictions.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate_image", "true_text", "predicted_text", "exact",
                    "format_warnings", "postprocessed_text",
                    "exact_after_postprocessing", "fixes_applied"])
        w.writerows(plate_rows)
    with open(out / "confusion_matrix.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        names = CLASS_NAMES + ["none"]
        w.writerow(["true \\ predicted"] + names)
        for i, n in enumerate(names):
            w.writerow([n] + cm[i].tolist())
    save_matrix_png(cm, out / "confusion_matrix.png")

    try:
        print(report)
    except UnicodeEncodeError:
        print(f"Done - open {out / 'summary.txt'} (console can't show Arabic).")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
