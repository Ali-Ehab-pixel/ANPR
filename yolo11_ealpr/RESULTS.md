# Results: Egyptian licence plate recognition on EALPR

All numbers are on the **test split**, which was never used for training or model selection, unless marked otherwise.
Hardware: laptop RTX 3050 (4 GB), Windows, Ultralytics 8.4, PyTorch 2.14.

## 1. Comparison with the EALPR authors' results

The ACIRS 2022 dataset paper (Youssef et al.) could not be retrieved: IEEE Xplore returned an empty page.
The same authors' companion paper uses the same dataset and the same two-stage design, and reports their recognition results:

> A. R. Youssef, A. A. Ali, F. R. Sayed, "Real-time Egyptian License Plate Detection and Recognition using YOLO",
> IJACSA vol. 13 no. 7, 2022. DOI 10.14569/IJACSA.2022.0130799

| | Youssef et al. 2022 | This project |
|---|---|---|
| Plate detector | Tiny-YOLOv3, 608×608 | YOLO11n, 640 |
| Character detector | Tiny-YOLOv3, 384×192, 27 classes | YOLO11n, 320 (long side), 27 classes |
| Framework / GPU | Darknet, GTX 1660 Ti | Ultralytics, RTX 3050 Laptop |
| Training length | 5000 epochs (6,000 / 54,000 max batches), batch 64 | 100 epochs (plates), 97 epochs with early stopping (characters), batch 8 / 32 |
| Data split | random 80% train / 20% test | 70 / 15 / 15 train / val / test, **by vehicle ID** |
| **Plate detection mAP@0.5** | **97.89%** | **99.48%** |
| **Character recognition mAP@0.5** | **92.46%** | **98.8%** |
| Character accuracy (correct box + class) | not reported | 99.36% |
| **Whole-plate exact match** | not reported | **94.83%** (on plate crops) |
| **End-to-end: photo → correct plate text** | not reported | **94.48%** |
| Speed | "real-time", no figure given | 23 ms per photo (≈44 photos/s), detection + reading |

Both papers use the same metric definition (mAP at IoU 0.5), so the two mAP rows can be compared directly.
The largest gain is in character recognition: **+6.3 mAP points**. Most likely reasons:
- **A newer architecture:** YOLO11n vs Tiny-YOLOv3.
- **Augmentation chosen for plates:** horizontal flipping off, small rotations only.
- **Character labels cleaned before training:** see §4.

## 2. Plate detector (stage 1)

| Metric | Val | Test |
|---|---|---|
| mAP@0.5 | 0.994 | **0.995** |
| mAP@0.5:0.95 | 0.892 | 0.903 |
| Precision | 0.990 | 0.990 |
| Recall | 0.982 | 0.994 |

- No sign of overfitting: validation loss was still decreasing at the last epoch, and the test results match the validation results.
- Some of the remaining "false positives" are real background plates, which the dataset does not label.

## 3. Character recognizer (stage 2) and post-processing (stage 3)

Evaluated on the 290 labelled test plate crops (1,561 characters):

| Metric | Result |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.988 / 0.724 |
| Character accuracy | 99.36% (8 wrong class, 2 missed, 12 extra boxes) |
| Whole-plate exact match, raw | 94.48% |
| Whole-plate exact match, after post-processing | **94.83%** (val: 95.68% → 96.35%) |

**Post-processing**
1. Sort the boxes left to right.
2. Fix the alef / one confusion (أ ↔ ١) using the digit/letter boundary.
3. Drop the lowest-confidence extra boxes when a plate has more than 4 digits or 3 letters.
4. Reverse the letters.
5. Check the format.

It only fixes plates that would otherwise be wrong: no correct plate was changed.

**Most common confusions:** ص→س (2), ن→ل (2), ق/ل→ط (1 each).
These letter pairs differ only in dots or small strokes, which become 1–2 pixels on ~125×65 px crops.
The pairs expected to be hard (ب/ن, ٦/٧, ٢/٣) caused at most one error each.

## 4. End-to-end pipeline (stage 4)

The whole pipeline runs on full vehicle photos: detect, crop with the dataset's margin, read, post-process.

| | Val | Test |
|---|---|---|
| Labelled plates | 301 | 290 |
| **Read exactly right** | **97.01%** | **94.48%** |
| Photos with no plate found | 0 | 0 |

The end-to-end result is within one plate of the result on hand-cut crops.
So detection and cropping cost almost nothing, and the remaining errors come from character reading.

## 5. Fairness notes and limitations (state these in the report)

1. **Different splits.** We split by vehicle ID and train on 70% of the data; the paper split randomly and trained on 80%.
   Our setup is the stricter one: less training data, and no plate of a test car is ever seen in training.
2. **Label cleaning.** Before training, plates with incomplete labels were removed: 28 in total, **4 of them in the test split**.
   These are plates where only the letters or only the digits were boxed, or where alef was labelled as ١.
   One zero-size box (plate 1981) was also dropped.
   The paper most likely kept them.
   Worst case, counting the 4 removed test plates as failures, end-to-end accuracy would be **274/294 = 93.2%**.
3. **Label errors remain in the test set.** A manual check of the 16 end-to-end failures found at least three where the model is right and the label is wrong:
   - **1434:** the plate shows ٤٤٤٤, the label has 3 digits.
   - **0994:** an alef is visible but not labelled.
   - **0138:** 4 digits are visible, the label has 2.

   The reported 94.48% is therefore a lower bound.
   The test labels were **not** changed.
4. **The digit ٠ never occurs in EALPR.** The models cannot recognise it, so a real plate containing ٠ would be misread.
5. **One dataset, one seed.** All images come from the same source (social-media photos of Egyptian cars).
   Accuracy on other cameras, at night, or on video is untested.
   Results are from a single training run (seed 42).
   Differences of ±1 plate (≈0.3%) between runs are within normal variation.

## 6. How to reproduce

```powershell
python prepare_dataset.py --overwrite
python train_plate_detector.py
python train_char_detector.py
python eval_char_recognizer.py --weights runs/chars_det/yolo11n/weights/best.pt
python eval_pipeline.py
```

Main outputs:
- `runs/plates_det/yolo11n/` and `runs/chars_det/yolo11n/`: weights and training curves.
- `runs/chars_det/yolo11n_eval_test/`: character confusion matrix and per-plate predictions.
- `runs/pipeline/eval_test/`: end-to-end summary and annotated failure images.
