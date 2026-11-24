# Egyptian Licence Plate Detection and Recognition

Graduation project: read Egyptian licence plates (Arabic letters and Arabic-Indic digits) from car photos, using two YOLO11n models trained on the public **EALPR** dataset.

```
car photo ──► plate detector ──► plate crop ──► character detector ──► post-processing ──► "٩١٦٢ ص ب ف"
              (YOLO11n, 640)                    (YOLO11n, 320, 27 classes)
```

## Results (test split, split by vehicle)

| | EALPR authors (Tiny-YOLOv3) | This project (YOLO11n) |
|---|---|---|
| Plate detection mAP@0.5 | 97.89% | **99.48%** |
| Character recognition mAP@0.5 | 92.46% | **98.8%** |
| Character accuracy | – | 99.36% |
| End-to-end: photo → exact plate text | – | **94.48%** |
| Speed (RTX 3050 Laptop) | – | ~23 ms per photo |

See [RESULTS.md](RESULTS.md) for the full comparison, confusion analysis and limitations.

## File map

```
yolo11_ealpr/
├── README.md                     this file
├── RESULTS.md                    full results, comparison with the EALPR paper, limitations
├── requirements.txt              Python packages
├── .gitignore                    keeps the dataset and large outputs out of git
│
├── prepare_dataset.py            step 0: validate + split EALPR by vehicle ID -> prepared/
├── train_plate_detector.py       step 1: train + test the plate detector
├── train_char_detector.py        step 2: train + test the character detector
├── eval_char_recognizer.py       step 2: character / whole-plate accuracy on plate crops
├── plate_reader.py               steps 3-4: post-processing + full pipeline (PlateReader, CLI)
├── eval_pipeline.py              step 5: end-to-end accuracy on vehicle photos
├── plate_gui.py                  desktop app: open a photo, read the plate
│
└── runs/                         outputs of the runs reported in RESULTS.md
    ├── plates_det/
    │   ├── yolo11n/weights/best.pt      TRAINED PLATE DETECTOR (used by the pipeline)
    │   ├── yolo11n/                     training settings (args.yaml), curves, results.csv/png
    │   └── yolo11n_test/                PR/F1 curves + confusion matrix on the test split
    ├── chars_det/
    │   ├── yolo11n/weights/best.pt      TRAINED CHARACTER DETECTOR (used by the pipeline)
    │   ├── yolo11n/                     training settings, curves, results.csv/png
    │   ├── yolo11n_test/                YOLO curves + confusion matrix on the test split
    │   ├── yolo11n_eval_test/           reading accuracy: summary.txt, per-plate CSV, confusion matrix
    │   ├── yolo11n_eval_val/            the same on the validation split
    │   └── yolo11n_eval/                first evaluation run (before post-processing was added)
    └── pipeline/
        ├── eval_test/                   end-to-end summary.txt + per_image.csv (test split)
        └── eval_val/                    the same on the validation split
```

Not in git (created when you run the scripts): `EALPR-master/` (the dataset), `prepared/` (built by `prepare_dataset.py`), and the annotated photos.

## Project files

| File | What it does |
|---|---|
| `prepare_dataset.py` | Validates and pairs the EALPR files, splits 70/15/15 **by vehicle ID** (no leakage), and writes YOLO datasets to `prepared/` plus a report |
| `train_plate_detector.py` | Trains the plate detector and evaluates it on the test split |
| `train_char_detector.py` | Trains the character detector (27 classes) |
| `eval_char_recognizer.py` | Character accuracy, whole-plate accuracy and confusion matrix on plate crops |
| `plate_reader.py` | The full pipeline (`PlateReader`) and the post-processing (reading order, أ/١ fix, format check); also works from the command line |
| `eval_pipeline.py` | End-to-end accuracy on whole vehicle photos |
| `plate_gui.py` | Desktop app: open a photo and see the plate read |
| `runs/plates_det/yolo11n/weights/best.pt` | Trained plate detector |
| `runs/chars_det/yolo11n/weights/best.pt` | Trained character detector |

## Quick start (use the trained models)

```bash
cd yolo11_ealpr
pip install -r requirements.txt
python plate_gui.py                    # desktop app
python plate_reader.py path/to/car.jpg # command line; annotated copies go to runs/pipeline/annotated/
```

Or from Python:

```python
from plate_reader import PlateReader
reader = PlateReader()
for plate in reader.read("car.jpg"):
    print(plate["text"], plate["valid"])
```

## Reproduce the training

1. Download the EALPR dataset (Youssef et al., ACIRS 2022) and place it in `yolo11_ealpr/EALPR-master/`, with the folders `EALPR Vechicles dataset`, `EALPR- Plates dataset` and `EALPR- LP characters dataset`.
   The dataset is not included in this repository.
2. Run:

```bash
python prepare_dataset.py
python train_plate_detector.py        # ~55 min on an RTX 3050 Laptop
python train_char_detector.py         # ~10 min
python eval_char_recognizer.py --weights runs/chars_det/yolo11n/weights/best.pt
python eval_pipeline.py
```

On Windows with 16 GB of RAM, keep `--workers` at 1. Each data-loader worker is a separate process that loads PyTorch again and needs about 1–1.5 GB.

## Key decisions

- **Reading order:** the boxes are sorted by x. Digits are read left→right; letters are read right→left. Urdu heh (U+06BE) is normalised to ه.
- **No horizontal flipping during training:** mirrored Arabic characters don't exist on real plates. Rotation is limited to ±5°.
- **Label cleaning:** 28 plates with incomplete character labels and one zero-size box were excluded. The details are in `prepared/report.txt`, which `prepare_dataset.py` writes.
- **Known limitation:** the digit ٠ never appears in EALPR, so the models can't read it.

## Dataset citation

```
@INPROCEEDINGS{9845514,
  author={Youssef, Ahmed Ramadan and Sayed, Fawzya Ramadan and Ali, Abdelmgeid Ameen},
  booktitle={2022 7th Asia-Pacific Conference on Intelligent Robot Systems (ACIRS)},
  title={A New Benchmark Dataset for Egyptian License Plate Detection and Recognition},
  year={2022}, pages={106-111}, doi={10.1109/ACIRS55390.2022.9845514}}
```
