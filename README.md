# Spalling Detection on PEER Hub ImageNet (Φ-Net), Task 3

Binary image classification: **0 = non-spalling, 1 = spalling** (material loss on structural surfaces).
Course mini-project for **UE24CS352A – Machine Learning**.

## Team

| Name | SRN | Contribution |
|---|---|---|
| Sudhanwa | PES2UG24CS531 | _(fill in, e.g. data pipeline, baseline, ...)_ |
| S Bindu | PES2UG25CS805 | _(fill in, e.g. transfer learning, Grad-CAM, demo, ...)_ |

## Quick demo

```bash
git clone <this-repo-url> && cd <repo-folder>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download best.pt (link in "Downloads" below) and place it at outputs/resnet50/best.pt
python src/app.py --ckpt outputs/resnet50/best.pt      # opens http://127.0.0.1:7860
```

Sample test images for trying the demo are in `sample_images/`.

## Downloads

| File | Where to get it | Where to put it |
|---|---|---|
| Trained ResNet50 weights `best.pt` (~95 MB) | _<paste GitHub Release / Google Drive link>_ | `outputs/resnet50/best.pt` |
| Prepared dataset (uint8 `.npy` files, ~1 GB) | _<paste Google Drive link>_ | `data/` |
| Original dataset | https://apps.peer.berkeley.edu/phi-net/ (Task 3: Spalling condition) | any folder, e.g. `~/Downloads/task3` |

The dataset and large model files are **not stored in this repository** (see `.gitignore`).

## Dataset

Φ-Net Task 3 (Gao & Mosalam, PEER Report 2019/07; https://apps.peer.berkeley.edu/phi-net/).
Dataset author: Yuqing Gao. Licensed under **CC BY-NC-SA 4.0** (non-commercial, attribution required, share-alike).
See `license.txt` for the full licence text and `DATASET_README.txt` for the original dataset notes.

| split | images | non-spalling | spalling |
|---|---|---|---|
| train (official) | 6,898 | 4,294 | 2,604 (37.8%) |
| test (official) | 837 | 527 | 310 (37.0%) |

Format of the downloaded `.npy` files: float32, 224x224x3, **BGR order with the ImageNet channel means already
subtracted** (Keras "caffe" style); labels are one-hot. `src/prepare_data.py` converts images losslessly to uint8
(4x smaller). **Both label files are sorted by class** (all 0s, then all 1s), so never take "the first N" images or use
un-shuffled folds. The code always uses stratified, shuffled splits.

No data files are modified in content other than the lossless uint8 conversion described above.

## Setup (Mac)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py --src ~/Downloads/task3 --dst data     # -> data/task3_*_u8.npy + labels
```

If you downloaded the prepared uint8 files from the link above, skip `prepare_data.py` and place them directly in `data/`.

## Method

1. **Splits:** official test set kept untouched until the end. 15% of train held out (stratified) for validation / early stopping.
2. **Classical baseline** (`baseline.py`): HOG + colour histograms -> standardise -> PCA(200) -> logistic regression or RBF-SVM, tuned by 3-fold CV on train only.
3. **Transfer learning** (`train.py`): ImageNet-pretrained ResNet50 (also EfficientNet-B0, MobileNetV2) with a new dropout + linear head.
   Stage 1 trains the head only; stage 2 fine-tunes the last block(s) at a lower learning rate (AdamW, cosine schedule).
   Class-weighted cross-entropy for the 38/62 imbalance, mild augmentation (flip, small affine, brightness/contrast), early stopping and
   model selection on validation F1 (spalling).
4. **Evaluation** (`evaluate.py`): accuracy, precision/recall/F1 for spalling, macro-F1, ROC-AUC, confusion matrix, and Grad-CAM on the most confident mistakes.
5. **Demo** (`app.py`): Gradio app, upload a photo -> spalling probability + Grad-CAM.

## Results (official test set, evaluated once per final model)

| Model | Accuracy | Spalling precision | Spalling recall | Spalling F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| HOG + SVM / LogReg (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ResNet50 (fine-tuned) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| EfficientNet-B0 (optional) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MobileNetV2 (optional) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Confusion matrix and Grad-CAM examples are saved in `outputs/` (small `.png` / `.json` result files are included in the repo).

## Run on Google Colab (GPU: Runtime -> Change runtime type -> T4)

Put the repo folder and the 4 data files (`*_u8.npy` x2 + `y_*.npy` x2) in Google Drive, then:

```python
from google.colab import drive; drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/ml_project/data /content/data          # local disk is much faster than Drive
%cd /content/drive/MyDrive/ml_project/spalling-detection
!pip install -q scikit-image
OUT = "/content/drive/MyDrive/ml_project/outputs"                    # results survive a Colab disconnect
!python src/baseline.py --data_dir /content/data --out_dir {OUT}/baseline
!python src/train.py    --data_dir /content/data --arch resnet50 --out_dir {OUT}/resnet50
!python src/evaluate.py --ckpt {OUT}/resnet50/best.pt --data_dir /content/data
```

Optional comparisons: `--arch efficientnet_b0`, `--arch mobilenet_v2`. Evaluate the test set **once per final model**; don't tune on it.

## Demo on the Mac

Download `best.pt` (see Downloads) to `outputs/resnet50/best.pt`, then:

```bash
python src/export_examples.py --data_dir data          # a few test images for the example gallery
python src/app.py --ckpt outputs/resnet50/best.pt      # opens http://127.0.0.1:7860
```

## Repository layout

```
README.md             this file
license.txt           CC BY-NC-SA 4.0 licence text (dataset licence)
DATASET_README.txt    original dataset notes (label registration, author)
requirements.txt      Python dependencies
.gitignore            excludes data/, *.npy, *.pt, *.pkl, .venv/
sample_images/        a few test images for the demo
outputs/              small result files (metrics, plots)
src/prepare_data.py   convert float32 BGR-mean-subtracted -> uint8
src/data.py           loading (memory-mapped), stratified split, BGR->RGB
src/baseline.py       HOG + SVM / logistic regression
src/train.py          two-stage transfer learning
src/evaluate.py       test metrics, confusion matrix, Grad-CAM on errors
src/gradcam.py        Grad-CAM implementation
src/app.py            Gradio demo
src/export_examples.py
```

## Known limitations

- Φ-Net images come from many sources; near-duplicate images may exist across train/val/test, which would inflate scores. Not checked.
- Labels are about local material loss, not overall severity: heavily collapsed buildings can be labelled non-spalling, and a hairline crack can be too.
- Single seed, single split; variance can be estimated by re-running with `--seed`.
- Ho & Troncoso (CS229, 2018) use the same image collection for a different task (damaged vs undamaged), so their numbers are **not** a benchmark for this task.

## References

- Gao, Y. & Mosalam, K. M. (2019). *PEER Hub ImageNet (Φ-Net): A Large-Scale Multi-Attribute Benchmark Dataset of Structural Images.* PEER Report 2019/07.
- Ho, M. & Troncoso, J. (2018). *Structural Damage Image Classification.* Stanford CS229 project report.
