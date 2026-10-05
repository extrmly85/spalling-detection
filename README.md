# Spalling detection on PEER Hub ImageNet (Φ-Net), Task 3

Binary image classification: **0 = non-spalling, 1 = spalling** (material loss on structural surfaces).
Course mini-project (UE24CS352A Machine Learning). Related work: Ho & Troncoso, *Structural Damage Image
Classification* (CS229, 2018), which uses the same image collection for a different task (damaged vs undamaged),
so its numbers are **not** a benchmark for this task.

## Dataset
Φ-Net Task 3 (Gao & Mosalam, PEER Report 2019/07; https://apps.peer.berkeley.edu/phi-net/), licence CC BY-NC-SA 4.0
(non-commercial, attribution required). **The data is not included in this repo** (see `.gitignore`).

| split | images | non-spalling | spalling |
|---|---|---|---|
| train (official) | 6,898 | 4,294 | 2,604 (37.8%) |
| test (official) | 837 | 527 | 310 (37.0%) |

Format of the downloaded `.npy` files: float32, 224x224x3, **BGR order with the ImageNet channel means already
subtracted** (Keras "caffe" style); labels are one-hot. `src/prepare_data.py` converts images losslessly to uint8
(4x smaller). **Both label files are sorted by class** (all 0s, then all 1s), so never take "the first N" images or use
un-shuffled folds. The code always uses stratified, shuffled splits.

## Setup (Mac)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py --src ~/Downloads/task3 --dst data     # -> data/task3_*_u8.npy + labels
```

## Method
1. **Splits:** official test set kept untouched until the end. 15% of train held out (stratified) for validation / early stopping.
2. **Classical baseline** (`baseline.py`): HOG + colour histograms -> standardise -> PCA(200) -> logistic regression or RBF-SVM, tuned by 3-fold CV on train only.
3. **Transfer learning** (`train.py`): ImageNet-pretrained ResNet50 (also EfficientNet-B0, MobileNetV2) with a new dropout + linear head.
   Stage 1 trains the head only; stage 2 fine-tunes the last block(s) at a lower learning rate (AdamW, cosine schedule).
   Class-weighted cross-entropy for the 38/62 imbalance, mild augmentation (flip, small affine, brightness/contrast), early stopping and
   model selection on validation F1 (spalling).
4. **Evaluation** (`evaluate.py`): accuracy, precision/recall/F1 for spalling, macro-F1, ROC-AUC, confusion matrix, and Grad-CAM on the most confident mistakes.
5. **Demo** (`app.py`): Gradio app, upload a photo -> spalling probability + Grad-CAM.

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
Download `best.pt` from Drive to `outputs/resnet50/best.pt`, then:
```bash
python src/export_examples.py --data_dir data          # a few test images for the example gallery
python src/app.py --ckpt outputs/resnet50/best.pt      # opens http://127.0.0.1:7860
```

## Layout
```
src/prepare_data.py   convert float32 BGR-mean-subtracted -> uint8
src/data.py           loading (memory-mapped), stratified split, BGR->RGB
src/baseline.py       HOG + SVM / logistic regression
src/train.py          two-stage transfer learning
src/evaluate.py       test metrics, confusion matrix, Grad-CAM on errors
src/gradcam.py        Grad-CAM implementation
src/app.py            Gradio demo
src/export_examples.py
```

## Known limitations (state these in the write-up)
- Φ-Net images come from many sources; near-duplicate images may exist across train/val/test, which would inflate scores. Not checked.
- Labels are about local material loss, not overall severity: heavily collapsed buildings can be labelled non-spalling, and a hairline crack can be too.
- Single seed, single split; report variance if time allows (re-run with `--seed`).

## References
- Gao, Y. & Mosalam, K. M. (2019). *PEER Hub ImageNet (Φ-Net): A Large-Scale Multi-Attribute Benchmark Dataset of Structural Images.* PEER Report 2019/07.
- Ho, M. & Troncoso, J. (2018). *Structural Damage Image Classification.* Stanford CS229 project report.
