"""Classical baselines: HOG (texture/edges) + colour histograms -> scaler -> PCA -> logistic regression / RBF-SVM.
Hyper-parameters are tuned with cross-validation on the TRAIN set only; the test set is used once at the end.
"""
import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import batch_to_rgb_uint8, load_data
from utils import compute_metrics, save_json


def features(img_bgr_or_float):
    img = batch_to_rgb_uint8(img_bgr_or_float)                       # uint8 RGB
    h = hog(rgb2gray(img), orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
    hist = np.concatenate([np.histogram(img[..., c], bins=16, range=(0, 256), density=True)[0] for c in range(3)])
    return np.concatenate([h, hist]).astype(np.float32)


def extract_all(X, n_jobs):
    return np.stack(Parallel(n_jobs=n_jobs)(delayed(features)(X[i]) for i in range(len(X))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs/baseline")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--max_train", type=int, default=0, help="subsample train (smoke tests)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.data_dir)
    cache = out / "features.npz"
    if cache.exists() and not args.max_train:
        z = np.load(cache)
        F_tr, F_te = z["train"], z["test"]
    else:
        if args.max_train:
            # NB: the label files are sorted by class, so never slice "the first N" -- sample at random
            sel = np.sort(np.random.default_rng(0).choice(len(y_train), args.max_train, replace=False))
            X_train, y_train = X_train[sel], y_train[sel]
        print("extracting HOG + colour features ...")
        F_tr, F_te = extract_all(X_train, args.n_jobs), extract_all(X_test, args.n_jobs)
        if not args.max_train:
            np.savez(cache, train=F_tr, test=F_te)
    print("feature matrix:", F_tr.shape)

    n_pca = min(200, F_tr.shape[0] - 1, F_tr.shape[1])
    candidates = {
        "logreg": (LogisticRegression(max_iter=3000, class_weight="balanced"), {"clf__C": [0.01, 0.1, 1]}),
        "svm_rbf": (SVC(kernel="rbf", class_weight="balanced"), {"clf__C": [1, 10], "clf__gamma": ["scale", 1e-4]}),
    }
    results = {}
    for name, (clf, grid) in candidates.items():
        pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=n_pca, random_state=0)), ("clf", clf)])
        cv = StratifiedKFold(3, shuffle=True, random_state=0)   # shuffled: the data files are sorted by class
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=args.n_jobs).fit(F_tr, y_train)
        score = gs.decision_function(F_te)
        m = compute_metrics(y_test, (score > 0).astype(int), score)
        m.update(best_params={k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()},
                 cv_f1=float(gs.best_score_))
        results[name] = m
        np.save(out / f"{name}_test_scores.npy", score)
        print(f"{name}: cv F1 {gs.best_score_:.3f} | TEST acc {m['accuracy']:.3f} F1(spalling) {m['f1_spalling']:.3f} "
              f"recall {m['recall_spalling']:.3f} AUC {m.get('roc_auc', float('nan')):.3f} | {gs.best_params_}")
    save_json(results, out / "test_metrics.json")


if __name__ == "__main__":
    main()
