"""Two-stage transfer learning.
Stage 1: freeze the backbone and train only the new head.
Stage 2: unfreeze the last block(s) and fine-tune at a lower learning rate.
The checkpoint with the best validation F1 (spalling) is kept. The test set is never used here.
"""
import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from data import batch_to_rgb_uint8, load_data, split_train_val
from models import ARCHS, MEAN, STD, build_model, head_and_blocks
from utils import compute_metrics, get_device, save_json, set_seed


class SpallingDataset(Dataset):
    def __init__(self, X, y, indices, train):
        self.X, self.y, self.indices = X, y, np.asarray(indices)
        # mild, physically plausible augmentation only (no vertical flips, no heavy distortion)
        self.aug = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if train else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        t = torch.from_numpy(batch_to_rgb_uint8(self.X[j])).permute(2, 0, 1)  # uint8 CHW, RGB
        if self.aug is not None:
            t = self.aug(t)
        return (t.float() / 255 - MEAN[0]) / STD[0], int(self.y[j])


def keep_frozen_bn_in_eval(model, trainable):
    """BatchNorm layers whose weights are frozen should not update their running statistics."""
    ids = {id(p) for p in trainable}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(id(p) in ids for p in m.parameters(recurse=False)):
                m.eval()


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False, trainable=()):
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_bn_in_eval(model, trainable)
    total, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        amp = torch.autocast("cuda") if use_amp else contextlib.nullcontext()
        with torch.set_grad_enabled(training), amp:
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += loss.item() * len(y)
        probs.append(logits.float().softmax(1)[:, 1].detach().cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return total / len(labels), labels, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--arch", default="resnet50", choices=ARCHS)
    ap.add_argument("--out_dir", default=None, help="default: outputs/<arch>")
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true", help="random init (debugging only)")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--max_train", type=int, default=0, help="use only N training images (smoke tests)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    out_dir = Path(args.out_dir or f"outputs/{args.arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.num_workers if args.num_workers is not None else (2 if device.type == "cuda" else 0)

    X_train, y_train, _, _ = load_data(args.data_dir)
    tr_idx, va_idx = split_train_val(y_train, args.val_frac, args.seed)
    if args.max_train:
        rng = np.random.default_rng(args.seed)
        tr_idx = np.sort(rng.choice(tr_idx, min(args.max_train, len(tr_idx)), replace=False))
    np.savez(out_dir / "split.npz", train=tr_idx, val=va_idx)
    print(f"device={device} | train={len(tr_idx)} val={len(va_idx)} | arch={args.arch}")

    mk = lambda idx, train: DataLoader(
        SpallingDataset(X_train, y_train, idx, train), batch_size=args.batch_size, shuffle=train,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    train_loader, val_loader = mk(tr_idx, True), mk(va_idx, False)

    counts = np.bincount(y_train[tr_idx], minlength=2)
    weight = None if args.no_class_weights else torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print("class counts (non-spalling, spalling):", counts.tolist(), "| loss weights:", None if weight is None else weight.tolist())

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history, best_f1, best_state = [], -1.0, None
    for stage, (epochs, unfreeze) in enumerate([(args.epochs_head, False), (args.epochs_ft, True)], 1):
        if epochs == 0:
            continue
        if stage == 2 and best_state is not None:
            model.load_state_dict(best_state)          # start fine-tuning from the best head-only model
        for p in model.parameters():
            p.requires_grad = False
        head, blocks = head_and_blocks(model, args.arch)
        groups = [{"params": list(head.parameters()), "lr": args.lr_head / (3 if unfreeze else 1)}]
        if unfreeze:
            groups.append({"params": [p for b in blocks for p in b.parameters()], "lr": args.lr_ft})
        trainable = [p for g in groups for p in g["params"]]
        for p in trainable:
            p.requires_grad = True
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        n_train = sum(p.numel() for p in trainable)
        print(f"\n--- stage {stage} ({'fine-tune last block(s) + head' if unfreeze else 'head only'}) | trainable params: {n_train:,} ---")
        bad = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, ytr, ptr = run_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp, trainable)
            va_loss, yva, pva = run_epoch(model, val_loader, device, criterion)
            scheduler.step()
            tr_m, va_m = compute_metrics(ytr, ptr > 0.5), compute_metrics(yva, pva > 0.5, pva)
            history.append({"stage": stage, "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                            "train_acc": tr_m["accuracy"], "val": va_m})
            improved = va_m["f1_spalling"] > best_f1
            print(f"s{stage} ep{ep:02d} | train loss {tr_loss:.3f} acc {tr_m['accuracy']:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_m['accuracy']:.3f} F1(spalling) {va_m['f1_spalling']:.3f} "
                  f"recall {va_m['recall_spalling']:.3f} | {time.time() - t0:.0f}s{'  *best*' if improved else ''}")
            if improved:
                best_f1, bad = va_m["f1_spalling"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"arch": args.arch, "state_dict": best_state, "stage": stage, "epoch": ep,
                            "val_metrics": va_m}, out_dir / "best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stopping this stage")
                    break
            save_json({"args": vars(args), "history": history, "best_val_f1_spalling": best_f1}, out_dir / "history.json")
    save_json({"args": vars(args), "history": history, "best_val_f1_spalling": best_f1}, out_dir / "history.json")
    print(f"\nbest validation F1 (spalling): {best_f1:.4f}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
